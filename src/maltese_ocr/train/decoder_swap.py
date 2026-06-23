"""Swap TrOCR's BPE decoder vocabulary for the Maltese char vocab (T6).

This is the highest-risk surgery in the staged pipeline.  We start from
``microsoft/trocr-base-printed`` (ViT-384 encoder + 12-layer transformer decoder,
``d_model=1024``, ``cross_attention_hidden_size=768`` already bridging the
768-dim encoder to the 1024-dim decoder) and keep *all* pretrained weights
except the decoder vocabulary projection.  We replace only:

  * the decoder token embedding  ->  ``[vocab x 1024]``
  * the tied output projection    ->  ``[1024 x vocab]``

with freshly random-initialised (normal, ``std=0.02``) weights for the 120-token
character vocabulary, and we set consistent decoder-start / bos / eos / pad ids.
The pretrained self-attention, cross-attention and FFN layers are untouched.

``tests/test_decoder_swap.py`` proves the wiring is correct by overfitting a
tiny batch to near-zero teacher-forced cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from maltese_ocr.train.char_tokenizer import CharTokenizer

DEFAULT_BASE = "microsoft/trocr-base-printed"
INIT_STD = 0.02  # matches the decoder's config.init_std


def swap_decoder_vocab(model, tokenizer: CharTokenizer):
    """Replace the decoder embedding + tied lm_head with a fresh char vocab.

    Operates in place on a loaded ``VisionEncoderDecoderModel`` and returns it.
    The decoder transformer layers (and the encoder) are left exactly as loaded.
    """
    decoder = model.decoder  # TrOCRForCausalLM
    new_vocab = tokenizer.vocab_size

    old_emb = decoder.get_input_embeddings()
    d_model = old_emb.embedding_dim
    expected = int(model.config.decoder.d_model)
    if d_model != expected:
        raise RuntimeError(f"decoder embedding dim {d_model} != config d_model {expected}")

    # Resize the input + (tied) output embeddings to the char vocab.  This keeps
    # the embedding subclass and the input<->output tie, and updates the decoder
    # config's vocab_size for us.
    decoder.resize_token_embeddings(new_vocab)

    # The char vocab is unrelated to the BPE vocab, so *every* row is new.
    # Re-initialise from scratch rather than keeping the first ``new_vocab`` BPE
    # rows that resize_token_embeddings copied over.
    new_emb = decoder.get_input_embeddings()
    nn.init.normal_(new_emb.weight, mean=0.0, std=INIT_STD)
    new_emb.padding_idx = tokenizer.pad_token_id
    with torch.no_grad():
        new_emb.weight[tokenizer.pad_token_id].zero_()

    # Re-tie so the output projection shares the re-initialised embedding weight.
    decoder.tie_weights()

    _verify_swap(decoder, new_vocab, d_model)
    _set_token_ids(model, decoder, tokenizer, new_vocab)
    return model


def _verify_swap(decoder, new_vocab: int, d_model: int) -> None:
    """Fail loudly if the embedding / lm_head shapes or tie are wrong."""
    emb = decoder.get_input_embeddings()
    head = decoder.get_output_embeddings()
    if tuple(emb.weight.shape) != (new_vocab, d_model):
        raise RuntimeError(f"embedding shape {tuple(emb.weight.shape)} != ({new_vocab}, {d_model})")
    if tuple(head.weight.shape) != (new_vocab, d_model):
        raise RuntimeError(f"lm_head shape {tuple(head.weight.shape)} != ({new_vocab}, {d_model})")
    if head.weight.data_ptr() != emb.weight.data_ptr():
        raise RuntimeError("output projection is not tied to the input embedding")


def _set_token_ids(model, decoder, tokenizer: CharTokenizer, new_vocab: int) -> None:
    """Propagate the char-vocab ids to every config that generation consults."""
    ids = {
        "decoder_start_token_id": tokenizer.bos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    for cfg in (model.config, model.config.decoder, decoder.config):
        for key, value in ids.items():
            setattr(cfg, key, value)
        cfg.vocab_size = new_vocab

    gen = getattr(model, "generation_config", None)
    if gen is not None:
        for key, value in ids.items():
            setattr(gen, key, value)


def load_char_trocr(
    tokenizer: CharTokenizer | None = None,
    *,
    base: str = DEFAULT_BASE,
    charset_path: str | None = None,
):
    """Load the base TrOCR model and swap in the char vocab.

    Returns ``(model, tokenizer)``.  Builds the tokenizer from
    ``configs/charset.txt`` (or ``charset_path``) when one is not supplied.
    """
    from transformers import VisionEncoderDecoderModel

    if tokenizer is None:
        tokenizer = (
            CharTokenizer.from_charset_file(charset_path)
            if charset_path is not None
            else CharTokenizer.from_charset_file()
        )
    model = VisionEncoderDecoderModel.from_pretrained(base)
    swap_decoder_vocab(model, tokenizer)
    return model, tokenizer
