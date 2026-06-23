"""De-risk the Stage-6 character-level decoder swap (T6).

The swap keeps TrOCR's pretrained ViT encoder and 12 transformer decoder layers
but replaces the 50 265-token BPE embedding / lm_head with a 120-token character
vocabulary.  If the embedding, the tied lm_head, or the decoder-start token are
wired up wrong, the model *cannot* learn — so we prove the wiring by overfitting
a single tiny batch to near-zero teacher-forced cross-entropy.  A loss that
refuses to drop below the threshold is a loud, early signal that the surgery is
broken, before we spend GPU-days on real training.

Marked ``slow``: needs torch, the cached base model, and (in practice) a GPU —
the loop fine-tunes the full TrOCR-base on every step.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from maltese_ocr.render import RenderConfig, load_fonts, render  # noqa: E402
from maltese_ocr.train.char_tokenizer import CharTokenizer  # noqa: E402
from maltese_ocr.train.decoder_swap import DEFAULT_BASE, load_char_trocr  # noqa: E402

# Short Maltese phrases (every character lives in configs/charset.txt), chosen to
# exercise the diacritics ħ ġ ż ċ that the BPE vocab handled very differently.
SAMPLE_TEXTS = [
    "Il-kelb tal-baħar",
    "Ix-xemx tielgħa",
    "Għawdex u Malta",
    "Ħamsa u għoxrin",
    "Iż-żiffa friska",
    "Ċar bħall-kristall",
    "Tagħlim kontinwu",
    "Ġmiel tan-natura",
]

EXPECTED_VOCAB = 120  # 117 charset chars + <pad> + <bos> + <eos>
OVERFIT_STEPS = 200
LOSS_THRESHOLD = 0.05
LEARNING_RATE = 5e-4


def _image_processor(base: str = DEFAULT_BASE):
    from transformers import TrOCRProcessor

    return TrOCRProcessor.from_pretrained(base)


def _build_batch(texts, tokenizer, processor, device):
    """Render each text once and build padded (decoder_input, label) pairs.

    Teacher forcing: ``decoder_input = <bos> c1..cN`` predicts ``label = c1..cN
    <eos>``; both are right-padded with <pad> and the loss ignores <pad>.
    """
    fonts = load_fonts()
    rng = random.Random(0)
    pixel_values = []
    for text in texts:
        cfg = RenderConfig.sample(fonts[0], rng, augment=False)
        cfg.image_width = 600
        img, _, _ = render(text, cfg, p_hyphen=0.0, rng=rng)
        px = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values[0]
        pixel_values.append(px)
    pixel_values = torch.stack(pixel_values).to(device)

    seqs = [tokenizer.encode(t) for t in texts]
    max_len = max(len(s) for s in seqs) + 1  # room for the leading <bos> / trailing <eos>
    pad = tokenizer.pad_token_id
    dec_in, labels = [], []
    for s in seqs:
        di = [tokenizer.bos_token_id, *s]
        lab = [*s, tokenizer.eos_token_id]
        di += [pad] * (max_len - len(di))
        lab += [pad] * (max_len - len(lab))
        dec_in.append(di)
        labels.append(lab)
    return (
        pixel_values,
        torch.tensor(dec_in, device=device),
        torch.tensor(labels, device=device),
    )


@pytest.mark.slow
def test_decoder_swap_overfits_tiny_batch(capsys):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_char_trocr()
    model.to(device)
    # Disable dropout so the overfit is deterministic and can actually reach ~0;
    # eval() leaves autograd fully active, we just want training=False everywhere.
    model.eval()

    # --- wiring sanity: vocab size, config ids, embedding/lm_head shapes -----
    assert tokenizer.vocab_size == EXPECTED_VOCAB, tokenizer.vocab_size
    assert model.config.decoder.vocab_size == EXPECTED_VOCAB
    assert model.config.decoder_start_token_id == tokenizer.bos_token_id
    assert model.config.pad_token_id == tokenizer.pad_token_id
    assert model.config.eos_token_id == tokenizer.eos_token_id
    emb = model.decoder.get_input_embeddings().weight
    head = model.decoder.get_output_embeddings().weight
    assert tuple(emb.shape) == (EXPECTED_VOCAB, 1024)
    assert tuple(head.shape) == (EXPECTED_VOCAB, 1024)
    assert head.data_ptr() == emb.data_ptr(), "lm_head must be tied to the embedding"

    processor = _image_processor()
    pixel_values, dec_in, labels = _build_batch(SAMPLE_TEXTS, tokenizer, processor, device)

    # --- forward pass produces [B, T, new_vocab] logits ----------------------
    with torch.no_grad():
        logits = model(pixel_values=pixel_values, decoder_input_ids=dec_in).logits
    assert tuple(logits.shape) == (len(SAMPLE_TEXTS), dec_in.shape[1], EXPECTED_VOCAB)

    # --- overfit the single batch -------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    pad = tokenizer.pad_token_id
    curve: list[tuple[int, float]] = []
    final_loss = float("inf")
    for step in range(OVERFIT_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values=pixel_values, decoder_input_ids=dec_in).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, EXPECTED_VOCAB),
            labels.reshape(-1),
            ignore_index=pad,
        )
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if step % 20 == 0 or step == OVERFIT_STEPS - 1:
            curve.append((step, final_loss))
            with capsys.disabled():
                print(f"step {step:4d}  loss {final_loss:.4f}")

    assert final_loss < LOSS_THRESHOLD, (
        f"decoder swap failed to overfit: final loss {final_loss:.4f} >= {LOSS_THRESHOLD}. "
        f"The embedding / tied lm_head / decoder_start_token_id wiring is likely wrong. "
        f"Loss curve: {curve}"
    )

    # --- generate() runs and emits only char-vocab ids ----------------------
    with torch.no_grad():
        generated = model.generate(pixel_values=pixel_values[:2], max_new_tokens=40)
    assert generated.dtype == torch.long
    assert int(generated.min()) >= 0
    assert int(generated.max()) < EXPECTED_VOCAB, "generate() emitted an out-of-vocab id"


def test_char_tokenizer_vocab_size_and_roundtrip(tmp_path):
    """Light, fast guard (no model): vocab size, special ids, encode/decode, save/load."""
    tok = CharTokenizer.from_charset_file()
    assert tok.vocab_size == EXPECTED_VOCAB
    assert (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) == (0, 1, 2)

    text = "Il-baħar"
    ids = tok.encode(text, add_special_tokens=True)
    assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id
    assert tok.decode(ids) == text

    path = tmp_path / "char_tokenizer.json"
    tok.save(path)
    reloaded = CharTokenizer.load(path)
    assert reloaded.id_to_token == tok.id_to_token
    assert reloaded.encode(text, add_special_tokens=True) == ids
