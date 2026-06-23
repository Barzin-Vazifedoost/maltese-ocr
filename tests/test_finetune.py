"""Tests for the Stage-2 supervised fine-tune loop (T6).

Two fast, model-free guards (the CER metric and the encoder-checkpoint hook) run
under ``make test``; the two heavy guards (a forward pass and a tiny overfit on
the real char-vocab TrOCR) are marked ``slow`` and need the cached base model.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from maltese_ocr.train.run import (  # noqa: E402
    compute_cer,
    load_encoder_checkpoint,
    train_step,
)

EXPECTED_VOCAB = 120  # 117 charset chars + <pad>/<bos>/<eos>

# Short Maltese phrases whose every character is in configs/charset.txt.
SAMPLE_TEXTS = ["Il-kelb tal-baħar", "Għawdex u Malta", "Iż-żiffa friska"]


# ---------------------------------------------------------------------------
# Fast: CER metric
# ---------------------------------------------------------------------------


def test_compute_cer_matches_hand_value():
    # "hello" -> "hallo": exactly one substitution over 5 reference chars = 0.2.
    cer = compute_cer(["hello"], ["hallo"])
    assert isinstance(cer, float)
    assert 0.0 <= cer <= 2.0
    assert cer == pytest.approx(0.2, abs=1e-9)

    # Perfect match is 0.0; empty input is defined as 0.0 (no division by zero).
    assert compute_cer(["malta"], ["malta"]) == pytest.approx(0.0)
    assert compute_cer([], []) == 0.0


# ---------------------------------------------------------------------------
# Fast: encoder-checkpoint hook
# ---------------------------------------------------------------------------


class _FakeModel(nn.Module):
    """A stand-in with a small ``.encoder`` so the loader can be tested cheaply."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)


def _zeroed_encoder() -> nn.Linear:
    enc = nn.Linear(4, 4)
    with torch.no_grad():
        for p in enc.parameters():
            p.zero_()
    return enc


def test_load_encoder_checkpoint_raw_state_dict(tmp_path):
    path = tmp_path / "encoder.pt"
    torch.save(_zeroed_encoder().state_dict(), path)

    model = _FakeModel()
    assert not torch.all(model.encoder.weight == 0)  # fresh init is non-zero

    missing, unexpected = load_encoder_checkpoint(model, path)
    assert missing == [] and unexpected == []
    assert torch.all(model.encoder.weight == 0)
    assert torch.all(model.encoder.bias == 0)


def test_load_encoder_checkpoint_full_checkpoint(tmp_path):
    """A full SeqCLR checkpoint: only the encoder.* sub-state is applied."""
    enc_state = _zeroed_encoder().state_dict()
    ckpt = {
        "step": 5000,
        "model": {f"encoder.{k}": v for k, v in enc_state.items()}
        | {"head.mlp.weight": torch.randn(3, 3)},  # non-encoder keys must be ignored
        "optimizer": {},
        "scheduler": None,
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(ckpt, path)

    model = _FakeModel()
    missing, unexpected = load_encoder_checkpoint(model, path)
    assert missing == [] and unexpected == []  # head.* filtered out before load
    assert torch.all(model.encoder.weight == 0)


# ---------------------------------------------------------------------------
# Slow: real model forward + tiny overfit
# ---------------------------------------------------------------------------


def _build_batch(texts, tokenizer, processor, device):
    """Render each text once and build padded teacher-forced tensors."""
    from maltese_ocr.render import RenderConfig, load_fonts, render

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

    seqs = [tokenizer.encode(t, add_special_tokens=True) for t in texts]  # <bos>..<eos>
    pad = tokenizer.pad_token_id
    dec_in = [s[:-1] for s in seqs]
    labels = [s[1:] for s in seqs]
    max_len = max(len(d) for d in dec_in)
    to_tensor = lambda rows: torch.tensor(  # noqa: E731
        [r + [pad] * (max_len - len(r)) for r in rows], dtype=torch.long, device=device
    )
    return pixel_values, to_tensor(dec_in), to_tensor(labels)


@pytest.mark.slow
def test_forward_pass_emits_char_vocab_logits():
    pytest.importorskip("transformers")
    from transformers import TrOCRProcessor

    from maltese_ocr.train.decoder_swap import DEFAULT_BASE, load_char_trocr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_char_trocr()
    model.to(device).eval()
    assert tokenizer.vocab_size == EXPECTED_VOCAB

    processor = TrOCRProcessor.from_pretrained(DEFAULT_BASE)
    pixel_values, dec_in, _ = _build_batch(SAMPLE_TEXTS, tokenizer, processor, device)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values, decoder_input_ids=dec_in).logits
    assert tuple(logits.shape) == (len(SAMPLE_TEXTS), dec_in.shape[1], EXPECTED_VOCAB)


@pytest.mark.slow
def test_train_step_drives_loss_down(capsys):
    """30 steps of the real train_step should clearly reduce CE (sanity, not the gate)."""
    pytest.importorskip("transformers")
    from transformers import TrOCRProcessor

    from maltese_ocr.train.decoder_swap import DEFAULT_BASE, load_char_trocr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_char_trocr()
    model.to(device)

    processor = TrOCRProcessor.from_pretrained(DEFAULT_BASE)
    pixel_values, dec_in, labels = _build_batch(SAMPLE_TEXTS, tokenizer, processor, device)

    # LR 5e-5 is the project's validated fine-tune rate (CLAUDE.md T6 note).
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    losses = []
    for step in range(30):
        loss = train_step(
            model,
            optimizer,
            None,
            pixel_values,
            dec_in,
            labels,
            pad_id=tokenizer.pad_token_id,
            vocab_size=EXPECTED_VOCAB,
            device_type=device.type,
        )
        losses.append(float(loss))
    with capsys.disabled():
        print(f"first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f} min={min(losses):.4f}")

    assert losses[-1] < losses[0], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert losses[0] - min(losses) > 0.2, f"insufficient learning over 30 steps: {losses}"
