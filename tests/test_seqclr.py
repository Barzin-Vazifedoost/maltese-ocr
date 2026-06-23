"""Unit tests for SeqCLR contrastive pretraining (T5, Stage 1).

Tests 1-4 run entirely on CPU with a tiny stand-in encoder, so they need no
network and no model download.  Test 5 exercises the real TrOCR processor and is
marked ``slow`` (skipped by ``make test`` / ``pytest -m "not slow"``).
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from maltese_ocr.pretrain.seqclr import (
    SeqCLRHead,
    SeqCLRModel,
    load_checkpoint,
    nt_xent_loss,
    save_checkpoint,
    train_step,
)

# Tiny geometry so the tests are fast: an 8x8 patch grid (64 patches) from a
# 32x32 image with a 4-px patch, hidden size 32.
HIDDEN = 32
GRID = 8
N_PATCHES = GRID * GRID
IMG = 32
PATCH = 4


class DummyEncoder(nn.Module):
    """Trainable ViT-patch-embed stand-in: pixels -> (B, 1 + N_PATCHES, HIDDEN)."""

    def __init__(self, hidden: int = HIDDEN, patch: int = PATCH) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        x = self.proj(pixel_values)  # (B, hidden, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, N_PATCHES, hidden)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + N_PATCHES, hidden)
        return SimpleNamespace(last_hidden_state=x)


def _head(n_windows: int = 4, proj_dim: int = 16) -> SeqCLRHead:
    return SeqCLRHead(
        hidden_dim=HIDDEN,
        n_windows=n_windows,
        proj_dim=proj_dim,
        proj_hidden=24,
        n_patches=N_PATCHES,
    )


def _model() -> SeqCLRModel:
    return SeqCLRModel(DummyEncoder(), _head())


def _pixels(batch: int, *, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, IMG, IMG, generator=gen)


# ---------------------------------------------------------------------------
# 1. Head output shape + L2 norm
# ---------------------------------------------------------------------------


def test_head_output_shape_and_l2_norm():
    head = _head(n_windows=4, proj_dim=16)
    # 1 special token (e.g. ViT CLS) and 2 special tokens (e.g. DeiT) both work.
    for n_special in (1, 2):
        hidden = torch.randn(3, n_special + N_PATCHES, HIDDEN)
        z = head(hidden)
        assert z.shape == (3, 4, 16)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# 2. NT-Xent loss is a finite positive scalar
# ---------------------------------------------------------------------------


def test_nt_xent_loss_scalar_finite_positive():
    torch.manual_seed(0)
    z_a = torch.nn.functional.normalize(torch.randn(6, 4, 16), dim=-1)
    z_b = torch.nn.functional.normalize(torch.randn(6, 4, 16), dim=-1)
    loss = nt_xent_loss(z_a, z_b, temperature=0.07)

    assert loss.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(loss), "loss must be finite"
    assert loss.item() > 0, "NT-Xent loss is strictly positive"

    # A perfectly-aligned pair should score far lower than a random one.
    aligned = nt_xent_loss(z_a, z_a.clone(), temperature=0.07)
    assert aligned.item() < loss.item()


# ---------------------------------------------------------------------------
# 3. Loss decreases over 10 CPU steps on dummy data
# ---------------------------------------------------------------------------


def test_loss_decreases_over_ten_steps():
    torch.manual_seed(0)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    pix_a = _pixels(8, seed=1)
    pix_b = _pixels(8, seed=2)

    losses = []
    for _ in range(10):
        loss = train_step(model, optimizer, None, pix_a, pix_b, temperature=0.1)
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(loss_value)) for loss_value in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# 4. Checkpoint save/resume -> identical next-step loss
# ---------------------------------------------------------------------------


def test_checkpoint_save_resume_identical_next_step_loss(tmp_path):
    from transformers import get_cosine_schedule_with_warmup

    torch.manual_seed(0)
    pix_a = _pixels(6, seed=3)
    pix_b = _pixels(6, seed=4)

    # Model 1: one step, then checkpoint the post-step state.
    model1 = _model()
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    sched1 = get_cosine_schedule_with_warmup(opt1, num_warmup_steps=2, num_training_steps=100)
    train_step(model1, opt1, sched1, pix_a, pix_b, temperature=0.1)

    ckpt = tmp_path / "checkpoint_step1.pt"
    save_checkpoint(ckpt, model=model1, optimizer=opt1, scheduler=sched1, step=1)

    # Model 2: fresh random init, then restored from the checkpoint.
    torch.manual_seed(123)  # deliberately different init
    model2 = _model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = get_cosine_schedule_with_warmup(opt2, num_warmup_steps=2, num_training_steps=100)
    restored_step = load_checkpoint(ckpt, model=model2, optimizer=opt2, scheduler=sched2)
    assert restored_step == 1

    # Both now sit at the identical post-step-1 state -> identical next-step loss.
    loss1 = train_step(model1, opt1, sched1, pix_a, pix_b, temperature=0.1)
    loss2 = train_step(model2, opt2, sched2, pix_a, pix_b, temperature=0.1)
    assert torch.allclose(loss1, loss2, atol=1e-6), f"{loss1.item()} != {loss2.item()}"


# ---------------------------------------------------------------------------
# 5. render_pair -> correct tensor shapes through the TrOCR processor
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_render_pair_through_processor_shapes():
    from transformers import TrOCRProcessor

    from maltese_ocr.render import RenderConfig, load_fonts, render_pair

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    fonts = load_fonts()
    rng = random.Random(0)

    cfg_a = RenderConfig.sample(fonts[0], rng)
    cfg_b = RenderConfig.sample(fonts[1], rng)
    cfg_b.image_width = cfg_a.image_width
    cfg_b.margin = cfg_a.margin

    (img_a, img_b), ground_truth, _ = render_pair(
        "Il-kelb tal-baħar għadda mill-port filgħodu kmieni ħafna.",
        cfg_a,
        cfg_b,
        rng=rng,
    )

    pix_a = processor(images=img_a.convert("RGB"), return_tensors="pt").pixel_values
    pix_b = processor(images=img_b.convert("RGB"), return_tensors="pt").pixel_values

    assert pix_a.shape == (1, 3, 384, 384)
    assert pix_b.shape == pix_a.shape
    assert pix_a.dtype == torch.float32
    assert isinstance(ground_truth, str) and ground_truth
