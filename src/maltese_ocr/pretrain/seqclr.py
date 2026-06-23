"""SeqCLR contrastive pretraining of the TrOCR ViT encoder (Stage 1).

Self-supervised pretraining with no labels.  Two independently-styled renders of
the *same* paragraph — identical horizontal wrapping, different font/appearance —
form a positive pair (see :func:`maltese_ocr.render.render_pair`).  The encoder's
patch grid is pooled into a short horizontal sequence of *windows*; matching
windows across the two views are pulled together with a frame-level NT-Xent loss
while every other (image, window) pair in the batch is pushed apart.

Public surface:

* :class:`SeqCLRHead`     — patch grid -> (B, n_windows, proj_dim) L2-normalised.
* :func:`nt_xent_loss`    — frame-level NT-Xent over the two views.
* :class:`SeqCLRModel`    — encoder + head, ``forward`` returns the loss.
* :class:`SeqCLRTrainer`  — full BF16 / AdamW / cosine-warmup training loop with
  checkpoint save+resume, separate encoder export, and a non-fatal linear probe.
* :func:`train_step`, :func:`save_checkpoint`, :func:`load_checkpoint` — small,
  framework-agnostic helpers used by the trainer and the unit tests.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from maltese_ocr.data.corpus import CORPUS_ID, load_charset, stream_paragraphs
from maltese_ocr.render import RenderConfig, load_fonts, render, render_pair

logger = logging.getLogger(__name__)

# Repo root: src/maltese_ocr/pretrain/seqclr.py -> parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# A 384x384 image with a 16-px ViT patch yields a 24x24 = 576 patch grid.
DEFAULT_N_PATCHES = 576


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------


class SeqCLRHead(nn.Module):
    """Project ViT patch tokens into L2-normalised per-window frame embeddings.

    The ViT encoder emits ``(B, S, H)`` hidden states where ``S`` is the patch
    grid plus one or two leading special tokens (CLS, and a distillation token
    for DeiT-style encoders).  We slice the trailing ``n_patches`` tokens — which
    is robust to either count — reshape them to the ``grid x grid`` layout, mean-
    pool the vertical axis into a left-to-right sequence of columns, adaptively
    pool that into ``n_windows`` windows, run a 2-layer MLP, and L2-normalise.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        n_windows: int = 8,
        proj_dim: int = 128,
        proj_hidden: int = 512,
        n_patches: int = DEFAULT_N_PATCHES,
    ) -> None:
        super().__init__()
        grid = int(round(math.isqrt(n_patches)))
        if grid * grid != n_patches:
            raise ValueError(f"n_patches={n_patches} is not a perfect square")
        self.hidden_dim = hidden_dim
        self.n_windows = n_windows
        self.n_patches = n_patches
        self.grid = grid
        self.pool = nn.AdaptiveAvgPool1d(n_windows)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, proj_hidden),
            nn.ReLU(),
            nn.LayerNorm(proj_hidden),
            nn.Linear(proj_hidden, proj_dim),
        )

    def pooled_columns(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Slice patch tokens and vertical-mean-pool into ``(B, grid, hidden)``."""
        b, s, h = hidden_states.shape
        if s < self.n_patches:
            raise ValueError(f"expected >= {self.n_patches} tokens, got {s}")
        patches = hidden_states[:, -self.n_patches :, :]  # (B, grid*grid, H)
        patches = patches.reshape(b, self.grid, self.grid, h)  # (B, rows, cols, H)
        return patches.mean(dim=1)  # mean-pool vertical -> (B, cols, H)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cols = self.pooled_columns(hidden_states)  # (B, grid, H)
        windows = self.pool(cols.transpose(1, 2)).transpose(1, 2)  # (B, n_windows, H)
        z = self.mlp(windows)  # (B, n_windows, proj_dim)
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Frame-level NT-Xent over two views of L2-normalised window embeddings.

    Both inputs are ``(B, W, D)``.  Flattening to ``N = B*W`` frames per view and
    concatenating gives ``2N`` frames.  For each anchor frame the single positive
    is the same ``(image, window)`` in the other view; every other frame in the
    batch — including different windows of the same image — is a negative.
    """
    if z_a.shape != z_b.shape:
        raise ValueError(f"view shapes differ: {tuple(z_a.shape)} vs {tuple(z_b.shape)}")
    b, w, d = z_a.shape
    n = b * w
    reps = torch.cat([z_a.reshape(n, d), z_b.reshape(n, d)], dim=0)  # (2N, D)
    sim = reps @ reps.t() / temperature  # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))  # mask self-similarity
    targets = torch.cat(
        [
            torch.arange(n, 2 * n, device=sim.device),
            torch.arange(0, n, device=sim.device),
        ]
    )
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Encoder + head wrapper
# ---------------------------------------------------------------------------


class SeqCLRModel(nn.Module):
    """A ViT encoder paired with a :class:`SeqCLRHead`; ``forward`` -> loss."""

    def __init__(self, encoder: nn.Module, head: SeqCLRHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def _hidden(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self._hidden(pixel_values))

    def forward(
        self, pix_a: torch.Tensor, pix_b: torch.Tensor, temperature: float = 0.07
    ) -> torch.Tensor:
        return nt_xent_loss(self.encode(pix_a), self.encode(pix_b), temperature)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _load_fallback_corpus(path: str | Path, min_chars: int, max_chars: int) -> list[str]:
    """Read paragraph strings from a local ``texts.json``-style file."""
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    texts: list[str] = []
    for item in data:
        text = " ".join((item.get("text") or "").split())
        if min_chars <= len(text) <= max_chars:
            texts.append(text)
    if not texts:
        raise RuntimeError(f"fallback corpus {p} produced no usable paragraphs")
    logger.warning("Loaded %d local paragraphs from %s (fallback corpus).", len(texts), p)
    return texts


def build_corpus(data_cfg: dict, *, limit: int | None = 50000) -> list[str]:
    """Materialise a list of paragraph strings for rendering.

    Streams from the gated ``MLRS/korpus_malti`` corpus when HuggingFace is
    authenticated; otherwise logs a clear warning and falls back to the local
    ``corpus_fallback`` file (``data/texts.json``).
    """
    min_chars = int(data_cfg.get("min_chars", 10))
    max_chars = int(data_cfg.get("max_chars", 500))
    fallback = data_cfg.get("corpus_fallback", "data/texts.json")

    token = None
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:  # pragma: no cover - huggingface_hub always importable here
        token = None

    if token:
        try:
            paras = list(stream_paragraphs(min_chars=min_chars, max_chars=max_chars, limit=limit))
            if paras:
                logger.info("Loaded %d paragraphs from %s.", len(paras), CORPUS_ID)
                return paras
            logger.warning("%s stream yielded no paragraphs; falling back.", CORPUS_ID)
        except Exception as exc:
            logger.warning("%s stream failed (%s); falling back.", CORPUS_ID, exc)
    else:
        logger.warning(
            "HuggingFace is not authenticated and %s is gated; falling back to local "
            "corpus %s. Real corpus training needs `huggingface-cli login`.",
            CORPUS_ID,
            fallback,
        )
    return _load_fallback_corpus(fallback, min_chars, max_chars)


# ---------------------------------------------------------------------------
# Contrastive pair dataset
# ---------------------------------------------------------------------------


class RenderPairDataset(IterableDataset):
    """Infinite stream of ``(pixel_values_a, pixel_values_b)`` contrastive pairs.

    Each pair renders the same paragraph in two independently-sampled configs.
    The two views share the canvas width and base margin and disable per-view
    ``margin_jitter`` so their horizontal layout stays aligned window-for-window;
    everything else (font, style, size, background, noise) varies freely.  No
    horizontal flips, crops, or re-wrapping are applied between the views.
    """

    def __init__(
        self,
        paragraphs: list[str],
        fonts: list[dict],
        processor,
        *,
        seed: int = 0,
        augment: bool = True,
    ) -> None:
        self.paragraphs = paragraphs
        self.fonts = fonts
        self.processor = processor
        self.seed = seed
        self.augment = augment

    def _to_pixels(self, img) -> torch.Tensor:
        out = self.processor(images=img.convert("RGB"), return_tensors="pt")
        return out.pixel_values[0]

    def _make_pair(self, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
        text = rng.choice(self.paragraphs)
        cfg_a = RenderConfig.sample(rng.choice(self.fonts), rng, augment=self.augment)
        cfg_b = RenderConfig.sample(rng.choice(self.fonts), rng, augment=self.augment)
        # Lock horizontal layout so window i means the same thing in both views.
        cfg_b.image_width = cfg_a.image_width
        cfg_b.margin = cfg_a.margin
        for cfg in (cfg_a, cfg_b):
            cfg.augmentations.pop("margin_jitter", None)
        (img_a, img_b), _, _ = render_pair(text, cfg_a, cfg_b, rng=rng)
        return self._to_pixels(img_a), self._to_pixels(img_b)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info is not None else 0))
        while True:
            yield self._make_pair(rng)


# ---------------------------------------------------------------------------
# Training step + checkpointing
# ---------------------------------------------------------------------------


def train_step(
    model: SeqCLRModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    pix_a: torch.Tensor,
    pix_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """One non-accumulating optimizer step; returns the pre-update loss tensor."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(pix_a, pix_b, temperature)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.detach()


def save_checkpoint(path, *, model, optimizer, scheduler, step: int, config: dict | None = None):
    """Save full training state (model + optimizer + scheduler + step)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, *, model, optimizer=None, scheduler=None, map_location="cpu") -> int:
    """Restore training state in place; returns the saved step count."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("step", 0))


# ---------------------------------------------------------------------------
# Linear probe (frozen encoder -> tiny CTC head)
# ---------------------------------------------------------------------------


def _greedy_ctc_decode(logits: torch.Tensor, idx_to_char: dict[int, str]) -> str:
    """Collapse repeats and drop the blank (index 0) from a ``(T, V)`` logit map."""
    ids = logits.argmax(dim=-1).tolist()
    out: list[str] = []
    prev = -1
    for i in ids:
        if i != prev and i != 0:
            out.append(idx_to_char.get(i, ""))
        prev = i
    return "".join(out)


def run_linear_probe(
    model: SeqCLRModel,
    processor,
    paragraphs: list[str],
    fonts: list[dict],
    charset: set[str],
    *,
    n_samples: int = 2000,
    epochs: int = 1,
    device="cpu",
    seed: int = 0,
    max_label: int = 18,
) -> float:
    """Freeze the encoder, train a 1-layer CTC head on short lines, report CER.

    A deliberately small diagnostic: the 24-column encoder sequence only fits
    short labels, so this measures *relative* representation quality, not OCR
    accuracy.  Raises on degenerate inputs; the trainer treats any failure as
    non-fatal so a probe error never discards trained encoder checkpoints.
    """
    from jiwer import cer as jiwer_cer

    rng = random.Random(seed)
    chars = sorted(charset)
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = CTC blank
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    vocab = len(chars) + 1

    short = [p for p in paragraphs if 3 <= len(p) <= max_label]
    if len(short) < 10:
        raise RuntimeError(f"only {len(short)} short paragraphs available for the probe")

    texts = [rng.choice(short) for _ in range(n_samples)]
    split = max(1, int(0.9 * len(texts)))
    train_texts, val_texts = texts[:split], texts[split:] or texts[-1:]

    ctc_head = nn.Linear(model.head.hidden_dim, vocab).to(device)
    optimizer = torch.optim.Adam(ctc_head.parameters(), lr=1e-3)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    model.eval()

    def featurize(text: str) -> torch.Tensor:
        cfg = RenderConfig.sample(rng.choice(fonts), rng)
        img, _, _ = render(text, cfg, rng=rng)
        pix = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            cols = model.head.pooled_columns(model._hidden(pix))  # (1, grid, hidden)
        return cols.squeeze(0).float()  # (grid, hidden)

    batch_size = 32
    for _ in range(epochs):
        for i in range(0, len(train_texts), batch_size):
            batch = train_texts[i : i + batch_size]
            feats = torch.stack([featurize(t) for t in batch])  # (B, T, H)
            t_len = feats.shape[1]
            targets = [
                torch.tensor([char_to_idx[c] for c in t if c in char_to_idx], dtype=torch.long)
                for t in batch
            ]
            target_lengths = torch.tensor([len(t) for t in targets])
            if (target_lengths > t_len).any() or (target_lengths == 0).any():
                continue
            logits = ctc_head(feats.to(device))  # (B, T, V)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
            input_lengths = torch.full((len(batch),), t_len, dtype=torch.long)
            loss = ctc_loss(log_probs, torch.cat(targets).to(device), input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    refs, hyps = [], []
    for text in val_texts:
        with torch.no_grad():
            logits = ctc_head(featurize(text).unsqueeze(0).to(device))
        refs.append(text)
        hyps.append(_greedy_ctc_decode(logits[0], idx_to_char))
    return float(jiwer_cer(refs, hyps))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeqCLRTrainer:
    """End-to-end SeqCLR pretraining loop with checkpointing and a linear probe."""

    def __init__(
        self,
        config: dict,
        *,
        device: str | None = None,
        num_workers: int | None = None,
        corpus_limit: int | None = 50000,
    ) -> None:
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            get_cosine_schedule_with_warmup,
        )

        self.config = config
        tr = config["training"]
        self.batch_size = int(tr["batch_size"])
        self.lr = float(tr["learning_rate"])
        self.warmup_steps = int(tr["warmup_steps"])
        self.total_steps = int(tr["total_steps"])
        self.grad_accum = max(1, int(tr.get("grad_accumulation", 1)))
        self.precision = str(tr.get("precision", "bf16"))
        self.seed = int(tr.get("seed", 42))
        self.temperature = float(config["loss"]["temperature"])

        ck = config["checkpointing"]
        self.save_dir = Path(ck["save_dir"])
        self.save_every = int(ck["save_every_n_steps"])
        self.keep_last_n = int(ck["keep_last_n"])
        self.log_every = int(config.get("logging", {}).get("log_every_n_steps", 100))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_bf16 = self.device.type == "cuda" and self.precision == "bf16"
        set_seed(self.seed)

        base = config["model"]["base"]
        logger.info("Loading base encoder + processor from %s ...", base)
        self.processor = TrOCRProcessor.from_pretrained(base)
        encoder = VisionEncoderDecoderModel.from_pretrained(base).encoder
        hidden = encoder.config.hidden_size
        head = SeqCLRHead(
            hidden_dim=hidden,
            n_windows=int(config["model"]["n_windows"]),
            proj_dim=int(config["model"]["proj_dim"]),
            proj_hidden=int(config["model"]["proj_hidden"]),
        )
        self.model = SeqCLRModel(encoder, head).to(self.device)

        self.paragraphs = build_corpus(config["data"], limit=corpus_limit)
        self.fonts = load_fonts()
        if num_workers is None:
            num_workers = 4 if self.device.type == "cuda" else 0
        self.dataset = RenderPairDataset(
            self.paragraphs, self.fonts, self.processor, seed=self.seed
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, self.warmup_steps, self.total_steps
        )
        self.step = 0

    # -- checkpointing ------------------------------------------------------

    def resume(self, path) -> None:
        self.step = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        logger.info("Resumed from %s at step %d.", path, self.step)

    def _prune(self, prefix: str) -> None:
        ckpts = sorted(
            self.save_dir.glob(f"{prefix}*.pt"),
            key=lambda p: int(p.stem.split("step")[-1]),
        )
        for old in ckpts[: -self.keep_last_n] if self.keep_last_n > 0 else []:
            old.unlink(missing_ok=True)

    def save(self, step: int) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.save_dir / f"checkpoint_step{step}.pt"
        save_checkpoint(
            ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            config=self.config,
        )
        enc_path = self.save_dir / f"encoder_step{step}.pt"
        torch.save(self.model.encoder.state_dict(), enc_path)
        self._prune("checkpoint_step")
        self._prune("encoder_step")
        logger.info("Saved %s and %s.", ckpt_path.name, enc_path.name)

    # -- training -----------------------------------------------------------

    def _forward_loss(self, pix_a: torch.Tensor, pix_b: torch.Tensor) -> torch.Tensor:
        if self.use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self.model(pix_a, pix_b, self.temperature)
        return self.model(pix_a, pix_b, self.temperature)

    def train(self, max_steps: int | None = None) -> dict:
        target = self.total_steps if max_steps is None else max_steps
        truncated = max_steps is not None and target < self.total_steps
        logger.info(
            "Training to step %d on %s (bf16=%s, grad_accum=%d, batch=%d)%s.",
            target,
            self.device,
            self.use_bf16,
            self.grad_accum,
            self.batch_size,
            " [truncated]" if truncated else "",
        )

        data_iter = iter(self.loader)
        losses: list[float] = []
        while self.step < target:
            t0 = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(self.grad_accum):
                pix_a, pix_b = next(data_iter)
                pix_a = pix_a.to(self.device, non_blocking=True)
                pix_b = pix_b.to(self.device, non_blocking=True)
                loss = self._forward_loss(pix_a, pix_b)
                (loss / self.grad_accum).backward()
                step_loss += loss.item() / self.grad_accum
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            losses.append(step_loss)

            dt = time.time() - t0
            if self.step % self.log_every == 0 or self.step == 1 or self.step == target:
                logger.info(
                    "step %d/%d  loss %.4f  lr %.2e  %.3fs/step",
                    self.step,
                    target,
                    step_loss,
                    self.scheduler.get_last_lr()[0],
                    dt,
                )
            if self.save_every > 0 and self.step % self.save_every == 0:
                self.save(self.step)

        self.save(self.step)

        probe_cer = None
        probe_cfg = self.config.get("linear_probe", {})
        if not truncated and probe_cfg.get("enabled", False):
            try:
                probe_cer = run_linear_probe(
                    self.model,
                    self.processor,
                    self.paragraphs,
                    self.fonts,
                    load_charset(),
                    n_samples=int(probe_cfg.get("n_samples", 2000)),
                    epochs=int(probe_cfg.get("epochs", 1)),
                    device=self.device,
                    seed=self.seed,
                )
                logger.info("Linear probe CER: %.4f", probe_cer)
            except Exception as exc:  # non-fatal: never discard trained checkpoints
                logger.warning("Linear probe failed (non-fatal): %s", exc)
        elif truncated:
            logger.info("Skipping linear probe on truncated run.")

        return {
            "final_step": self.step,
            "losses": losses,
            "truncated": truncated,
            "probe_cer": probe_cer,
        }
