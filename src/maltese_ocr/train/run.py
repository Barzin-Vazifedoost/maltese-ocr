"""Stage 2 supervised fine-tune of the char-vocab TrOCR decoder (T6).

This is the loop that actually produces the fine-tuned OCR model.  It loads the
validated character-vocab TrOCR (``train/decoder_swap.load_char_trocr``) and
fine-tunes it on synthetic ``(image, joined-paragraph)`` pairs rendered on the
fly from the Maltese corpus, supervising with teacher-forced cross-entropy.

Design (every choice de-risked earlier in the T6 work):

* **LR 5e-5** for the decoder — the standard TrOCR fine-tune rate.  5e-4 floors
  the loss at unigram entropy (~3.0); see the T6 note in ``CLAUDE.md``.
* **Encoder warmup freeze.**  The encoder is frozen for the first ``freeze_pct``
  of steps so the fresh decoder embedding/lm_head can settle without dragging the
  pretrained encoder around.  It is then unfrozen with a *fresh* optimizer and a
  short LR warmup, training at a differential rate ``decoder_lr / encoder_lr_divisor``.
* **Optional T5 encoder.**  ``--encoder-checkpoint`` loads a Stage-1 SeqCLR
  encoder ``state_dict`` before fine-tuning; omitted, the stock
  ``trocr-base-printed`` encoder is used.  The chosen path is logged loudly.
* **bf16** autocast, **AdamW**, **cosine schedule with linear warmup** (counted in
  optimizer steps), full checkpoint save/resume, and a best-by-dev-CER export.
* **Dev CER** is computed on the *real* 422-image dev set with beam search and
  ``jiwer`` on RAW strings — Maltese is case / diacritic / dash sensitive, so no
  lowercasing or punctuation stripping.

Usage::

    python -m maltese_ocr.train.run --config configs/stage2.yaml
    python -m maltese_ocr.train.run --config configs/stage2.yaml \
        --encoder-checkpoint models/stage1/encoder_step100000.pt
    python -m maltese_ocr.train.run --config configs/stage2.yaml \
        --total-steps 100 --eval-every 50          # short GPU smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

from maltese_ocr.pretrain.seqclr import build_corpus, set_seed
from maltese_ocr.render import RenderConfig, clean_ground_truth, load_fonts, render
from maltese_ocr.train.char_tokenizer import CharTokenizer
from maltese_ocr.train.decoder_swap import load_char_trocr

logger = logging.getLogger(__name__)

# Repo root: src/maltese_ocr/train/run.py -> parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    """Character error rate over parallel reference/hypothesis lists.

    A thin wrapper over ``jiwer.cer`` on RAW strings — no lowercasing, no
    punctuation removal — because Maltese OCR is case, diacritic and dash
    sensitive and the competition is scored that way.  Returns ``0.0`` for an
    empty input rather than raising.
    """
    from jiwer import cer as jiwer_cer

    if not refs:
        return 0.0
    return float(jiwer_cer(refs, hyps))


# ---------------------------------------------------------------------------
# Encoder checkpoint hook
# ---------------------------------------------------------------------------


def load_encoder_checkpoint(model, path, *, map_location: str = "cpu") -> tuple[list, list]:
    """Load a Stage-1 (T5 SeqCLR) encoder ``state_dict`` into ``model.encoder``.

    Accepts either a raw encoder ``state_dict`` (what ``seqclr`` exports as
    ``encoder_step*.pt``) or a full SeqCLR training checkpoint dict, from which
    the ``encoder.*`` sub-state is extracted.  Returns the
    ``(missing, unexpected)`` key lists from ``load_state_dict(strict=False)``;
    a large count of either is logged as a warning since it usually means an
    architecture mismatch.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model" in state and "step" in state:
        # Full SeqCLR checkpoint: keep only the encoder.* sub-state.
        prefix = "encoder."
        state = {k[len(prefix) :]: v for k, v in state["model"].items() if k.startswith(prefix)}
    result = model.encoder.load_state_dict(state, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    level = logging.WARNING if (missing or unexpected) else logging.INFO
    logger.log(
        level,
        "Loaded encoder checkpoint %s (missing=%d, unexpected=%d).",
        path,
        len(missing),
        len(unexpected),
    )
    return missing, unexpected


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------


def set_encoder_trainable(model, trainable: bool) -> None:
    """Toggle ``requires_grad`` on every encoder parameter."""
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def split_param_groups(model) -> tuple[list, list]:
    """Return ``(decoder_params, encoder_params)`` partitioned by name.

    Everything that is not under ``encoder.`` (the decoder, plus any
    encoder->decoder bridge) lands in the decoder group, so no parameter is
    dropped from the optimizer.
    """
    encoder_params, decoder_params = [], []
    for name, p in model.named_parameters():
        (encoder_params if name.startswith("encoder.") else decoder_params).append(p)
    return decoder_params, encoder_params


# ---------------------------------------------------------------------------
# Teacher-forced training step
# ---------------------------------------------------------------------------


def train_step(
    model,
    optimizer,
    scheduler,
    pixel_values: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    pad_id: int,
    vocab_size: int,
    label_smoothing: float = 0.0,
    use_bf16: bool = False,
    device_type: str = "cuda",
) -> torch.Tensor:
    """One non-accumulating optimizer step of teacher-forced cross-entropy.

    ``decoder_input_ids`` is ``<bos> c1..cN`` and ``labels`` is ``c1..cN <eos>``
    (right-padded with ``pad_id``, which the loss ignores).  Returns the detached
    pre-update loss.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if use_bf16:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits
    else:
        logits = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits
    # Compute the loss in fp32 for numerical stability under bf16 autocast.
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        labels.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.detach()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RenderTextDataset(IterableDataset):
    """Infinite stream of ``(pixel_values, label_ids)`` supervised pairs.

    Each example renders a randomly-styled image of a randomly-chosen paragraph
    and pairs it with that paragraph's char-token ids (``<bos> .. <eos>``).  The
    paragraph pool is pre-filtered to the tokenizer's charset and to
    ``max_target_len`` tokens, so encoding never raises and labels never exceed
    the decoder's positional limit.
    """

    def __init__(
        self,
        paragraphs: list[str],
        fonts: list[dict],
        processor,
        tokenizer: CharTokenizer,
        *,
        seed: int = 0,
        augment: bool = True,
        max_target_len: int = 512,
    ) -> None:
        charset = set(tokenizer.token_to_id)
        usable: list[str] = []
        for raw in paragraphs:
            text = clean_ground_truth(raw)
            if not text or any(ch not in charset for ch in text):
                continue
            if len(text) + 2 > max_target_len:  # +2 for <bos>/<eos>
                continue
            usable.append(text)
        if not usable:
            raise RuntimeError("no paragraphs survived charset/length filtering")
        logger.info(
            "Fine-tune corpus: %d/%d paragraphs usable after filtering.",
            len(usable),
            len(paragraphs),
        )
        self.paragraphs = usable
        self.fonts = fonts
        self.processor = processor
        self.tokenizer = tokenizer
        self.seed = seed
        self.augment = augment

    def _make_example(self, rng: random.Random) -> tuple[torch.Tensor, list[int]]:
        text = rng.choice(self.paragraphs)
        cfg = RenderConfig.sample(rng.choice(self.fonts), rng, augment=self.augment)
        img, ground_truth, _ = render(text, cfg, rng=rng)
        px = self.processor(images=img.convert("RGB"), return_tensors="pt").pixel_values[0]
        ids = self.tokenizer.encode(ground_truth, add_special_tokens=True)
        return px, ids

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info is not None else 0))
        while True:
            yield self._make_example(rng)


def collate_finetune(batch: list[tuple[torch.Tensor, list[int]]], pad_id: int):
    """Collate ``(pixel_values, ids)`` into teacher-forced tensors.

    ``ids`` is ``<bos> .. <eos>``; the standard shift gives
    ``decoder_input = ids[:-1]`` and ``labels = ids[1:]``, both right-padded.
    """
    pixel_values = torch.stack([px for px, _ in batch])
    dec_in = [ids[:-1] for _, ids in batch]
    labels = [ids[1:] for _, ids in batch]
    max_len = max(len(row) for row in dec_in)

    def pad(rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(
            [row + [pad_id] * (max_len - len(row)) for row in rows], dtype=torch.long
        )

    return pixel_values, pad(dec_in), pad(labels)


# ---------------------------------------------------------------------------
# Dev set
# ---------------------------------------------------------------------------


def load_dev_set(labels_path, dev_dir) -> list[dict]:
    """Load the dev-set ``{image, text}`` records, keeping only present images."""
    labels_path = Path(labels_path)
    if not labels_path.is_absolute():
        labels_path = _REPO_ROOT / labels_path
    dev_dir = Path(dev_dir)
    if not dev_dir.is_absolute():
        dev_dir = _REPO_ROOT / dev_dir
    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)
    items = [it for it in data if (dev_dir / it["image"]).exists()]
    if not items:
        raise RuntimeError(f"no dev images found under {dev_dir}")
    return items


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FineTuneTrainer:
    """Supervised fine-tune loop with freeze->unfreeze, eval, and checkpointing."""

    def __init__(
        self,
        config: dict,
        *,
        device: str | None = None,
        num_workers: int | None = None,
    ) -> None:
        from transformers import TrOCRProcessor

        self.config = config
        m, d, tr = config["model"], config["data"], config["training"]
        ev, ck = config["eval"], config["checkpointing"]

        self.batch_size = int(d["batch_size"])
        self.decoder_lr = float(tr["decoder_lr"])
        self.encoder_lr_divisor = float(tr.get("encoder_lr_divisor", 8))
        self.weight_decay = float(tr.get("weight_decay", 0.0))
        self.warmup_steps = int(tr["warmup_steps"])
        self.unfreeze_warmup_steps = int(tr.get("unfreeze_warmup_steps", 0))
        self.total_steps = int(tr["total_steps"])
        self.freeze_pct = float(tr.get("freeze_pct", 0.0))
        self.freeze_steps = int(self.freeze_pct * self.total_steps)
        self.grad_accum = max(1, int(tr.get("grad_accumulation", 1)))
        self.precision = str(tr.get("precision", "bf16"))
        self.seed = int(tr.get("seed", 42))
        self.label_smoothing = float(tr.get("label_smoothing", 0.0))
        self.max_target_len = int(d.get("max_target_len", 512))

        self.decode_cfg = dict(config["decode"])
        self.eval_every = int(ev["eval_every"])
        self.eval_batch_size = int(ev.get("eval_batch_size", 16))
        self.eval_max_images = ev.get("max_images")
        self.dev_items = load_dev_set(ev["dev_labels"], ev["dev_dir"])
        self.dev_dir = Path(ev["dev_dir"])
        if not self.dev_dir.is_absolute():
            self.dev_dir = _REPO_ROOT / self.dev_dir

        self.save_dir = Path(ck["save_dir"])
        self.save_every = int(ck["save_every_n_steps"])
        self.keep_last_n = int(ck.get("keep_last_n", 3))
        self.log_every = int(config.get("logging", {}).get("log_every_n_steps", 50))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_bf16 = self.device.type == "cuda" and self.precision == "bf16"
        set_seed(self.seed)

        base = m["base"]
        charset_file = m.get("charset_file")
        logger.info("Loading char-vocab TrOCR from %s ...", base)
        self.model, self.tokenizer = load_char_trocr(base=base, charset_path=charset_file)
        self.processor = TrOCRProcessor.from_pretrained(base)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_token_id

        # --- optional T5 SeqCLR encoder -------------------------------------
        enc_ckpt = m.get("encoder_checkpoint")
        if enc_ckpt:
            logger.info("Encoder init: loading T5 SeqCLR encoder from %s.", enc_ckpt)
            load_encoder_checkpoint(self.model, enc_ckpt, map_location="cpu")
        else:
            logger.info("Encoder init: stock %s encoder (no T5 checkpoint).", base)
        self.model.to(self.device)

        # --- data ------------------------------------------------------------
        paragraphs = build_corpus(d, limit=d.get("corpus_limit", 50000))
        fonts = load_fonts()
        if num_workers is None:
            num_workers = int(d.get("num_workers", 4)) if self.device.type == "cuda" else 0
        self.dataset = RenderTextDataset(
            paragraphs,
            fonts,
            self.processor,
            self.tokenizer,
            seed=self.seed,
            max_target_len=self.max_target_len,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=num_workers > 0,
            collate_fn=partial(collate_finetune, pad_id=self.pad_id),
        )

        # --- optimizer / schedule (phase 1: frozen encoder) -----------------
        self.step = 0
        self.unfrozen = False
        self.best_cer = float("inf")
        if self.freeze_steps > 0:
            set_encoder_trainable(self.model, False)
            self._build_phase1()
        else:
            self._unfreeze()

    # -- optimizer construction --------------------------------------------

    def _cosine_scheduler(self, optimizer, warmup: int, total: int):
        from transformers import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(optimizer, warmup, max(total, warmup + 1))

    def _build_phase1(self) -> None:
        """Decoder-only AdamW + cosine schedule while the encoder is frozen."""
        decoder_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            decoder_params, lr=self.decoder_lr, weight_decay=self.weight_decay
        )
        self.scheduler = self._cosine_scheduler(self.optimizer, self.warmup_steps, self.total_steps)
        logger.info(
            "Phase 1 (encoder frozen): decoder-only AdamW, lr=%.1e, until step %d.",
            self.decoder_lr,
            self.freeze_steps,
        )

    def _unfreeze(self) -> None:
        """Unfreeze the encoder with a fresh differential-LR optimizer + warmup."""
        set_encoder_trainable(self.model, True)
        decoder_params, encoder_params = split_param_groups(self.model)
        encoder_lr = self.decoder_lr / self.encoder_lr_divisor
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decoder_params, "lr": self.decoder_lr},
                {"params": encoder_params, "lr": encoder_lr},
            ],
            weight_decay=self.weight_decay,
        )
        remaining = max(1, self.total_steps - self.step)
        self.scheduler = self._cosine_scheduler(
            self.optimizer, self.unfreeze_warmup_steps, remaining
        )
        self.unfrozen = True
        logger.info(
            "Unfroze encoder at step %d: decoder_lr=%.1e, encoder_lr=%.1e, "
            "fresh optimizer, %d-step warmup over %d remaining steps.",
            self.step,
            self.decoder_lr,
            encoder_lr,
            self.unfreeze_warmup_steps,
            remaining,
        )

    # -- checkpointing ------------------------------------------------------

    def _state(self) -> dict:
        return {
            "step": self.step,
            "unfrozen": self.unfrozen,
            "best_cer": self.best_cer,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }

    def _prune(self) -> None:
        ckpts = sorted(
            self.save_dir.glob("checkpoint_step*.pt"),
            key=lambda p: int(p.stem.split("step")[-1]),
        )
        if self.keep_last_n > 0:
            for old in ckpts[: -self.keep_last_n]:
                old.unlink(missing_ok=True)

    def save(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"checkpoint_step{self.step}.pt"
        torch.save(self._state(), path)
        self._prune()
        logger.info("Saved %s.", path.name)

    def resume(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.step = int(ckpt.get("step", 0))
        self.best_cer = float(ckpt.get("best_cer", float("inf")))
        self.model.load_state_dict(ckpt["model"])
        # Rebuild the optimizer/scheduler matching the saved phase before loading
        # their state, since the param groups differ across the unfreeze boundary.
        if ckpt.get("unfrozen"):
            self._unfreeze()
        elif not self.unfrozen:
            self._build_phase1()
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        logger.info("Resumed from %s at step %d (unfrozen=%s).", path, self.step, self.unfrozen)

    def save_best(self, cer: float) -> None:
        best_dir = self.save_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(best_dir)
        self.processor.save_pretrained(best_dir)
        self.tokenizer.save(best_dir / "char_tokenizer.json")
        with open(best_dir / "best.json", "w", encoding="utf-8") as f:
            json.dump({"step": self.step, "dev_cer": cer}, f, indent=2)
        logger.info("New best dev CER %.4f at step %d -> %s.", cer, self.step, best_dir)

    # -- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Generate on the real dev set and return raw-string CER."""
        self.model.eval()
        items = self.dev_items
        if self.eval_max_images:
            items = items[: int(self.eval_max_images)]

        gen_kwargs = dict(
            num_beams=int(self.decode_cfg.get("num_beams", 2)),
            repetition_penalty=float(self.decode_cfg.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(self.decode_cfg.get("no_repeat_ngram_size", 0)),
            max_new_tokens=int(self.decode_cfg.get("max_new_tokens", 256)),
        )
        refs, hyps = [], []
        for start in range(0, len(items), self.eval_batch_size):
            chunk = items[start : start + self.eval_batch_size]
            images = [Image.open(self.dev_dir / it["image"]).convert("RGB") for it in chunk]
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(
                self.device
            )
            if self.use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self.model.generate(pixel_values=pixel_values, **gen_kwargs)
            else:
                out = self.model.generate(pixel_values=pixel_values, **gen_kwargs)
            for it, row in zip(chunk, out):
                refs.append(it["text"])
                hyps.append(self.tokenizer.decode(row.tolist(), skip_special_tokens=True))
        return compute_cer(refs, hyps)

    def _eval_and_track(self) -> float:
        cer = self.evaluate()
        logger.info("step %d  dev_cer %.4f  (best %.4f)", self.step, cer, min(cer, self.best_cer))
        if cer < self.best_cer:
            self.best_cer = cer
            self.save_best(cer)
        return cer

    # -- training -----------------------------------------------------------

    def train(self) -> dict:
        target = self.total_steps
        logger.info(
            "Fine-tuning to step %d on %s (bf16=%s, batch=%d, grad_accum=%d).",
            target,
            self.device,
            self.use_bf16,
            self.batch_size,
            self.grad_accum,
        )
        data_iter = iter(self.loader)
        losses: list[float] = []
        last_cer: float | None = None
        while self.step < target:
            if not self.unfrozen and self.step >= self.freeze_steps:
                self._unfreeze()

            t0 = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(self.grad_accum):
                pixel_values, dec_in, labels = next(data_iter)
                pixel_values = pixel_values.to(self.device, non_blocking=True)
                dec_in = dec_in.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.model.train()
                if self.use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = self.model(
                            pixel_values=pixel_values, decoder_input_ids=dec_in
                        ).logits
                else:
                    logits = self.model(pixel_values=pixel_values, decoder_input_ids=dec_in).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size).float(),
                    labels.reshape(-1),
                    ignore_index=self.pad_id,
                    label_smoothing=self.label_smoothing,
                )
                (loss / self.grad_accum).backward()
                step_loss += loss.item() / self.grad_accum
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            losses.append(step_loss)

            dt = time.time() - t0
            if self.step % self.log_every == 0 or self.step in (1, target):
                logger.info(
                    "step %d/%d  loss %.4f  lr %.2e  %.3fs/step",
                    self.step,
                    target,
                    step_loss,
                    self.scheduler.get_last_lr()[0],
                    dt,
                )
            if self.eval_every > 0 and self.step % self.eval_every == 0:
                last_cer = self._eval_and_track()
            if self.save_every > 0 and self.step % self.save_every == 0:
                self.save()

        # Final eval + checkpoint so a run always ends with a measured CER.
        last_cer = self._eval_and_track()
        self.save()
        return {
            "final_step": self.step,
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "dev_cer": last_cer,
            "best_cer": self.best_cer,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _apply_overrides(config: dict, args) -> None:
    if args.encoder_checkpoint is not None:
        config["model"]["encoder_checkpoint"] = args.encoder_checkpoint
    if args.total_steps is not None:
        config["training"]["total_steps"] = args.total_steps
    if args.eval_every is not None:
        config["eval"]["eval_every"] = args.eval_every
    if args.eval_max_images is not None:
        config["eval"]["max_images"] = args.eval_max_images
    if args.corpus_limit is not None:
        config["data"]["corpus_limit"] = args.corpus_limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the stage2 YAML config.")
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help="Load a Stage-1 (T5 SeqCLR) encoder state_dict before fine-tuning.",
    )
    parser.add_argument("--resume", default=None, help="Resume from a full checkpoint .pt.")
    parser.add_argument(
        "--total-steps", type=int, default=None, help="Override training.total_steps."
    )
    parser.add_argument(
        "--eval-every", type=int, default=None, help="Override eval.eval_every (steps)."
    )
    parser.add_argument(
        "--eval-max-images",
        type=int,
        default=None,
        help="Cap dev images per eval (default: all 422).",
    )
    parser.add_argument(
        "--corpus-limit", type=int, default=None, help="Override data.corpus_limit."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _apply_overrides(config, args)

    trainer = FineTuneTrainer(config)
    if args.resume:
        trainer.resume(args.resume)
    summary = trainer.train()

    print(
        f"Done. final_step={summary['final_step']} "
        f"final_loss={summary['final_loss']:.4f} "
        f"dev_cer={summary['dev_cer']:.4f} best_cer={summary['best_cer']:.4f}"
    )


if __name__ == "__main__":
    main()
