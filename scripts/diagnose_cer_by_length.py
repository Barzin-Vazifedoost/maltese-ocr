"""Per-length CER breakdown of the finished T6 (Stage 2) char-decoder TrOCR.

Diagnostic only — read-only on ``models/stage2/best``, no training. The finished
T6 run plateaued at dev CER ~0.50 while train loss collapsed to ~0.002 (classic
overfit signature). This script separates two hypotheses for *why* the dev CER
is stuck, so the next experiments can be aimed:

  (A) synthetic->real domain gap / overfitting
        => CER is roughly UNIFORM across paragraph lengths (~0.5 everywhere).
        Fix points at better/realistic synthetic data + early stopping.

  (B) the 512-position decoder ceiling
        => CER is LOW on short paragraphs and HIGH only on long ones, driven by
        truncation (the char decoder has a hard 512-position limit, so any
        paragraph longer than ~511 chars is un-emittable).
        Fix points at line-tiling the input.

It runs the SAME decoding config the training eval used (``train/run.py``
``FineTuneTrainer.evaluate``): num_beams=2, repetition_penalty=1.2,
no_repeat_ngram_size=3, and ``max_new_tokens`` clamped to the decoder's
positional limit minus one (``decoder.max_position_embeddings - 1`` = 511). CER
is raw-string ``jiwer.cer`` with no normalization (Maltese is case / diacritic /
dash sensitive), exactly as training scored it.

Usage (needs the GPU `ocr` conda env)::

    conda activate ocr
    python3 scripts/diagnose_cer_by_length.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch
from jiwer import cer as jiwer_cer
from PIL import Image

# Allow running as `python scripts/diagnose_cer_by_length.py` without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from maltese_ocr.train.char_tokenizer import CharTokenizer  # noqa: E402

BEST_DIR = _REPO_ROOT / "models" / "stage2" / "best"
DEV_DIR = _REPO_ROOT / "data" / "dev_set"
DEV_LABELS = _REPO_ROOT / "data" / "texts.json"
EVAL_BATCH_SIZE = 16

# Length buckets keyed on ground-truth char length. (lo, hi) is [lo, hi); the
# last bucket has hi=None meaning [lo, inf).
BUCKETS: list[tuple[int, int | None]] = [
    (0, 128),
    (128, 256),
    (256, 384),
    (384, 512),
    (512, 768),
    (768, 1024),
    (1024, None),
]


def bucket_label(lo: int, hi: int | None) -> str:
    return f"[{lo}-{hi}]" if hi is not None else f"[{lo}+]"


def assign_bucket(gt_len: int) -> int:
    """Index into BUCKETS for a given ground-truth length."""
    for i, (lo, hi) in enumerate(BUCKETS):
        if gt_len >= lo and (hi is None or gt_len < hi):
            return i
    return len(BUCKETS) - 1  # unreachable: last bucket is open-ended


def load_model_and_tokenizer(device: torch.device):
    """Load the best checkpoint (swapped char decoder) + char tokenizer + processor."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(BEST_DIR)
    model.to(device)
    model.eval()
    tokenizer = CharTokenizer.load(BEST_DIR / "char_tokenizer.json")
    # The processor (ViT image processor, 384x384, mean/std 0.5) only preprocesses
    # images; text decoding goes through the char tokenizer above.
    processor = TrOCRProcessor.from_pretrained(BEST_DIR)
    return model, tokenizer, processor


def load_dev_items() -> list[dict]:
    with open(DEV_LABELS, encoding="utf-8") as f:
        data = json.load(f)
    items = [it for it in data if (DEV_DIR / it["image"]).exists()]
    if not items:
        raise RuntimeError(f"no dev images found under {DEV_DIR}")
    return items


@torch.no_grad()
def run_inference(model, tokenizer, processor, items, device, *, use_bf16: bool):
    """Generate predictions for every dev image; return a list of per-image records.

    Mirrors FineTuneTrainer.evaluate exactly: same clamp on max_new_tokens, same
    gen kwargs, same bf16 autocast, same raw-string decode.
    """
    decoder_max_pos = int(model.config.decoder.max_position_embeddings)
    # Training eval: min(decode.max_new_tokens=512, decoder_max_pos - 1) = 511.
    max_new_tokens = min(512, decoder_max_pos - 1)
    gen_kwargs = dict(
        num_beams=2,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        max_new_tokens=max_new_tokens,
    )
    eos_id = tokenizer.eos_token_id

    records: list[dict] = []
    for start in range(0, len(items), EVAL_BATCH_SIZE):
        chunk = items[start : start + EVAL_BATCH_SIZE]
        images = [Image.open(DEV_DIR / it["image"]).convert("RGB") for it in chunk]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.generate(pixel_values=pixel_values, **gen_kwargs)
        else:
            out = model.generate(pixel_values=pixel_values, **gen_kwargs)
        for it, row in zip(chunk, out):
            ids = row.tolist()
            hyp = tokenizer.decode(ids, skip_special_tokens=True)
            ref = it["text"]
            # The decoder never emitted <eos> => it ran to the max_new_tokens cap.
            truncated = eos_id not in ids
            records.append(
                {
                    "image": it["image"],
                    "ref": ref,
                    "hyp": hyp,
                    "gt_len": len(ref),
                    "pred_len": len(hyp),
                    "cer": float(jiwer_cer(ref, hyp)),
                    "truncated": truncated,
                    "bucket": assign_bucket(len(ref)),
                }
            )
        print(
            f"  ...inferred {min(start + EVAL_BATCH_SIZE, len(items))}/{len(items)}",
            end="\r",
            flush=True,
        )
    print()
    return records, max_new_tokens


def print_bucket_table(records: list[dict]) -> None:
    header = (
        f"{'bucket':>12}  {'n':>4}  {'mean_CER':>9}  {'median_CER':>10}  "
        f"{'mean_GT':>8}  {'mean_pred':>9}  {'frac_trunc':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, (lo, hi) in enumerate(BUCKETS):
        rows = [r for r in records if r["bucket"] == i]
        n = len(rows)
        if n == 0:
            print(
                f"{bucket_label(lo, hi):>12}  {0:>4}  {'-':>9}  {'-':>10}  "
                f"{'-':>8}  {'-':>9}  {'-':>10}"
            )
            continue
        mean_cer = statistics.mean(r["cer"] for r in rows)
        median_cer = statistics.median(r["cer"] for r in rows)
        mean_gt = statistics.mean(r["gt_len"] for r in rows)
        mean_pred = statistics.mean(r["pred_len"] for r in rows)
        frac_trunc = sum(r["truncated"] for r in rows) / n
        print(
            f"{bucket_label(lo, hi):>12}  {n:>4}  {mean_cer:>9.4f}  {median_cer:>10.4f}  "
            f"{mean_gt:>8.1f}  {mean_pred:>9.1f}  {frac_trunc:>10.2f}"
        )


def print_examples(records: list[dict], bucket_idx: int, title: str, k: int = 3) -> None:
    rows = [r for r in records if r["bucket"] == bucket_idx]
    lo, hi = BUCKETS[bucket_idx]
    print(
        f"\n=== {title}: {k} examples from bucket {bucket_label(lo, hi)} ({len(rows)} images) ==="
    )
    if not rows:
        print("  (bucket is empty)")
        return
    for r in rows[:k]:
        print(
            f"\n  [{r['image']}] gt_len={r['gt_len']} pred_len={r['pred_len']} "
            f"cer={r['cer']:.3f} truncated={r['truncated']}"
        )
        print(f"    GT  : {r['ref'][:120]!r}")
        print(f"    PRED: {r['hyp'][:120]!r}")


def pick_long_bucket(records: list[dict], *, min_n: int = 3) -> int:
    """Longest bucket holding at least ``min_n`` images (falls back to most-populated)."""
    counts = {i: sum(1 for r in records if r["bucket"] == i) for i in range(len(BUCKETS))}
    for i in range(len(BUCKETS) - 1, -1, -1):
        if counts[i] >= min_n:
            return i
    # No bucket has >= min_n; return the most-populated non-empty one.
    return max(counts, key=lambda i: counts[i])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    print(f"Device: {device} (bf16 autocast={use_bf16})")
    print(f"Checkpoint: {BEST_DIR}")

    model, tokenizer, processor = load_model_and_tokenizer(device)
    items = load_dev_items()
    print(
        f"Loaded {len(items)} dev images; running inference (beam=2, "
        f"rep_pen=1.2, no_repeat_ngram=3)..."
    )

    records, max_new_tokens = run_inference(
        model, tokenizer, processor, items, device, use_bf16=use_bf16
    )
    print(f"max_new_tokens cap used: {max_new_tokens}")

    # --- per-length table -------------------------------------------------
    print("\n========== CER by ground-truth length ==========")
    print_bucket_table(records)

    # --- overall sanity ---------------------------------------------------
    all_refs = [r["ref"] for r in records]
    all_hyps = [r["hyp"] for r in records]
    corpus_cer = float(jiwer_cer(all_refs, all_hyps))
    mean_per_image_cer = statistics.mean(r["cer"] for r in records)
    n_trunc = sum(r["truncated"] for r in records)
    print("\n========== Overall ==========")
    print(
        f"corpus-level CER (matches training eval): {corpus_cer:.4f}  (best.json reported 0.5045)"
    )
    print(f"mean of per-image CER                   : {mean_per_image_cer:.4f}")
    print(
        f"images that hit the {max_new_tokens}-token cap        : "
        f"{n_trunc}/{len(records)} ({n_trunc / len(records):.1%})"
    )

    # --- eyeball examples -------------------------------------------------
    long_idx = pick_long_bucket(records)
    print_examples(records, 0, "SHORT")
    print_examples(records, long_idx, "LONG")


if __name__ == "__main__":
    main()
