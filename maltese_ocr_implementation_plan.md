# Maltese OCR Competition — Implementation Plan (DocEng 2026)

This plan breaks the project into self-contained tasks. Each task has a **prompt**
you can paste directly into Claude Code (or hand to an engineer), plus acceptance
criteria. Tasks are ordered by dependency; T1–T4 and T8 can start in parallel
with the modeling tasks once T1 is done.

---

## Project context (include this block at the top of every prompt)

```
CONTEXT — Maltese OCR competition (DocEng 2026):
- Task: transcribe images of Maltese paragraphs cropped from real PDFs into text.
- Training data: synthetic only. We render text sampled from the korpus_malti
  corpus (HuggingFace: MLRS/korpus_malti). A small labeled dev set of real crops
  is provided for evaluation ONLY — never train on it (images or labels).
- Metric: Character Error Rate (CER), lower is better.
- Maltese specifics: special characters ħ Ħ ġ Ġ ċ Ċ ż Ż and the għ digraph.
  Structural dashes inside words ("il-kelb") must be kept; hyphens introduced
  by line breaks must be removed when joining lines. The organizers provide the
  complete character set covering dev + test.
- Submission interface: a Python class exposing transcribe(image: PIL.Image) -> str,
  model weights hosted on HuggingFace. Evaluation machine: single RTX 2080 Ti
  (11 GB VRAM), batch size 1, total runtime budget 5 hours, 20 GB disk, no
  internet access during evaluation.
- Architecture decision (already made): TrOCR-style encoder-decoder.
  Stage 1 = SeqCLR contrastive pretraining of the encoder (no labels).
  Stage 2 = supervised fine-tuning on synthetic data with an auxiliary
  style-invariance contrastive loss. Stage 3 = hard-negative minimal pairs
  for confusable characters. Init from microsoft/trocr-base-printed.
  Decoder vocabulary: character-level over the provided closed character set
  (replace the RoBERTa BPE tokenizer).
- Repo layout: src/data, src/render, src/pretrain, src/train, src/infer,
  scripts/, tests/, configs/ (YAML per stage).
```

---

## T1 — Repo scaffold and environment

**Prompt:**
```
[paste CONTEXT block]

Create the repository scaffold: pyproject.toml (Python 3.11, torch,
transformers, datasets, Pillow, albumentations, jiwer for CER, pyyaml,
fonttools), the src/ package layout from the context block, a configs/
directory with stage1.yaml, stage2.yaml, stage3.yaml stubs, pre-commit with
ruff, and a Makefile with targets: setup, test, render-sample, pretrain,
train, eval, package. Add a smoke test that loads
microsoft/trocr-base-printed and runs a dummy forward pass. Pin all versions.
```

**Accept when:** `make setup && make test` passes on a clean machine; dummy forward
pass runs on CPU.

---

## T2 — Font audit and acquisition

**Prompt:**
```
[paste CONTEXT block]

Write src/render/fonts.py and scripts/audit_fonts.py. The script scans a
fonts/ directory of .ttf/.otf files and, using fonttools, verifies each font
has real glyphs (not .notdef, not silent fallback) for every character in the
provided competition character set — especially ħ Ħ ġ Ġ ċ Ċ ż Ż. Output a
report (fonts_ok.json) listing usable fonts with metadata (serif/sans/mono,
weight). Also write scripts/fetch_fonts.py that downloads a diverse set of
open-license fonts (Google Fonts: at minimum Noto Serif/Sans, Lato, Merriweather,
Source Serif, PT Serif, Gentium, Charis SIL, Libertinus, EB Garamond) and runs
the audit on them. Document in fonts/README.md which fonts passed.
```

**Accept when:** ≥ 12 fonts pass the audit, spanning serif/sans, regular/bold/italic;
report shows zero missing-glyph fonts in the usable list.

---

## T3 — Synthetic paragraph renderer

**Prompt:**
```
[paste CONTEXT block]

Implement src/render/renderer.py: a configurable synthetic paragraph renderer.
Input: a text string + a RenderConfig (font, size, line width in chars or px,
justification on/off, line spacing, margins). Behavior:
- Wrap text into lines; with probability p_hyphen, break words at line ends
  with a hyphen (record ground truth as the JOINED paragraph with line-break
  hyphens removed but structural dashes like "il-kelb" preserved).
- Render with PIL at 150–400 simulated DPI.
- Degradation pipeline (albumentations or custom): gaussian/speckle noise,
  motion + gaussian blur, JPEG compression (quality 30–95), brightness/
  contrast jitter, slight rotation (±1.5°), mild perspective, bleed-through
  (faint flipped text from a 'previous page'), salt-and-pepper, downscale-
  upscale. Each degradation independently sampled.
- CRITICAL for contrastive pairs: expose render_pair(text, cfg_a, cfg_b) that
  renders the SAME text with the SAME line breaks in two different
  font/degradation configs (identical horizontal layout, different appearance).
- Output: (PIL.Image, ground_truth_str, metadata dict).
Also implement src/data/corpus.py to stream paragraphs from MLRS/korpus_malti
with length filtering and character-set filtering, and
scripts/build_dataset.py to materialize N samples to webdataset shards.
Write tests: ground truth never contains line-break hyphens; structural
dashes preserved; render_pair produces identical line wrapping.
```

**Accept when:** `make render-sample` writes 50 sample images + labels for visual
inspection; all tests pass; 100k-sample shard build completes and reports
characters-per-second throughput.

---

## T4 — Real unlabeled crop collection (pretraining pool only)

**Prompt:**
```
[paste CONTEXT block]

Write scripts/collect_real_crops.py: given a directory of Maltese-language
PDFs that I will provide (government gazettes, university theses, news —
collected manually, NOT the competition dev set), extract paragraph-level
image crops. Use pdfplumber or PyMuPDF to rasterize pages at 200–300 DPI and
a simple layout heuristic (or docTR detection) to crop paragraph blocks.
De-duplicate near-identical crops. No transcription needed — these are
unlabeled and used only for Stage 1 contrastive pretraining. Output webdataset
shards. Add a manifest recording source file + page for provenance, and a
README stating these are excluded from any supervised training and that the
competition dev set must never enter this pool.
```

**Accept when:** Given 20 sample PDFs, produces ≥ 1,000 clean paragraph crops;
manifest complete; spot-check of 30 random crops shows ≥ 90% are real paragraphs.

---

## T5 — Stage 1: SeqCLR contrastive pretraining

**Prompt:**
```
[paste CONTEXT block]

Implement src/pretrain/seqclr.py following this design (a reference sketch
exists in seqclr_trocr_pretrain.py — productionize it):
- Detach the ViT encoder from VisionEncoderDecoderModel
  ("microsoft/trocr-base-printed").
- SeqCLRHead: drop CLS token, reshape 576 patch tokens to 24x24, mean-pool the
  vertical axis -> 24 horizontal frames, AdaptiveAvgPool1d to n_windows
  (config, default 8), 2-layer MLP projection to 128-d, L2 normalize.
- Frame-level NT-Xent loss: positives = same window index across the two
  views; negatives = all other (image, window) instances in batch.
- Dataloader yields view pairs from two sources, mixed by config ratio:
  (a) renderer.render_pair (same text, two configs) and (b) real crops from T4
  with photometric-only augmentation pairs (blur/noise/JPEG/contrast; NO
  horizontal flips, NO horizontal crops/translation, NO re-wrapping).
- Training: AdamW, cosine schedule with warmup, bf16/fp16, gradient
  accumulation to fit 11 GB if needed, checkpointing, wandb-optional logging,
  resume support. Save encoder state_dict separately at each checkpoint.
- Config-driven via configs/stage1.yaml.
Include a linear-probe sanity eval: freeze encoder, train a tiny CTC head on
10k synthetic lines for 1 epoch, report CER — used only to compare encoder
checkpoints, not as the final model.
```

**Accept when:** Loss decreases steadily over a 100k-step run; linear-probe CER of
the pretrained encoder beats the un-pretrained baseline encoder; checkpoint
save/resume verified.

---

## T6 — Character-level decoder + Stage 2 supervised fine-tuning

**Prompt:**
```
[paste CONTEXT block]

Implement src/train/finetune.py:
- Build a character-level tokenizer over the provided competition character
  set plus BOS/EOS/PAD (src/data/char_tokenizer.py, with save/load).
- Replace the TrOCR decoder vocabulary: re-init decoder embedding + lm_head at
  the new (small) vocab size; keep decoder transformer layers. Make decoder
  depth configurable (option to shrink to 6 layers for the runtime budget).
- Load the Stage 1 encoder weights.
- Train end-to-end on synthetic (image, joined-paragraph) pairs with
  cross-entropy. Auxiliary loss: with probability p, a batch contains
  render_pair duplicates; add the SeqCLR-style style-invariance contrastive
  term on encoder features, weighted by lambda (config, default 0.1).
- Eval loop: CER (jiwer) on a held-out synthetic split AND on the official
  dev set (read-only, evaluation only). Log both; early-stop on dev CER.
- Mixed precision, grad accumulation, configs/stage2.yaml, resume support.
```

**Accept when:** Training runs end-to-end; dev-set CER reported and below the
trocr-base-printed zero-shot baseline; checkpoints + tokenizer artifacts saved
in HuggingFace-loadable format.

---

## T7 — Stage 3: confusable-character hard negatives

**Prompt:**
```
[paste CONTEXT block]

Implement src/train/hard_negatives.py:
- Minimal-pair generator: sample real korpus_malti words containing ħ/ġ/ċ/ż
  and produce counterfeit twins with the undotted/unbarred counterpart
  (ħin->hin), plus dash cases: same word rendered with a structural dash vs
  split across a line break with a hyphen.
- Two uses, both config-toggleable:
  (a) oversample minimal-pair renders into the Stage 2 training mix;
  (b) a character-contrastive auxiliary loss that pushes apart encoder frame
      embeddings at the differing character position between twin renders
      (positions known from the renderer's per-character layout metadata —
      extend renderer to emit char x-positions if not already available).
- Confusion-matrix eval: scripts/confusion_report.py aligns predictions vs
  ground truth (Levenshtein alignment) on dev and reports per-character
  substitution rates, highlighting ħ/h, ġ/g, ċ/c, ż/z and dash/hyphen errors.
```

**Accept when:** Confusion report runs on dev; after Stage 3 training the targeted
substitution rates drop relative to the Stage 2 checkpoint without overall dev
CER regressing.

---

## T8 — Submission packaging and resource validation

**Prompt:**
```
[paste CONTEXT block]

Implement src/infer/submission.py exposing the exact competition interface:
a class with transcribe(image: PIL.Image) -> str. Requirements:
- Loads model + char tokenizer from a HuggingFace repo id (weights pushed by
  scripts/push_to_hub.py) with local cache; must work fully offline once
  cached (pre-download step documented).
- Inference: preprocess, greedy + optional small beam (config), batch size 1.
- Optional fallback path (config flag): line-level OCR + the `malti` package
  rule-based line joiner, for A/B comparison against end-to-end paragraph
  decoding.
- scripts/validate_resources.py: simulates evaluation — loads the model cold,
  runs N dev images at batch size 1, reports peak VRAM (must be < 10 GB to
  leave headroom), per-image latency, and extrapolated total runtime vs the
  5-hour budget; checks total model+cache disk footprint < 20 GB.
Write an end-to-end test: fresh process, no network (mock/iptables-style env
var guard), transcribe 5 cached dev images successfully.
```

**Accept when:** Offline end-to-end test passes; resource report shows VRAM,
latency, and disk within budget with stated margins.

---

## T9 — Experiment tracking and final ablation

**Prompt:**
```
[paste CONTEXT block]

Write scripts/run_ablation.py and an EXPERIMENTS.md template. Run and record
dev CER for: (1) trocr-base-printed zero-shot, (2) Stage 2 only (no SeqCLR),
(3) Stage 1+2, (4) Stage 1+2 with real crops removed from pretraining,
(5) full pipeline with Stage 3, (6) end-to-end paragraph decoding vs
line+malti joiner. One config file per run, fixed seeds, results table in
EXPERIMENTS.md with checkpoint paths. The best run becomes the submission;
document the choice.
```

**Accept when:** Table complete for all 6 runs; submission checkpoint identified
and validated through T8.

---

## Dependency graph

```
T1 ──> T2 ──> T3 ──> T5 ──> T6 ──> T7 ──> T9
        └───> T4 ──┘         └──────> T8 ──┘
```

## Hard rules for the implementer

1. The official dev set is **evaluation only** — never in any training or
   pretraining pool, images or labels.
2. Ground-truth strings must follow the joined-paragraph convention:
   line-break hyphens removed, structural dashes preserved.
3. No horizontal flips / horizontal translation / re-wrapping between
   contrastive view pairs.
4. Everything config-driven and resumable; assume training will be
   interrupted.
5. Validate against the 11 GB VRAM / 5 h / 20 GB disk / offline constraints
   early (T8 can run with the zero-shot baseline before T6 finishes).
