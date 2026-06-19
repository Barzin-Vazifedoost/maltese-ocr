# Maltese OCR Competition — Project Log

## Goal
Build the best OCR pipeline for Maltese text for a university competition.
Baseline: Tesseract with the official Maltese language model (`mlt.traineddata`), CER **0.036** on 422 dev-set images.

---

## Git practices

- One commit per logical piece; use Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, …).
- Run `python3 -m ruff format` and `python3 -m ruff check` before each commit.
- **Do NOT add a `Co-Authored-By: Claude` trailer** to commit messages. The co-author line is appended automatically by the local git setup, so adding it manually duplicates it.

---

## Repository layout

```
maltese-ocr/
├── competition_transcriber.py   # Primary OCR class used for scoring
├── test_baseline.py             # Evaluates transcriber on dev set, produces results.json
├── generate_data.py             # Generates 5000 synthetic training images from real Maltese text
├── train.py                     # Fine-tunes TrOCR — auto-detects CUDA / MPS / CPU
├── run_colab.ipynb              # 4-cell Colab notebook (mount Drive, pip, check GPU, run train)
├── transcribe.py                # Stub for final competition submission
├── requirements.txt             # Python dependencies
├── results.json                 # Output of test_baseline.py — 422 entries sorted by CER desc
├── data/
│   ├── texts.json               # 422 ground-truth transcriptions for the dev set
│   ├── dev_set/                 # 423 JPG images from the competition (001.jpg … 422.jpg)
│   └── synthetic/
│       ├── images/              # 5000 rendered paragraph images (syn_000001.jpg …)
│       └── transcriptions.json  # Ground-truth text for each synthetic image
└── models/
    └── trocr-maltese/           # Fine-tuned TrOCR checkpoint (from Colab training)
```

---

## CER progression

| Stage | CER | Notes |
|-------|-----|-------|
| Tesseract PSM 3 (original baseline) | 0.036 | Default Tesseract mode |
| Tesseract PSM 6 | 0.0237 | Single block mode — matched organizers' 0.023 |
| + ImageMagick preprocessing fallback (2× at <150 px) | 0.0225 | Fallback for empty-output images |
| + 3× upscale at <200 px (**current best**, `targeted-fixes` branch) | **0.0221** | Stronger upscaling in fallback path |

---

## Phase 1 — Tesseract baseline

**What was done**
- Installed Tesseract via Homebrew (`brew install tesseract tesseract-lang`).
- Downloaded the NOMOCRAT fine-tuned Maltese model (`mlt.traineddata`) and placed it in the Tesseract `tessdata` directory.
- Wrote `competition_transcriber.py` — a class with a public `.transcribe(image)` method.
- Wrote `test_baseline.py` — loads `data/texts.json`, runs the transcriber on every dev-set image, computes CER with `jiwer.cer()`, saves `results.json` (sorted highest→lowest CER), prints the 10 hardest images and the average CER.

**Tesseract result: average CER 0.036** (on 422 images)

---

## Phase 2 — Synthetic training data (`generate_data.py`)

**Purpose**: Create 5000 labelled paragraph images to fine-tune TrOCR.

**Key design decisions**

| Choice | Value / Reason |
|--------|---------------|
| Corpus | `MLRS/korpus_malti` (HuggingFace, gated — need `huggingface-cli login`) |
| Sentence splitting | `malti.KMSentSplitter` |
| Character filter | Exact competition vocab (see `ALLOWED_CHARS` in the script) |
| Font families | Times New Roman, Georgia (serif); Arial, Verdana, Trebuchet MS (sans-serif) — all from `/System/Library/Fonts/Supplemental/` |
| Font sizes | 10–14 pt, converted to pixels at 96 DPI |
| Image widths | 400–900 px (random) |
| Paragraph length | 1–15 consecutive sentences from same source document |
| Background | 80% white, 20% light pastel (pre-assigned to guarantee exactly 20%) |
| Style weights | Regular 10%, Bold 30%, Italic 30%, Bold-Italic 30% |
| Justification | Full justification; left-aligned if line fill < 75% to avoid ugly gaps |
| Height safety | 20% buffer on top of measured text-block height to prevent clipping |
| Bottom padding | Capped at 1.5× line height (never less than top inset) |

**Bugs fixed during development**

1. `random.randint(3, 2)` crash on short documents — fixed to `random.randint(1, min(8, len(doc)))` (later widened to 15).
2. `trust_remote_code` parameter rejected by newer `datasets` — removed it.
3. Coloured backgrounds never appearing in smoke tests (statistical bad luck) — fixed with pre-assignment.
4. Text clipping for bold/italic fonts — `getmetrics()` under-reports descenders. Fixed by measuring actual pixel height with `font.getbbox("Ħġpqjy|")`.
5. Ugly word gaps on short justified lines — threshold raised from 60% to 75%.

---

## Phase 3 — TrOCR fine-tuning

### Model
`microsoft/trocr-base-handwritten` — a Vision Encoder–Decoder transformer.

### Training script (`train.py`) — single file, auto-detects the device
`train.py` was originally split into a Mac version (`train.py`) and a Colab
version (`train_colab.py`); these were near-identical copies that had to be
kept in sync by hand. They were merged into one `train.py` (Section 18 in
`documentation.md`) that picks the device and matching settings at runtime via
a `select_device()` helper:

| Device | `num_workers` | `pin_memory` | Epochs | Notes |
|--------|---------------|--------------|--------|-------|
| CUDA (NVIDIA T4, e.g. Colab) | 2 | True | 5 | T4 is fast; each epoch ~25–35 min |
| MPS (Apple Silicon GPU)      | 0 | False | 10 | `num_workers=0` required on macOS (DataLoader hangs otherwise) |
| CPU (fallback)               | 0 | False | 10 | Slow |

Common to all: batch size 8, learning rate 5e-5, saves the best checkpoint
(lowest val loss) to the model directory. Data/save paths auto-detect Colab:
if a mounted Drive exists at `/content/drive/MyDrive/maltese-OCR/`, it reads/
writes there; otherwise it uses the local repo paths (`data/synthetic/`,
`models/trocr-maltese/`).

- **Mac note**: MPS is ~18× slower than a T4; a full MPS run was estimated at
  15–18 hours, so training was done on Colab instead.

**Important bug**: `train.py` calls `SAVE_DIR.mkdir()` at the start of the training loop (before any epoch completes). So the directory `models/trocr-maltese/` can exist but be empty. `competition_transcriber.py` was updated to check for actual model files (`config.json`, `pytorch_model.bin`, or `model.safetensors`) rather than just directory existence — otherwise it would crash with an `OSError` trying to load from an empty directory.

### Colab notebook (`run_colab.ipynb`)
4 cells:
1. Mount Google Drive
2. `pip install torch transformers pillow datasets malti tqdm jiwer`
3. Check CUDA availability
4. `!python train.py`

**Training was completed on Colab**. The resulting checkpoint was downloaded from Google Drive and placed in `models/trocr-maltese/` locally.

---

## Phase 4 — Inference pipeline (`competition_transcriber.py`)

**Inference order**
1. Fine-tuned TrOCR (if `models/trocr-maltese/` contains `config.json` or a model weight file)
2. Tesseract fallback (if TrOCR is unavailable or returns an empty string)

**TrOCR inference settings**
- Beam search: `num_beams=4`, `max_new_tokens=256`, `early_stopping=True`
- Device: MPS if available, else CPU
- Image preprocessing: `.convert("RGB")` then `TrOCRProcessor`

**Tesseract settings**
- Language: `-l mlt`
- PSM: `--psm 6` (single uniform block of text — critical for accuracy; default PSM 3 gives CER 0.034, PSM 6 gives CER 0.024 matching organizer baseline)
- Post-processing: `RBLineJoiner.join_lines(..., fix_hyphenated_words=False)` (`fix_hyphenated_words` has no measurable effect on this dataset)

**ImageMagick preprocessing fallback** (`wand` library, `brew install imagemagick`)
- Triggered only when Tesseract returns < 3 characters (7 images had empty output under old PSM 3 config)
- Pipeline: upscale 3× if height < 200 px → grayscale → contrast +50 → adaptive threshold
- `MAGICK_HOME=/opt/homebrew` set at module level so wand finds the Homebrew dylib
- Install: `brew install imagemagick && pip install wand`

---

## Phase 5 — Targeted fixes investigation (`targeted-fixes` branch)

### PSM investigation
Compared PSM 3 vs PSM 6 vs PSM 7/11 on the 7 images returning empty output under PSM 3.
**Finding**: PSM 6 ("single uniform block") is best for competition images (text snippets/crops).
PSM 3 uses full auto layout detection which fails on narrow or small images.

### What worked
- **PSM 6**: CER 0.036 → 0.0237 (matches organizers' 0.023)
- **ImageMagick preprocessing fallback** (3× upscale at <200 px): CER 0.0237 → 0.0221
  - Only triggers on images where Tesseract returns < 3 chars
  - 102.jpg: CER 1.000 → 0.354 with 3× upscaling (was 0.523 with 2× at <150 px)

### What did NOT work (tried and reverted)
- **2% border crop**: CER jumped from 0.0225 → 0.0348. Competition images have text flush to the edge; cropping removed real characters.
- **Digit → em-dash replacement** (`203-249` → `203—249`): Made 12 images worse. Maltese academic texts use real hyphens in index ranges and ISBNs; the ground truth has hyphens, not em dashes.
- **Leading lowercase char removal** (`f word` → `word`): Only helped 1 image (225.jpg), harmed 0, but not worth keeping given the narrow scope.
- **`fix_hyphenated_words=True`**: No measurable effect vs `False` on this dataset.

---

## Evaluation (`test_baseline.py`)

Loads `data/texts.json` (422 entries), runs `.transcribe()` on each image, computes `jiwer.cer(reference, hypothesis)`, saves `results.json` sorted by CER descending. Now also prints:
- CER without preprocessing fallback vs with
- Side-by-side comparison for the 7 previously-blank images with `[IMPROVED/SAME/WORSE]` tags

**Current best CER: 0.0221** (Tesseract PSM 6 + ImageMagick fallback with 3× upscale, `targeted-fixes` branch)

---

## How to reproduce

### Prerequisites
```bash
brew install tesseract tesseract-lang imagemagick
pip install torch transformers pillow datasets malti tqdm jiwer huggingface_hub wand
huggingface-cli login   # needed for MLRS/korpus_malti
```

### Run evaluation
```bash
python3 test_baseline.py
```

### Generate synthetic data
```bash
python3 generate_data.py
# Produces data/synthetic/images/ (5000 jpg) and data/synthetic/transcriptions.json
```

### Fine-tune locally (Mac MPS / CPU — slow)
```bash
python3 train.py
# Auto-detects MPS or CPU and uses local paths (data/synthetic/, models/trocr-maltese/).
```

### Fine-tune on Colab (recommended)
1. Upload `data/synthetic/` to `MyDrive/maltese-OCR/synthetic/` on Google Drive.
2. Upload `train.py` to your Colab session.
3. Open `run_colab.ipynb` in Colab and run all cells.
   `train.py` auto-detects the CUDA GPU and the mounted Drive paths.
4. Download `MyDrive/maltese-OCR/models/trocr-maltese/` to `models/trocr-maltese/` locally.

### Evaluate fine-tuned model
```bash
python3 test_baseline.py
# competition_transcriber.py auto-detects the model in models/trocr-maltese/
```

---

## Key numbers

| Metric | Value |
|--------|-------|
| Dev set images | 422 |
| Original Tesseract baseline CER | 0.036 |
| Current best CER | **0.0221** |
| Organizers' reference CER | 0.023 |
| Synthetic training images | 5000 |
| TrOCR fine-tune epochs (CUDA/Colab) | 5 |
| TrOCR fine-tune epochs (MPS/CPU) | 10 |
| Train/val split | 90% / 10% |
| Batch size | 8 |
| Learning rate | 5e-5 |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch — model training and inference |
| `transformers` | HuggingFace — TrOCR model and processor |
| `pillow` | Image loading and rendering |
| `datasets` | Stream `MLRS/korpus_malti` from HuggingFace |
| `malti` | Maltese NLP: `RBLineJoiner`, `KMSentSplitter` |
| `tqdm` | Progress bars |
| `jiwer` | CER calculation |
| `tesseract` | Tesseract OCR engine (system install via brew) |
| `wand` | Python bindings for ImageMagick (preprocessing fallback) |
| `imagemagick` | Image processing library (system install via brew) |

---

## 16. Preprocessing Improvements — Round 2

### Full CER progression

| Stage | CER | Delta |
|-------|-----|-------|
| Tesseract PSM 3 (original baseline) | 0.036 | — |
| Tesseract PSM 6 | 0.0237 | −0.0123 |
| + ImageMagick fallback (2× upscale at <150 px) | 0.0225 | −0.0012 |
| + 3× upscale at <200 px (`targeted-fixes` branch) | **0.0221** | −0.0004 |

### What was tried

**1. 2% border crop** (`_crop_border` method, now removed)
Cropped 2% from each edge before passing to Tesseract, intending to remove border noise.
Result: CER jumped from 0.0225 → **0.0348** (+0.0123 regression).
Why it failed: Competition images regularly have text printed flush to the edge. A 2% crop on a 210 px-wide image removes ~4 px — enough to cut off the leading character of a word. Reverted.

**2. Digit → em-dash replacement** (post-processing regex, now removed)
Replaced any `digit-digit` pattern with an em dash, e.g. `203-249` → `203—249`.
Result: Made **12 images worse**, 0 better. Net CER increase.
Why it failed: Maltese academic texts use ordinary hyphens in index page ranges (`16(29-31)`) and ISBNs (`978-87-92387-48-6`). The ground truth has hyphens, not em dashes. Reverted.

**3. Leading lowercase character removal** (post-processing regex, now removed)
Stripped a single lowercase letter at the start of output, e.g. `f Kelma` → `Kelma`.
Result: Helped 1 image (225.jpg), harmed 0. Not worth keeping given the minimal scope. Reverted.

**4. 3× upscale at <200 px** (kept — current implementation)
Changed the ImageMagick preprocessing fallback from 2× at <150 px to 3× at <200 px.
Only triggers when Tesseract returns fewer than 3 characters on the raw image.
Result: CER 0.0225 → **0.0221**.

### Per-image impact: 102.jpg

102.jpg is the hardest remaining image — 359×105 px with small text and a cluttered background. It was blank under PSM 3 (CER 1.000).

| Fallback config | Predicted (truncated) | CER |
|---|---|---|
| No fallback (PSM 6 only) | *(empty)* | 1.000 |
| 2× upscale at <150 px | `ika iii i ii ge al g, Il-provvista ta' tagħlim...` | 0.523 |
| 3× upscale at <200 px | `aq A) il-provvista ta' tagħlim q professjonali kontinwu dwar` | **0.354** |

Ground truth: `Il-provvista ta' tagħlim professjonali kontinwu dwar l inklużjoni`

The stronger upscale gives Tesseract more pixel resolution to work with, recovering more of the correct text despite the noise.

### Final state of `_apply_preprocessing`

```python
def _apply_preprocessing(self, image):
    if wimg.height < 200:
        wimg.resize(wimg.width * 3, wimg.height * 3)  # was: 2× at <150 px
    wimg.transform_colorspace("gray")
    wimg.brightness_contrast(brightness=0, contrast=50)
    wimg.adaptive_threshold(width=wimg.width//8, height=wimg.height//8, offset=0)
```

Triggered only when `_run_tesseract` returns fewer than 3 characters on the original image.
