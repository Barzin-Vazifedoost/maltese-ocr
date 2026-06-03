# Maltese OCR Competition — Project Log

## Goal
Build the best OCR pipeline for Maltese text for a university competition.
Baseline: Tesseract with the official Maltese language model (`mlt.traineddata`), CER **0.036** on 422 dev-set images.

---

## Repository layout

```
maltese-ocr/
├── competition_transcriber.py   # Primary OCR class used for scoring
├── test_baseline.py             # Evaluates transcriber on dev set, produces results.json
├── generate_data.py             # Generates 5000 synthetic training images from real Maltese text
├── train.py                     # Fine-tunes TrOCR locally on Apple Silicon (MPS)
├── train_colab.py               # Fine-tunes TrOCR on Google Colab (CUDA / T4)
├── run_colab.ipynb              # 4-cell Colab notebook (mount Drive, pip, check GPU, run train)
├── transcribe.py                # Stub for final competition submission
├── requirements.txt             # Python dependencies (add as needed)
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

### Mac version (`train.py`)
- Device: MPS (Apple Silicon GPU)
- `num_workers=0` — required on macOS (multiprocessing in DataLoader hangs otherwise)
- 10 epochs, batch size 8, learning rate 5e-5
- Saves best checkpoint (lowest val loss) to `models/trocr-maltese/`
- **Problem**: MPS is ~18× slower than a T4 GPU; estimated 15–18 hours per run.
- Mac training was abandoned in favour of Colab.

**Important bug**: `train.py` calls `SAVE_DIR.mkdir()` at the start of the training loop (before any epoch completes). So the directory `models/trocr-maltese/` can exist but be empty. `competition_transcriber.py` was updated to check for actual model files (`config.json`, `pytorch_model.bin`, or `model.safetensors`) rather than just directory existence — otherwise it would crash with an `OSError` trying to load from an empty directory.

### Colab version (`train_colab.py`)
- Device: CUDA (NVIDIA T4)
- `num_workers=2`, `pin_memory=True` — safe on Linux, faster data loading
- 5 epochs (T4 is fast; each epoch ~25–35 min)
- Data paths: `/content/drive/MyDrive/maltese-OCR/synthetic/`
- Save path: `/content/drive/MyDrive/maltese-OCR/models/trocr-maltese/`

### Colab notebook (`run_colab.ipynb`)
4 cells:
1. Mount Google Drive
2. `pip install torch transformers pillow datasets malti tqdm jiwer`
3. Check CUDA availability
4. `!python train_colab.py`

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

---

## Evaluation (`test_baseline.py`)

Loads `data/texts.json` (422 entries), runs `.transcribe()` on each image, computes `jiwer.cer(reference, hypothesis)`, saves `results.json` sorted by CER descending, prints top 10 hardest images and average CER.

**Tesseract-only baseline: CER 0.036**

TrOCR fine-tuned evaluation: pending (model in `models/trocr-maltese/` but `test_baseline.py` has not successfully completed a run yet as of the end of the last session).

---

## How to reproduce

### Prerequisites
```bash
brew install tesseract tesseract-lang
pip install torch transformers pillow datasets malti tqdm jiwer huggingface_hub
huggingface-cli login   # needed for MLRS/korpus_malti
```

### Run Tesseract baseline
```bash
python3 test_baseline.py
```

### Generate synthetic data
```bash
python3 generate_data.py
# Produces data/synthetic/images/ (5000 jpg) and data/synthetic/transcriptions.json
```

### Fine-tune on Mac (slow)
```bash
python3 train.py
```

### Fine-tune on Colab (recommended)
1. Upload `data/synthetic/` to `MyDrive/maltese-OCR/synthetic/` on Google Drive.
2. Upload `train_colab.py` to your Colab session.
3. Open `run_colab.ipynb` in Colab and run all cells.
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
| Tesseract baseline CER | 0.036 |
| Synthetic training images | 5000 |
| TrOCR fine-tune epochs (Colab) | 5 |
| TrOCR fine-tune epochs (Mac) | 10 (abandoned) |
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
