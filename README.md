# Maltese OCR — DocEng 2026 Competition

## Overview
- **Competition:** OCRs for Corpus Extraction for the Maltese Language
- **Event:** 26th ACM Symposium on Document Engineering (DocEng 2026)
- **Deadline:** June 30, 2026
- **Task:** Transcribe cropped paragraph images from Maltese PDFs into text
- **Metric:** Character Error Rate (CER) — lower is better
- **Organizers:** Marc Tanti, Stefania Cristina, Alexandra Bonnici (University of Malta)

## Current Best Result
| Model | CER | Branch |
|---|---|---|
| Tesseract PSM6 + NOMOCRAT + ImageMagick + em-dash post-processing | **0.0196** | targeted-fixes |
| Organizers' best baseline | 0.0234 | — |
| Donut (zero-shot) | 0.217 | — |
| GOT-OCR2.0 (zero-shot) | 0.156 | — |

## Project Structure
```
maltese-ocr/
├── src/maltese_ocr/             # New staged pipeline (in development)
│   ├── render/                  # Synthetic image renderer (T3 ✅)
│   ├── data/                    # Corpus streamer (T3 ✅)
│   ├── pretrain/                # SeqCLR pretraining (T5 ✅)
│   ├── train/                   # Supervised fine-tuning (T6 ⏳)
│   └── infer/                   # Submission interface (T8 ⏳)
├── scripts/                     # Dataset generation, font audit, packaging
├── configs/                     # Stage YAML configs + charset
├── fonts/                       # 76 validated Maltese-capable fonts
├── data/
│   └── dev_set/                 # 422 real competition images (evaluation only)
├── competition_transcriber.py   # Current working submission (CER 0.0196)
├── test_baseline.py             # Evaluation script
├── train.py                     # Legacy TrOCR trainer
└── documentation.md             # Full project documentation
```

## Pipeline

### Current submission (working, beats baseline)
```
Image → Tesseract PSM6 + NOMOCRAT → malti line joiner → em-dash post-processing → text
  ↓ (if empty)
ImageMagick preprocessing → Tesseract retry
```

### New staged pipeline (in development)
- **Stage 1 (T5):** SeqCLR contrastive pretraining of ViT encoder
- **Stage 2 (T6):** Supervised fine-tuning with character-level decoder
- **Stage 3 (T7):** Hard-negative minimal pairs for Maltese confusable chars

## Setup
```bash
git clone https://github.com/Barzin-Vazifedoost/maltese-ocr.git
cd maltese-ocr
pip install -e ".[dev]"
make setup
make test
```

## Generate Synthetic Training Data
```bash
python3 scripts/build_dataset.py --n 5000 --out data/synthetic_v3
```

## Evaluation
```bash
python3 test_baseline.py
```

## Makefile Targets
| Target | Status | Description |
|---|---|---|
| `make setup` | ✅ Working | Install deps + pre-commit |
| `make test` | ✅ Working | Run fast tests |
| `make test-all` | ✅ Working | Run all tests including slow |
| `make render-sample` | ✅ Working | Generate 20 sample images |
| `make eval` | ✅ Working | Run CER evaluation |
| `make pretrain` | ✅ Working | T5 SeqCLR Stage 1 (needs base model; GPU for full runs) |
| `make pretrain-smoke` | ✅ Working | 2-step CPU/offline end-to-end check of the pretrain loop |
| `make train` | ⏳ Stub | T6 fine-tuning (`train/run.py` not yet implemented) |
| `make package` | ⏳ Blocked | `scripts/package_model.py` is implemented but needs a Stage 3 checkpoint (`make train`) |

## Competition Rules
- Model must be published on HuggingFace
- Must run offline in under 5 hours on RTX 2080 Ti
- Max 20GB disk space
- Batch size 1 enforced during evaluation
- No internet access during evaluation except HuggingFace model download

## Key Dates
- Competition deadline: June 30, 2026, 23:59 AoE
- Conference: August 25–28, 2026, Fribourg Switzerland
