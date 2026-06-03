# Maltese OCR Competition — Full Project Documentation

**Competition:** OCRs for Corpus Extraction for the Maltese Language  
**Event:** 26th ACM Symposium on Document Engineering (DocEng 2026\)  
**Venue:** HES-SO / University of Fribourg, Switzerland — August 25–28, 2026  
**Submission Deadline:** June 30, 2026, 23:59 AoE  
**Team:** First-year CS undergraduate

---

## Table of Contents

1. [Competition Overview](#1-competition-overview)  
2. [Development Environment](#2-development-environment)  
3. [Dev Set Analysis](#3-dev-set-analysis)  
4. [Baseline — Tesseract \+ NOMOCRAT](#4-baseline--tesseract--nomocrat)  
5. [Baseline Evaluation Results](#5-baseline-evaluation-results)  
6. [Error Analysis](#6-error-analysis)  
7. [Synthetic Data Generation](#7-synthetic-data-generation)  
8. [TrOCR Fine-tuning](#8-trocr-fine-tuning)  
9. [TrOCR Evaluation Results](#9-trocr-evaluation-results)  
10. [GOT-OCR2.0 Evaluation](#10-got-ocr20-evaluation)  
11. [Donut Evaluation](#11-donut-evaluation)  
12. [Model Comparison](#12-model-comparison)  
13. [Organizer Baselines](#13-organizer-baselines)  
14. [Next Steps](#14-next-steps)  
15. [Project File Structure](#15-project-file-structure)

---

## 1\. Competition Overview

### Task

Build an OCR model that transcribes images of Maltese text paragraphs extracted from PDFs. The input is a rectangular JPG image containing a single paragraph of text. The output must be the full paragraph as a single clean string with line-break hyphens resolved.

### Key Challenges

- **No training set provided** — all training data must be generated synthetically  
- **Paragraph-level output required** — not a list of lines  
- **Maltese-specific characters** — ħ, għ, ċ, ż, ġ, Ħ, Ġ, Ċ, Ż  
- **Hyphen ambiguity** — Maltese uses dashes structurally (e.g. "il-kelb" \= "the dog"), making it non-trivial to distinguish line-break hyphens from structural dashes  
- **Mixed languages** — some paragraphs contain English or other languages

### Evaluation Metric

Character Error Rate (CER) — lower is better. Ties broken by shortest runtime on the organizers' computer.

### Submission Requirements

- Python class with `__init__` and `transcribe(image: PIL.Image) -> str` methods  
- Model publicly published on HuggingFace  
- Must run in under 5 hours on the organizers' Windows 11 machine  
- All resources must fit within 20GB disk space

### Organizers

- Marc Tanti — University of Malta, Institute of Linguistics and Language Technology  
- Stefania Cristina — University of Malta, Department of Systems & Control Engineering  
- Alexandra Bonnici — University of Malta, Department of Systems & Control Engineering

### Provided Resources

- Development set: cropped paragraph images \+ `texts.json` ground truth  
- `char_set.json`: complete list of characters in both dev and test sets  
- `competition_transcriber.py`: submission template  
- `example_competition_transcriber_tesseract.py`: Tesseract example  
- `example_competition_transcriber_donut.py`: Donut example  
- `requirements.txt`: organizer dependencies  
- MLRS/korpus\_malti corpus on HuggingFace  
- `malti` Python package (v0.3.1) with line joiner and sentence splitter

### Evaluation Computer Specifications

- OS: Windows 11  
- Processor: Intel Core i7-9700K @ 3.60GHz  
- RAM: 32 GB  
- GPU: NVIDIA GeForce RTX 2080 Ti (11 GB VRAM)  
- Python: v3.9 with conda v4.12

---

## 2\. Development Environment

### Hardware

- MacBook Pro with Apple Silicon (M-series chip)  
- MPS (Metal Performance Shaders) GPU available via PyTorch  
- Python 3.13

### Key Packages Installed

malti==0.3.1

sentence-splitter==1.4

Pillow

torch (with MPS support)

transformers

datasets

tqdm

jiwer

huggingface\_hub

### malti Package — Submodules Discovered

malti.data

malti.line\_joiner

malti.line\_joiner.rb\_line\_joiner.rb\_line\_joiner  ← RBLineJoiner

malti.sent\_splitter

malti.sent\_splitter.km\_sent\_splitter.km\_sent\_splitter  ← KMSentSplitter

malti.tokeniser

### Key malti APIs Used

**Sentence Splitter:**

from malti.sent\_splitter.km\_sent\_splitter.km\_sent\_splitter import KMSentSplitter

splitter \= KMSentSplitter()

sentences \= splitter.split("Il-kelb qabad il-gurdien. Il-qattus kien hemm ukoll.")

\# → \['Il-kelb qabad il-gurdien.', 'Il-qattus kien hemm ukoll.'\]

**Line Joiner:**

from malti.line\_joiner.rb\_line\_joiner.rb\_line\_joiner import RBLineJoiner

joiner \= RBLineJoiner()

result \= joiner.join\_lines(\["Il-kelb qabad", "il-gurdien."\], fix\_hyphenated\_words=True)

\# → 'Il-kelb qabad il-gurdien.'

---

## 3\. Dev Set Analysis

### Dataset Statistics

- **Total images:** 422 JPG files  
- **Ground truth:** `texts.json` — list of objects with `image`, `text`, and `as_lines` fields

### JSON Structure

{

  "image": "013.jpg",

  "text": "Madankollu, is-serje ta' ambjenti...",

  "as\_lines": \[

    "Madankollu, is-serje ta' ambjenti u tipi ta'...",

    "wieħed isib fil-pajjiżi jagħfsu..."

  \]

}

### Visual Analysis of Dev Set Images

By examining sample images (013.jpg, 014.jpg, 015.jpg, 016.jpg, 017.jpg, 018.jpg), the following characteristics were identified:

| Property | Observation |
| :---- | :---- |
| Background | Mostly white; some coloured (yellow, green) |
| Font families | Serif (Times New Roman style) and sans-serif (Arial style) |
| Text styles | Regular, bold, italic, bold-italic mixed within paragraphs |
| Alignment | Justified |
| Paragraph length | 1 line to 15+ lines |
| Font sizes | \~10–13pt for body text; larger for headings |
| Special characters | ħ, għ, ċ, ż, ġ and uppercase variants |
| Languages | Mostly Maltese; some English; occasional other languages |

### Challenging Images Identified

- `054.jpg`, `096.jpg`, `106.jpg` — bold white text on yellow background  
- `214.jpg` — hand-drawn/comic style font  
- `227.jpg` — red 3D graffiti-style text on green background  
- `417.jpg` — very long single line getting truncated

### Allowed Character Set (from char\_set.json)

space \! " & ' ( ) \+ , \- . / 0–9 : ; \= ? A–Z \[ \] \_ a–z

© ² ¹ Ø à ç é ì ñ ò ó ô ö ø ú ü ā Ċ ċ Ġ ġ Ħ ħ ł Ż ż ỹ

— ' ' " " • ⁴ € ♢

---

## 4\. Baseline — Tesseract \+ NOMOCRAT

### Approach

Use Tesseract OCR with the NOMOCRAT fine-tuned Maltese language model, followed by the malti rule-based line joiner to produce paragraph-level output.

### Setup Steps

**1\. Install Tesseract on macOS:**

brew install tesseract tesseract-lang

**2\. Download NOMOCRAT fine-tuned model:**

curl \-L https://raw.githubusercontent.com/vanyagelfo/NOMOCRAT-OCR/main/tessdata\_custom/mlt\_custom\_v1.traineddata \\

  \-o /opt/homebrew/opt/tesseract/share/tessdata/mlt.traineddata

**3\. competition\_transcriber.py (Mac-adapted):**

import subprocess

import tempfile

import os

import PIL.Image

from malti.line\_joiner.rb\_line\_joiner.rb\_line\_joiner import RBLineJoiner

class CompetitionTranscriber:

    def \_\_init\_\_(self) \-\> None:

        self.line\_joiner \= RBLineJoiner()

    def transcribe(self, image: PIL.Image) \-\> str:

        with tempfile.TemporaryDirectory() as path:

            image.save(os.path.join(path, 'img.jpg'))

            subprocess.run(

                \['tesseract', '-l', 'mlt',

                 os.path.join(path, 'img.jpg'),

                 os.path.join(path, 'out')\],

                stdout=subprocess.DEVNULL,

                stderr=subprocess.DEVNULL,

            )

            with open(os.path.join(path, 'out.txt'), encoding='utf-8') as f:

                text \= self.line\_joiner.join\_lines(

                    f.read().strip().split('\\n'),

                    fix\_hyphenated\_words=True

                )

        return text

---

## 5\. Baseline Evaluation Results

### Overall Statistics

| Metric | Value |
| :---- | :---- |
| Total images | 422 |
| Perfect predictions (CER \= 0.0) | 225 (53%) |
| Blank predictions (CER \= 1.0) | 7 (1.7%) |
| Partial errors (0 \< CER \< 1\) | 190 (45%) |
| **Average CER** | **0.036** |

### Sample Results

\=== 013.jpg \===

PREDICTED: Madankollu, is-serje ta' ambjenti...

EXPECTED:  Madankollu, is-serje ta' ambjenti...

CER: 0.000

\=== 016.jpg \===

PREDICTED: Is-sistema — ara Figura 1 — tikkonsisti minn tliet elementi: /nput u riżorsi...

EXPECTED:  Is-sistema — ara Figura 1 — tikkonsisti minn tliet elementi: Input u riżorsi...

CER: 0.003

---

## 6\. Error Analysis

### Error Categories Identified

**1\. Blank predictions (CER 1.0) — 7 images** Caused by unusual visual styles that Tesseract cannot handle:

- Bold white text on coloured backgrounds (yellow, green)  
- Hand-drawn/comic fonts  
- 3D graffiti-style text

**2\. Character substitutions**

- Capital `I` read as `/` (e.g. `Input` → `/nput`)  
- `y` read as `v` (e.g. `complementary` → `complementarv`, `Ruddy` → `Ruddv`)  
- `ø` dropped (e.g. `Jørgen` → `Jorgen`)

**3\. Punctuation errors**

- Curly quotes `"` `"` read as straight quotes `'` `'`

**4\. Line truncation**

- Very long single lines getting cut off mid-sentence

### Mid-Range Error Examples (CER 0.05–0.4)

231.jpg (CER=0.053): complementary → complementarv, payments → pavments

218.jpg (CER=0.061): "Lil Anaktorja" → 'Lil Anaktorja' (quote style)

082.jpg (CER=0.083): Jørgen Greve → Jorgen Greve (ø dropped)

---

## 7\. Synthetic Data Generation

### Motivation

No training set is provided. A synthetic dataset must be created by rendering real Maltese text as images mimicking the visual style of the dev set.

### Text Source

MLRS/korpus\_malti dataset on HuggingFace — a large corpus of real Maltese text.

### Fonts Available on macOS Supporting Maltese Characters

| Font | Style | Type |
| :---- | :---- | :---- |
| Times New Roman | Regular, Bold, Italic, Bold-Italic | Serif |
| Georgia | Regular, Bold, Italic, Bold-Italic | Serif |
| Arial | Regular, Bold, Italic, Bold-Italic | Sans-serif |
| Verdana | Regular, Bold, Italic, Bold-Italic | Sans-serif |
| Trebuchet MS | Regular, Bold, Italic, Bold-Italic | Sans-serif |

### Generation Pipeline (generate\_data.py)

1. **Load corpus** — stream MLRS/korpus\_malti from HuggingFace  
2. **Split sentences** — use `KMSentSplitter` to split corpus into sentences  
3. **Filter characters** — remove sentences containing characters outside the allowed set  
4. **Build coherent paragraphs** — group consecutive sentences from the same document (not random mixing)  
5. **Render images** — use Pillow with:  
   - Random image width: 400–900px  
   - Random font family (from 5 families above)  
   - Random style: regular, bold, italic, bold-italic  
   - Justified text alignment (only for lines ≥75% full)  
   - Font size: 10–14pt  
   - 80% white background, 20% light coloured background  
   - Dynamic image height (no clipping)  
6. **Save outputs** — JPG images \+ `transcriptions.json`

### Dataset Statistics

| Metric | Value |
| :---- | :---- |
| Total images generated | 5000 |
| Total disk size | 78 MB |
| Average image size | \~15.6 KB |
| JSON entries | 5000 |

### Output Format

{

  "syn\_000001.jpg": "JESMOND MUGLIETT: Mr Speaker, naturalment...",

  "syn\_000002.jpg": "b'kapaċità tal-proċess ta' 15000 litru,"

}

---

## 8\. TrOCR Fine-tuning

### Model

`microsoft/trocr-base-handwritten` — a Vision Encoder-Decoder model from Microsoft that takes an image and outputs text. Pre-trained on handwritten text datasets.

### Why TrOCR

- Available on HuggingFace (required by competition)  
- Designed specifically for OCR (image → text)  
- Beginner-friendly with good documentation  
- Works with standard PyTorch training loop

### Training Configuration

| Parameter | Value |
| :---- | :---- |
| Base model | microsoft/trocr-base-handwritten |
| Training images | 4500 (90% of 5000\) |
| Validation images | 500 (10% of 5000\) |
| Batch size | 8 |
| Learning rate | 5e-5 |
| Max target length | 256 tokens |
| Optimizer | AdamW |

### Infrastructure

- **Initial attempt:** Mac with Apple Silicon (MPS) — estimated 15–18 hours for 10 epochs  
- **Solution:** Google Colab with free Tesla T4 GPU — \~25 minutes per epoch

### Initial Training Run (5 epochs on Colab)

| Epoch | Train Loss | Val Loss |
| :---- | :---- | :---- |
| 1 | 3.8251 | 3.0640 ✓ |
| 2 | 2.3295 | 2.1157 ✓ |
| 3 | 2.0579 | 1.9330 ✓ |
| 4 | 1.6599 | 1.8418 ✓ |
| 5 | 1.6228 | 1.8138 ✓ |

Model saved to Google Drive at `maltese-OCR/models/trocr-maltese/`.

### Continued Training (20 epochs)

- Learning rate reduced to 1e-5 (standard practice when continuing from checkpoint)  
- Loaded from saved 5-epoch checkpoint  
- Currently running on Colab T4

---

## 9\. TrOCR Evaluation Results (5 epochs)

### Issues Observed

The 5-epoch model showed significant problems:

**Repetition loops:**

PREDICTED: Kummissjoni għall-Kummissjoni għall-Kummissjoni għall-Kummissjoni...

EXPECTED:  Madankollu, is-serje ta' ambjenti u tipi ta' provedimenti...

**Hallucinations:**

PREDICTED: Brinet, Joseph, Joseph M. Maltese and other lanġuġes \- A għandhom jiġi l-appoġġġġġġ...

EXPECTED:  Brincat, Joseph M. Maltese and other languages — A Linguistic History of Malta.

### Root Cause

The model was significantly undertrained at 5 epochs. The repetition problem is a classic symptom of a model that has not learned when to stop generating. More epochs are expected to resolve this.

---

## 10\. GOT-OCR2.0 Evaluation

### Model

`ucaslcl/GOT-OCR2_0` — General OCR Theory model, designed specifically for OCR across multiple languages and document types.

### Setup Notes

- Required `transformers==4.37.2` (newer versions incompatible due to `QWenTokenizer` changes)  
- Used `torch.float16` for memory efficiency on T4  
- Inference via `model.chat(tokenizer, image_path, ocr_type='ocr')`

### Results

| Metric | Value |
| :---- | :---- |
| Images evaluated | 422 |
| **Average CER** | **0.1560** |
| Runtime | \~30 minutes on T4 |

### Conclusion

GOT-OCR2.0 without fine-tuning scores significantly worse than the Tesseract baseline. Fine-tuning on Maltese data would likely improve this substantially, but was not attempted due to time and resource constraints.

---

## 11\. Donut Evaluation

### Model

`naver-clova-ix/donut-base-finetuned-cord-v2` — provided as an example by competition organizers.

### Notes

- Originally fine-tuned on receipt data (CORD dataset), not Maltese text  
- Used as provided in `example_competition_transcriber_donut.py`

### Results

| Metric | Value |
| :---- | :---- |
| **Average CER** | **0.217** |

### Conclusion

Donut performs poorly without Maltese-specific fine-tuning, consistent with the organizers' published baseline of 0.217.

---

## 12\. Model Comparison

| Model | CER | Notes |
| :---- | :---- | :---- |
| ✅ Tesseract \+ NOMOCRAT \+ line joiner | **0.036** | Best result achieved |
| ❌ GOT-OCR2.0 (zero-shot) | 0.156 | No fine-tuning |
| ❌ Donut (zero-shot) | 0.217 | No fine-tuning |
| ⏳ TrOCR (5 epochs) | \~0.8+ | Undertrained, repetition loops |
| ⏳ TrOCR (20 epochs) | TBD | Currently training |

---

## 13\. Organizer Baselines

The competition organizers published the following baseline scores on the dev set:

| Model | CER | Duration on dev (s) | Duration on test (s) |
| :---- | :---- | :---- | :---- |
| Tesseract \+ NOMOCRAT \+ line joiner | **0.02344** | 91.26 | 106.13 |
| Tesseract \+ mlt tessdata best \+ line joiner | 0.02387 | 91.81 | 102.90 |
| Tesseract \+ mlt tessdata fast \+ line joiner | 0.02390 | 72.59 | 84.38 |
| Tesseract \+ mlt tessdata default \+ line joiner | 0.02429 | 128.86 | 156.22 |
| Donut on GPU | 0.21669 | 242.37 | 284.62 |
| Donut on CPU | 0.21669 | 1759.77 | 2130.68 |

### Gap Analysis

Our Tesseract implementation scores **0.036** vs the organizers' **0.023** using the same model. The likely cause is a subtle difference in the line joiner configuration or preprocessing. Closing this gap is the highest priority fix.

---

## 14\. Next Steps

### Immediate (before submission)

1. **Fix line joiner** — investigate why our Tesseract scores 0.036 vs organizers' 0.023 with the same setup. Check `fix_hyphenated_words` parameter and line splitting logic.  
2. **Image preprocessing** — add binarization to fix the 7 blank predictions on coloured background images. Using PIL or ImageMagick.  
3. **Evaluate 20-epoch TrOCR** — once training completes, evaluate on dev set and compare to Tesseract baseline.

### Potential Improvements

4. **Ensemble approach** — use Tesseract as primary, fall back to TrOCR or GOT-OCR2.0 when Tesseract returns empty string (for the coloured/decorative images)  
5. **Fine-tune GOT-OCR2.0** — on synthetic Maltese data, may outperform TrOCR  
6. **More synthetic data** — generate 10,000–20,000 images with more font variety  
7. **HuggingFace upload** — required for submission; upload final model with README

### Submission Checklist

- [ ] Close CER gap from 0.036 to \~0.023  
- [ ] Fix blank predictions on stylized images  
- [ ] Evaluate 20-epoch TrOCR  
- [ ] Build ensemble if TrOCR improves  
- [ ] Upload model to HuggingFace  
- [ ] Write HuggingFace README  
- [ ] Test on Windows (evaluation computer is Windows 11\)  
- [ ] Verify total runtime \< 5 hours  
- [ ] Verify total disk space \< 20 GB  
- [ ] Submit by June 30, 2026 23:59 AoE

---

## 15\. Project File Structure

maltese-ocr/

├── data/

│   ├── dev\_set/                    ← 422 real competition JPGs

│   ├── texts.json                  ← ground truth for dev set

│   └── synthetic/

│       ├── images/                 ← 5000 synthetic training JPGs

│       └── transcriptions.json    ← ground truth for synthetic data

├── models/

│   └── trocr-maltese/             ← fine-tuned TrOCR checkpoint

├── competition\_transcriber.py     ← submission file

├── example\_competition\_transcriber\_tesseract.py

├── example\_competition\_transcriber\_donut.py

├── generate\_data.py               ← synthetic data generator

├── train.py                       ← TrOCR training script (Mac/MPS)

├── train\_colab.py                 ← TrOCR training script (Colab/CUDA)

├── test\_baseline.py               ← evaluation script

├── results.json                   ← evaluation results sorted by CER

├── char\_set.json                  ← allowed character set

└── requirements.txt               ← organizer dependencies

Google Drive (maltese-OCR/):

├── synthetic/                     ← uploaded for Colab training

├── dev\_set/                       ← uploaded for Colab evaluation

├── texts.json

└── models/

    └── trocr-maltese/             ← trained model saved from Colab

---

---

## 17. Pipeline Refactor — Tesseract First, TrOCR as Fallback

### What Was Changed and Why

Two changes were made to `competition_transcriber.py`:

**1. Swapped inference order (Tesseract first)**

Previously the pipeline ran TrOCR first and fell back to Tesseract only when TrOCR returned empty text. This was the wrong order because:

- Tesseract (PSM 6 + NOMOCRAT) achieves CER **0.0221** on the dev set.
- Our 5-epoch TrOCR fine-tune scored much worse (~0.8+ CER) and showed repetition loops on many images.
- Running TrOCR on all 422 images also makes inference much slower (~seconds per image vs milliseconds for Tesseract).

The new order is:
1. Run Tesseract. If it returns ≥ 3 characters, return the result immediately.
2. If Tesseract gets < 3 characters, apply ImageMagick preprocessing (upscale, grayscale, contrast, binarise) and retry Tesseract.
3. Only if both Tesseract attempts fail (< 3 chars), use TrOCR as a last resort.

**2. Added repetition penalty to TrOCR**

The 5-epoch fine-tune produced outputs like:
```
Kummissjoni għall-Kummissjoni għall-Kummissjoni għall-...
```
Two generation parameters were added to the `generate()` call to fix this:

- `repetition_penalty=2.0` — penalises the model for repeating tokens it has already output. Any value above 1.0 reduces repetition; 2.0 is a firm penalty.
- `no_repeat_ngram_size=3` — hard-blocks any 3-word sequence from appearing more than once in the output.

These parameters only affect TrOCR (which is now a last-resort fallback). They have no impact on the Tesseract path.

---

### Before / After CER Table

| Stage | CER | Notes |
|-------|-----|-------|
| Tesseract PSM 3 (original baseline) | 0.0360 | Default Tesseract mode |
| Tesseract PSM 6 | 0.0237 | Matched organizers' 0.023 |
| + ImageMagick fallback (2× at <150 px) | 0.0225 | Fallback for empty-output images |
| + 3× upscale at <200 px | 0.0221 | Stronger upscaling in fallback path |
| **Pipeline refactor (Tesseract first + TrOCR repetition penalty)** | **0.0221** | No regression — CER unchanged |

The refactor did **not** change the CER because almost all images are handled entirely by Tesseract. The TrOCR path is only reached on images where Tesseract produces fewer than 3 characters even after ImageMagick preprocessing, which is an extremely rare edge case in the dev set.

---

### New Inference Order (Diagram)

```
transcribe(image)
│
├─ Tesseract (PSM 6, --l mlt)
│   ├─ result ≥ 3 chars? ──→ return result  (the common case, ~99% of images)
│   └─ result < 3 chars?
│       ├─ wand available?
│       │   └─ ImageMagick preprocessing (upscale 3× if h<200, grayscale, contrast, threshold)
│       │       └─ Tesseract again
│       │           ├─ result ≥ 3 chars? ──→ return result
│       │           └─ result < 3 chars? ──→ fall through to TrOCR
│       └─ no wand ──→ fall through to TrOCR
│
└─ TrOCR fallback (only if fine-tuned model exists at models/trocr-maltese/)
    └─ generate(repetition_penalty=2.0, no_repeat_ngram_size=3)
        └─ return result (or "" if empty)
```

---

## 18. Code Refactor

This section records a round of code-cleanup changes that removed duplication
and dead logic without altering the transcription pipeline's behaviour. The
competition CER is unchanged (still **0.0221**) — these are maintainability
fixes, not accuracy changes.

### 18.1 — `transcribe()` reuses `_run_tesseract_with_preprocessing`

**What changed.** `competition_transcriber.py` had inlined the
"run Tesseract → if < 3 chars, preprocess and retry" logic directly inside
`transcribe()`, duplicating the body of the existing
`_run_tesseract_with_preprocessing` helper (which had become dead code as a
result). `transcribe()` now calls the helper instead of re-implementing it.

**Why.** Two copies of the same fallback logic is a maintenance hazard — a fix
to one can silently drift from the other. The two copies had in fact already
diverged slightly in how they treated a 1–2 character preprocessed result.

**Behaviour-preserving.** Every branch was checked by hand (raw ≥ 3 chars,
preprocessed ≥ 3, preprocessed 1–2 chars, preprocessed empty, no-wand,
no-TrOCR) and returns the identical value to the previous inline version. The
Tesseract-first inference order is unchanged.

### 18.2 — `test_baseline.py` calls `transcribe()` once per image

**What changed.** The evaluation loop used to run Tesseract twice for every
image: once directly via `transcriber._run_tesseract(image)` to compute a
"raw" (no-fallback) CER, and again via `transcriber.transcribe(image)` for the
real result. It now calls `transcribe()` exactly once per image.

**Why.** For the ~415 images that pass on the first Tesseract call, the second
run was identical wasted work — it doubled the subprocess cost across the whole
422-image dev set. The before/after "raw vs fallback" comparison columns and
the per-image `IMPROVED/SAME/WORSE` tags that depended on the raw value were
removed; the script still reports the overall CER, the 10 hardest images, and
the current status of the 7 previously-blank images.

### 18.3 — Merged `train.py` and `train_colab.py` into one `train.py`

**What changed.** `train_colab.py` was a near-verbatim copy of `train.py`
(identical `MalteseOCRDataset` and `run_epoch`); only the device, DataLoader
settings, epoch count, and data paths differed. The two files are now a single
`train.py` that auto-detects the device and selects the right settings:

| Device | num_workers | pin_memory | epochs |
|--------|-------------|------------|--------|
| CUDA (e.g. Colab T4) | 2 | True | 5 |
| MPS (Apple Silicon)  | 0 | False | 10 |
| CPU (fallback)       | 0 | False | 10 |

`num_workers=0` is kept for MPS/CPU because multiprocessing in the macOS
DataLoader hangs. Data paths also auto-detect Colab: if a mounted Drive is
present at `MyDrive/maltese-OCR/`, the script reads/writes there; otherwise it
uses the local repo paths.

**Why.** Two copies meant every shared bug fix had to be applied twice — the
exact duplication hazard that had already bitten this project. One device-aware
file removes that risk.

**Follow-on edits.** `train_colab.py` was deleted, and `run_colab.ipynb` was
updated to upload and run `train.py` instead of `train_colab.py`.

---

## 19. Targeted Post-processing Fix

(Numbered 19 because `## 18` above is already the Code Refactor section.)

A single targeted regex was added to `competition_transcriber._postprocess`,
applied to the Tesseract result inside `_run_tesseract_with_preprocessing` (and
therefore to every value returned by `transcribe()`):

```python
# digit-hyphen-Capital (no surrounding spaces) → digit em-dash Capital
text = re.sub(r'(\d)-([A-ZĄĦĊĠŻ])', r'\1 — \2', text)
```

### Motivation

Three dev-set images (`154.jpg`, `236.jpg`, `354.jpg`) shared the identical
error: Tesseract read the heading `1 — Ippjanata` as `1-Ippjanata` (a tight
hyphen instead of a spaced em dash). The same misread appears as `2-Parzjalment`
on three more images (`166.jpg`, `207.jpg`, `337.jpg`).

### Note on the pattern — why "no surrounding spaces"

The fix was first prototyped as `(\d)\s*-\s*([A-ZĄĦĊĠŻ])` (allowing spaces
around the hyphen). That version regressed **`121.jpg`** (CER 0.007 → 0.009):
its prediction contains `...Malti2 - Għaliex`, where the hyphen is a genuine
hyphen in the ground truth, and the `\s*` form rewrote it to an em dash. The
target misreads are always *tight* (`1-Ippjanata`, no spaces), so requiring the
hyphen to sit directly between the digit and the capital — `(\d)-([A-Z…])` —
fixes all six target images and leaves the spaced, legitimate hyphen in
`121.jpg` alone. This mirrors the earlier finding (Section 5/Round 2) that a
broad digit→em-dash rule made 12 images worse; the rule must stay narrow.

The pattern intentionally does **not** match:
`978-87-92387-68-4` (ISBN), `14(1,13-15)` (page range), `DK-5000` (letter
before the hyphen) — confirmed unchanged.

### Results

Evaluated by applying the shipping regex to the predictions in `results.json`
(the 0.0221 run) and recomputing CER with `jiwer` — equivalent to re-running
`test_baseline.py`.

| Metric | Before | After |
|--------|--------|-------|
| Overall CER (422 images) | 0.0221 | **0.0196** (−0.00253) |
| Images changed | — | 6 |
| Improved | — | 6 |
| Regressions | — | 0 |

Per-image before/after (all six changed images):

| Image | Prediction (before → after) | CER before | CER after |
|-------|-----------------------------|-----------:|----------:|
| 154.jpg | `1-Ippjanata` → `1 — Ippjanata` | 0.231 | **0.000** |
| 236.jpg | `1-Ippjanata` → `1 — Ippjanata` | 0.231 | **0.000** |
| 354.jpg | `1-Ippjanata` → `1 — Ippjanata` | 0.231 | **0.000** |
| 166.jpg | `2-Parzjalment fis-seħħ` → `2 — Parzjalment fis-seħħ` | 0.125 | **0.000** |
| 207.jpg | `2-Parzjalment fis-seħħ` → `2 — Parzjalment fis-seħħ` | 0.125 | **0.000** |
| 337.jpg | `2-Parzjalment fis-seħħ` → `2 — Parzjalment fis-seħħ` | 0.125 | **0.000** |

The three explicitly targeted images (`154`, `236`, `354`) all reached CER
0.000; three further images with the same misread improved as a bonus; no other
image changed.

### CER progression (updated)

| Stage | CER |
|-------|-----|
| Tesseract PSM 6 + ImageMagick fallback (3× upscale) | 0.0221 |
| + digit-hyphen-Capital → em-dash post-processing | **0.0196** |

---

## 20. Synthetic Data Evaluation

### What This Test Measures

We ran our current best transcriber (Tesseract PSM 6 + NOMOCRAT + ImageMagick
preprocessing + digit-hyphen-Capital post-processing) on 500 randomly sampled
synthetic training images to measure how well it performs on the data TrOCR was
trained on, and to understand the gap between synthetic and real performance.

### Results

| Metric | Value |
|--------|-------|
| Images tested | 500 |
| Average CER on synthetic images | **0.0070** |
| Real dev set CER | 0.0196 |
| Gap | 0.0126 |

### Analysis

- **Synthetic CER (0.0070) is ~3× lower than real CER (0.0196).** This means our synthetic images are noticeably easier to read than real PDF-extracted images.
- **Tesseract reads our synthetic fonts very well** because they are clean, high-contrast, and rendered at a consistent DPI with no scanning artefacts.
- **Real PDFs have much more variety** — compression artefacts, slight blurring, tilted scans, irregular margins, uneven paper backgrounds, and font rendering differences from the original PDF renderer.
- **This gap explains why TrOCR trained on V1 synthetic data struggles to generalize** to the real dev set: the model learned to read clean text and is not robust to the noise in real images.
- **Solution:** Add realistic noise augmentation to synthetic data (blur, JPEG compression artefacts, rotation, background texture, margin variation) to close this gap. This is implemented in the V2 generator described in Section 21.

---

## 21. Synthetic Data V2 — Noise Augmentation

### What Was Changed and Why

The original V1 synthetic generator (`generate_data.py`) produced clean, high-contrast images. Section 20 showed that Tesseract achieves CER 0.0070 on these synthetic images vs 0.0196 on the real dev set — a gap of 0.0126. This gap means TrOCR trained on V1 data learns features of *clean rendered text* rather than *scanned PDF text*, hurting generalization.

`generate_data.py` was updated to add a `--v2` flag that enables five augmentations:

| # | Augmentation | Probability | What it simulates |
|---|---|---|---|
| 1 | **Gaussian blur** (radius 0–0.8 px) | 30 % | Slightly out-of-focus scanner |
| 2 | **Background texture** (±5 px noise) | 30 % | Paper grain / scanner sensor noise |
| 3 | **Slight rotation** (±1.5°) | 20 % | Tilted page in a scanner or PDF crop |
| 4 | **JPEG compression** (quality 70–85) | 40 % | Low-quality PDF image extraction |
| 5 | **Margin jitter** (±10 px) | 100 % (always) | Variation in crop alignment |

Each augmentation is applied independently, so any combination (0 through 4) can appear on the same image.

**Key implementation notes for a first-year CS student:**

- *Blur*: `ImageFilter.GaussianBlur(radius)` from Pillow. Radius 0 = no blur, radius 0.8 = very subtle softening.
- *Background texture*: We convert the image to a numpy array, add random integers in [-5, +5] to every pixel channel, then `np.clip` to keep values in [0, 255]. `int16` arithmetic avoids overflow.
- *Rotation*: `img.rotate(angle, expand=True, fillcolor=(255,255,255))`. `expand=True` makes the canvas bigger so no text is cut off; white fills the new corners.
- *JPEG compression*: Encode to a BytesIO buffer at quality 70–85, then decode back. The compression artefacts ("blocks" / colour fringing) are baked into the pixel values before the final save.
- *Margin jitter*: `random.randint(-10, 10)` is added to the margin in `render_paragraph()`, clamped at 10 px minimum so text never touches the edge.

### How to Run

```bash
# Generate 1000 V2 augmented images (does NOT overwrite V1 data):
python3 generate_data.py --v2
#   → data/synthetic_v2/images/  (1000 JPGs)
#   → data/synthetic_v2/transcriptions.json

# Original V1 generation (5000 clean images):
python3 generate_data.py
```

### Sample Images Generated

The V2 run completed successfully, producing 1000 images in ~5 seconds locally.
Three sample filenames with their ground-truth text (first 80 chars):

| Filename | Text (truncated) |
|----------|-----------------|
| `syn_000001.jpg` (793×196 px) | `Fit-tielet lok, Repsol tikkunsidra li s-sentenza tikser l-Artikolu 261 TFUE...` |
| `syn_000050.jpg` (694×230 px) | *(varies by run — check transcriptions.json for ground truth)* |
| `syn_000200.jpg` (532×224 px) | *(varies by run — check transcriptions.json for ground truth)* |

### Font Families Used in V2 (9 verified — 4 removed after glyph testing)

All fonts were tested with `fontTools` (checking the font's cmap — the internal character→glyph map) against every Maltese special character: `ħ ġ ċ ż Ħ Ġ Ċ Ż`.

**Removed fonts (render □ boxes for Maltese chars):**

| Font | Missing glyphs |
|------|---------------|
| PT Serif | ġ ċ Ġ Ċ (all 4 styles) |
| Charter | ħ Ħ (all 4 styles) |
| Gill Sans | ħ ġ ċ Ħ Ġ Ċ (3 of 4 styles) |
| Optima | ALL 8 Maltese chars (all 4 styles) |

Including these would corrupt TrOCR training — the model would learn to associate □ boxes with Maltese characters, the opposite of what we need.

**Verified font families (9 — full Maltese glyph coverage confirmed):**

| # | Family | Type | Why chosen |
|---|--------|------|-----------|
| 1 | Times New Roman | Classic serif | Academic standard; in V1 |
| 2 | Georgia | Screen serif | Legible at small sizes; in V1 |
| 3 | Arial | Sans-serif | Universal sans; in V1 |
| 4 | Verdana | Sans-serif | Wide, legible; in V1 |
| 5 | Trebuchet MS | Humanist sans | Slightly informal; in V1 |
| 6 | Palatino | Calligraphic serif | Very common in academic/book PDFs |
| 7 | Baskerville | Transitional serif | Classic British journals and books |
| 8 | Rockwell | Slab serif | Distinctive square serifs; textbooks |
| 9 | Courier New | Monospace | Typewriter-style appendices |

**Why font variety matters for TrOCR generalisation:**
- TrOCR is a transformer that learns visual representations of text. If it only ever sees 5 fonts during training, it builds shortcuts tied to those specific glyph shapes.
- When presented with a real PDF that uses Palatino or Baskerville (very common in European academic publishing), the model has no reference and makes more errors.
- Each new font family teaches TrOCR a different stroke width, glyph proportions, and character spacing — making it extract the *abstract letter shape* rather than memorising specific pixels.
- This is the same principle as data augmentation in image classifiers: more variety → better generalisation.

### Plan for Retraining TrOCR on V2 Data

1. **Upload V2 data to Google Drive** — copy `data/synthetic_v2/` to `MyDrive/maltese-OCR/synthetic_v2/`.
2. **Update `train.py`** — add a `--data-dir` argument (or edit the `DATA_DIR` constant) pointing to `synthetic_v2/` instead of `synthetic/`.
3. **Retrain from scratch** — start from `microsoft/trocr-base-handwritten` (not the V1 checkpoint), run 5–10 epochs on Colab T4.
4. **Evaluate** — run `test_baseline.py` with the new V2 model loaded. Target: TrOCR CER should drop significantly compared to the V1 model because training distribution is now closer to real PDF images.
5. **Consider combining V1 + V2** — if V2 alone doesn't generalise, training on a mixture of V1 (5000 clean) + V2 (5000 noisy) images (10,000 total) may give the best of both worlds.

### Expected Outcome

If the noise augmentation is well-calibrated, retraining on V2 data should reduce the gap between TrOCR and Tesseract. Whether TrOCR then beats Tesseract (CER 0.0196) remains to be seen — that depends on how well the model handles real Maltese-specific characters and typography.

---

*Documentation last updated: June 2026* *Competition: ACM DocEng 2026 — Maltese OCR*  

---

## 22. Synthetic Data V2 — Extended Augmentation

### What Was Changed and Why

After V2 (basic augmentation) gave a CER of **0.0094** — still 3× easier than the real dev set (0.0196) — we added 6 more aggressive augmentations to close the gap further. The goal: make TrOCR train on images that look like real PDFs, not perfect renders.

All augmentations fire independently at low probability, so most images have 0–2 effects applied. The 5000-image evaluation measures an average that reflects the mix.

### All 8 Augmentations (Final Settings)

| # | Augmentation | Probability | Settings | Purpose |
|---|---|---|---|---|
| 1 | Gaussian blur | 30% | radius 0–0.5 px | Simulates slightly out-of-focus scan |
| 2 | JPEG compression | 40% | quality 70–85 | Simulates PDF extraction artefacts |
| 3 | Slight rotation | 20% | ±1.5 degrees | Simulates tilted scan |
| 4 | Background texture | 30% | ±5 pixel noise | Simulates paper grain |
| 5 | Low-resolution simulation | 25% | downscale to 70%, upscale with NEAREST | Simulates low-DPI scans |
| 6 | Brightness variation | 40% | ±8 units via `ImageEnhance.Brightness` | Simulates uneven lighting / exposure |
| 7 | Ink bleed | 20% | `MaxFilter(3)` blended at 40% opacity | Makes strokes slightly thicker (inkbleed) |
| 8 | Bleed-through | 15% | flipped copy at 1–3% opacity | Text showing through from other side of page |

**Render-time augmentations (applied before post-processing):**

| # | Augmentation | Probability | Settings | Purpose |
|---|---|---|---|---|
| A | Per-line font size variation | 20% | ±1 pt around base size | Inconsistent PDF rendering |
| B | Uneven margins (x-shift) | 30% | shift 5–20 px left/right | Non-centred scans |

### Why MaxFilter Must Be Size 3 (Not 2)

PIL's `ImageFilter.MaxFilter` requires an **odd** filter size. `MaxFilter(2)` raises a `ValueError`. To get a gentler effect than `MaxFilter(3)`, we blend the filtered version back at 40% opacity:

```python
bleed = img.filter(ImageFilter.MaxFilter(3))   # dilate strokes
img = Image.blend(img, bleed, alpha=0.40)       # only 40% of the effect
```

### CER Comparison Table

| Dataset | CER | Notes |
|---------|-----|-------|
| V1 synthetic (clean, 5 fonts) | 0.0070 | Too easy — no noise |
| V2 old (basic augmentation, 9 fonts) | 0.0094 | Better but still too easy |
| **V2 new (extended augmentation, 9 fonts)** | **0.0282** | Slightly harder than real PDFs |
| Real dev set (422 images) | 0.0196 | Target difficulty |

### Analysis

The V2 new CER of **0.0282** slightly overshoots the real dev-set CER of 0.0196 — meaning synthetic images are now a little **harder** than real PDFs rather than easier. This is a reasonable outcome:

- **Having training data slightly harder than the test set helps generalisation.** TrOCR will learn to read imperfect images, so clean real PDFs should be easier for it.
- The previous V2 (0.0094) had synthetic images 2× easier than real PDFs, likely contributing to poor generalisation.
- We went from 0.0102 below target → 0.0086 above target. A well-calibrated sweet spot would be exactly 0.0196, but +0.0086 is a much better position for training than −0.0102.

### Sample Images

Five representative samples from the regenerated V2 dataset (visual inspection confirms text is clearly legible in all cases):

- `syn_002520.jpg` (425×260 px) — italic serif, slight brightness drop
- `syn_002365.jpg` (505×212 px) — bold-italic, clean render
- `syn_001249.jpg` (784×176 px) — ink bleed visible on serif bold, text readable
- `syn_003728.jpg` (819×256 px) — clean single-sentence, Rockwell font
- `syn_001104.jpg` (721×269 px) — heavy ink bleed case; strokes merge slightly but remain legible

All special Maltese characters (ħ, ġ, ċ, ż, Ħ, Ġ, Ċ, Ż) are correctly rendered — confirmed by the 9-font glyph audit in section 21.

### Next Step: Retrain TrOCR on V2 Data

1. Upload `data/synthetic_v2/` to `MyDrive/maltese-OCR/synthetic_v2/` on Google Drive.
2. Update `train.py` / `train_colab.py` `DATA_DIR` constant to point to `synthetic_v2/`.
3. Retrain from `microsoft/trocr-base-handwritten` baseline (not from V1 checkpoint).
4. Run 5 epochs on Colab T4, evaluate with `test_baseline.py`.
5. If V2 alone does not beat Tesseract (CER 0.0196), consider training on **V1 + V2 combined** (10 000 images total).

---

*Documentation last updated: June 2026* *Competition: ACM DocEng 2026 — Maltese OCR*
