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

*Documentation last updated: June 2026* *Competition: ACM DocEng 2026 — Maltese OCR*  
