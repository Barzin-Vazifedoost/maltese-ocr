# Session Log — 2026-06-15

## Context

Maltese OCR competition (DocEng 2026). Task: transcribe images of Maltese
paragraphs (cropped from real PDFs) into text. Metric: Character Error Rate
(CER), lower is better. Evaluation machine: single RTX 2080 Ti (11 GB VRAM),
batch size 1, 5-hour runtime budget, 20 GB disk, no internet during evaluation.

Prior best CER: **0.0221** (Tesseract PSM 6 + ImageMagick 3× upscale fallback,
`targeted-fixes` branch).

---

## Architecture Decision (pre-existing, confirmed this session)

Three-stage TrOCR-style encoder-decoder pipeline:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | `seqclr_pretrain` | SeqCLR contrastive pretraining of the ViT encoder — no labels, synthetic images only |
| 2 | `supervised_finetune` | Supervised fine-tuning on synthetic data + auxiliary style-invariance contrastive loss; character-level decoder replaces RoBERTa BPE |
| 3 | `hard_negative_finetune` | Hard-negative minimal pairs for Maltese-specific confusable characters (ħ/h, ġ/g, ċ/c, ż/z, għ/gh) |

Base model: `microsoft/trocr-base-printed`

---

## What Was Built This Session

### Repository scaffold created on branch `training_basic_step`

#### `pyproject.toml`
- Python 3.11+, all dependencies pinned to versions available on the user's
  Python 3.13 environment (discovered iteratively — older pins like torch 2.4.0
  are not available for Python 3.13).
- Final pinned versions:

| Package | Version |
|---------|---------|
| torch | 2.12.0 |
| transformers | 5.12.1 |
| datasets | 5.0.0 |
| Pillow | 12.2.0 |
| albumentations | 2.0.8 |
| opencv-python-headless | 4.13.0.92 |
| jiwer | 4.0.0 |
| PyYAML | 6.0.3 |
| fonttools | 4.63.0 |
| tqdm | 4.68.2 |
| huggingface_hub | 1.19.0 |
| accelerate | 1.14.0 |
| sentencepiece | 0.2.1 |
| malti | >=0.3 |
| wand | >=0.6 |
| pytest (dev) | 9.1.0 |
| ruff (dev) | 0.15.17 |
| pre-commit (dev) | 4.6.0 |

- Build backend: `setuptools.build_meta` (not `setuptools.backends.legacy` —
  that API was unavailable on the installed setuptools version).
- `torchvision` deliberately excluded: TrOCR uses a ViT from `transformers`,
  not torchvision, and albumentations handles augmentation.

#### `.pre-commit-config.yaml`
- Single repo: `astral-sh/ruff-pre-commit` at `v0.15.17`
- Two hooks: `ruff` (lint + auto-fix) and `ruff-format`

#### `Makefile`
Targets:

| Target | Command |
|--------|---------|
| `setup` | `pip install -e ".[dev]" && pre-commit install` |
| `test` | `pytest tests/ -v -m "not slow"` (skips network tests) |
| `test-all` | `pytest tests/ -v` (includes smoke tests) |
| `render-sample` | `python scripts/render_sample.py` |
| `pretrain` | `python -m maltese_ocr.pretrain.run --config configs/stage1.yaml` |
| `train` | Runs stage2 then stage3 configs sequentially |
| `eval` | `python test_baseline.py` |
| `package` | `python scripts/package_model.py` |

#### `src/maltese_ocr/` — package layout
```
src/maltese_ocr/
  __init__.py       version = "0.1.0"
  data/__init__.py
  render/__init__.py
  pretrain/__init__.py
  train/__init__.py
  infer/__init__.py
```
Installed as `maltese_ocr` via `pip install -e .` (src layout,
`[tool.setuptools.packages.find] where = ["src"]`).

#### `configs/`

**`stage1.yaml`** — SeqCLR contrastive pretraining
- Encoder only (`trocr-base-printed`), decoder unused
- 2 augmented views per image, temperature 0.07, projection dim 256
- AdamW lr 1e-4, 20 epochs, saves every 5
- Output: `models/stage1/`

**`stage2.yaml`** — Supervised fine-tuning
- Init encoder from `models/stage1/best_encoder.pt`
- Character-level tokenizer over closed competition charset (`configs/charset.txt`)
- CE loss weight 1.0 + contrastive aux loss weight 0.1
- AdamW lr 5e-5, 10 epochs, saves best by val CER
- Beam search: 4 beams, max 256 tokens
- Output: `models/stage2/`

**`stage3.yaml`** — Hard-negative minimal pairs
- Init from `models/stage2/best.pt`
- Confusable pairs: ħ/h, Ħ/H, ġ/g, Ġ/G, ċ/c, Ċ/C, ż/z, Ż/Z, għ/gh
- 50% hard-negative ratio per batch
- AdamW lr 1e-5, 5 epochs
- Final output: `models/trocr-maltese-v2/`

#### `tests/test_smoke.py`
Two tests, both marked `@pytest.mark.slow` (require network, skipped by
`make test`):

1. `test_trocr_forward_pass` — loads `microsoft/trocr-base-printed`, runs
   `.generate()` on a white 384×384 dummy image, asserts output is a string.
2. `test_trocr_encoder_output_shape` — asserts encoder output is 3-D with
   `hidden_size == 768` (trocr-base spec).

#### `scripts/render_sample.py`
Stub that calls `maltese_ocr.render.render_sample(n=10, ...)` — to be
implemented when the render module is filled in.

#### `scripts/package_model.py`
Copies the Stage 3 checkpoint to `models/trocr-maltese-v2/`, checks for
required files (`config.json`, `tokenizer_config.json`, weight file), prints
directory size.

---

## Issues Encountered & Fixed

| Problem | Fix |
|---------|-----|
| `setuptools.backends.legacy` not available | Changed build-backend to `setuptools.build_meta` |
| `torch==2.4.0` not available for Python 3.13 | Queried `pip index versions` for all deps; updated all pins to actual available versions |
| `opencv-python-headless==4.13.0` not found | OpenCV requires 4-part version: `4.13.0.92` |

---

## Current State

- Branch: `training_basic_step`
- Working tree: clean (scaffold committed or staged — no commits made this session)
- Package: installed and importable (`python3 -c "import maltese_ocr"` → `0.1.0`)

---

## Next Steps (after initial scaffold session)

1. Implement `src/maltese_ocr/render/` — image renderer (can reuse logic from
   existing `generate_data.py`).
2. Implement `src/maltese_ocr/pretrain/run.py` — Stage 1 SeqCLR training loop.
3. Implement `src/maltese_ocr/train/run.py` — Stage 2/3 fine-tuning loop.
4. Implement `src/maltese_ocr/infer/` — `Transcriber` class for competition
   submission.
5. Run `make test-all` to verify smoke tests pass (requires network).

---

## T2 — Font Audit & Acquisition (second session)

Branch: `training_basic_step`
Commits: `91923f5`, `9d15021`, `08e6ac3`

### Background & motivation

The V2 synthetic-data generation (V2 run in `generate_data.py`) previously
discovered four macOS system fonts that rendered hollow boxes □ for
Maltese-specific characters, and removed them manually.  That process was
ad-hoc.  T2 formalises glyph validation with proper tooling so that no
missing-glyph font can enter the usable set undetected.

### What was built

#### `configs/charset.txt` (new)
117 characters, one per line, sorted by Unicode codepoint.  Derived from
`ALLOWED_CHARS` in `generate_data.py` — the exact competition vocabulary.
Referenced by `configs/stage2.yaml` (tokenizer charset) and consumed by the
audit script.

#### `src/maltese_ocr/render/fonts.py` (new)
`FontFace` dataclass + `MACOS_FACES` list — the font catalog for synthetic
rendering, migrated from the inline dict in `generate_data.py`.

| Field | Type | Meaning |
|-------|------|---------|
| `family` | `str` | e.g. "Times New Roman" |
| `style` | `str` | "regular" / "bold" / "italic" / "bold_italic" |
| `category` | `str` | "serif" / "sans" / "mono" |
| `path` | `str` | Absolute path to font file |
| `index` | `int` | TTC face index (0 for single-face files) |

`MACOS_FACES` lists all 36 faces across 9 macOS families (Times New Roman,
Georgia, Palatino, Baskerville, Rockwell, Arial, Verdana, Trebuchet MS,
Courier New).  Fonts removed in V2 (PT Serif, Charter, Gill Sans, Optima)
are absent.

#### `scripts/audit_fonts.py` (new)
Scans `fonts/` + `MACOS_FACES` catalog with `fontTools.getBestCmap()`.
A glyph is considered present if the codepoint exists in the font's best
Unicode cmap AND maps to a name other than `.notdef`.

**Two-tier character classification:**

| Tier | Characters | Rule |
|------|-----------|------|
| Hard | All 114 non-symbol chars including **ħ Ħ ġ Ġ ċ Ċ ż Ż** | Missing → font excluded from `fonts_ok.json` |
| Soft | `♢` (U+2662), `⁴` (U+2074), `ỹ` (U+1EF9) | Missing → warning only, font still passes |

The three soft characters appear in the competition corpus but are so rare
that no professional font includes ♢ (White Diamond Suit); hard-failing on
them would reject every font in the library.

**Category detection** (for `fonts_ok.json` metadata):
1. Name-fragment lookup table (`_NAME_CATEGORY` dict, covers all known
   families)
2. `font["post"].isFixedPitch` → "mono"
3. IBM font family class from `font["OS/2"].sFamilyClass >> 8`

**Style detection:** reads name-table entry 2 (Font Subfamily Name);
normalises to "regular" / "bold" / "italic" / "bold_italic".

**De-duplication:** when a font ships as both `.otf` and `.ttf` (Libertinus
does this), the output keeps one entry per `(family, style)` pair, preferring
`.ttf`.

Output: `fonts/fonts_ok.json` — a list of passing faces with family, style,
category, path, index, and `soft_missing`.  Exits with code 1 if fewer than
12 families pass.

#### `scripts/fetch_fonts.py` (new)
Downloads open-licence fonts to `fonts/`.  Two strategies:

**Google Fonts** — direct raw files from `github.com/google/fonts`:

| Family | Dir in repo | Format |
|--------|-------------|--------|
| Noto Serif | `ofl/notoserif` | Variable (wdth, wght axes) |
| Noto Sans | `ofl/notosans` | Variable |
| Lato | `ofl/lato` | Static (Regular/Bold/Italic/BoldItalic) |
| Merriweather | `ofl/merriweather` | Variable (opsz, wdth, wght) |
| Source Serif 4 | `ofl/sourceserif4` | Variable (opsz, wght) |
| PT Serif | `ofl/ptserif` | Static (`PT_Serif-Web-*.ttf`) |
| EB Garamond | `ofl/ebgaramond` | Variable (wght) |

URL-encoded bracket characters in variable-font filenames
(`NotoSerif[wdth,wght].ttf` → `NotoSerif%5Bwdth%2Cwght%5D.ttf`) via
`urllib.parse.quote(filename, safe=".-_")`.

**GitHub Releases API** — for fonts not on Google Fonts:

| Font | Repository |
|------|-----------|
| Gentium Plus | `silnrsi/font-gentium` |
| Charis SIL | `silnrsi/font-charis` |
| Libertinus | `alerque/libertinus` |

Fetches `/releases/latest` JSON, picks the smallest `.zip` asset, extracts
all `.ttf`/`.otf` files to `fonts/`.

**SSL fix:** `python.org` macOS installer doesn't wire in the system trust
store.  The script detects `certifi` and builds `ssl.create_default_context(
cafile=certifi.where())` automatically.

Flags: `--skip-existing` (skip already-downloaded files), `--no-audit`
(skip running audit after download).  By default, calls `audit_fonts.main()`
after all downloads complete.

#### `fonts/README.md` (new)
Documents: the Ħ-missing-glyph incident that motivated the tooling, the
hard/soft audit methodology, all 27 passing families grouped by category with
source and notes, all 5 failing fonts with specific missing characters and
root cause, and instructions for adding new fonts.

### Issues encountered and fixes

| Problem | Root cause | Fix |
|---------|-----------|-----|
| `from fonttools.ttLib import TTFont` fails | The PyPI package is `fonttools` but the Python module is `fontTools` (capital T); case-sensitive on Python 3.13 even on macOS | Changed import to `from fontTools.ttLib import …` |
| Google Fonts download endpoint returns HTML | `fonts.google.com/download?family=…` now requires a login session | Switched to direct `raw.githubusercontent.com/google/fonts/main/…` URLs |
| Merriweather, Noto, EB Garamond 404 with guessed filenames | Modern Google Fonts uses variable fonts with axis-tag brackets in filenames | Used GitHub API to list actual filenames first, then hard-coded them |
| SSL certificate verification failure | macOS `python.org` installer ships without system trust store | Added `certifi` detection + `ssl.create_default_context(cafile=certifi.where())` |
| Charis shows `category: "unknown"` | SIL renamed "Charis SIL" to "Charis" in v7 (2024) | Added `"charis": "serif"` to `_NAME_CATEGORY` lookup |
| Libertinus ships both `.otf` and `.ttf` — duplicate entries | GitHub release zip contains both formats | Added `(family, style)` de-duplication, preferring `.ttf` |

### Audit results

```
Total faces checked : 98
Passed              : 87  (87/98 faces)
Failed (hard miss)  : 11
Passing families    : 27  ✓ (target ≥ 12)
De-duped entries    : 76  (in fonts_ok.json)
```

**Failed fonts:**

| Font | Missing hard chars | Notes |
|------|--------------------|-------|
| PT Serif (all 4 styles) | Ċ ċ Ġ ġ | Same gap on GF version as on macOS copy — confirmed not fixed upstream |
| EB Garamond italic | ² ¹ | Italic variant only; regular passes |
| Libertinus Serif Initials | All lowercase + Maltese chars | Initial-caps decorative font — not a text face |
| Libertinus Keyboard | © ² ¹ — ' ' " " • € | Keyboard-symbol font — not a text face |
| Libertinus Mono | € | Only one missing char; otherwise complete |

**Passing families by category:**

| Category | Families |
|----------|---------|
| Serif (19) | Times New Roman, Georgia, Palatino, Baskerville, Rockwell, EB Garamond (regular), Merriweather, Noto Serif, Source Serif 4, Gentium Book (+Medium +SemiBold), Charis (+Medium +SemiBold), Libertinus Serif (+Display +Semibold +Semibold Italic), Libertinus Math |
| Sans (6) | Arial, Verdana, Trebuchet MS, Lato, Noto Sans, Libertinus Sans |
| Mono (1) | Courier New |

### Acceptance criteria

- [x] ≥ 12 families pass — **27 families pass**
- [x] Spanning serif/sans — **19 serif, 6 sans, 1 mono**
- [x] Regular/bold/italic coverage — all core styles present
- [x] Zero missing-glyph fonts in `fonts_ok.json` — **confirmed**
- [x] Three commits in Conventional Commits format — `91923f5`, `9d15021`, `08e6ac3`
- [x] ruff check + ruff format pass — run before each commit
