# fonts/ — Maltese OCR font library

This directory holds the open-licence font files used to render synthetic
training images.  Every font here was validated against the 117-character
competition charset (`configs/charset.txt`) by `scripts/audit_fonts.py`.

---

## Why this tooling exists

During V2 synthetic-data generation four macOS system fonts (PT Serif,
Charter, Gill Sans, Optima) were found to render hollow boxes □ for
Maltese-specific characters.  Training on images with boxes for ħ or Ħ
would teach the OCR model to predict garbage for those characters.
`audit_fonts.py` formalises the check so that no font with a missing
Maltese glyph can slip into the usable set undetected.

---

## Audit methodology

`scripts/audit_fonts.py` classifies each character in `configs/charset.txt`
as **hard** or **soft**:

| Tier | Characters | Effect of missing glyph |
|------|-----------|------------------------|
| Hard | All 114 non-symbol chars, **including ħ Ħ ġ Ġ ċ Ċ ż Ż** | Font **fails** — excluded from `fonts_ok.json` |
| Soft | `♢` (U+2662), `⁴` (U+2074), `ỹ` (U+1EF9) | Warning only — font **still passes** |

The three soft characters appear in the corpus but are rare enough that
an occasional box in training data has no measurable impact.  No standard
professional font includes ♢ (White Diamond Suit); demoting it to soft
avoids rejecting every useful font in the library.

Glyph presence is determined with `fontTools.getBestCmap()`: a codepoint
must have a cmap entry that maps to a glyph name other than `.notdef`.

---

## Passing fonts

**98 faces checked → 87 passed → 76 de-duplicated entries in `fonts_ok.json`
→ 27 passing families**  (target ≥ 12 ✓)

### Serif

| Family | Source | Notes |
|--------|--------|-------|
| Times New Roman | macOS system | Classic transitional serif |
| Georgia | macOS system | Designed for screen readability |
| Palatino | macOS system | Wide-stroke Renaissance serif (TTC) |
| Baskerville | macOS system | British transitional serif (TTC) |
| Rockwell | macOS system | Slab serif — distinctive in headings (TTC) |
| EB Garamond\* | Google Fonts | Old-style serif; *italic variant fails (see below)* |
| Merriweather | Google Fonts | Variable font (opsz, wdth, wght axes) |
| Noto Serif | Google Fonts | Universal coverage; variable font |
| Source Serif 4 | Google Fonts | Variable font (opsz, wght axes) |
| Gentium Book | SIL International | Designed for multilingual Latin/Greek/Cyrillic |
| Charis | SIL International | Humanist serif for minority language texts |
| Libertinus Serif | Libertinus project | Fork of Linux Libertine; full Latin Extended-A |

\* EB Garamond **regular** passes; italic fails on `²` `¹` — only the regular
weight is included in `fonts_ok.json`.

### Sans-serif

| Family | Source | Notes |
|--------|--------|-------|
| Arial | macOS system | Grotesque sans |
| Verdana | macOS system | Humanist sans, designed for screen |
| Trebuchet MS | macOS system | Humanist sans with ink-trap design |
| Lato | Google Fonts | Humanist sans; extensive weight range |
| Noto Sans | Google Fonts | Universal coverage; variable font |
| Libertinus Sans | Libertinus project | Companion sans to Libertinus Serif |

### Monospace

| Family | Source | Notes |
|--------|--------|-------|
| Courier New | macOS system | Classic typewriter face |

---

## Failed fonts

| Font | Source | Missing hard chars | Notes |
|------|--------|--------------------|-------|
| PT Serif (all 4 styles) | Google Fonts | Ċ ċ Ġ ġ | Same gap as the macOS copy — these codepoints are absent from all PT Serif releases tested, confirming the prior finding in `generate_data.py` |
| EB Garamond italic | Google Fonts | ² ¹ | Superscript digits absent from italic variant; regular included |
| Libertinus Serif Initials | Libertinus | All lowercase + Maltese chars | Decorative initials-only font; not a text font |
| Libertinus Keyboard | Libertinus | ©  ² ¹ — ' ' " " • € | Keyboard-symbol font; not a text font |
| Libertinus Mono | Libertinus | € | Missing euro sign; otherwise complete |

---

## Adding fonts

1. Drop the `.ttf` or `.otf` file into `fonts/`.
2. Run `python scripts/audit_fonts.py` — it will scan the directory and
   update `fonts_ok.json` automatically.
3. If the font passes, add an entry to the table above.

To download additional open-licence fonts from the same sources:

```bash
python scripts/fetch_fonts.py --skip-existing
```

---

## File index

The `fonts_ok.json` file is the machine-readable registry consumed by
`generate_data.py` (and future rendering code in `src/maltese_ocr/render/`).
Each entry has:

```json
{
  "family": "Noto Serif",
  "style": "regular",
  "category": "serif",
  "path": "fonts/NotoSerif[wdth,wght].ttf",
  "index": 0,
  "soft_missing": ["♢"]
}
```

`soft_missing` lists which soft-tier characters are absent — recorded for
transparency, not a barrier to use.
