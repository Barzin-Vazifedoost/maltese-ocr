"""
generate_data.py — Synthetic Maltese OCR training data generator.

Default run (V1):
    Renders 5000 paragraph images from real Maltese text (MLRS/korpus_malti),
    saved to data/synthetic/images/ with ground truths in transcriptions.json.

V2 run (with noise augmentation):
    python3 generate_data.py --v2
    Renders 5000 images with realistic noise (blur, JPEG artifacts, rotation,
    background texture, margin variation) into data/synthetic_v2/.
    Purpose: close the gap between synthetic CER (0.0070) and real CER (0.0196).

    Uses 9 font families verified to have complete Maltese glyph coverage.
    4 fonts were removed (PT Serif, Charter, Gill Sans, Optima) after fontTools
    testing revealed they render □ boxes for ħ, ġ, ċ and related characters.

Before running, authenticate with HuggingFace:
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np                               # for adding pixel-level noise
from datasets import load_dataset
from malti.line_joiner.rb_line_joiner.rb_line_joiner import RBLineJoiner
from malti.sent_splitter.km_sent_splitter.km_sent_splitter import KMSentSplitter
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# V1 defaults (original synthetic set)
TOTAL_IMAGES = 5000
IMAGES_DIR   = Path("data/synthetic/images")
OUTPUT_JSON  = Path("data/synthetic/transcriptions.json")

# V2 defaults (noise-augmented set — activated with --v2 flag)
V2_TOTAL_IMAGES = 5000
V2_IMAGES_DIR   = Path("data/synthetic_v2/images")
V2_OUTPUT_JSON  = Path("data/synthetic_v2/transcriptions.json")

# Allowed characters — sentences containing anything outside this set are
# discarded so that every character the model sees is in the competition vocab
ALLOWED_CHARS = set(
    ' !"&\'()+,-./:;=?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_'
    'abcdefghijklmnopqrstuvwxyz'
    '0123456789'
    '©²¹Øàçéìñòóôöøúüā'
    'ĊċĠġĦħłŻżỹ'
    '\u2014'   # — em dash
    '\u2018'   # ' left single quote
    '\u2019'   # ' right single quote
    '\u201c'   # " left double quote
    '\u201d'   # " right double quote
    '•⁴€♢'
)

# ---------------------------------------------------------------------------
# Font families — verified present on macOS and supporting Maltese characters.
#
# V1 used 5 families. V2 originally added 8 more, but 4 were removed after
# glyph coverage testing with fontTools revealed they render tofu boxes □ for
# Maltese-specific characters:
#
#   REMOVED (missing glyphs):
#     PT Serif   — missing ġ ċ Ġ Ċ in all 4 styles
#     Charter    — missing ħ Ħ in all 4 styles
#     Gill Sans  — missing ħ ġ ċ Ħ Ġ Ċ in most styles
#     Optima     — missing ALL 8 Maltese chars in all 4 styles
#
# Including fonts with missing glyphs would corrupt training data, teaching
# TrOCR to associate □ boxes with Maltese characters — the opposite of what
# we want.
#
# TTC fonts (font collections that bundle multiple faces in one file) need a
# face index. The style values for TTC fonts are (path, index) tuples instead
# of plain strings. render_paragraph() handles both forms automatically.
#
# ── How face indices were determined ───────────────────────────────────────
# For each .ttc file, ImageFont.truetype(path, 14, index=N).getname() was used
# to enumerate and identify every face. Only the Regular / Bold / Italic /
# Bold-Italic faces are mapped here (extras like SemiBold are ignored).
#
# ── 9 families (verified full Maltese coverage) ────────────────────────────
#   Classic serif : Times New Roman, Georgia, Palatino, Baskerville
#   Slab serif    : Rockwell
#   Sans-serif    : Arial, Verdana, Trebuchet MS
#   Monospace     : Courier New
# ---------------------------------------------------------------------------

_SUPP = "/System/Library/Fonts/Supplemental"
_SYS  = "/System/Library/Fonts"   # non-Supplemental system fonts (TTC files)

FONT_FAMILIES = [
    # ── Classic serif ─────────────────────────────────────────────────────
    {
        "name":        "Times New Roman",
        "regular":     f"{_SUPP}/Times New Roman.ttf",
        "bold":        f"{_SUPP}/Times New Roman Bold.ttf",
        "italic":      f"{_SUPP}/Times New Roman Italic.ttf",
        "bold_italic": f"{_SUPP}/Times New Roman Bold Italic.ttf",
    },
    {
        "name":        "Georgia",
        "regular":     f"{_SUPP}/Georgia.ttf",
        "bold":        f"{_SUPP}/Georgia Bold.ttf",
        "italic":      f"{_SUPP}/Georgia Italic.ttf",
        "bold_italic": f"{_SUPP}/Georgia Bold Italic.ttf",
    },
    {
        # Palatino — extremely common in academic and book PDFs; visually very
        # different from Times (wider, more calligraphic stroke contrast).
        # TTC face order: [0]=Regular [1]=Italic [2]=Bold [3]=Bold Italic
        "name":        "Palatino",
        "regular":     (f"{_SYS}/Palatino.ttc", 0),
        "bold":        (f"{_SYS}/Palatino.ttc", 2),
        "italic":      (f"{_SYS}/Palatino.ttc", 1),
        "bold_italic": (f"{_SYS}/Palatino.ttc", 3),
    },
    {
        # Baskerville — classic British transitional serif, used in many books
        # and European academic journals.
        # TTC face order: [0]=Regular [1]=Bold [2]=Italic [3]=Bold Italic
        "name":        "Baskerville",
        "regular":     (f"{_SUPP}/Baskerville.ttc", 0),
        "bold":        (f"{_SUPP}/Baskerville.ttc", 1),
        "italic":      (f"{_SUPP}/Baskerville.ttc", 2),
        "bold_italic": (f"{_SUPP}/Baskerville.ttc", 3),
    },
    # ── Slab serif ────────────────────────────────────────────────────────
    {
        # Rockwell — square slab serifs give it a very distinctive look; appears
        # in textbook headings and older print-era PDFs.
        # TTC face order: [0]=Regular [1]=Italic [2]=Bold [3]=Bold Italic
        "name":        "Rockwell",
        "regular":     (f"{_SUPP}/Rockwell.ttc", 0),
        "bold":        (f"{_SUPP}/Rockwell.ttc", 2),
        "italic":      (f"{_SUPP}/Rockwell.ttc", 1),
        "bold_italic": (f"{_SUPP}/Rockwell.ttc", 3),
    },
    # ── Sans-serif ─────────────────────────────────────────────────────────
    {
        "name":        "Arial",
        "regular":     f"{_SUPP}/Arial.ttf",
        "bold":        f"{_SUPP}/Arial Bold.ttf",
        "italic":      f"{_SUPP}/Arial Italic.ttf",
        "bold_italic": f"{_SUPP}/Arial Bold Italic.ttf",
    },
    {
        "name":        "Verdana",
        "regular":     f"{_SUPP}/Verdana.ttf",
        "bold":        f"{_SUPP}/Verdana Bold.ttf",
        "italic":      f"{_SUPP}/Verdana Italic.ttf",
        "bold_italic": f"{_SUPP}/Verdana Bold Italic.ttf",
    },
    {
        "name":        "Trebuchet MS",
        "regular":     f"{_SUPP}/Trebuchet MS.ttf",
        "bold":        f"{_SUPP}/Trebuchet MS Bold.ttf",
        "italic":      f"{_SUPP}/Trebuchet MS Italic.ttf",
        "bold_italic": f"{_SUPP}/Trebuchet MS Bold Italic.ttf",
    },
    # ── Monospace ─────────────────────────────────────────────────────────
    {
        # Courier New — the classic typewriter monospace; appears in technical
        # appendices, code listings, and older word-processor PDFs.
        "name":        "Courier New",
        "regular":     f"{_SUPP}/Courier New.ttf",
        "bold":        f"{_SUPP}/Courier New Bold.ttf",
        "italic":      f"{_SUPP}/Courier New Italic.ttf",
        "bold_italic": f"{_SUPP}/Courier New Bold Italic.ttf",
    },
]

# Light background colours used 20 % of the time (the other 80 % is white)
LIGHT_BG_COLORS = [
    (255, 255, 210),  # pale yellow
    (210, 245, 210),  # pale green
    (210, 240, 255),  # pale blue
    (255, 225, 210),  # pale orange
    (245, 210, 255),  # pale lavender
]

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def is_allowed(sentence: str) -> bool:
    """Return True only if every character is in the competition character set."""
    return all(ch in ALLOWED_CHARS for ch in sentence)


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[list[str]]:
    """
    Word-wrap `text` so no line exceeds `max_width` pixels.

    Returns a list of lines; each line is itself a list of word strings so
    the justified-rendering function can redistribute spacing between words.
    """
    words = text.split()
    lines: list[list[str]] = []
    current_words: list[str] = []
    current_width = 0.0
    space_width = font.getlength(" ")

    for word in words:
        word_width = font.getlength(word)
        # How wide would this line be if we appended the next word?
        extra = space_width if current_words else 0.0
        test_width = current_width + extra + word_width

        if test_width <= max_width or not current_words:
            # Word fits — add it to the current line
            current_words.append(word)
            current_width = test_width
        else:
            # Word doesn't fit — start a new line
            lines.append(current_words)
            current_words = [word]
            current_width = word_width

    if current_words:
        lines.append(current_words)

    return lines


def draw_justified(
    draw: ImageDraw.ImageDraw,
    lines: list[list[str]],
    font: ImageFont.FreeTypeFont,
    x: int,
    y: int,
    text_width: int,
    line_height: int,
    fonts_per_line: list[ImageFont.FreeTypeFont] | None = None,
) -> None:
    """
    Draw word-wrapped text with full justification.

    Interior lines are stretched to fill `text_width` by spreading extra
    whitespace evenly between words.  Lines that are less than 75 % full are
    left-aligned to avoid ugly large gaps on short lines.

    `fonts_per_line` is an optional list of fonts — one per line — used when
    font-size variation is enabled in V2 mode.  If None, every line uses the
    shared `font` argument.
    """
    for i, words in enumerate(lines):
        # Use the per-line font if provided, otherwise fall back to shared font.
        # Per-line fonts simulate the slight size inconsistency seen in PDFs
        # that were converted from mixed-format Word documents.
        line_font = fonts_per_line[i] if fonts_per_line is not None else font

        is_last_line = (i == len(lines) - 1)
        total_word_width = sum(line_font.getlength(w) for w in words)
        fill_ratio = total_word_width / text_width if text_width > 0 else 1.0

        if len(words) == 1 or is_last_line or fill_ratio < 0.75:
            # Left-align: single-word lines, last line, or underfull lines
            draw.text((x, y), " ".join(words), font=line_font, fill=(0, 0, 0))
        else:
            # Justified: spread the surplus pixels as inter-word gaps
            extra_space = text_width - total_word_width
            gap = extra_space / (len(words) - 1)

            cx = float(x)
            for j, word in enumerate(words):
                draw.text((round(cx), y), word, font=line_font, fill=(0, 0, 0))
                cx += line_font.getlength(word)
                if j < len(words) - 1:
                    cx += gap

        y += line_height


# ---------------------------------------------------------------------------
# Image augmentation (V2 only)
# ---------------------------------------------------------------------------

def apply_augmentations(img: Image.Image) -> Image.Image:
    """
    Apply a random mix of realistic noise augmentations to a rendered image.

    Each augmentation fires independently with its own probability, so any
    combination (0 to all 8) can occur on a single image.  Called only during
    V2 generation — V1 images are always clean.

    Goal: make synthetic images resemble real PDF scans, to close the gap
    between synthetic CER (0.0094) and real dev-set CER (0.0196).

    Augmentations (in application order):
      1. Gaussian blur      (30%) — slightly out-of-focus scan
      2. Background texture (30%) — paper grain / scanner noise
      3. Rotation           (20%) — tilted page on the scanner glass
      4. Low resolution     (25%) — low-DPI scan (downscale then upscale)
      5. Brightness         (40%) — over/under-exposed scan
      6. Ink bleed          (20%) — ink spread on absorbent paper
      7. Bleed-through      (15%) — text from the reverse side showing faintly
      8. JPEG compression   (40%) — lossy PDF extraction artefacts

    JPEG is applied last so it bakes all other effects together, just as a
    real PDF encoder would compress a pre-processed scan.
    """

    # ── 1. Slight Gaussian blur ──────────────────────────────────────────────
    # radius=0 is a no-op; radius=0.5 gives a very soft blur.
    if random.random() < 0.30:
        radius = random.uniform(0.0, 0.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # ── 2. Background texture (pixel-level noise) ────────────────────────────
    # Adds ±5 to every colour channel to simulate paper grain or scanner noise.
    # We use int16 arithmetic first so that adding noise can't overflow uint8.
    if random.random() < 0.30:
        arr = np.array(img, dtype=np.int16)
        noise = np.random.randint(-5, 6, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # ── 3. Slight rotation ───────────────────────────────────────────────────
    # Rotates ±1.5°.  expand=True grows the canvas so no text is cropped.
    if random.random() < 0.20:
        angle = random.uniform(-1.5, 1.5)
        img = img.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

    # ── 4. Low-resolution simulation ────────────────────────────────────────
    # Shrinks the image to 70 % then enlarges it back using nearest-neighbour
    # interpolation (no smoothing).  This creates mild pixel-stepping that
    # simulates a slightly low-DPI scan without making the text unreadable.
    if random.random() < 0.25:
        orig_w, orig_h = img.size
        small_w = max(50, round(orig_w * 0.70))
        small_h = max(50, round(orig_h * 0.70))
        if small_w >= 50 and small_h >= 50:
            small = img.resize((small_w, small_h), resample=Image.LANCZOS)
            img = small.resize((orig_w, orig_h), resample=Image.NEAREST)

    # ── 5. Random brightness variation ──────────────────────────────────────
    # Adjusts overall brightness by ±8 units — subtle enough that text stays
    # clearly legible, but enough to simulate exposure variation in scans.
    if random.random() < 0.40:
        shift = random.randint(-8, 8)
        factor = (255 + shift) / 255.0
        img = ImageEnhance.Brightness(img).enhance(factor)

    # ── 6. Ink bleed simulation ──────────────────────────────────────────────
    # MaxFilter requires an odd size; minimum is 3.  To get a gentler effect
    # than full MaxFilter(3), we blend the filtered result at 40 % opacity —
    # only 40 % of the dilation comes through, giving a subtle thickening.
    if random.random() < 0.20:
        bled = img.filter(ImageFilter.MaxFilter(3))
        img  = Image.blend(img, bled, alpha=0.40)  # 40 % dilation, 60 % original

    # ── 7. Bleed-through simulation ─────────────────────────────────────────
    # Blends a horizontally-flipped copy of the image at 1–3 % opacity —
    # barely perceptible, just enough to simulate thin paper.
    if random.random() < 0.15:
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        opacity = random.uniform(0.01, 0.03)    # very faint — 1–3 %
        img = Image.blend(img, flipped, alpha=opacity)

    # ── 8. JPEG compression artefacts ───────────────────────────────────────
    # Encode at low quality then decode back.  Bakes all previous effects in,
    # just as a real PDF encoder would.  .copy() detaches from the BytesIO
    # buffer before it's garbage-collected.
    if random.random() < 0.40:
        quality = random.randint(70, 85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).copy()

    return img


# ---------------------------------------------------------------------------
# Image renderer
# ---------------------------------------------------------------------------

def render_paragraph(
    text: str,
    bg_color: tuple[int, int, int] | None = None,
    margin_jitter: int = 0,
    font_size_vary: bool = False,
    x_shift: int = 0,
) -> Image.Image:
    """
    Render one paragraph of Maltese text as a PIL RGB image.

    All visual parameters (font, size, width, style, background) are chosen
    randomly to make the synthetic set as varied as possible.  Pass `bg_color`
    explicitly to override the random background (used by main() to guarantee
    exactly 20 % coloured images).

    V2 parameters:
      `margin_jitter`  — adds ±10 px to the layout margin, simulating slight
                         scanner alignment variation.
      `font_size_vary` — when True, each line is rendered at a slightly
                         different font size (±1pt around the base), simulating
                         the font-size inconsistencies found in PDFs converted
                         from Word documents or old typesetters.
      `x_shift`        — shifts the text block left (negative) or right
                         (positive) by 5–20 px, simulating non-centred scans
                         where the paper wasn't placed flush against the guide.
    """
    # Pick a random font family and style variant
    family = random.choice(FONT_FAMILIES)
    # Weights: regular 10 %, bold 30 %, italic 30 %, bold-italic 30 %
    style = random.choices(
        ["regular", "bold", "italic", "bold_italic"],
        weights=[0.10, 0.30, 0.30, 0.30],
    )[0]
    font_spec = family[style]

    # Convert point size to pixels at 96 DPI  (px = pt × 96 ÷ 72)
    font_size_pt = random.randint(10, 14)
    font_size_px = round(font_size_pt * 96 / 72)

    # Font loading: most families store each style as a separate .ttf file
    # (plain string path). TTC families bundle all styles into one file and
    # need a face index — those are stored as (path, index) tuples.
    if isinstance(font_spec, tuple):
        font_path, font_index = font_spec
        font = ImageFont.truetype(font_path, font_size_px, index=font_index)
    else:
        font_path, font_index = font_spec, None
        font = ImageFont.truetype(font_spec, font_size_px)

    # Image width and inner layout margins.
    # margin_jitter offsets the margin by ±10 px in V2 to add positional variety.
    img_width = random.randint(400, 900)
    margin  = max(10, random.randint(20, 60) + margin_jitter)  # clamped to ≥10 px
    padding = random.randint(10, 30)   # extra space inside the text box

    # x_shift moves the text block sideways.  We clamp so the text never
    # starts off-canvas (left edge ≥ 0) and the right edge stays inside the image.
    draw_x = max(0, margin + padding + x_shift)
    text_width = max(50, img_width - draw_x - padding)   # ensure at least 50 px wide

    # Background colour — use the caller-supplied value if given, else random
    if bg_color is None:
        bg_color = (255, 255, 255) if random.random() < 0.80 else random.choice(LIGHT_BG_COLORS)

    # Word-wrap the paragraph text using the base font
    lines = wrap_text(text, font, text_width)

    # ── Font-size variation (V2 only) ────────────────────────────────────────
    # Build a per-line font list where each line is ±1pt around the base size.
    # This simulates the slight font-size inconsistency seen in some PDFs.
    # The wrapping above was done with the base font; using slightly different
    # sizes per line is intentional and adds realistic imperfection.
    fonts_per_line = None
    if font_size_vary and lines:
        fonts_per_line = []
        for _ in lines:
            variation_pt = random.randint(-1, 1)               # -1, 0, or +1 pt
            varied_pt    = max(8, font_size_pt + variation_pt) # never below 8pt
            varied_px    = round(varied_pt * 96 / 72)
            if font_index is not None:
                varied_font = ImageFont.truetype(font_path, varied_px, index=font_index)
            else:
                varied_font = ImageFont.truetype(font_path, varied_px)
            fonts_per_line.append(varied_font)

    # Use getbbox on representative tall/deep chars for accurate pixel height.
    # getmetrics() can under-report for bold/italic faces and diacritics (ħ, ġ).
    bbox = font.getbbox("Ħġpqjy|")
    glyph_height = bbox[3] - bbox[1]   # bottom - top in pixels
    line_spacing = random.uniform(1.1, 1.5)
    line_height  = round(glyph_height * line_spacing)

    # Measure the full text block height before creating the image so we never clip.
    # text_block_height = (N-1) line spacings + the actual glyph height of the last line.
    text_block_height = max(0, len(lines) - 1) * line_height + glyph_height

    # Bottom padding: cap at 1.5× line_height so short paragraphs don't get a
    # huge empty gap, but always leave at least the standard top inset.
    bottom_pad = min(2 * (margin + padding), round(1.5 * line_height))
    bottom_pad = max(bottom_pad, margin + padding)  # never less than top inset

    img_height = (margin + padding) + text_block_height + bottom_pad

    # 20 % safety buffer to guarantee no clipping from font rendering edge cases
    img_height = round(img_height * 1.20)

    # Draw everything
    img  = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw_justified(
        draw, lines, font,
        x=draw_x,
        y=margin + padding,
        text_width=text_width,
        line_height=line_height,
        fonts_per_line=fonts_per_line,
    )
    return img


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_sentences(needed: int) -> list[list[str]]:
    """
    Stream MLRS/korpus_malti from HuggingFace, split into sentences with
    KMSentSplitter, and return sentences grouped by source document so that
    paragraphs drawn from consecutive sentences stay coherent.

    Returns a list of documents; each document is a list of allowed sentences.
    The dataset is gated — run `huggingface-cli login` beforehand.
    """
    print("Connecting to MLRS/korpus_malti on HuggingFace (streaming)…")
    try:
        dataset = load_dataset(
            "MLRS/korpus_malti",
            split="train",
            streaming=True,
        )
    except Exception as exc:
        print(f"\nERROR loading dataset: {exc}")
        print(
            "\nThe corpus is gated. To fix this:\n"
            "  1. Create a free account at https://huggingface.co\n"
            "  2. Request access at https://huggingface.co/datasets/MLRS/korpus_malti\n"
            "  3. Run:  huggingface-cli login\n"
            "  4. Then re-run this script."
        )
        sys.exit(1)

    splitter = KMSentSplitter()
    docs: list[list[str]] = []   # one inner list per corpus document
    total_sents = 0

    print(f"Collecting sentences from corpus (target: {needed:,})…")
    with tqdm(total=needed, desc="Sentences collected", unit="sent") as pbar:
        for item in dataset:
            raw = (item.get("text") or item.get("sentence") or "").strip()
            if not raw:
                continue

            # Split the whole document into sentences, keep only allowed ones
            doc_sents = [
                s.strip()
                for s in splitter.split(raw)
                if len(s.strip()) >= 5 and is_allowed(s.strip())
            ]
            if doc_sents:
                docs.append(doc_sents)
                pbar.update(len(doc_sents))
                total_sents += len(doc_sents)

            if total_sents >= needed:
                break

    print(f"Collected {total_sents:,} sentences across {len(docs):,} documents.\n")
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Parse command-line arguments ─────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Generate synthetic Maltese OCR training images."
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help=(
            "Generate the noise-augmented V2 dataset (1000 images → "
            "data/synthetic_v2/). Without this flag the original V1 behaviour "
            "(5000 images → data/synthetic/) is used."
        ),
    )
    args = parser.parse_args()

    # ── Select output paths and image count based on --v2 flag ───────────────
    if args.v2:
        total   = V2_TOTAL_IMAGES
        out_dir = V2_IMAGES_DIR
        out_json = V2_OUTPUT_JSON
        augment = True   # apply noise augmentations in V2 mode
        print("Running in V2 mode — noise augmentation ENABLED.")
        print(f"Output: {out_dir}  ({total} images)\n")
    else:
        total   = TOTAL_IMAGES
        out_dir = IMAGES_DIR
        out_json = OUTPUT_JSON
        augment = False  # V1: clean images, no augmentation
        print("Running in V1 mode — noise augmentation DISABLED.")
        print(f"Output: {out_dir}  ({total} images)\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sentences grouped by source document for coherent paragraphs
    docs = load_sentences(needed=total * 15)

    if not docs:
        print("ERROR: no sentences collected — cannot generate images.")
        sys.exit(1)

    # Pre-assign background colours to guarantee exactly 20 % are coloured.
    # Doing this upfront avoids statistical bad luck with per-image random draws.
    n_coloured = round(total * 0.20)
    bg_colors = (
        [(255, 255, 255)] * (total - n_coloured)
        + [random.choice(LIGHT_BG_COLORS) for _ in range(n_coloured)]
    )
    random.shuffle(bg_colors)

    joiner = RBLineJoiner()
    transcriptions: dict[str, str] = {}

    print(f"Rendering {total:,} synthetic images…")
    for img_num in tqdm(range(1, total + 1), desc="Images", unit="img"):

        # Pick a random document, then take consecutive sentences from it so
        # the paragraph reads coherently (same topic / writing style)
        doc = random.choice(docs)
        n_sents = random.randint(1, min(15, len(doc)))
        start = random.randint(0, len(doc) - n_sents)
        para_sents = doc[start : start + n_sents]

        # Join into one paragraph string; RBLineJoiner de-hyphenates any words
        # broken across lines in the original corpus source
        paragraph_text = joiner.join_lines(
            " ".join(para_sents).split("\n"),
            fix_hyphenated_words=True,
        )

        # ── V2: pick render-time augmentation parameters ─────────────────────
        # These must be decided before render_paragraph() is called because
        # they affect how the image is drawn (not post-processed).

        # font_size_vary: 20 % chance — each line in the paragraph will use a
        # slightly different font size (±1pt around the base), simulating
        # inconsistent PDF rendering.
        font_size_vary = (random.random() < 0.20) if augment else False

        # x_shift: 30 % chance — shift the text block left or right by 5–20 px,
        # simulating a page that wasn't aligned flush on the scanner.
        if augment and random.random() < 0.30:
            direction = random.choice([-1, 1])
            x_shift   = direction * random.randint(5, 20)
        else:
            x_shift = 0

        # ── V2: pick a random margin jitter of ±10 px ────────────────────────
        # This simulates slight positional variation in where text sits inside
        # the image, as seen in real PDF-extracted paragraph crops.
        margin_jitter = random.randint(-10, 10) if augment else 0

        # Render the clean paragraph image with the pre-assigned background
        img = render_paragraph(
            paragraph_text,
            bg_color=bg_colors[img_num - 1],
            margin_jitter=margin_jitter,
            font_size_vary=font_size_vary,
            x_shift=x_shift,
        )

        # ── V2: apply noise augmentations after rendering ────────────────────
        # Blur, texture, rotation, and JPEG artifacts are applied here so the
        # ground-truth text is always the clean pre-noise paragraph string.
        if augment:
            img = apply_augmentations(img)

        # Save the final image (quality=90 for V1; already compressed in V2)
        filename = f"syn_{img_num:06d}.jpg"
        img.save(out_dir / filename, "JPEG", quality=90)

        transcriptions[filename] = paragraph_text

    # Write the ground-truth JSON  {filename: paragraph_text, …}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Images  → {out_dir}/  ({total} files)")
    print(f"  Labels  → {out_json}")


if __name__ == "__main__":
    main()
