"""
generate_data.py — Synthetic Maltese OCR training data generator.

Renders 5000 paragraph images from real Maltese text (MLRS/korpus_malti),
saved to data/synthetic/images/ with ground truths in transcriptions.json.

Before running, authenticate with HuggingFace:
    pip install huggingface_hub
    huggingface-cli login
"""

import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset
from malti.line_joiner.rb_line_joiner.rb_line_joiner import RBLineJoiner
from malti.sent_splitter.km_sent_splitter.km_sent_splitter import KMSentSplitter
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOTAL_IMAGES = 5000
IMAGES_DIR   = Path("data/synthetic/images")
OUTPUT_JSON  = Path("data/synthetic/transcriptions.json")

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
# Five complete families (each has Regular / Bold / Italic / Bold-Italic):
#   Serif    : Times New Roman, Georgia
#   Sans-serif: Arial, Verdana, Trebuchet MS
# ---------------------------------------------------------------------------

_SUPP = "/System/Library/Fonts/Supplemental"

FONT_FAMILIES = [
    # ── Serif ──────────────────────────────────────────────────────────────
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
) -> None:
    """
    Draw word-wrapped text with full justification.

    Interior lines are stretched to fill `text_width` by spreading extra
    whitespace evenly between words.  Lines that are less than 75 % full are
    left-aligned to avoid ugly large gaps on short lines.
    """
    for i, words in enumerate(lines):
        is_last_line = (i == len(lines) - 1)
        total_word_width = sum(font.getlength(w) for w in words)
        fill_ratio = total_word_width / text_width if text_width > 0 else 1.0

        if len(words) == 1 or is_last_line or fill_ratio < 0.75:
            # Left-align: single-word lines, last line, or underfull lines
            draw.text((x, y), " ".join(words), font=font, fill=(0, 0, 0))
        else:
            # Justified: spread the surplus pixels as inter-word gaps
            extra_space = text_width - total_word_width
            gap = extra_space / (len(words) - 1)

            cx = float(x)
            for j, word in enumerate(words):
                draw.text((round(cx), y), word, font=font, fill=(0, 0, 0))
                cx += font.getlength(word)
                if j < len(words) - 1:
                    cx += gap

        y += line_height


# ---------------------------------------------------------------------------
# Image renderer
# ---------------------------------------------------------------------------

def render_paragraph(text: str, bg_color: tuple[int, int, int] | None = None) -> Image.Image:
    """
    Render one paragraph of Maltese text as a PIL RGB image.

    All visual parameters (font, size, width, style, background) are chosen
    randomly to make the synthetic set as varied as possible.  Pass `bg_color`
    explicitly to override the random background (used by main() to guarantee
    exactly 20 % coloured images).
    """
    # Pick a random font family and style variant
    family = random.choice(FONT_FAMILIES)
    # Weights: regular 10 %, bold 30 %, italic 30 %, bold-italic 30 %
    style = random.choices(
        ["regular", "bold", "italic", "bold_italic"],
        weights=[0.10, 0.30, 0.30, 0.30],
    )[0]
    font_path = family[style]

    # Convert point size to pixels at 96 DPI  (px = pt × 96 ÷ 72)
    font_size_pt = random.randint(10, 14)
    font_size_px = round(font_size_pt * 96 / 72)
    font = ImageFont.truetype(font_path, font_size_px)

    # Image width and inner layout margins
    img_width = random.randint(400, 900)
    margin  = random.randint(20, 60)   # space between image edge and text box
    padding = random.randint(10, 30)   # extra space inside the text box
    text_width = img_width - 2 * (margin + padding)

    # Background colour — use the caller-supplied value if given, else random
    if bg_color is None:
        bg_color = (255, 255, 255) if random.random() < 0.80 else random.choice(LIGHT_BG_COLORS)

    # Word-wrap the paragraph text
    lines = wrap_text(text, font, text_width)

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
        x=margin + padding,
        y=margin + padding,
        text_width=text_width,
        line_height=line_height,
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
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load sentences grouped by source document for coherent paragraphs
    docs = load_sentences(needed=TOTAL_IMAGES * 15)

    if not docs:
        print("ERROR: no sentences collected — cannot generate images.")
        sys.exit(1)

    # Pre-assign background colours to guarantee exactly 20 % are coloured.
    # Doing this upfront avoids statistical bad luck with per-image random draws.
    n_coloured = round(TOTAL_IMAGES * 0.20)
    bg_colors = (
        [(255, 255, 255)] * (TOTAL_IMAGES - n_coloured)
        + [random.choice(LIGHT_BG_COLORS) for _ in range(n_coloured)]
    )
    random.shuffle(bg_colors)

    joiner = RBLineJoiner()
    transcriptions: dict[str, str] = {}

    print(f"Rendering {TOTAL_IMAGES:,} synthetic images…")
    for img_num in tqdm(range(1, TOTAL_IMAGES + 1), desc="Images", unit="img"):

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

        # Render and save the image, using the pre-assigned background colour
        img      = render_paragraph(paragraph_text, bg_color=bg_colors[img_num - 1])
        filename = f"syn_{img_num:06d}.jpg"
        img.save(IMAGES_DIR / filename, "JPEG", quality=90)

        transcriptions[filename] = paragraph_text

    # Write the ground-truth JSON  {filename: paragraph_text, …}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Images  → {IMAGES_DIR}/  ({TOTAL_IMAGES} files)")
    print(f"  Labels  → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
