"""Configurable synthetic Maltese paragraph renderer.

Ports and improves the ``generate_data.py --v2`` rendering logic into a small,
configurable API built around :class:`RenderConfig`:

    img, ground_truth, metadata = render(text, config)

The renderer wraps text into lines (optionally breaking words with a line-break
hyphen), draws them with optional justification, and applies a configurable set
of realistic noise augmentations.  The returned ``ground_truth`` is always the
clean joined paragraph: line-break hyphens are never part of it, while
structural dashes such as ``il-kelb`` are preserved (they live in the source
text and are never inserted by the renderer).

:func:`render_pair` renders the *same* text with the *same* line wrapping in two
different font/augmentation configs — identical horizontal layout, different
appearance — for SeqCLR-style contrastive pretraining.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# Repo root: src/maltese_ocr/render/renderer.py -> parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
FONTS_OK_JSON = _REPO_ROOT / "fonts" / "fonts_ok.json"

# Light background colours used in place of pure white for some images.
LIGHT_BG_COLORS = [
    (255, 255, 210),  # pale yellow
    (210, 245, 210),  # pale green
    (210, 240, 255),  # pale blue
    (255, 225, 210),  # pale orange
    (245, 210, 255),  # pale lavender
]

# Default augmentation probabilities.  Each augmentation fires independently;
# ``margin_jitter`` is part of layout (applied before drawing), the rest are
# post-render image effects.  Ranges are fixed in ``_apply_augmentations``.
DEFAULT_AUGMENTATIONS: dict[str, float] = {
    "blur": 0.30,  # Gaussian blur, radius 0–0.4
    "texture": 0.30,  # background pixel noise, ±3
    "rotation": 0.20,  # rotation, ±1 degree
    "jpeg": 0.40,  # JPEG recompression, quality 80–95
    "brightness": 0.40,  # brightness shift, ±5 units
    "margin_jitter": 1.0,  # margin offset, ±10 px (always)
}


@dataclass
class RenderConfig:
    """All visual parameters for a single rendered paragraph image."""

    font_path: str
    font_index: int = 0
    font_size: int = 12  # point size, competition range 10–14
    image_width: int = 600  # pixels, competition range 400–900
    line_spacing_factor: float = 1.15  # range 0.9–1.3
    margin: int = 40  # pixels, range 20–60
    justify: bool = True
    background_color: tuple[int, int, int] = (255, 255, 255)
    augmentations: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_AUGMENTATIONS))
    # Carried for metadata; populated by ``RenderConfig.sample``.
    font_family: str = ""
    font_style: str = ""
    category: str = ""

    @classmethod
    def sample(
        cls,
        font: dict,
        rng: random.Random | None = None,
        *,
        augment: bool = True,
    ) -> RenderConfig:
        """Build a randomly-sampled config for a font dict from fonts_ok.json."""
        rng = rng or random
        bg = (255, 255, 255) if rng.random() < 0.80 else tuple(rng.choice(LIGHT_BG_COLORS))
        return cls(
            font_path=font["path"],
            font_index=int(font.get("index", 0)),
            font_size=rng.randint(10, 14),
            image_width=rng.randint(400, 900),
            line_spacing_factor=round(rng.uniform(0.9, 1.3), 3),
            margin=rng.randint(20, 60),
            justify=rng.random() < 0.85,
            background_color=bg,
            augmentations=dict(DEFAULT_AUGMENTATIONS) if augment else {},
            font_family=font.get("family", ""),
            font_style=font.get("style", ""),
            category=font.get("category", ""),
        )


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------


def load_fonts(path: str | Path = FONTS_OK_JSON) -> list[dict]:
    """Load validated fonts from fonts_ok.json.

    Skips any font that records hard-missing characters.  ``soft_missing``
    characters (e.g. ⁴ and ♢) are tolerated — they fall outside the core
    Maltese glyph set and rarely occur in body text.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [font for font in data if not font.get("hard_missing")]


def _load_font(config: RenderConfig) -> ImageFont.FreeTypeFont:
    """Load the PIL font for a config, converting point size to pixels (96 DPI)."""
    size_px = round(config.font_size * 96 / 72)
    return ImageFont.truetype(config.font_path, size_px, index=config.font_index)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def clean_ground_truth(text: str) -> str:
    """Normalise whitespace to produce the canonical joined paragraph.

    The renderer only ever inserts line-break hyphens into the *drawn* lines,
    never into this string, so the result keeps structural dashes (``il-kelb``)
    while never containing a line-break hyphen.
    """
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Wrapping
# ---------------------------------------------------------------------------


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[list[str]]:
    """Word-wrap text so no line exceeds ``max_width`` pixels.

    Returns a list of lines; each line is a list of word tokens so the
    justified renderer can redistribute inter-word spacing.
    """
    words = text.split()
    lines: list[list[str]] = []
    current: list[str] = []
    current_width = 0.0
    space_width = font.getlength(" ")

    for word in words:
        extra = space_width if current else 0.0
        test_width = current_width + extra + font.getlength(word)
        if test_width <= max_width or not current:
            current.append(word)
            current_width = test_width
        else:
            lines.append(current)
            current = [word]
            current_width = font.getlength(word)

    if current:
        lines.append(current)
    return lines


def _wrap_hyphenated(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[list[str]]:
    """Like :func:`_wrap`, but break long words at line ends with a hyphen.

    When a word does not fit on the current line, the longest prefix that still
    fits (with a trailing hyphen) is placed on the current line and the
    remainder starts the next line.  The hyphen is purely visual — it never
    enters the ground truth.
    """
    words = text.split()
    lines: list[list[str]] = []
    current: list[str] = []
    current_width = 0.0
    space_width = font.getlength(" ")
    hyphen_width = font.getlength("-")

    for word in words:
        extra = space_width if current else 0.0
        if current_width + extra + font.getlength(word) <= max_width or not current:
            current.append(word)
            current_width += extra + font.getlength(word)
            continue

        # Try to hyphenate the word onto the remaining space of the current line.
        remaining = max_width - current_width - extra - hyphen_width
        prefix_len = 0
        if remaining > 0 and len(word) >= 4:
            for k in range(2, len(word) - 1):
                if font.getlength(word[:k]) <= remaining:
                    prefix_len = k
                else:
                    break

        if prefix_len >= 2:
            current.append(word[:prefix_len] + "-")
            lines.append(current)
            current = [word[prefix_len:]]
            current_width = font.getlength(word[prefix_len:])
        else:
            lines.append(current)
            current = [word]
            current_width = font.getlength(word)

    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[list[str]],
    font: ImageFont.FreeTypeFont,
    x: int,
    y: int,
    text_width: int,
    line_height: int,
    justify: bool,
) -> None:
    """Draw wrapped lines, optionally fully justified.

    Justification spreads surplus pixels evenly between words.  Single-word
    lines, the last line, and lines less than 75 % full are always left-aligned
    to avoid ugly gaps.
    """
    for i, words in enumerate(lines):
        is_last = i == len(lines) - 1
        total_word_width = sum(font.getlength(w) for w in words)
        fill_ratio = total_word_width / text_width if text_width > 0 else 1.0

        if not justify or len(words) == 1 or is_last or fill_ratio < 0.75:
            draw.text((x, y), " ".join(words), font=font, fill=(0, 0, 0))
        else:
            gap = (text_width - total_word_width) / (len(words) - 1)
            cx = float(x)
            for j, word in enumerate(words):
                draw.text((round(cx), y), word, font=font, fill=(0, 0, 0))
                cx += font.getlength(word)
                if j < len(words) - 1:
                    cx += gap
        y += line_height


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _apply_augmentations(
    img: Image.Image,
    aug: dict[str, float],
    rng: random.Random,
    applied: list[str],
) -> Image.Image:
    """Apply post-render noise augmentations and record which fired.

    ``margin_jitter`` is handled during layout, not here.  JPEG is applied last
    so it bakes the other effects together, as a real PDF encoder would.
    """
    # 1. Gaussian blur — slightly out-of-focus scan.
    if rng.random() < aug.get("blur", 0.0):
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.4)))
        applied.append("blur")

    # 2. Background texture — paper grain / scanner noise (±3 per channel).
    if rng.random() < aug.get("texture", 0.0):
        arr = np.array(img, dtype=np.int16)
        noise = np.random.randint(-3, 4, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        applied.append("texture")

    # 3. Rotation — tilted page on the scanner glass (expand to avoid clipping).
    if rng.random() < aug.get("rotation", 0.0):
        img = img.rotate(
            rng.uniform(-1.0, 1.0),
            resample=Image.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )
        applied.append("rotation")

    # 4. Brightness — over/under-exposed scan (±5 units).
    if rng.random() < aug.get("brightness", 0.0):
        factor = (255 + rng.randint(-5, 5)) / 255.0
        img = ImageEnhance.Brightness(img).enhance(factor)
        applied.append("brightness")

    # 5. JPEG recompression — lossy PDF extraction artefacts (quality 80–95).
    if rng.random() < aug.get("jpeg", 0.0):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(80, 95))
        buf.seek(0)
        img = Image.open(buf).copy()
        applied.append("jpeg")

    return img


# ---------------------------------------------------------------------------
# Core render
# ---------------------------------------------------------------------------


def _render_from_lines(
    display_lines: list[list[str]],
    config: RenderConfig,
    font: ImageFont.FreeTypeFont,
    rng: random.Random,
) -> tuple[Image.Image, list[str]]:
    """Lay out and draw pre-computed lines, then augment.

    Returns the image and the list of augmentations that fired (which may
    include ``margin_jitter``).
    """
    aug = config.augmentations
    applied: list[str] = []

    jitter = 0
    if rng.random() < aug.get("margin_jitter", 0.0):
        jitter = rng.randint(-10, 10)
        if jitter != 0:
            applied.append("margin_jitter")
    margin = max(10, config.margin + jitter)
    text_width = max(50, config.image_width - 2 * margin)

    # Measure tall/deep glyphs for an accurate line height (getmetrics under-
    # reports descenders for bold/italic faces and diacritics like ħ, ġ).
    bbox = font.getbbox("Ħġpqjy|")
    glyph_height = bbox[3] - bbox[1]
    line_height = round(glyph_height * config.line_spacing_factor)

    n = len(display_lines)
    text_block_height = max(0, n - 1) * line_height + glyph_height
    # 2× margin top+bottom plus a 15 % safety buffer against descender clipping.
    img_height = round((2 * margin + text_block_height) * 1.15)

    img = Image.new("RGB", (config.image_width, img_height), color=config.background_color)
    draw = ImageDraw.Draw(img)
    _draw_lines(
        draw,
        display_lines,
        font,
        x=margin,
        y=margin,
        text_width=text_width,
        line_height=line_height,
        justify=config.justify,
    )

    img = _apply_augmentations(img, aug, rng, applied)
    return img, applied


def _build_display_lines(
    text: str,
    config: RenderConfig,
    font: ImageFont.FreeTypeFont,
    p_hyphen: float,
    rng: random.Random,
) -> tuple[list[list[str]], bool]:
    """Wrap text using the config's base layout, optionally with hyphenation."""
    wrap_margin = max(10, config.margin)
    wrap_width = max(50, config.image_width - 2 * wrap_margin)
    hyphenated = rng.random() < p_hyphen
    if hyphenated:
        lines = _wrap_hyphenated(text, font, wrap_width)
    else:
        lines = _wrap(text, font, wrap_width)
    return lines, hyphenated


def _metadata(
    config: RenderConfig,
    image_size: tuple[int, int],
    display_lines: list[list[str]],
    applied: list[str],
    hyphenated: bool,
) -> dict:
    return {
        "font_family": config.font_family,
        "font_style": config.font_style,
        "category": config.category,
        "image_size": image_size,
        "num_lines": len(display_lines),
        "lines": [list(line) for line in display_lines],
        "hyphenated": hyphenated,
        "augmentations_applied": applied,
    }


def render(
    text: str,
    config: RenderConfig,
    *,
    p_hyphen: float = 0.3,
    rng: random.Random | None = None,
) -> tuple[Image.Image, str, dict]:
    """Render ``text`` as a synthetic paragraph image.

    Returns ``(image, ground_truth, metadata)`` where ``ground_truth`` is the
    clean joined paragraph (line-break hyphens removed, structural dashes kept)
    and ``metadata`` records font, image size, line count, the wrapped lines,
    and which augmentations were applied.
    """
    rng = rng or random
    font = _load_font(config)
    display_lines, hyphenated = _build_display_lines(text, config, font, p_hyphen, rng)
    img, applied = _render_from_lines(display_lines, config, font, rng)
    ground_truth = clean_ground_truth(text)
    metadata = _metadata(config, img.size, display_lines, applied, hyphenated)
    return img, ground_truth, metadata


def render_pair(
    text: str,
    config_a: RenderConfig,
    config_b: RenderConfig,
    *,
    p_hyphen: float = 0.3,
    rng: random.Random | None = None,
) -> tuple[tuple[Image.Image, Image.Image], str, dict]:
    """Render the same text with the same line wrapping in two configs.

    The line wrapping (and hyphenation) is computed once using ``config_a`` and
    reused verbatim for ``config_b``, guaranteeing identical horizontal layout
    with different appearance — the contrastive pair used by SeqCLR (T5).

    Returns ``((image_a, image_b), ground_truth, metadata)`` where ``metadata``
    contains a ``config_a`` / ``config_b`` sub-dict each and the shared line
    count.
    """
    rng = rng or random
    font_a = _load_font(config_a)
    font_b = _load_font(config_b)

    display_lines, hyphenated = _build_display_lines(text, config_a, font_a, p_hyphen, rng)

    img_a, applied_a = _render_from_lines(display_lines, config_a, font_a, rng)
    img_b, applied_b = _render_from_lines(display_lines, config_b, font_b, rng)

    ground_truth = clean_ground_truth(text)
    metadata = {
        "num_lines": len(display_lines),
        "lines": [list(line) for line in display_lines],
        "hyphenated": hyphenated,
        "config_a": _metadata(config_a, img_a.size, display_lines, applied_a, hyphenated),
        "config_b": _metadata(config_b, img_b.size, display_lines, applied_b, hyphenated),
    }
    return (img_a, img_b), ground_truth, metadata
