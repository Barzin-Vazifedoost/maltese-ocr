"""Tests for the synthetic paragraph renderer (T3)."""

from __future__ import annotations

import random

from PIL import Image

from maltese_ocr.render import (
    RenderConfig,
    clean_ground_truth,
    load_fonts,
    render,
    render_pair,
)

# A paragraph long enough to wrap over several lines, exercising both structural
# dashes (il-baħar, tal-..., x-xemx) and Maltese diacritics.
SAMPLE_TEXT = (
    "Il-kelb tal-baħar għadda mill-port filgħodu kmieni, meta ż-żiffa kienet "
    "għadha friska u x-xemx bdiet tielgħa fuq il-baħar kwiet ħafna."
)


def _font():
    """First validated font, for tests that just need a working face."""
    return load_fonts()[0]


def _no_aug_config(width: int = 500) -> RenderConfig:
    """A config with augmentations disabled so image size is deterministic."""
    font = _font()
    return RenderConfig(
        font_path=font["path"],
        font_index=int(font.get("index", 0)),
        image_width=width,
        augmentations={},
        font_family=font["family"],
        font_style=font["style"],
        category=font["category"],
    )


def test_ground_truth_has_no_line_break_hyphens():
    """Forcing hyphenation must not leak line-break hyphens into the ground truth.

    Structural dashes from the source text (il-baħar) must be preserved.
    """
    config = _no_aug_config(width=360)  # narrow -> guaranteed wrapping
    _, ground_truth, meta = render(SAMPLE_TEXT, config, p_hyphen=1.0, rng=random.Random(0))

    # Hyphenation actually happened: at least one drawn line ends with a hyphen.
    assert meta["hyphenated"]
    assert any(line and line[-1].endswith("-") for line in meta["lines"])

    # Ground truth is exactly the whitespace-normalised source: no inserted
    # hyphens, structural dashes intact.
    assert ground_truth == clean_ground_truth(SAMPLE_TEXT)
    assert "il-baħar" in ground_truth
    assert "tal-baħar" in ground_truth

    # No drawn line-break hyphen survives as a real token in the ground truth.
    line_break_fragments = [line[-1] for line in meta["lines"] if line and line[-1].endswith("-")]
    for fragment in line_break_fragments:
        assert fragment not in ground_truth.split()


def test_render_pair_identical_line_wrapping():
    """render_pair must produce identical line wrapping for both configs."""
    fonts = load_fonts()
    serif = next(f for f in fonts if f["category"] == "serif")
    sans = next(f for f in fonts if f["category"] == "sans")

    config_a = RenderConfig.sample(serif, random.Random(1))
    config_b = RenderConfig.sample(sans, random.Random(2))
    # Same canvas width so the layouts are directly comparable.
    config_a.image_width = config_b.image_width = 520

    (img_a, img_b), ground_truth, meta = render_pair(
        SAMPLE_TEXT, config_a, config_b, p_hyphen=0.5, rng=random.Random(3)
    )

    assert isinstance(img_a, Image.Image)
    assert isinstance(img_b, Image.Image)
    assert meta["config_a"]["lines"] == meta["config_b"]["lines"]
    assert meta["config_a"]["num_lines"] == meta["config_b"]["num_lines"]
    assert ground_truth == clean_ground_truth(SAMPLE_TEXT)


def test_all_fonts_load_without_error():
    """Every font in fonts_ok.json must render a paragraph without raising."""
    fonts = load_fonts()
    assert len(fonts) == 76

    for font in fonts:
        config = RenderConfig(
            font_path=font["path"],
            font_index=int(font.get("index", 0)),
            augmentations={},
            font_family=font["family"],
            font_style=font["style"],
            category=font["category"],
        )
        img, _, _ = render("Ħġ ċ ż test", config, p_hyphen=0.0, rng=random.Random(0))
        assert isinstance(img, Image.Image)


def test_image_width_matches_config():
    """The rendered image width must equal the configured width."""
    for width in (400, 600, 900):
        config = _no_aug_config(width=width)
        img, _, _ = render(SAMPLE_TEXT, config, rng=random.Random(0))
        assert img.width == width


def test_render_returns_image_str_dict():
    """render() must return (PIL.Image, str, dict) with the documented keys."""
    config = _no_aug_config()
    result = render(SAMPLE_TEXT, config, rng=random.Random(0))

    assert isinstance(result, tuple) and len(result) == 3
    img, ground_truth, meta = result
    assert isinstance(img, Image.Image)
    assert isinstance(ground_truth, str) and ground_truth
    assert isinstance(meta, dict)
    for key in (
        "font_family",
        "font_style",
        "image_size",
        "num_lines",
        "augmentations_applied",
    ):
        assert key in meta
