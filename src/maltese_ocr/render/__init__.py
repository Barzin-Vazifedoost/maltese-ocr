"""Synthetic image rendering from Maltese text."""

from maltese_ocr.render.renderer import (
    DEFAULT_AUGMENTATIONS,
    FONTS_OK_JSON,
    LIGHT_BG_COLORS,
    RenderConfig,
    clean_ground_truth,
    load_fonts,
    render,
)

__all__ = [
    "DEFAULT_AUGMENTATIONS",
    "FONTS_OK_JSON",
    "LIGHT_BG_COLORS",
    "RenderConfig",
    "clean_ground_truth",
    "load_fonts",
    "render",
]
