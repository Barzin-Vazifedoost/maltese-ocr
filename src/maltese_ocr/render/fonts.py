"""Font catalog for Maltese OCR synthetic data generation.

Defines every font face available for rendering.  The audit script
(scripts/audit_fonts.py) verifies glyph coverage against configs/charset.txt
and writes fonts_ok.json with the subset that passes.
"""

from dataclasses import dataclass, field

_SUPP = "/System/Library/Fonts/Supplemental"
_SYS = "/System/Library/Fonts"


@dataclass
class FontFace:
    family: str
    style: str  # "regular" | "bold" | "italic" | "bold_italic"
    category: str  # "serif" | "sans" | "mono"
    path: str
    index: int = field(default=0)  # TTC face index; 0 for single-face files


# macOS system fonts with verified path structure.
# Styles within a TTC collection are identified by face index.
# Fonts that FAILED prior glyph-coverage testing are omitted:
#   PT Serif, Charter, Gill Sans, Optima — missing ħ/Ħ/ġ/Ġ/ċ/Ċ or more.
MACOS_FACES: list[FontFace] = [
    # ── Classic serif ──────────────────────────────────────────────────────
    FontFace("Times New Roman", "regular", "serif", f"{_SUPP}/Times New Roman.ttf"),
    FontFace("Times New Roman", "bold", "serif", f"{_SUPP}/Times New Roman Bold.ttf"),
    FontFace("Times New Roman", "italic", "serif", f"{_SUPP}/Times New Roman Italic.ttf"),
    FontFace("Times New Roman", "bold_italic", "serif", f"{_SUPP}/Times New Roman Bold Italic.ttf"),
    FontFace("Georgia", "regular", "serif", f"{_SUPP}/Georgia.ttf"),
    FontFace("Georgia", "bold", "serif", f"{_SUPP}/Georgia Bold.ttf"),
    FontFace("Georgia", "italic", "serif", f"{_SUPP}/Georgia Italic.ttf"),
    FontFace("Georgia", "bold_italic", "serif", f"{_SUPP}/Georgia Bold Italic.ttf"),
    # Palatino TTC: [0]=Regular [1]=Italic [2]=Bold [3]=Bold Italic
    FontFace("Palatino", "regular", "serif", f"{_SYS}/Palatino.ttc", index=0),
    FontFace("Palatino", "italic", "serif", f"{_SYS}/Palatino.ttc", index=1),
    FontFace("Palatino", "bold", "serif", f"{_SYS}/Palatino.ttc", index=2),
    FontFace("Palatino", "bold_italic", "serif", f"{_SYS}/Palatino.ttc", index=3),
    # Baskerville TTC: [0]=Regular [1]=Bold [2]=Italic [3]=Bold Italic
    FontFace("Baskerville", "regular", "serif", f"{_SUPP}/Baskerville.ttc", index=0),
    FontFace("Baskerville", "bold", "serif", f"{_SUPP}/Baskerville.ttc", index=1),
    FontFace("Baskerville", "italic", "serif", f"{_SUPP}/Baskerville.ttc", index=2),
    FontFace("Baskerville", "bold_italic", "serif", f"{_SUPP}/Baskerville.ttc", index=3),
    # ── Slab serif ─────────────────────────────────────────────────────────
    # Rockwell TTC: [0]=Regular [1]=Italic [2]=Bold [3]=Bold Italic
    FontFace("Rockwell", "regular", "serif", f"{_SUPP}/Rockwell.ttc", index=0),
    FontFace("Rockwell", "italic", "serif", f"{_SUPP}/Rockwell.ttc", index=1),
    FontFace("Rockwell", "bold", "serif", f"{_SUPP}/Rockwell.ttc", index=2),
    FontFace("Rockwell", "bold_italic", "serif", f"{_SUPP}/Rockwell.ttc", index=3),
    # ── Sans-serif ─────────────────────────────────────────────────────────
    FontFace("Arial", "regular", "sans", f"{_SUPP}/Arial.ttf"),
    FontFace("Arial", "bold", "sans", f"{_SUPP}/Arial Bold.ttf"),
    FontFace("Arial", "italic", "sans", f"{_SUPP}/Arial Italic.ttf"),
    FontFace("Arial", "bold_italic", "sans", f"{_SUPP}/Arial Bold Italic.ttf"),
    FontFace("Verdana", "regular", "sans", f"{_SUPP}/Verdana.ttf"),
    FontFace("Verdana", "bold", "sans", f"{_SUPP}/Verdana Bold.ttf"),
    FontFace("Verdana", "italic", "sans", f"{_SUPP}/Verdana Italic.ttf"),
    FontFace("Verdana", "bold_italic", "sans", f"{_SUPP}/Verdana Bold Italic.ttf"),
    FontFace("Trebuchet MS", "regular", "sans", f"{_SUPP}/Trebuchet MS.ttf"),
    FontFace("Trebuchet MS", "bold", "sans", f"{_SUPP}/Trebuchet MS Bold.ttf"),
    FontFace("Trebuchet MS", "italic", "sans", f"{_SUPP}/Trebuchet MS Italic.ttf"),
    FontFace("Trebuchet MS", "bold_italic", "sans", f"{_SUPP}/Trebuchet MS Bold Italic.ttf"),
    # ── Monospace ──────────────────────────────────────────────────────────
    FontFace("Courier New", "regular", "mono", f"{_SUPP}/Courier New.ttf"),
    FontFace("Courier New", "bold", "mono", f"{_SUPP}/Courier New Bold.ttf"),
    FontFace("Courier New", "italic", "mono", f"{_SUPP}/Courier New Italic.ttf"),
    FontFace("Courier New", "bold_italic", "mono", f"{_SUPP}/Courier New Bold Italic.ttf"),
]
