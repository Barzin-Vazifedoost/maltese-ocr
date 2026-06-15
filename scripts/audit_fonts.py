"""Audit font files for complete Maltese character glyph coverage.

Checks every font in fonts/ (and the macOS catalog in fonts.py) against
configs/charset.txt.  A font PASSES if every *hard* character has a real
glyph (not .notdef, not absent from cmap).

Hard chars = the full charset MINUS the small set of rare Unicode symbols
in SOFT_CHARS.  Missing soft chars are reported as warnings but do NOT
disqualify a font — they appear so infrequently in the corpus that the
training impact is negligible.

Maltese-specific characters (ħ Ħ ġ Ġ ċ Ċ ż Ż) are always hard, and the
audit exit-code is non-zero if any passing font is missing one of them.

Output: fonts/fonts_ok.json — list of passing faces with metadata.

Usage:
    python scripts/audit_fonts.py [--fonts-dir PATH] [--output PATH]
"""

import argparse
import json
import sys
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont

REPO_ROOT = Path(__file__).resolve().parent.parent
CHARSET_FILE = REPO_ROOT / "configs" / "charset.txt"
DEFAULT_FONTS_DIR = REPO_ROOT / "fonts"
DEFAULT_OUTPUT = REPO_ROOT / "fonts" / "fonts_ok.json"

# Maltese-specific characters.  Missing any of these is always a hard failure.
MALTESE_CHARS = frozenset("ħĦġĠċĊżŻ")

# Rare Unicode symbols absent from most professional fonts.
# Missing these is logged as a warning; the font still passes.
# ♢ U+2662 white diamond suit
# ⁴ U+2074 superscript four
# ỹ U+1EF9 y with tilde (Latin Extended Additional)
SOFT_CHARS = frozenset("♢⁴ỹ")

# Name-fragment → category mapping for reliable detection.
_NAME_CATEGORY: dict[str, str] = {
    "noto serif": "serif",
    "noto sans": "sans",
    "lato": "sans",
    "merriweather": "serif",
    "source serif": "serif",
    "pt serif": "serif",
    "gentium": "serif",
    "charis sil": "serif",
    "libertinus serif": "serif",
    "libertinus sans": "sans",
    "libertinus mono": "mono",
    "eb garamond": "serif",
    "times new roman": "serif",
    "georgia": "serif",
    "palatino": "serif",
    "baskerville": "serif",
    "rockwell": "serif",
    "arial": "sans",
    "verdana": "sans",
    "trebuchet ms": "sans",
    "trebuchet": "sans",
    "courier new": "mono",
    "courier": "mono",
}


def load_charset(path: Path) -> list[str]:
    chars = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            chars.append(line[0])
    return chars


def detect_category(font: TTFont, family_name: str) -> str:
    lower = family_name.lower()
    for fragment, cat in _NAME_CATEGORY.items():
        if fragment in lower:
            return cat

    try:
        if font["post"].isFixedPitch:
            return "mono"
    except (KeyError, AttributeError):
        pass

    try:
        ibm_class = font["OS/2"].sFamilyClass >> 8
        if ibm_class in (1, 2, 3, 4, 5, 7):
            return "serif"
        if ibm_class == 8:
            return "sans"
    except (KeyError, AttributeError):
        pass

    return "unknown"


def detect_style(font: TTFont) -> str:
    try:
        raw = (font["name"].getDebugName(2) or "").lower()
    except (KeyError, AttributeError):
        return "regular"

    if "bold italic" in raw or "bold oblique" in raw:
        return "bold_italic"
    if "bold" in raw:
        return "bold"
    if "italic" in raw or "oblique" in raw:
        return "italic"
    return "regular"


def get_family_name(font: TTFont) -> str:
    try:
        return font["name"].getDebugName(1) or "Unknown"
    except (KeyError, AttributeError):
        return "Unknown"


def check_coverage(font: TTFont, chars: list[str]) -> tuple[list[str], list[str]]:
    """Return (hard_missing, soft_missing) for this font."""
    cmap = font.getBestCmap() or {}
    hard_missing = []
    soft_missing = []
    for ch in chars:
        if ord(ch) not in cmap or cmap[ord(ch)] == ".notdef":
            if ch in SOFT_CHARS:
                soft_missing.append(ch)
            else:
                hard_missing.append(ch)
    return hard_missing, soft_missing


def iter_faces(path: Path):
    """Yield (TTFont, face_index) for every face in a font file."""
    suffix = path.suffix.lower()
    if suffix in (".ttc", ".otc"):
        try:
            coll = TTCollection(str(path))
            for i, font in enumerate(coll.fonts):
                yield font, i
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}", file=sys.stderr)
    else:
        try:
            yield TTFont(str(path), lazy=True), 0
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}", file=sys.stderr)


def audit_face(
    font: TTFont,
    chars: list[str],
    path: Path,
    index: int,
    category_override: str | None = None,
) -> dict:
    family = get_family_name(font)
    style = detect_style(font)
    category = category_override or detect_category(font, family)
    hard_missing, soft_missing = check_coverage(font, chars)
    maltese_missing = [ch for ch in hard_missing if ch in MALTESE_CHARS]

    return {
        "family": family,
        "style": style,
        "category": category,
        "path": str(path),
        "index": index,
        "pass": len(hard_missing) == 0,
        "hard_missing": hard_missing,
        "soft_missing": soft_missing,
        "maltese_missing": maltese_missing,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fonts-dir",
        default=str(DEFAULT_FONTS_DIR),
        help="Directory of downloaded .ttf/.otf/.ttc files (default: fonts/)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for fonts_ok.json",
    )
    parser.add_argument(
        "--include-macos",
        action="store_true",
        default=True,
        help="Also audit macOS system fonts from fonts.py catalog (default: true)",
    )
    parser.add_argument(
        "--no-macos",
        dest="include_macos",
        action="store_false",
    )
    args = parser.parse_args(argv)

    fonts_dir = Path(args.fonts_dir)
    output_path = Path(args.output)

    if not CHARSET_FILE.exists():
        sys.exit(f"charset file not found: {CHARSET_FILE}")

    chars = load_charset(CHARSET_FILE)
    hard_chars = [ch for ch in chars if ch not in SOFT_CHARS]
    print(f"Charset: {len(chars)} chars  ({len(hard_chars)} hard, {len(SOFT_CHARS)} soft)")
    print("Maltese chars (always hard): " + " ".join(sorted(MALTESE_CHARS, key=ord)))
    print("Soft chars (warn only):      " + " ".join(sorted(SOFT_CHARS, key=ord)))

    all_results: list[dict] = []

    # ── macOS catalog ──────────────────────────────────────────────────────
    if args.include_macos:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from maltese_ocr.render.fonts import MACOS_FACES

        print(f"\nAuditing {len(MACOS_FACES)} macOS catalog faces …")
        for face in MACOS_FACES:
            p = Path(face.path)
            if not p.exists():
                print(f"  [missing file] {p}")
                continue
            try:
                font = TTFont(str(p), fontNumber=face.index, lazy=True)
                result = audit_face(font, chars, p, face.index, category_override=face.category)
                result["family"] = face.family
                result["style"] = face.style
                all_results.append(result)
                _print_result(result)
            except Exception as exc:
                print(f"  [error] {face.family} {face.style}: {exc}", file=sys.stderr)

    # ── fonts/ directory ───────────────────────────────────────────────────
    if fonts_dir.exists():
        font_files = sorted(
            p for p in fonts_dir.iterdir() if p.suffix.lower() in (".ttf", ".otf", ".ttc", ".otc")
        )
        print(f"\nAuditing {len(font_files)} files in {fonts_dir} …")
        for path in font_files:
            for font, idx in iter_faces(path):
                result = audit_face(font, chars, path, idx)
                all_results.append(result)
                _print_result(result)
    else:
        print(f"\nfonts/ directory not found — skipping downloaded fonts ({fonts_dir})")

    # ── Summary ───────────────────────────────────────────────────────────
    passed = [r for r in all_results if r["pass"]]
    failed = [r for r in all_results if not r["pass"]]
    passing_families = sorted({r["family"] for r in passed})

    print(f"\n{'─' * 60}")
    print(f"Total faces checked : {len(all_results)}")
    print(f"Passed              : {len(passed)}")
    print(f"Failed (hard miss)  : {len(failed)}")
    print(f"Passing families    : {len(passing_families)}")
    for fam in passing_families:
        cat = next(r["category"] for r in passed if r["family"] == fam)
        print(f"  ✓ {fam} ({cat})")

    if failed:
        print("\nFailed faces (hard missing chars):")
        for r in failed:
            print(
                f"  ✗ {r['family']:25s} {r['style']:12s}  "
                f"hard={r['hard_missing']}  maltese={r['maltese_missing']}"
            )

    # ── Write fonts_ok.json ───────────────────────────────────────────────
    fonts_ok = [
        {
            "family": r["family"],
            "style": r["style"],
            "category": r["category"],
            "path": r["path"],
            "index": r["index"],
            "soft_missing": r["soft_missing"],
        }
        for r in passed
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fonts_ok, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(fonts_ok)} entries → {output_path}")

    if len(passing_families) < 12:
        print(
            f"\nWARNING: only {len(passing_families)} passing families (target ≥ 12)",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_result(r: dict) -> None:
    if r["pass"]:
        soft_note = f"  (soft missing: {''.join(r['soft_missing'])})" if r["soft_missing"] else ""
        print(f"  {r['family']:30s} {r['style']:12s}  PASS{soft_note}")
    else:
        maltese_warn = (
            f"  ← MALTESE MISSING: {r['maltese_missing']}" if r["maltese_missing"] else ""
        )
        print(f"  {r['family']:30s} {r['style']:12s}  FAIL  hard={r['hard_missing']}{maltese_warn}")


if __name__ == "__main__":
    main()
