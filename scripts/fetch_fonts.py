"""Download open-license fonts and save them to fonts/.

Sources:
  - github.com/google/fonts raw files (Google Fonts)
  - GitHub releases API — for SIL (Gentium Plus, Charis SIL) and Libertinus

After downloading, runs scripts/audit_fonts.py on the combined set
(macOS system fonts + fonts/ directory).

Usage:
    python scripts/fetch_fonts.py [--skip-existing] [--no-audit]
"""

import argparse
import io
import json
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "fonts"

_UA = "maltese-ocr-font-fetcher/1.0"

# Build an SSL context that works on macOS Python installed from python.org.
# That installer ships without the system trust store; certifi provides a
# bundled CA bundle that fixes certificate verification.
try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

# ---------------------------------------------------------------------------
# Google Fonts via raw.githubusercontent.com/google/fonts
# Each tuple is (repo_path, local_filename).
# Variable fonts carry axis tags in brackets (URL-encoded on the wire).
# ---------------------------------------------------------------------------
_GF = "https://raw.githubusercontent.com/google/fonts/main"

GOOGLE_FONT_FILES: list[tuple[str, str]] = [
    # Noto Serif — variable font (roman + italic axis file)
    ("ofl/notoserif", "NotoSerif[wdth,wght].ttf"),
    ("ofl/notoserif", "NotoSerif-Italic[wdth,wght].ttf"),
    # Noto Sans — variable font
    ("ofl/notosans", "NotoSans[wdth,wght].ttf"),
    ("ofl/notosans", "NotoSans-Italic[wdth,wght].ttf"),
    # Lato — static fonts
    ("ofl/lato", "Lato-Regular.ttf"),
    ("ofl/lato", "Lato-Bold.ttf"),
    ("ofl/lato", "Lato-Italic.ttf"),
    ("ofl/lato", "Lato-BoldItalic.ttf"),
    # Merriweather — variable font
    ("ofl/merriweather", "Merriweather[opsz,wdth,wght].ttf"),
    ("ofl/merriweather", "Merriweather-Italic[opsz,wdth,wght].ttf"),
    # Source Serif 4 — variable font
    ("ofl/sourceserif4", "SourceSerif4[opsz,wght].ttf"),
    ("ofl/sourceserif4", "SourceSerif4-Italic[opsz,wght].ttf"),
    # PT Serif — static fonts
    ("ofl/ptserif", "PT_Serif-Web-Regular.ttf"),
    ("ofl/ptserif", "PT_Serif-Web-Bold.ttf"),
    ("ofl/ptserif", "PT_Serif-Web-Italic.ttf"),
    ("ofl/ptserif", "PT_Serif-Web-BoldItalic.ttf"),
    # EB Garamond — variable font
    ("ofl/ebgaramond", "EBGaramond[wght].ttf"),
    ("ofl/ebgaramond", "EBGaramond-Italic[wght].ttf"),
]

# GitHub release sources for fonts not on Google Fonts.
GITHUB_RELEASES: list[dict] = [
    {"label": "Gentium Plus", "repo": "silnrsi/font-gentium", "asset_suffix": ".zip"},
    {"label": "Charis SIL", "repo": "silnrsi/font-charis", "asset_suffix": ".zip"},
    {"label": "Libertinus", "repo": "alerque/libertinus", "asset_suffix": ".zip"},
]


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return resp.read()


def _extract_zip(data: bytes, dest: Path, skip_existing: bool) -> list[Path]:
    saved: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".ttf", ".otf")):
                fname = Path(name).name
                target = dest / fname
                if skip_existing and target.exists():
                    print(f"    skip (exists) {fname}")
                    continue
                target.write_bytes(zf.read(name))
                saved.append(target)
                print(f"    saved {fname}")
    return saved


# ---------------------------------------------------------------------------
# Download strategies
# ---------------------------------------------------------------------------


def fetch_google_fonts(dest: Path, skip_existing: bool) -> list[Path]:
    saved: list[Path] = []
    for repo_dir, filename in GOOGLE_FONT_FILES:
        target = dest / filename
        if skip_existing and target.exists():
            print(f"  skip (exists) {filename}")
            continue
        # URL-encode brackets so the request parses correctly.
        encoded_name = urllib.parse.quote(filename, safe=".-_")
        url = f"{_GF}/{repo_dir}/{encoded_name}"
        print(f"  {filename} …", end=" ", flush=True)
        try:
            data = _get(url, timeout=60)
            target.write_bytes(data)
            saved.append(target)
            print(f"OK ({len(data) // 1024} KB)")
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}", file=sys.stderr)
        except Exception as exc:
            print(f"error — {exc}", file=sys.stderr)
    return saved


def fetch_github_release(
    label: str, repo: str, asset_suffix: str, dest: Path, skip_existing: bool
) -> list[Path]:
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    print(f"  Fetching {label} from github.com/{repo} …")
    try:
        meta = json.loads(_get(api, timeout=30))
    except Exception as exc:
        print(f"    error reading release metadata — {exc}", file=sys.stderr)
        return []

    assets = [a for a in meta.get("assets", []) if a["name"].endswith(asset_suffix)]
    if not assets:
        print(f"    no {asset_suffix} asset found in latest release", file=sys.stderr)
        return []

    # Pick smallest zip to avoid huge extra-files bundles.
    asset = min(assets, key=lambda a: a["size"])
    print(f"    downloading {asset['name']} ({asset['size'] // 1024} KB) …")

    try:
        data = _get(asset["browser_download_url"], timeout=180)
    except Exception as exc:
        print(f"    download failed — {exc}", file=sys.stderr)
        return []

    return _extract_zip(data, dest, skip_existing)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading if font file already exists in fonts/",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Do not run audit_fonts.py after downloading",
    )
    args = parser.parse_args(argv)

    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    all_saved: list[Path] = []

    print("=== Google Fonts (via github.com/google/fonts) ===")
    all_saved.extend(fetch_google_fonts(FONTS_DIR, args.skip_existing))

    print("\n=== GitHub Releases (SIL + Libertinus) ===")
    for spec in GITHUB_RELEASES:
        saved = fetch_github_release(
            spec["label"], spec["repo"], spec["asset_suffix"], FONTS_DIR, args.skip_existing
        )
        all_saved.extend(saved)

    print(f"\nDownloaded {len(all_saved)} font file(s) to {FONTS_DIR}")

    if not args.no_audit:
        print("\n=== Running combined font audit ===")
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from audit_fonts import main as run_audit  # noqa: PLC0415

        try:
            run_audit()
        except SystemExit as exc:
            sys.exit(exc.code)


if __name__ == "__main__":
    main()
