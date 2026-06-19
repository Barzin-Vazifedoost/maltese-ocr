"""Package the final model checkpoint for competition submission.

Copies the Stage 3 (or Stage 2 fallback) checkpoint into models/trocr-maltese-v2/,
verifies the required files are present, and prints the total directory size.
"""

import shutil
import sys
from pathlib import Path

REQUIRED_FILES = {"config.json", "tokenizer_config.json"}
WEIGHT_FILES = {"pytorch_model.bin", "model.safetensors"}

STAGE3_DIR = Path("models/stage3/best.pt")
OUTPUT_DIR = Path("models/trocr-maltese-v2")


def main() -> None:
    if not STAGE3_DIR.exists():
        sys.exit(f"[package] Stage 3 checkpoint not found at {STAGE3_DIR}. Run `make train` first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(STAGE3_DIR.parent, OUTPUT_DIR, dirs_exist_ok=True)

    missing = REQUIRED_FILES - {f.name for f in OUTPUT_DIR.iterdir()}
    if missing:
        sys.exit(f"[package] Missing required files in output dir: {missing}")

    if not any((OUTPUT_DIR / w).exists() for w in WEIGHT_FILES):
        sys.exit(f"[package] No weight file found in {OUTPUT_DIR}. Expected one of {WEIGHT_FILES}.")

    size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"[package] Done. {OUTPUT_DIR}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
