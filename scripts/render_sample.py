"""Quick sanity-check: render 10 synthetic images and print their paths."""

import sys
from pathlib import Path

# Allow running as `python scripts/render_sample.py` without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maltese_ocr.render import render_sample  # noqa: E402  (populated in Stage 1 work)

if __name__ == "__main__":
    paths = render_sample(n=10, out_dir=Path("data/synthetic/sample"))
    for p in paths:
        print(p)
