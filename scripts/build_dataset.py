"""Materialise N synthetic training samples to data/synthetic_v3/.

Streams clean Maltese paragraphs from MLRS/korpus_malti, renders each with a
randomly-sampled RenderConfig (font from fonts_ok.json), and writes:

    data/synthetic_v3/images/syn_000001.jpg, ...
    data/synthetic_v3/transcriptions.json   {filename: ground_truth, ...}

Usage:
    python3 scripts/build_dataset.py --n 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running as `python scripts/build_dataset.py` without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from maltese_ocr.data import stream_paragraphs  # noqa: E402
from maltese_ocr.render import RenderConfig, load_fonts, render  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="Number of samples to generate.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/synthetic_v3"),
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--min-chars", type=int, default=10, help="Minimum paragraph length.")
    parser.add_argument("--max-chars", type=int, default=500, help="Maximum paragraph length.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fonts = load_fonts()
    if not fonts:
        print("ERROR: no usable fonts in fonts_ok.json.")
        sys.exit(1)
    print(f"Loaded {len(fonts)} fonts.")

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "transcriptions.json"

    paragraphs = stream_paragraphs(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        limit=args.n,
    )

    transcriptions: dict[str, str] = {}
    count = 0
    for paragraph in tqdm(paragraphs, total=args.n, desc="Rendering", unit="img"):
        config = RenderConfig.sample(rng.choice(fonts), rng)
        img, ground_truth, _ = render(paragraph, config, rng=rng)

        count += 1
        filename = f"syn_{count:06d}.jpg"
        img.save(images_dir / filename, "JPEG", quality=90)
        transcriptions[filename] = ground_truth

        if count >= args.n:
            break

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {count} images -> {images_dir}/")
    print(f"Labels -> {out_json}")


if __name__ == "__main__":
    main()
