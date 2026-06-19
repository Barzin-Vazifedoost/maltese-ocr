"""Render 20 sample synthetic images to data/samples/ for visual inspection.

Samples are spread across font categories (serif / sans / mono) so the output
covers the full range of rendered appearances.  Runs fully offline using a
handful of built-in Maltese sentences (no corpus download needed) and prints
the path of every image it saves.

Usage:
    python3 scripts/render_sample.py
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from itertools import cycle
from pathlib import Path

# Allow running as `python scripts/render_sample.py` without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from maltese_ocr.render import RenderConfig, load_fonts, render  # noqa: E402

N_SAMPLES = 20
OUT_DIR = Path("data/samples")

# Built-in Maltese paragraphs covering ħ Ħ ġ Ġ ċ Ċ ż Ż, the għ digraph, and
# structural dashes (il-, tal-, għall-) that must survive into the ground truth.
SAMPLE_TEXTS = [
    "Il-kelb tal-baħar għadda mill-port filgħodu kmieni, meta ż-żiffa kienet "
    "għadha friska u x-xemx bdiet tielgħa fuq il-baħar kwiet.",
    "Iċ-ċumnija l-qadima ġiet imġedda mill-ġdid, u l-għalliema spjegat lit-tfal "
    "kif l-irħula tal-gżira nbnew madwar il-knejjes.",
    "Għall-ewwel darba, il-ġurnalista kiteb dwar iż-żmien meta l-Maltin kienu "
    "jaħdmu fl-għelieqi taħt ix-xemx tas-sajf.",
    "Ħafna mill-kotba l-antiki nżammu fil-librerija nazzjonali, fejn ir-riċerkaturi "
    "jistgħu jaqraw dwar l-istorja tal-pajjiż.",
    "Iż-żgħażagħ inġabru fil-pjazza biex jiċċelebraw il-festa, bid-daqq tal-banda "
    "u l-logħob tan-nar fuq il-knisja.",
]


def main() -> None:
    rng = random.Random(0)
    fonts = load_fonts()

    by_category: dict[str, list[dict]] = defaultdict(list)
    for font in fonts:
        by_category[font.get("category", "other")].append(font)

    categories = sorted(by_category)
    print(f"Loaded {len(fonts)} fonts across categories: {', '.join(categories)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Round-robin across categories so all of serif / sans / mono are sampled.
    cat_cycle = cycle(categories)
    saved: list[Path] = []
    for i in tqdm(range(N_SAMPLES), desc="Rendering samples", unit="img"):
        category = next(cat_cycle)
        font = rng.choice(by_category[category])
        config = RenderConfig.sample(font, rng)
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        img, _, meta = render(text, config, rng=rng)

        family = meta["font_family"].replace(" ", "")
        path = OUT_DIR / f"sample_{i + 1:02d}_{category}_{family}.jpg"
        img.save(path, "JPEG", quality=90)
        saved.append(path)

    print(f"\nSaved {len(saved)} sample images:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
