# test_baseline.py — Evaluate the OCR pipeline on the 422-image dev set.
#
# Outputs:
#   - Per-image CER printed to stdout
#   - results.json sorted by CER descending
#   - Summary: overall CER and the status of images that previously returned
#     empty output (now recovered by the preprocessing / TrOCR fallback path)
#
# Each image is transcribed exactly once via CompetitionTranscriber.transcribe(),
# which is the real competition path (Tesseract PSM 6 + ImageMagick fallback).

import json
import os

import PIL.Image
import jiwer
from competition_transcriber import CompetitionTranscriber

# Images that returned empty string under the old config (Tesseract PSM 3,
# no preprocessing).  Kept here so we can report their current status.
_PREVIOUSLY_BLANK = {
    "054.jpg", "067.jpg", "096.jpg", "102.jpg",
    "106.jpg", "165.jpg", "214.jpg",
}


def main():
    with open("data/texts.json", encoding="utf-8") as f:
        entries = json.load(f)

    transcriber = CompetitionTranscriber()
    print()

    results       = []
    cers          = []   # CER of the real transcribe() path, per image
    blank_details = []   # current status of the 7 previously-blank images

    for entry in entries:
        filename   = entry["image"]
        expected   = entry["text"]
        image_path = os.path.join("data", "dev_set", filename)

        image = PIL.Image.open(image_path)

        # The real competition path: Tesseract PSM 6 + ImageMagick fallback.
        predicted = transcriber.transcribe(image)
        cer = jiwer.cer(expected, predicted)
        cers.append(cer)

        results.append({
            "image":     filename,
            "predicted": predicted,
            "expected":  expected,
            "cer":       cer,
        })

        if filename in _PREVIOUSLY_BLANK:
            blank_details.append({
                "image":     filename,
                "expected":  expected,
                "predicted": predicted,
                "cer":       cer,
            })

        print(f"=== {filename} ===")
        print(f"PREDICTED: {predicted}")
        print(f"EXPECTED:  {expected}")
        print(f"CER: {cer:.3f}")
        print()

    if not results:
        return

    # Save results.json
    results_sorted = sorted(results, key=lambda r: r["cer"], reverse=True)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    print("Saved results.json (sorted by CER, highest first)\n")

    # --- Top 10 hardest images ---
    print("=== TOP 10 HARDEST IMAGES ===")
    for r in results_sorted[:10]:
        print(f"{r['image']:12s}  CER: {r['cer']:.3f}")
        print(f"  PREDICTED: {r['predicted']}")
        print(f"  EXPECTED:  {r['expected']}")
        print()

    # --- Overall CER ---
    avg_cer = sum(cers) / len(cers)
    print("=" * 60)
    print("CER SUMMARY")
    print("=" * 60)
    print(f"  Overall CER ({len(cers)} images): {avg_cer:.4f}")
    print()

    # --- Previously-blank image status ---
    now_non_blank = sum(1 for d in blank_details if len(d["predicted"]) >= 3)
    print("=" * 60)
    print("PREVIOUSLY BLANK IMAGES (7 images that returned '' under PSM 3)")
    print(f"  Now have predictions: {now_non_blank} / {len(blank_details)}")
    print("=" * 60)
    for d in blank_details:
        print(f"\n  {d['image']}  (CER {d['cer']:.3f})")
        print(f"    EXPECTED  : {d['expected']}")
        print(f"    PREDICTED : {d['predicted'] or '(empty)'}")
    print()


if __name__ == "__main__":
    main()
