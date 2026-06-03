# test_baseline.py — Evaluate the OCR pipeline on the 422-image dev set.
#
# Outputs:
#   - Per-image CER printed to stdout
#   - results.json sorted by CER descending
#   - Summary: overall CER, side-by-side before/after comparison for images
#     that previously returned empty output (helped by preprocessing fallback)

import json
import os

import PIL.Image
import jiwer
from competition_transcriber import CompetitionTranscriber

# Images that returned empty string under the old config (Tesseract PSM 3,
# no preprocessing).  Kept here so we can show targeted before/after output.
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
    raw_cers      = []   # Tesseract PSM 6 with NO preprocessing fallback
    final_cers    = []   # Tesseract PSM 6 WITH preprocessing fallback (transcribe)
    blank_details = []   # side-by-side for the 7 previously-blank images

    for entry in entries:
        filename   = entry["image"]
        expected   = entry["text"]
        image_path = os.path.join("data", "dev_set", filename)

        image = PIL.Image.open(image_path)

        # --- without preprocessing fallback ---
        raw = (
            transcriber._run_tesseract(image)
            if transcriber._has_tesseract
            else ""
        )

        # --- with preprocessing fallback (the real transcribe path) ---
        final = transcriber.transcribe(image)

        cer_raw   = jiwer.cer(expected, raw)
        cer_final = jiwer.cer(expected, final)

        raw_cers.append(cer_raw)
        final_cers.append(cer_final)

        results.append({
            "image":     filename,
            "predicted": final,
            "expected":  expected,
            "cer":       cer_final,
        })

        if filename in _PREVIOUSLY_BLANK:
            blank_details.append({
                "image":    filename,
                "expected": expected,
                "raw":      raw,
                "final":    final,
                "cer_raw":  cer_raw,
                "cer_final":cer_final,
            })

        print(f"=== {filename} ===")
        print(f"PREDICTED: {final}")
        print(f"EXPECTED:  {expected}")
        print(f"CER: {cer_final:.3f}")
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

    # --- CER summary: before vs after preprocessing fallback ---
    avg_raw   = sum(raw_cers)   / len(raw_cers)
    avg_final = sum(final_cers) / len(final_cers)

    print("=" * 60)
    print("CER SUMMARY")
    print("=" * 60)
    print(f"  Without preprocessing fallback : {avg_raw:.4f}")
    print(f"  With    preprocessing fallback : {avg_final:.4f}")
    delta = avg_raw - avg_final
    print(f"  Improvement                    : {delta:+.4f}")
    print()

    # --- Previously-blank image comparison ---
    now_non_blank = sum(1 for d in blank_details if len(d["final"]) >= 3)
    print("=" * 60)
    print(f"PREVIOUSLY BLANK IMAGES (7 images that returned '' under PSM 3)")
    print(f"  Now have predictions: {now_non_blank} / {len(blank_details)}")
    print("=" * 60)
    for d in blank_details:
        tag = "IMPROVED" if d["cer_final"] < d["cer_raw"] else (
              "SAME"     if d["cer_final"] == d["cer_raw"] else "WORSE")
        print(f"\n  {d['image']}  [{tag}]")
        print(f"    EXPECTED : {d['expected']}")
        print(f"    RAW      : {d['raw'] or '(empty)'}  (CER {d['cer_raw']:.3f})")
        print(f"    FINAL    : {d['final'] or '(empty)'}  (CER {d['cer_final']:.3f})")
    print()


if __name__ == "__main__":
    main()
