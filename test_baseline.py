# Test script that transcribes dev_set images and measures accuracy using CER
# CER (Character Error Rate) = fraction of characters that are wrong (0.0 is perfect, 1.0 is all wrong)

import json
import os
import PIL.Image
import jiwer
from competition_transcriber import CompetitionTranscriber


def main():
    # Load ground truth transcriptions from JSON file (list of {image, text, as_lines} entries)
    with open("data/texts.json", encoding="utf-8") as f:
        entries = json.load(f)

    # Initialize the transcriber once and reuse it for all images
    transcriber = CompetitionTranscriber()

    results = []

    for entry in entries:
        filename = entry["image"]
        expected = entry["text"]
        image_path = os.path.join("data", "dev_set", filename)

        print(f"=== {filename} ===")

        # Open the image and run Tesseract OCR
        image = PIL.Image.open(image_path)
        predicted = transcriber.transcribe(image)

        # Calculate CER: jiwer.cer(reference, hypothesis) — reference is the ground truth
        cer = jiwer.cer(expected, predicted)
        results.append({"image": filename, "predicted": predicted, "expected": expected, "cer": cer})

        print(f"PREDICTED: {predicted}")
        print(f"EXPECTED:  {expected}")
        print(f"CER: {cer:.3f}")
        print()

    if not results:
        return

    # Sort results from highest CER to lowest and save to JSON
    results_sorted = sorted(results, key=lambda r: r["cer"], reverse=True)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    print("Saved results.json (sorted by CER, highest first)\n")

    # Print the 10 hardest images
    print("=== TOP 10 HARDEST IMAGES ===")
    for r in results_sorted[:10]:
        print(f"{r['image']:12s}  CER: {r['cer']:.3f}")
        print(f"  PREDICTED: {r['predicted']}")
        print(f"  EXPECTED:  {r['expected']}")
        print()

    average_cer = sum(r["cer"] for r in results) / len(results)
    print(f"AVERAGE CER: {average_cer:.3f}")


if __name__ == "__main__":
    main()
