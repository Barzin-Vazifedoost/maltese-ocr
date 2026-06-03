# competition_transcriber.py — OCR transcriber for the Maltese text competition.
#
# Inference order (as of Phase 5 / Section 17 refactor):
#   1. Tesseract (PSM 6)        — our best-performing model, CER 0.0221
#      - If Tesseract returns < 3 characters, apply ImageMagick preprocessing
#        (upscale, grayscale, contrast, adaptive threshold) and retry.
#   2. TrOCR fallback           — used ONLY when Tesseract returns < 3 chars
#                                  even after preprocessing, AND a fine-tuned
#                                  model exists at models/trocr-maltese/.
#
# Why Tesseract first?
#   During evaluation Tesseract (PSM 6) achieved CER 0.0221, which is better
#   than our 5-epoch TrOCR fine-tune.  TrOCR also suffered from repetition
#   loops on some images.  Using TrOCR only as a last-resort fallback keeps
#   the high-quality Tesseract output for almost all images.

import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import PIL.Image
import torch
import malti.line_joiner

# wand needs MAGICK_HOME set before it is imported on macOS with Homebrew
os.environ.setdefault("MAGICK_HOME", "/opt/homebrew")

# Path where train.py saves the fine-tuned model
_TROCR_MODEL_DIR = Path("models/trocr-maltese")

class CompetitionTranscriber:

    def __init__(self) -> None:
        self._line_joiner = malti.line_joiner.RBLineJoiner()
        self._load_trocr()
        self._check_tesseract()
        self._check_wand()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_trocr(self) -> None:
        """Load the fine-tuned TrOCR model if it exists."""
        model_file_exists = any(
            (_TROCR_MODEL_DIR / f).exists()
            for f in ("config.json", "pytorch_model.bin", "model.safetensors")
        )
        if not model_file_exists:
            print(
                f"[CompetitionTranscriber] TrOCR model not found at "
                f"'{_TROCR_MODEL_DIR}' — will use Tesseract only."
            )
            self._trocr   = None
            self._proc    = None
            self._device  = None
            return

        # Import here so the file can be imported even without transformers
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        # Prefer MPS (Apple Silicon GPU) for fast inference, fall back to CPU
        self._device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        self._proc  = TrOCRProcessor.from_pretrained(str(_TROCR_MODEL_DIR))
        self._trocr = VisionEncoderDecoderModel.from_pretrained(
            str(_TROCR_MODEL_DIR)
        )
        self._trocr.to(self._device)
        self._trocr.eval()   # disable dropout — we are doing inference, not training

        print(
            f"[CompetitionTranscriber] Loaded TrOCR from '{_TROCR_MODEL_DIR}' "
            f"on {self._device}."
        )

    def _check_tesseract(self) -> None:
        """Confirm Tesseract is on PATH; raise if neither backend is available."""
        self._has_tesseract = shutil.which("tesseract") is not None
        if not self._has_tesseract:
            if self._trocr is None:
                raise RuntimeError(
                    "No transcription backend found. "
                    "Either train the TrOCR model (python3 train.py) or install "
                    "Tesseract (brew install tesseract tesseract-lang)."
                )
            print(
                "[CompetitionTranscriber] Tesseract not found — "
                "TrOCR only, no fallback."
            )

    def _check_wand(self) -> None:
        """Check whether the wand / ImageMagick library is available."""
        try:
            from wand.image import Image  # noqa: F401
            self._has_wand = True
            print("[CompetitionTranscriber] wand (ImageMagick) available — "
                  "preprocessing fallback enabled.")
        except ImportError:
            self._has_wand = False
            print("[CompetitionTranscriber] wand not installed — "
                  "preprocessing fallback disabled.")

    # ------------------------------------------------------------------
    # Private inference methods
    # ------------------------------------------------------------------

    def _run_trocr(self, image: PIL.Image.Image) -> str:
        """
        Run the fine-tuned TrOCR model and return the decoded text.

        repetition_penalty=2.0 :
            Strongly penalises the model for repeating tokens it has already
            generated.  Values > 1.0 make repetition less likely; 2.0 is a
            firm penalty that fixes the looping behaviour seen in the 5-epoch
            fine-tune (e.g. "tal-tal-tal-tal..." on noisy images).

        no_repeat_ngram_size=3 :
            Forbids the model from generating any 3-word sequence it has
            already produced in the same output.  Works alongside
            repetition_penalty as a hard block on repeated phrases.
        """
        pixel_values = self._proc(
            image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self._device)

        with torch.no_grad():
            generated_ids = self._trocr.generate(
                pixel_values,
                max_new_tokens=256,
                num_beams=4,               # beam search — better accuracy than greedy
                early_stopping=True,
                repetition_penalty=2.0,    # penalise repeated tokens (fixes loops)
                no_repeat_ngram_size=3,    # hard-block any repeated 3-gram phrase
            )

        text = self._proc.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()

    def _run_tesseract(self, image: PIL.Image.Image) -> str:
        """Run Tesseract OCR (PSM 6) and return the joined text."""
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "img.png")
            out_stem  = os.path.join(tmp, "out")
            image.save(img_path)

            subprocess.run(
                ["tesseract", "-l", "mlt", "--psm", "6", img_path, out_stem],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            with open(out_stem + ".txt", encoding="utf-8") as f:
                return self._line_joiner.join_lines(
                    f.read().strip().split("\n"),
                    fix_hyphenated_words=False,
                )

    def _apply_preprocessing(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Apply ImageMagick preprocessing and return a new PIL image.

        Pipeline (applied only to images where raw Tesseract gets < 3 chars):
          1. Upscale 3× if height < 200 px   — helps Tesseract on tiny images
          2. Convert to grayscale             — removes colour-background confusion
          3. Enhance contrast by 150 %        — sharpens faint or low-contrast text
          4. Adaptive threshold (binarise)    — converts uneven backgrounds to white
        """
        from wand.image import Image as WandImage

        buf = io.BytesIO()
        image.save(buf, format="PNG")

        with WandImage(blob=buf.getvalue()) as wimg:
            if wimg.height < 200:
                wimg.resize(wimg.width * 3, wimg.height * 3)

            wimg.transform_colorspace("gray")

            # +50 units ≈ 150 % of the default contrast level
            wimg.brightness_contrast(brightness=0, contrast=50)

            wimg.adaptive_threshold(
                width=max(1, wimg.width // 8),
                height=max(1, wimg.height // 8),
                offset=0,
            )

            return PIL.Image.open(io.BytesIO(wimg.make_blob("PNG")))

    @staticmethod
    def _postprocess(text: str) -> str:
        """
        Targeted post-processing fix applied to the Tesseract result.

        Fix: digit-hyphen-Capital (no surrounding spaces) → digit em-dash Capital
          e.g. "1-Ippjanata" → "1 — Ippjanata"   (also "2-Parzjalment" → "2 — …")

        The hyphen must be tight against both characters. Requiring no spaces is
        what keeps this safe: it matches the tight "1-Ippjanata" misread but NOT
        a spaced "Malti2 - Għaliex", where the hyphen is a real one in the ground
        truth (rewriting that to an em dash regresses 121.jpg).

        Only this specific pattern is applied — broad replacements hurt other
        images (a general digit→em-dash rule was tried before and made 12 images
        worse, because Maltese texts use real hyphens in page ranges and ISBNs).
        """
        return re.sub(r'(\d)-([A-ZĄĦĊĠŻ])', r'\1 — \2', text)

    def _run_tesseract_with_preprocessing(self, image: PIL.Image.Image) -> str:
        """
        Run Tesseract; if the result is empty or < 3 chars, apply
        ImageMagick preprocessing and retry.  Returns whichever result
        is non-empty (raw result takes priority if it has ≥ 3 chars).
        The targeted post-processing fix is applied to the chosen result.
        """
        text = self._run_tesseract(image)
        if len(text) >= 3:
            return self._postprocess(text)

        if not self._has_wand:
            return self._postprocess(text)

        text_pp = self._run_tesseract(self._apply_preprocessing(image))
        return self._postprocess(text_pp if text_pp else text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, image: PIL.Image.Image) -> str:
        """
        Transcribe `image` to text and return the result as a plain string.

        Inference order
        ---------------
        Step 1 — Tesseract (always tried first, because it achieves CER 0.0221).
            • Run Tesseract with PSM 6 ("uniform block of text").
            • If output is >= 3 characters, return it immediately — we are done.
            • If output is < 3 characters, apply ImageMagick preprocessing
              (upscale, grayscale, contrast, binarise) and retry Tesseract.
            • If the preprocessed result is >= 3 chars, return it.

        Step 2 — TrOCR (last-resort fallback).
            • Only reached when BOTH raw and preprocessed Tesseract produced
              fewer than 3 characters (very rare — ~7 images in the dev set).
            • Requires the fine-tuned model at models/trocr-maltese/.
            • Uses repetition_penalty + no_repeat_ngram_size to suppress loops.
        """
        # ---- Step 1: Tesseract (+ ImageMagick preprocessing retry) ----
        if self._has_tesseract:
            # Runs raw Tesseract; if that yields < 3 chars it retries on a
            # preprocessed image and returns the best non-empty result.
            text = self._run_tesseract_with_preprocessing(image)
            if len(text) >= 3:
                # Tesseract produced a good result — return straight away.
                return text

            # Only 0-2 chars. If TrOCR can't help, return what we have
            # (a 1-2 char string) rather than an empty string.
            if self._trocr is None:
                return text

        # ---- Step 2: TrOCR fallback (only if Tesseract failed / unavailable) ----
        if self._trocr is not None:
            trocr_text = self._run_trocr(image)
            if trocr_text:
                return trocr_text

        # Nothing worked — return empty string.
        return ""
