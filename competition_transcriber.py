# competition_transcriber.py — OCR transcriber for the Maltese text competition.
#
# Inference order:
#   1. Fine-tuned TrOCR model  (loaded from models/trocr-maltese/ if present)
#   2. Tesseract fallback       (used when TrOCR returns empty text, or if the
#                                fine-tuned model has not been trained yet)
#      - First tries the raw image with Tesseract (PSM 6)
#      - If the result is empty or < 3 chars, applies ImageMagick preprocessing
#        and retries (handles coloured backgrounds, decorative fonts, tiny images)

import io
import os
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
        """Run the fine-tuned TrOCR model and return the decoded text."""
        pixel_values = self._proc(
            image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self._device)

        with torch.no_grad():
            generated_ids = self._trocr.generate(
                pixel_values,
                max_new_tokens=256,
                num_beams=4,          # beam search — better accuracy than greedy
                early_stopping=True,
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
          1. Upscale 2× if height < 150 px   — helps Tesseract on tiny images
          2. Convert to grayscale             — removes colour-background confusion
          3. Enhance contrast by 150 %        — sharpens faint or low-contrast text
          4. Adaptive threshold (binarise)    — converts uneven backgrounds to white
        """
        from wand.image import Image as WandImage

        buf = io.BytesIO()
        image.save(buf, format="PNG")

        with WandImage(blob=buf.getvalue()) as wimg:
            if wimg.height < 150:
                wimg.resize(wimg.width * 2, wimg.height * 2)

            wimg.transform_colorspace("gray")

            # +50 units ≈ 150 % of the default contrast level
            wimg.brightness_contrast(brightness=0, contrast=50)

            wimg.adaptive_threshold(
                width=max(1, wimg.width // 8),
                height=max(1, wimg.height // 8),
                offset=0,
            )

            return PIL.Image.open(io.BytesIO(wimg.make_blob("PNG")))

    def _run_tesseract_with_preprocessing(self, image: PIL.Image.Image) -> str:
        """
        Run Tesseract; if the result is empty or < 3 chars, apply
        ImageMagick preprocessing and retry.  Returns whichever result
        is non-empty (raw result takes priority if it has ≥ 3 chars).
        """
        text = self._run_tesseract(image)
        if len(text) >= 3:
            return text

        if not self._has_wand:
            return text

        text_pp = self._run_tesseract(self._apply_preprocessing(image))
        return text_pp if text_pp else text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, image: PIL.Image.Image) -> str:
        """
        Transcribe `image` to text.

        Tries TrOCR first.  If TrOCR is unavailable or returns empty text,
        falls back to Tesseract (with ImageMagick preprocessing for images
        where Tesseract alone returns too little text).
        """
        if self._trocr is not None:
            text = self._run_trocr(image)
            if text:
                return text

        if self._has_tesseract:
            return self._run_tesseract_with_preprocessing(image)

        return ""
