# competition_transcriber.py — OCR transcriber for the Maltese text competition.
#
# Inference order:
#   1. Fine-tuned TrOCR model  (loaded from models/trocr-maltese/ if present)
#   2. Tesseract fallback       (used when TrOCR returns empty text, or if the
#                                fine-tuned model has not been trained yet)

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import PIL.Image
import torch
import malti.line_joiner

# Path where train.py saves the fine-tuned model
_TROCR_MODEL_DIR = Path("models/trocr-maltese")


class CompetitionTranscriber:

    def __init__(self) -> None:
        self._line_joiner = malti.line_joiner.RBLineJoiner()
        self._load_trocr()
        self._check_tesseract()

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
            # TrOCR is available so we can continue without Tesseract
            print(
                "[CompetitionTranscriber] Tesseract not found — "
                "TrOCR only, no fallback."
            )

    # ------------------------------------------------------------------
    # Private inference methods
    # ------------------------------------------------------------------

    def _run_trocr(self, image: PIL.Image.Image) -> str:
        """Run the fine-tuned TrOCR model and return the decoded text."""
        # Resize + normalise the image into the tensor format TrOCR expects
        pixel_values = self._proc(
            image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self._device)

        # generate() runs beam search to produce the most likely token sequence
        with torch.no_grad():
            generated_ids = self._trocr.generate(
                pixel_values,
                max_new_tokens=256,
                num_beams=4,          # beam search — better accuracy than greedy
                early_stopping=True,
            )

        # Decode the token IDs back into a readable string
        text = self._proc.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()

    def _run_tesseract(self, image: PIL.Image.Image) -> str:
        """Run Tesseract OCR and return the joined text."""
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "img.jpg")
            out_stem = os.path.join(tmp, "out")
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, image: PIL.Image.Image) -> str:
        """
        Transcribe `image` to text.

        Tries TrOCR first.  If TrOCR is unavailable or returns empty text,
        falls back to Tesseract.
        """
        if self._trocr is not None:
            text = self._run_trocr(image)
            if text:
                return text
            # TrOCR produced nothing (can happen on very unusual images)

        if self._has_tesseract:
            return self._run_tesseract(image)

        return ""
