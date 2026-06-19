"""Smoke test: load microsoft/trocr-base-printed and run a dummy forward pass."""

import pytest
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

pytestmark = pytest.mark.slow  # requires network; skip with: pytest -m "not slow"


def test_trocr_forward_pass():
    model_id = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.eval()

    dummy = Image.new("RGB", (384, 384), color=(255, 255, 255))
    pixel_values = processor(images=dummy, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=16)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    assert isinstance(text, str), "decode output must be a string"


def test_trocr_encoder_output_shape():
    """Encoder must produce (batch, seq_len, hidden) with hidden=768 for trocr-base."""
    model_id = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.eval()

    dummy = Image.new("RGB", (384, 384), color=(200, 200, 200))
    pixel_values = processor(images=dummy, return_tensors="pt").pixel_values

    with torch.no_grad():
        encoder_out = model.encoder(pixel_values)

    hidden = encoder_out.last_hidden_state
    assert hidden.ndim == 3, "encoder output must be 3-D"
    assert hidden.shape[0] == 1, "batch size mismatch"
    assert hidden.shape[-1] == 768, "hidden size must be 768 for trocr-base"
