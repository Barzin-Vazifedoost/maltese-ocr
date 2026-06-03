"""
train_colab.py — Fine-tune TrOCR on synthetic Maltese OCR images on Google Colab.

This is the Colab version of train.py. Key differences from the Mac version:
  - Uses CUDA (NVIDIA T4 GPU) instead of MPS (Apple Silicon)
  - num_workers=2 in DataLoaders for faster data loading on Linux
  - 5 epochs instead of 10 (T4 is fast enough that 5 gives good results)
  - Paths point to Google Drive where you uploaded your data

Before running, make sure you have:
  1. Mounted Google Drive (Cell 1 in run_colab.ipynb)
  2. Uploaded data/synthetic/ to MyDrive/maltese-OCR/synthetic/
  3. Installed dependencies (Cell 2 in run_colab.ipynb)
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Google Drive paths — your uploaded data must be at these locations
IMAGES_DIR     = Path("/content/drive/MyDrive/maltese-OCR/synthetic/images")
TRANSCRIPTIONS = Path("/content/drive/MyDrive/maltese-OCR/synthetic/transcriptions.json")
SAVE_DIR       = Path("/content/drive/MyDrive/maltese-OCR/models/trocr-maltese")

PRETRAINED_MODEL = "microsoft/trocr-base-handwritten"

NUM_EPOCHS     = 5       # 5 epochs is enough on a T4; each takes ~25-35 minutes
BATCH_SIZE     = 8
LEARNING_RATE  = 5e-5
VAL_SPLIT      = 0.10    # 10% of images held out for validation
MAX_TARGET_LEN = 256     # token sequences longer than this are truncated
RANDOM_SEED    = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MalteseOCRDataset(Dataset):
    """
    Loads (image, text) pairs for TrOCR fine-tuning.

    __getitem__ returns a dict with two tensors:
      pixel_values — the image resized and normalised by the TrOCR processor
      labels       — target token IDs; padding positions set to -100 so the
                     loss function ignores them (standard HuggingFace convention)
    """

    def __init__(
        self,
        image_paths: list[Path],
        texts: list[str],
        processor: TrOCRProcessor,
    ) -> None:
        self.image_paths = image_paths
        self.texts       = texts
        self.processor   = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        # Open image and convert to RGB (handles RGBA or grayscale edge cases)
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # The processor resizes the image to 384x384 and normalises pixel values
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenise the ground-truth text into integer token IDs
        label_ids = self.processor.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=MAX_TARGET_LEN,
            truncation=True,
        ).input_ids

        # Replace pad token with -100 so cross-entropy loss ignores padding
        pad_id    = self.processor.tokenizer.pad_token_id
        label_ids = [t if t != pad_id else -100 for t in label_ids]

        return {
            "pixel_values": pixel_values,
            "labels":       torch.tensor(label_ids, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# One epoch of training or validation
# ---------------------------------------------------------------------------

def run_epoch(
    model:     VisionEncoderDecoderModel,
    loader:    DataLoader,
    device:    torch.device,
    optimizer: AdamW | None,
    desc:      str,
) -> float:
    """
    Run one full pass over `loader` and return the average loss.

    Pass optimizer=None for a validation pass (no weight updates).
    Pass the AdamW optimizer for a training pass (gradients + weight updates).
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(is_training):
        for batch in tqdm(loader, desc=desc, leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            # VisionEncoderDecoderModel computes cross-entropy loss internally
            # when labels are passed — no need to call a separate loss function
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss

            if is_training:
                optimizer.zero_grad()   # clear gradients from the previous batch
                loss.backward()         # compute gradients via backpropagation
                optimizer.step()        # update model weights

            total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Device — Colab uses CUDA (NVIDIA T4 GPU)
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Go to Runtime > Change runtime type "
              "and select GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load transcriptions
    # ------------------------------------------------------------------
    print("\nLoading transcriptions…")
    with open(TRANSCRIPTIONS, encoding="utf-8") as f:
        transcriptions: dict[str, str] = json.load(f)

    # Only include syn_*.jpg files (skip any smoke-test images)
    image_paths: list[Path] = []
    texts:       list[str]  = []
    for filename, text in sorted(transcriptions.items()):
        if filename.startswith("syn_"):
            path = IMAGES_DIR / filename
            if path.exists():
                image_paths.append(path)
                texts.append(text)

    print(f"Found {len(image_paths)} valid training images.")

    # ------------------------------------------------------------------
    # Load pretrained TrOCR model + processor from HuggingFace
    # ------------------------------------------------------------------
    print(f"\nLoading {PRETRAINED_MODEL}…")
    processor = TrOCRProcessor.from_pretrained(PRETRAINED_MODEL)
    model     = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_MODEL)

    # Required settings so the decoder and tokeniser work together correctly
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.eos_token_id           = processor.tokenizer.sep_token_id

    model.to(device)
    print(f"Model ready on {device}.")

    # ------------------------------------------------------------------
    # Build datasets and data loaders
    # ------------------------------------------------------------------
    full_dataset = MalteseOCRDataset(image_paths, texts, processor)

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    print(f"\nSplit: {train_size} train / {val_size} validation")

    # num_workers=2 speeds up data loading on Colab's Linux environment
    # (unlike macOS, Linux handles multiprocessing in DataLoader fine)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2,
        pin_memory=True,   # pin_memory=True speeds up CPU→GPU transfers
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(
        f"\nTraining for {NUM_EPOCHS} epochs "
        f"(batch_size={BATCH_SIZE}, lr={LEARNING_RATE})…\n"
    )

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss = run_epoch(
            model, train_loader, device,
            optimizer=optimizer,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} train",
        )
        val_loss = run_epoch(
            model, val_loader, device,
            optimizer=None,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} val  ",
        )

        saved = val_loss < best_val_loss
        if saved:
            best_val_loss = val_loss
            model.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} — "
            f"train_loss: {train_loss:.4f}  "
            f"val_loss: {val_loss:.4f}"
            + ("  ✓ best checkpoint saved" if saved else "")
        )

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
