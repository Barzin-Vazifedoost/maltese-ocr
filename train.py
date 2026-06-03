"""
train.py — Fine-tune TrOCR on synthetic Maltese OCR training images.

What this script does, step by step:
  1. Loads 5000 synthetic images + their ground-truth text from transcriptions.json
  2. Splits them 90% train / 10% validation
  3. Loads the pretrained TrOCR model from HuggingFace
  4. Fine-tunes for 10 epochs on the MPS (Apple Silicon) GPU
  5. After each epoch, evaluates on the validation split
  6. Saves the best checkpoint (lowest val loss) to models/trocr-maltese/

Run with:
    python3 train.py
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
# Configuration — change these if you want to experiment
# ---------------------------------------------------------------------------

IMAGES_DIR     = Path("data/synthetic/images")
TRANSCRIPTIONS = Path("data/synthetic/transcriptions.json")
SAVE_DIR       = Path("models/trocr-maltese")

PRETRAINED_MODEL = "microsoft/trocr-base-handwritten"

NUM_EPOCHS     = 10
BATCH_SIZE     = 8
LEARNING_RATE  = 5e-5
VAL_SPLIT      = 0.10    # fraction of data held out for validation
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
      labels       — target token IDs; padding positions are -100 so the loss
                     function ignores them (standard HuggingFace convention)
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
        # Open image and ensure it is RGB (some images can be RGBA or grayscale)
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # The processor handles resizing (to 384x384) and pixel normalisation
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenise the ground-truth text into a list of integer token IDs
        label_ids = self.processor.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=MAX_TARGET_LEN,
            truncation=True,
        ).input_ids

        # Replace every pad token with -100 so it is excluded from the loss.
        # PyTorch's cross-entropy loss ignores positions labelled -100.
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

    If `optimizer` is provided this is a training pass:
      gradients are computed and weights are updated after each batch.
    If `optimizer` is None this is a validation pass:
      no gradients, no weight updates — just measure the loss.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0

    # torch.set_grad_enabled controls whether PyTorch tracks operations for
    # backpropagation — we only need this overhead during training
    with torch.set_grad_enabled(is_training):
        for batch in tqdm(loader, desc=desc, leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            # Forward pass — VisionEncoderDecoderModel computes the
            # cross-entropy loss automatically when labels are supplied
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss

            if is_training:
                # Backpropagation: compute gradients and update model weights
                optimizer.zero_grad()   # clear gradients from previous batch
                loss.backward()         # compute new gradients
                optimizer.step()        # apply gradients to model weights

            total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Fix random seeds so results are reproducible across runs
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Choose compute device
    # ------------------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("MPS not available — using CPU (will be slow)")

    # ------------------------------------------------------------------
    # Load transcriptions and build file lists
    # ------------------------------------------------------------------
    print("\nLoading transcriptions…")
    with open(TRANSCRIPTIONS, encoding="utf-8") as f:
        transcriptions: dict[str, str] = json.load(f)

    # Only use syn_*.jpg files — skip any smoke-test images in the folder
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
    # Load pretrained TrOCR model + processor
    # ------------------------------------------------------------------
    print(f"\nDownloading / loading {PRETRAINED_MODEL}…")
    processor = TrOCRProcessor.from_pretrained(PRETRAINED_MODEL)
    model     = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_MODEL)

    # These settings configure the decoder to work with the TrOCR tokeniser
    # for both training (teacher forcing) and inference (generate())
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    # eos_token_id tells model.generate() when the output sequence is complete
    model.config.eos_token_id           = processor.tokenizer.sep_token_id

    model.to(device)
    print(f"Model ready on {device}.")

    # ------------------------------------------------------------------
    # Build PyTorch datasets and data loaders
    # ------------------------------------------------------------------
    full_dataset = MalteseOCRDataset(image_paths, texts, processor)

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    # random_split shuffles and divides the dataset reproducibly
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    print(f"\nSplit: {train_size} train / {val_size} validation")

    # num_workers=0 is required on macOS — multiprocessing in DataLoader
    # conflicts with the macOS process model and causes hangs
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------
    # Optimiser — AdamW is the standard choice for transformer fine-tuning
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

        # Save a checkpoint whenever validation loss improves
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
