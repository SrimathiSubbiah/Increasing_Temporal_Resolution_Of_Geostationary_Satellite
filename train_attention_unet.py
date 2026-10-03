"""
Train the Attention U-Net refinement model on precomputed TV-L1-warped
triplets. Identical to train.py / train_resunet.py in every respect (same
data, same loss, same schedule, same hyperparameters) except the model
class -- isolates the one variable: do attention gates on the skip
connections (smarter USE of encoder features) beat plain concatenation,
where ResUNet's approach (MORE COMPUTE via internal residual blocks)
already failed to help. Writes to a SEPARATE checkpoint file -- never
touches checkpoint.pt or checkpoint_resunet.pt.
"""

import glob
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append("model")
from attention_unet_refine import AttentionUNetRefine


# --- config -------------------------------------------------------------
TRAIN_DIR = "data/triplets/train"
VAL_DIR = "data/triplets/val"

# Same sample counts as train.py / train_resunet.py -- keeps all three
# models on an apples-to-apples comparison.
MAX_TRAIN_SAMPLES = 8000
MAX_VAL_SAMPLES = 1500

BATCH_SIZE = 16
EPOCHS = 14
LR = 1e-4

# Separate checkpoint -- checkpoint.pt (plain U-Net) and checkpoint_resunet.pt
# are used throughout eval/*.py; never overwrite either here.
CHECKPOINT_PATH = "checkpoint_attention_unet.pt"

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available -- refusing to silently train on CPU. "
        "Check the torch/CUDA install (torch.cuda.is_available() should be True)."
    )
DEVICE = torch.device("cuda")
# -------------------------------------------------------------------------


def ensure_chw(arr: np.ndarray) -> np.ndarray:
    """
    Ensures array shape is [C, H, W].

    Expected valid inputs:
    - [C, H, W]
    - [H, W, C]
    """

    arr = arr.astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    # Already [C, H, W]
    if arr.shape[0] in [1, 2, 3, 6]:
        return arr

    # Convert [H, W, C] -> [C, H, W]
    if arr.shape[-1] in [1, 2, 3, 6]:
        return np.transpose(arr, (2, 0, 1))

    raise ValueError(f"Could not understand array shape: {arr.shape}")


class TripletDataset(Dataset):
    """
    Loads triplets from:
        data/triplets/train/*.npz
        data/triplets/val/*.npz

    Each .npz file must contain:
        frame0
        frame1
        frame2

    The TV-L1 warped frame is cached inside the same .npz under the key
    "warped" (see data/precompute_warps.py, which overwrites each triplet
    file in place). If a file wasn't precomputed yet, fall back to a simple
    average -- this should only happen for stragglers past the precompute
    cap, never silently for the whole dataset.
    """

    def __init__(self, triplet_dir: str, split_name: str, max_samples: int = None):
        self.triplet_dir = Path(triplet_dir)
        self.split_name = split_name

        self.files = sorted(glob.glob(str(self.triplet_dir / "*.npz")))

        if max_samples:
            random.seed(42)
            self.files = random.sample(self.files, min(max_samples, len(self.files)))

        print(f"{split_name} triplets found:", len(self.files))

        n_with_warp = sum("warped" in np.load(f).files for f in self.files)
        print(f"{split_name} triplets with cached warp: {n_with_warp}/{len(self.files)}")
        if n_with_warp < len(self.files):
            print(
                f"WARNING: {len(self.files) - n_with_warp} {split_name} triplets have no "
                "cached warp and will fall back to a plain average. Run "
                "data/precompute_warps.py to cover the full split."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        triplet_path = Path(self.files[idx])

        data = np.load(triplet_path)

        f0 = ensure_chw(data["frame0"])
        f1_target = ensure_chw(data["frame1"])
        f2 = ensure_chw(data["frame2"])

        if "warped" in data.files:
            warped = ensure_chw(data["warped"])
        else:
            # fallback for triplets outside the precompute cap
            warped = (f0 + f2) / 2.0

        # Safety check
        if warped.shape != f0.shape:
            raise ValueError(
                f"Shape mismatch in {triplet_path.name}: "
                f"warped={warped.shape}, frame0={f0.shape}"
            )

        # Input = TV-L1 warped frame + frame0 + frame2
        # Each is 2 channels, so total = 6 channels
        model_input = np.concatenate([warped, f0, f2], axis=0)

        return {
            "input": torch.from_numpy(model_input).float(),
            "target": torch.from_numpy(f1_target).float(),
            "name": triplet_path.name,
        }


def train():
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print("Model: AttentionUNetRefine (attention gates on every skip connection)")

    train_ds = TripletDataset(TRAIN_DIR, split_name="train", max_samples=MAX_TRAIN_SAMPLES)
    val_ds = TripletDataset(VAL_DIR, split_name="val", max_samples=MAX_VAL_SAMPLES)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = AttentionUNetRefine(
        in_channels=6,
        out_channels=2,
        predict_uncertainty=False,
        residual=True,
    ).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.L1Loss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()
        train_losses = []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False)
        for batch in train_bar:
            x = batch["input"].to(DEVICE, non_blocking=True)
            y = batch["target"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        model.eval()
        val_losses = []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [val]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                x = batch["input"].to(DEVICE, non_blocking=True)
                y = batch["target"].to(DEVICE, non_blocking=True)

                pred = model(x)
                loss = criterion(pred, y)

                val_losses.append(loss.item())
                val_bar.set_postfix(loss=f"{np.mean(val_losses):.4f}")

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        elapsed = time.time() - start_time

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={lr_after:.2e} | "
            f"time={elapsed:.1f}s"
        )
        if lr_after < lr_before:
            print(f"  -> val_loss plateaued, LR reduced to {lr_after:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved checkpoint, val_loss improved to {val_loss:.4f}")

    print()
    print("Training done.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
