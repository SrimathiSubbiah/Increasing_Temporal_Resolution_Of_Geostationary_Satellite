"""
Evaluate the ResUNet checkpoint (checkpoint_resunet.pt) exactly the same
way evaluate_model.py scores the plain U-Net -- same test sampling (full
3207, seed=42), same metrics, same input construction. Writes to a SEPARATE
output directory (outputs/resunet_predictions) so it never touches or
overwrites outputs/unet_predictions, which every other eval script and
result in this project depends on.
"""

import glob
import random
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append("model")
from resunet_refine import ResUNetRefine


TEST_DIR = Path("data/triplets/test")
CHECKPOINT_PATH = Path("checkpoint_resunet.pt")
OUT_DIR = Path("outputs/resunet_predictions")

MAX_TEST_SAMPLES = 3207  # full test split, matching evaluate_model.py's final run
SAMPLE_SEED = 42

BAND_NAMES = ["band13", "band9"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_chw(arr):
    arr = arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if arr.shape[0] in [1, 2, 3, 6]:
        return arr
    if arr.shape[-1] in [1, 2, 3, 6]:
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unknown shape: {arr.shape}")


def normalize_for_metrics(x):
    return np.clip(x, 0.0, 1.0)


def print_band_results(name, mse_list, psnr_list, ssim_list):
    print()
    print(f"{name}")
    print("-" * len(name))
    print("MSE :", float(np.mean(mse_list)), "+/-", float(np.std(mse_list)))
    print("PSNR:", float(np.mean(psnr_list)), "+/-", float(np.std(psnr_list)))
    print("SSIM:", float(np.mean(ssim_list)), "+/-", float(np.std(ssim_list)))


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found. Run train_resunet.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(TEST_DIR / "*.npz")))
    random.seed(SAMPLE_SEED)
    files = sorted(random.sample(files, min(MAX_TEST_SAMPLES, len(files))))

    print("Using device:", DEVICE)
    print("Test samples:", len(files), "of", len(glob.glob(str(TEST_DIR / '*.npz'))), "available")

    n_with_warp = sum("warped" in np.load(f).files for f in files)
    print(f"Sampled files with cached TV-L1 warp: {n_with_warp}/{len(files)}")

    model = ResUNetRefine(
        in_channels=6,
        out_channels=2,
        predict_uncertainty=False,
        residual=True,
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    metrics = {b: {"mse": [], "psnr": [], "ssim": []} for b in BAND_NAMES}

    with torch.no_grad():
        for i, file in enumerate(files):
            data = np.load(file)

            f0 = ensure_chw(data["frame0"])
            f1 = ensure_chw(data["frame1"])
            f2 = ensure_chw(data["frame2"])

            if "warped" in data.files:
                warped = ensure_chw(data["warped"])
            else:
                warped = (f0 + f2) / 2.0

            model_input = np.concatenate([warped, f0, f2], axis=0)
            x = torch.from_numpy(model_input).unsqueeze(0).float().to(DEVICE)

            pred = model(x)[0].cpu().numpy()
            pred = normalize_for_metrics(pred)
            target = normalize_for_metrics(f1)

            for c, band in enumerate(BAND_NAMES):
                pred_c = pred[c]
                target_c = target[c]
                metrics[band]["mse"].append(np.mean((target_c - pred_c) ** 2))
                metrics[band]["psnr"].append(psnr(target_c, pred_c, data_range=1.0))
                metrics[band]["ssim"].append(ssim(target_c, pred_c, data_range=1.0))

            np.savez_compressed(
                OUT_DIR / f"pred_{i:05d}.npz",
                pred=pred,
                target=target,
                frame0=f0,
                frame2=f2,
                warped=normalize_for_metrics(warped),
            )

    print()
    print(f"ResUNet Evaluation Results (n={len(files)}, same sample as U-Net's full-test-set eval)")
    print("=" * 70)
    for band in BAND_NAMES:
        m = metrics[band]
        print_band_results(band, m["mse"], m["psnr"], m["ssim"])

    print()
    print("Predictions saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
