"""
Head-to-head: plain U-Net (unet_refine.py, checkpoint.pt) vs ResUNet
(resunet_refine.py, checkpoint_resunet.pt) -- same architecture shape,
same data, same training schedule, the ONLY difference is whether each
conv block has an internal residual connection.

Because both evaluate_model.py and evaluate_resunet.py sample the test set
with the identical seed (42) from the identical file list, pred_XXXXX.npz
in outputs/unet_predictions/ and outputs/resunet_predictions/ refer to the
SAME underlying test triplet at the same index -- so this compares them
pair-by-pair, not just aggregate-vs-aggregate.
"""

from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

UNET_DIR = Path("outputs/unet_predictions")
RESUNET_DIR = Path("outputs/resunet_predictions")

BAND_NAMES = ["band13", "band9"]
# Validated 3-slot palette (dataviz skill), reusing the U-Net/TV-L1 slots
# for continuity with earlier charts -- ResUNet gets the 3rd slot (aqua).
MODEL_COLORS = {"unet": "#2a78d6", "resunet": "#1baf7a"}


def main():
    unet_files = sorted(UNET_DIR.glob("pred_*.npz"))
    resunet_files = sorted(RESUNET_DIR.glob("pred_*.npz"))

    if not resunet_files:
        raise FileNotFoundError(
            f"No files in {RESUNET_DIR}. Run train_resunet.py then eval/evaluate_resunet.py first."
        )
    if len(unet_files) != len(resunet_files):
        print(f"WARNING: sample count mismatch -- U-Net has {len(unet_files)}, "
              f"ResUNet has {len(resunet_files)}. Comparing only the overlapping range.")

    n = min(len(unet_files), len(resunet_files))
    print(f"Comparing {n} paired test samples (same underlying triplets, same seed)")

    metrics = {
        band: {model: {"mse": [], "psnr": [], "ssim": []} for model in ["unet", "resunet"]}
        for band in BAND_NAMES
    }

    for i in range(n):
        u = np.load(unet_files[i])
        r = np.load(resunet_files[i])

        u_target = np.clip(u["target"], 0, 1)
        r_target = np.clip(r["target"], 0, 1)
        # sanity check the pairing assumption -- targets should be identical
        # since it's the same underlying triplet
        if not np.allclose(u_target, r_target, atol=1e-5):
            raise RuntimeError(
                f"pred_{i:05d}.npz targets differ between U-Net and ResUNet dirs -- "
                "the sampling seeds don't actually line up, don't trust a pairwise "
                "comparison here. Check MAX_TEST_SAMPLES/SAMPLE_SEED match in both scripts."
            )

        u_pred = np.clip(u["pred"], 0, 1)
        r_pred = np.clip(r["pred"], 0, 1)

        for c, band in enumerate(BAND_NAMES):
            target_c = u_target[c]
            for model, pred in [("unet", u_pred[c]), ("resunet", r_pred[c])]:
                m = metrics[band][model]
                m["mse"].append(np.mean((target_c - pred) ** 2))
                m["psnr"].append(psnr(target_c, pred, data_range=1.0))
                m["ssim"].append(ssim(target_c, pred, data_range=1.0))

    for band in BAND_NAMES:
        print()
        print(f"=== {band} ===")
        for model in ["unet", "resunet"]:
            m = metrics[band][model]
            label = "U-Net (plain)" if model == "unet" else "ResUNet"
            print()
            print(label)
            print("-" * len(label))
            print("MSE :", np.mean(m["mse"]), "+/-", np.std(m["mse"]))
            print("PSNR:", np.mean(m["psnr"]), "+/-", np.std(m["psnr"]))
            print("SSIM:", np.mean(m["ssim"]), "+/-", np.std(m["ssim"]))

        unet_mse = np.array(metrics[band]["unet"]["mse"])
        resunet_mse = np.array(metrics[band]["resunet"]["mse"])
        resunet_wins = int((resunet_mse < unet_mse).sum())
        print()
        print(f"[{band}] ResUNet better than plain U-Net on MSE: {resunet_wins} / {n} "
              f"({100 * resunet_wins / n:.1f}%)")

        mean_improvement = (np.mean(unet_mse) - np.mean(resunet_mse)) / np.mean(unet_mse) * 100
        print(f"[{band}] Mean MSE change (ResUNet vs U-Net): {mean_improvement:+.2f}%")


if __name__ == "__main__":
    main()
