"""
Fair three-way comparison on the exact same sampled test triplets that
evaluate_model.py wrote to outputs/unet_predictions:
  1. Naive average of frame0/frame2 (the "do nothing clever" baseline)
  2. TV-L1 optical-flow warp (cached in the pred files under "warped")
  3. U-Net refinement of the TV-L1 warp (the actual model output, "pred")

Both Band 13 (IR) and Band 9 (water vapor) are reported since the model
predicts both channels.
"""

from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

PRED_DIR = Path("outputs/unet_predictions")
BAND_NAMES = ["band13", "band9"]
METHODS = ["average", "tvl1_warp", "unet"]

files = sorted(PRED_DIR.glob("pred_*.npz"))
print("Files found:", len(files))
if not files:
    raise FileNotFoundError(
        f"No prediction files in {PRED_DIR}. Run eval/evaluate_model.py first."
    )

# metrics[band][method][metric] -> list of per-sample values
metrics = {
    band: {method: {"mse": [], "psnr": [], "ssim": []} for method in METHODS}
    for band in BAND_NAMES
}

n_no_warp = 0

for file in files:
    data = np.load(file)

    pred = np.clip(data["pred"], 0, 1)
    target = np.clip(data["target"], 0, 1)
    frame0 = np.clip(data["frame0"], 0, 1)
    frame2 = np.clip(data["frame2"], 0, 1)

    if "warped" in data.files:
        tvl1_warp = np.clip(data["warped"], 0, 1)
    else:
        # this file predates caching the warp -- fall back so the script
        # still runs, but flag it since it skews the tvl1_warp column
        tvl1_warp = (frame0 + frame2) / 2.0
        n_no_warp += 1

    for c, band in enumerate(BAND_NAMES):
        target_c = target[c]
        values = {
            "average": (frame0[c] + frame2[c]) / 2.0,
            "tvl1_warp": tvl1_warp[c],
            "unet": pred[c],
        }
        for method, arr in values.items():
            m = metrics[band][method]
            m["mse"].append(np.mean((target_c - arr) ** 2))
            m["psnr"].append(psnr(target_c, arr, data_range=1.0))
            m["ssim"].append(ssim(target_c, arr, data_range=1.0))

if n_no_warp:
    print(f"WARNING: {n_no_warp}/{len(files)} files had no cached TV-L1 warp "
          "(average fallback used for the tvl1_warp column).")

for band in BAND_NAMES:
    print()
    print(f"=== {band} ===")
    for method in METHODS:
        m = metrics[band][method]
        print()
        print(method)
        print("-" * len(method))
        print("MSE :", np.mean(m["mse"]), "+/-", np.std(m["mse"]))
        print("PSNR:", np.mean(m["psnr"]), "+/-", np.std(m["psnr"]))
        print("SSIM:", np.mean(m["ssim"]), "+/-", np.std(m["ssim"]))

    avg_mse = metrics[band]["average"]["mse"]
    tvl1_mse = metrics[band]["tvl1_warp"]["mse"]
    unet_mse = metrics[band]["unet"]["mse"]

    unet_vs_avg = sum(u < a for u, a in zip(unet_mse, avg_mse))
    unet_vs_tvl1 = sum(u < t for u, t in zip(unet_mse, tvl1_mse))
    tvl1_vs_avg = sum(t < a for t, a in zip(tvl1_mse, avg_mse))

    print()
    print(f"[{band}] U-Net better than average on MSE :", unet_vs_avg, "/", len(files))
    print(f"[{band}] U-Net better than TV-L1 on MSE    :", unet_vs_tvl1, "/", len(files))
    print(f"[{band}] TV-L1 better than average on MSE  :", tvl1_vs_avg, "/", len(files))