"""
Failure-mode analysis: WHEN does the U-Net's prediction break down, not just
how good is it on average?

Every other eval script reports one aggregate number. This script asks two
concrete questions per test sample, using data already available in
outputs/unet_predictions/*.npz (no retraining, no new data):

1. Does error increase with cloud speed? Fast-moving cells are the hard case
   for any interpolation method -- more displacement between frame0 and
   frame2 means more extrapolation for the model to get right. Cloud speed
   is proxied by the frame0->frame2 centroid displacement of matched cold
   cloud-top blobs (reusing downstream_tracking.py's own detector/matcher).
2. Does error increase with scene complexity/texture? A flat, cloud-free
   patch is trivially easy; a highly textured convective scene is hard.
   Proxied by the standard deviation of the true middle frame (Band 13).

Reports Pearson correlation for each, plus a binned bar chart (error by
speed quartile) -- "the model is fine except when clouds move faster than
X px/frame" is a much more useful, citable finding than a single mean error.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "eval"))
from downstream_tracking import detect_cloud_centroids, match_centroids  # noqa: E402

PRED_DIR = PROJECT_ROOT / "outputs" / "unet_predictions"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BT_THRESHOLD = 0.35
MIN_AREA = 20
MAX_MATCH_DIST = 40.0

# Validated palette (dataviz skill): blue for the primary single-series bars.
BAR_COLOR = "#2a78d6"


def pearson_r(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    files = sorted(PRED_DIR.glob("pred_*.npz"))
    print("Files found:", len(files))
    if not files:
        raise FileNotFoundError(f"No prediction files in {PRED_DIR}. Run eval/evaluate_model.py first.")

    cloud_speed = []      # px displacement frame0->frame2 for matched clouds, per sample
    scene_complexity = [] # std of true middle frame, Band13
    unet_mse = []          # U-Net Band13 pixel error, per sample (always defined)
    unet_tracking_err = [] # U-Net centroid tracking error, per sample (None if no valid track)

    n_no_cloud_match = 0

    for file in files:
        data = np.load(file)
        target = np.clip(data["target"], 0, 1)
        frame0 = np.clip(data["frame0"], 0, 1)
        frame2 = np.clip(data["frame2"], 0, 1)
        pred = np.clip(data["pred"], 0, 1)

        true_b13 = target[0]
        f0_b13 = frame0[0]
        f2_b13 = frame2[0]
        pred_b13 = pred[0]

        # --- cloud speed proxy ---
        c0, _ = detect_cloud_centroids(f0_b13, BT_THRESHOLD, MIN_AREA)
        c2, _ = detect_cloud_centroids(f2_b13, BT_THRESHOLD, MIN_AREA)
        matches = match_centroids(c0, c2, MAX_MATCH_DIST)
        if not matches:
            n_no_cloud_match += 1
            continue
        mean_speed = float(np.mean([d for _, _, d in matches]))

        # --- scene complexity proxy ---
        complexity = float(true_b13.std())

        # --- U-Net error (pixel-level, always defined) ---
        mse = float(np.mean((true_b13 - pred_b13) ** 2))

        # --- U-Net error (tracking-level, may be undefined) ---
        c_pred1, _ = detect_cloud_centroids(pred_b13, BT_THRESHOLD, MIN_AREA)
        c_true1, _ = detect_cloud_centroids(true_b13, BT_THRESHOLD, MIN_AREA)
        track_err = None
        if c_pred1 and c_true1:
            true_arr = np.array(c_true1)
            pred_arr = np.array(c_pred1)
            # nearest true centroid to each predicted centroid, averaged
            d = np.linalg.norm(true_arr[:, None, :] - pred_arr[None, :, :], axis=2)
            track_err = float(d.min(axis=0).mean())

        cloud_speed.append(mean_speed)
        scene_complexity.append(complexity)
        unet_mse.append(mse)
        unet_tracking_err.append(track_err)

    n = len(cloud_speed)
    print(f"Samples with a trackable cloud (used for this analysis): {n}/{len(files)}")
    print(f"Samples skipped (no persistent cloud to measure speed from): {n_no_cloud_match}")

    cloud_speed = np.array(cloud_speed)
    scene_complexity = np.array(scene_complexity)
    unet_mse = np.array(unet_mse)

    print()
    print("Correlation with U-Net pixel error (Band 13 MSE)")
    print("---------------------------------------------------")
    print(f"vs. cloud speed (px/2-frame displacement) : r = {pearson_r(cloud_speed, unet_mse):.3f}")
    print(f"vs. scene complexity (true-frame std)      : r = {pearson_r(scene_complexity, unet_mse):.3f}")

    # tracking error correlation, dropping samples where it's undefined
    valid = [i for i, e in enumerate(unet_tracking_err) if e is not None]
    if valid:
        track_err_arr = np.array([unet_tracking_err[i] for i in valid])
        speed_arr = cloud_speed[valid]
        print()
        print(f"Correlation with U-Net downstream tracking error (n={len(valid)})")
        print("---------------------------------------------------------------")
        print(f"vs. cloud speed : r = {pearson_r(speed_arr, track_err_arr):.3f}")

    # --- binned analysis: error by cloud-speed quartile ---------------------
    quartiles = np.percentile(cloud_speed, [25, 50, 75])
    bins = np.digitize(cloud_speed, quartiles)  # 0,1,2,3
    bin_labels = [
        f"Q1 (slowest)\n<{quartiles[0]:.1f}px",
        f"Q2\n{quartiles[0]:.1f}-{quartiles[1]:.1f}px",
        f"Q3\n{quartiles[1]:.1f}-{quartiles[2]:.1f}px",
        f"Q4 (fastest)\n>{quartiles[2]:.1f}px",
    ]
    bin_means = [unet_mse[bins == b].mean() if (bins == b).any() else 0.0 for b in range(4)]

    print()
    print("U-Net Band13 MSE by cloud-speed quartile")
    print("-------------------------------------------")
    for label, m in zip(bin_labels, bin_means):
        print(f"{label.splitlines()[0]:15s}: MSE = {m:.6f}")

    # --- chart ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(bin_labels, bin_means, color=BAR_COLOR)
    axes[0].set_ylabel("U-Net Band13 MSE")
    axes[0].set_title("Error by Cloud-Speed Quartile")

    axes[1].scatter(scene_complexity, unet_mse, s=10, alpha=0.35, color=BAR_COLOR)
    axes[1].set_xlabel("Scene complexity (true-frame std)")
    axes[1].set_ylabel("U-Net Band13 MSE")
    r = pearson_r(scene_complexity, unet_mse)
    axes[1].set_title(f"Error vs. Scene Complexity (r={r:.2f})")

    fig.suptitle(f"Failure-Mode Analysis (n={n})")
    fig.tight_layout()
    save_path = OUT_DIR / "failure_mode_analysis.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print()
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
