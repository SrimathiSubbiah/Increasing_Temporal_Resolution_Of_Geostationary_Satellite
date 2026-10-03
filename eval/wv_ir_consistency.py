"""
Physical consistency check: does the interpolated frame preserve the
relationship BETWEEN Band 13 and Band 9, not just get each band right
independently?

Meteorological background: forecasters use the Water-Vapor minus IR-window
Brightness Temperature Difference (WV-IR BTD = BT9 - BT13) as a real
diagnostic. Band 9 (6.9um) normally senses colder, higher-altitude moisture
than Band 13's (10.3um) cloud-top temperature, so BTD is typically negative
and large in magnitude. When a convective cloud top overshoots the
tropopause, BT13 rises toward what BT9 is already sensing at that altitude,
so BTD shrinks toward zero -- this is an operational signature of severe
convection (e.g. CIMSS/NOAA overshooting-top products).

This project trains the model on both bands but every existing eval script
(evaluate_model.py, compare_average_vs_unet.py) scores each band in
isolation. A model could get both bands individually "close enough" while
still scrambling the cross-band relationship a forecaster would actually
read off the imagery. This script checks that directly: for each method,
does the BTD field it produces track the true BTD field, in both error
magnitude and spatial correlation -- not whether each band looks right on
its own.

No retraining needed -- this runs entirely on outputs/unet_predictions/*.npz
already written by evaluate_model.py.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "data"))
from build_triplets import BT13_MIN, BT13_MAX, BT9_MIN, BT9_MAX  # noqa: E402

PRED_DIR = PROJECT_ROOT / "outputs" / "unet_predictions"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["average", "tvl1_warp", "unet"]
METHOD_LABELS = {"average": "Average Baseline", "tvl1_warp": "TV-L1 Warp", "unet": "U-Net Prediction"}
# Validated 3-slot categorical palette (blue/orange/aqua) -- see dataviz skill.
METHOD_COLORS = {"average": "#2a78d6", "tvl1_warp": "#eb6834", "unet": "#1baf7a"}


def denorm_bt13(x):
    return np.clip(x, 0.0, 1.0) * (BT13_MAX - BT13_MIN) + BT13_MIN


def denorm_bt9(x):
    return np.clip(x, 0.0, 1.0) * (BT9_MAX - BT9_MIN) + BT9_MIN


def btd(bt13, bt9):
    """Water-vapor minus IR-window brightness temp difference, in Kelvin."""
    return bt9 - bt13


def pearson_r(a, b):
    a = a.ravel()
    b = b.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    files = sorted(PRED_DIR.glob("pred_*.npz"))
    print("Files found:", len(files))
    if not files:
        raise FileNotFoundError(f"No prediction files in {PRED_DIR}. Run eval/evaluate_model.py first.")

    mae = {m: [] for m in METHODS}
    corr = {m: [] for m in METHODS}

    for file in files:
        data = np.load(file)

        target = np.clip(data["target"], 0, 1)
        frame0 = np.clip(data["frame0"], 0, 1)
        frame2 = np.clip(data["frame2"], 0, 1)
        pred = np.clip(data["pred"], 0, 1)
        tvl1_warp = np.clip(data["warped"], 0, 1) if "warped" in data.files else (frame0 + frame2) / 2.0

        true_btd = btd(denorm_bt13(target[0]), denorm_bt9(target[1]))

        candidates = {
            "average": btd(denorm_bt13((frame0[0] + frame2[0]) / 2.0), denorm_bt9((frame0[1] + frame2[1]) / 2.0)),
            "tvl1_warp": btd(denorm_bt13(tvl1_warp[0]), denorm_bt9(tvl1_warp[1])),
            "unet": btd(denorm_bt13(pred[0]), denorm_bt9(pred[1])),
        }

        for method, pred_btd in candidates.items():
            mae[method].append(float(np.mean(np.abs(pred_btd - true_btd))))
            corr[method].append(pearson_r(pred_btd, true_btd))

    print()
    print(f"WV-IR Brightness Temperature Difference (BTD) Consistency Check (n={len(files)})")
    print("=" * 70)
    print("BTD = BT(Band9, water vapor) - BT(Band13, IR window), in Kelvin.")
    print("Lower MAE / higher correlation = better preserves the true cross-band")
    print("relationship, not just each band's own pixel accuracy.")

    for method in METHODS:
        m = np.array(mae[method])
        c = np.array([v for v in corr[method] if not np.isnan(v)])
        print()
        print(METHOD_LABELS[method])
        print("-" * len(METHOD_LABELS[method]))
        print(f"BTD MAE (K)        : {m.mean():.3f} +/- {m.std():.3f}")
        print(f"BTD spatial r      : {c.mean():.4f} +/- {c.std():.4f}  (n valid={len(c)}/{len(files)})")

    unet_mae = np.array(mae["unet"])
    avg_mae = np.array(mae["average"])
    tvl1_mae = np.array(mae["tvl1_warp"])
    print()
    print("Head-to-head on BTD MAE (lower is better)")
    print("------------------------------------------")
    print("U-Net better than average:", int((unet_mae < avg_mae).sum()), "/", len(files))
    print("U-Net better than TV-L1  :", int((unet_mae < tvl1_mae).sum()), "/", len(files))

    # --- chart -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    mae_means = [np.mean(mae[m]) for m in METHODS]
    mae_stds = [np.std(mae[m]) for m in METHODS]
    axes[0].bar(
        [METHOD_LABELS[m] for m in METHODS], mae_means, yerr=mae_stds, capsize=4,
        color=[METHOD_COLORS[m] for m in METHODS],
    )
    axes[0].set_ylabel("BTD Mean Absolute Error (K)")
    axes[0].set_title("WV-IR Difference Field Error")
    axes[0].tick_params(axis="x", labelrotation=15)

    corr_means = [np.nanmean(corr[m]) for m in METHODS]
    axes[1].bar(
        [METHOD_LABELS[m] for m in METHODS], corr_means,
        color=[METHOD_COLORS[m] for m in METHODS],
    )
    axes[1].set_ylabel("Mean Spatial Correlation (Pearson r)")
    axes[1].set_title("WV-IR Difference Field Fidelity")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", labelrotation=15)

    fig.suptitle(f"Physical Consistency: WV-IR Brightness Temp Difference (n={len(files)})")
    fig.tight_layout()
    save_path = OUT_DIR / "wv_ir_consistency.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print()
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
