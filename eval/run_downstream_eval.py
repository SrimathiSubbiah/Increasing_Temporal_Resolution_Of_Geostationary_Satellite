import json
from pathlib import Path
import sys

import numpy as np

# Make imports work from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "eval"))

from downstream_tracking import evaluate_tracking_error


PRED_DIR = PROJECT_ROOT / "outputs" / "unet_predictions"
RESUNET_DIR = PROJECT_ROOT / "outputs" / "resunet_predictions"
ATTENTION_DIR = PROJECT_ROOT / "outputs" / "attention_unet_predictions"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "downstream_results.json"
MAX_SAMPLES = 3207  # match evaluate_model.py's full-test-split run

# Threshold settings for detecting cold cloud-top regions.
# If valid samples are too low, try BT_THRESHOLD = 0.40 or 0.45.
BT_THRESHOLD = 0.35
MIN_AREA = 20
MAX_MATCH_DIST = 40.0

# ResUNet / Attention U-Net are optional -- included only if their prediction
# dirs exist (train_*.py + eval/evaluate_*.py already run). All are paired
# 1:1 by index with outputs/unet_predictions -- same seed, same file list
# (see eval/compare_unet_vs_resunet.py / confusion_matrix_cloud_detection.py
# for the same pairing pattern).
INCLUDE_RESUNET = RESUNET_DIR.exists() and any(RESUNET_DIR.glob("pred_*.npz"))
INCLUDE_ATTENTION = ATTENTION_DIR.exists() and any(ATTENTION_DIR.glob("pred_*.npz"))

METHODS = (
    ["average", "tvl1_warp", "unet"]
    + (["resunet"] if INCLUDE_RESUNET else [])
    + (["attention_unet"] if INCLUDE_ATTENTION else [])
)
METHOD_LABELS = {
    "average": "Average Baseline",
    "tvl1_warp": "TV-L1 Warp",
    "unet": "U-Net Prediction",
    "resunet": "ResUNet Prediction",
    "attention_unet": "Attention U-Net Prediction",
}


def summarize(errors, cloud_counts):
    """
    Robust tracking statistics for one method.
    Mean can be affected by rare large failures, so median and 90th percentile
    are also reported.
    """
    errors = np.array(errors)
    return {
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "std_error_px": float(np.std(errors)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "mean_tracked_clouds": float(np.mean(cloud_counts)),
        "n_samples": len(errors),
    }


def print_summary(name, stats):
    print()
    print(name)
    print("-" * len(name))
    print("Mean centroid error px  :", stats["mean_error_px"])
    print("Median centroid error px:", stats["median_error_px"])
    print("Std centroid error px   :", stats["std_error_px"])
    print("90th percentile error px:", stats["p90_error_px"])
    print("Mean tracked clouds     :", stats["mean_tracked_clouds"])


def main():
    files = sorted(PRED_DIR.glob("pred_*.npz"))[:MAX_SAMPLES]

    print("Prediction files found:", len(files))
    print("BT threshold:", BT_THRESHOLD)
    print("Min area:", MIN_AREA)
    print("Max match distance:", MAX_MATCH_DIST)

    resunet_files = sorted(RESUNET_DIR.glob("pred_*.npz"))[:MAX_SAMPLES] if INCLUDE_RESUNET else []
    if INCLUDE_RESUNET:
        print("ResUNet files found:", len(resunet_files), "-- including as a method")
    attention_files = sorted(ATTENTION_DIR.glob("pred_*.npz"))[:MAX_SAMPLES] if INCLUDE_ATTENTION else []
    if INCLUDE_ATTENTION:
        print("Attention U-Net files found:", len(attention_files), "-- including as a method")
    print()

    if len(files) == 0:
        raise FileNotFoundError(f"No prediction files found in {PRED_DIR}")

    errors = {m: [] for m in METHODS}
    cloud_counts = {m: [] for m in METHODS}
    win_counts = {m: 0 for m in METHODS}
    tie_count = 0

    valid_samples = 0
    skipped_samples = 0
    n_no_warp = 0

    for i, file in enumerate(files):
        data = np.load(file)

        pred = np.clip(data["pred"], 0, 1)
        target = np.clip(data["target"], 0, 1)
        frame0 = np.clip(data["frame0"], 0, 1)
        frame2 = np.clip(data["frame2"], 0, 1)

        if "warped" in data.files:
            tvl1_warp = np.clip(data["warped"], 0, 1)
        else:
            tvl1_warp = (frame0 + frame2) / 2.0
            n_no_warp += 1

        # Band 13 only. Assumption: channel 0 = Band 13, channel 1 = Band 9.
        f0_b13 = frame0[0]
        f2_b13 = frame2[0]
        true_b13 = target[0]

        candidates = {
            "average": (f0_b13 + f2_b13) / 2.0,
            "tvl1_warp": tvl1_warp[0],
            "unet": pred[0],
        }

        if INCLUDE_RESUNET and i < len(resunet_files):
            r = np.load(resunet_files[i])
            r_target = np.clip(r["target"], 0, 1)
            if not np.allclose(target, r_target, atol=1e-5):
                raise RuntimeError(
                    f"{file.name} targets differ between U-Net and ResUNet dirs -- "
                    "sampling seeds don't line up, pairing assumption is broken."
                )
            candidates["resunet"] = np.clip(r["pred"], 0, 1)[0]

        if INCLUDE_ATTENTION and i < len(attention_files):
            a = np.load(attention_files[i])
            a_target = np.clip(a["target"], 0, 1)
            if not np.allclose(target, a_target, atol=1e-5):
                raise RuntimeError(
                    f"{file.name} targets differ between U-Net and Attention U-Net dirs -- "
                    "sampling seeds don't line up, pairing assumption is broken."
                )
            candidates["attention_unet"] = np.clip(a["pred"], 0, 1)[0]

        sample_errors = {}
        sample_clouds = {}
        for method, pred_b13 in candidates.items():
            result = evaluate_tracking_error(
                frame0_b13=f0_b13,
                true_frame1_b13=true_b13,
                pred_frame1_b13=pred_b13,
                frame2_b13=f2_b13,
                bt_threshold=BT_THRESHOLD,
                min_area=MIN_AREA,
                max_match_dist=MAX_MATCH_DIST,
            )
            sample_errors[method] = result["mean_error_px"]
            sample_clouds[method] = result["n_tracked_clouds"]

        # Skip samples where any method has no valid cloud track -- keeps
        # the three methods compared on an identical set of samples.
        if any(v is None for v in sample_errors.values()):
            skipped_samples += 1
            continue

        valid_samples += 1
        for method in METHODS:
            errors[method].append(sample_errors[method])
            cloud_counts[method].append(sample_clouds[method])

        best_method = min(sample_errors, key=sample_errors.get)
        best_value = sample_errors[best_method]
        tied = [m for m, v in sample_errors.items() if v == best_value]
        if len(tied) > 1:
            tie_count += 1
        else:
            win_counts[best_method] += 1

    print("Downstream Cloud-Tracking Evaluation")
    print("------------------------------------")
    print("Total files checked:", len(files))
    print("Valid samples:", valid_samples)
    print("Skipped samples:", skipped_samples)
    if n_no_warp:
        print(f"WARNING: {n_no_warp} files had no cached TV-L1 warp (average fallback used).")

    if valid_samples == 0:
        print()
        print("No valid cloud tracks found.")
        print("Try increasing BT_THRESHOLD to 0.40 or 0.45, or reducing MIN_AREA to 10.")
        return

    stats = {method: summarize(errors[method], cloud_counts[method]) for method in METHODS}
    for method in METHODS:
        print_summary(METHOD_LABELS[method], stats[method])

    print()
    print("Head-to-head (best method per sample)")
    print("--------------------------------------")
    for method in METHODS:
        print(f"{METHOD_LABELS[method]:<20}: {win_counts[method]} / {valid_samples}")
    print(f"{'Ties':<20}: {tie_count} / {valid_samples}")

    avg_mean = stats["average"]["mean_error_px"]
    avg_median = stats["average"]["median_error_px"]
    print()
    print("Relative improvement vs. Average baseline")
    print("------------------------------------------")
    for method in [m for m in METHODS if m != "average"]:
        mean_improvement = (avg_mean - stats[method]["mean_error_px"]) / avg_mean * 100
        median_improvement = (avg_median - stats[method]["median_error_px"]) / avg_median * 100
        print(f"{METHOD_LABELS[method]}: mean {mean_improvement:+.1f}%, median {median_improvement:+.1f}%")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "methods": METHOD_LABELS,
                "stats": stats,
                "win_counts": win_counts,
                "tie_count": tie_count,
                "valid_samples": valid_samples,
                "skipped_samples": skipped_samples,
                "bt_threshold": BT_THRESHOLD,
                "min_area": MIN_AREA,
                "max_match_dist": MAX_MATCH_DIST,
            },
            f,
            indent=2,
        )
    print()
    print("Results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()