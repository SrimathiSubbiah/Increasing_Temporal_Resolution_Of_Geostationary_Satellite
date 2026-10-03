"""
Confusion matrix for cloud-top pixel detection.

This project is fundamentally regression (predicting continuous brightness
temperature), so there's no natural classification target -- but there IS
one real binary decision buried in the pipeline already: downstream_tracking.py
thresholds every pixel into "cold cloud-top" vs "not," at BT_THRESHOLD=0.35
(~229K), before it ever looks for blobs to track. This script makes that
binary decision explicit and scores it directly: for every pixel, does each
method's predicted middle frame correctly say "cloud" or "clear" versus what
the REAL middle frame says?

This is a different, complementary question to the downstream tracking
metric. Tracking asks "is the storm's CENTROID in the right place." This
asks "is each individual PIXEL correctly classified as cloud or not" --
catching an entirely different kind of error (e.g. a cloud with roughly the
right shape but a systematically wrong size/coverage would still track
well but score badly here).

No retraining, no new data -- runs entirely on outputs/unet_predictions/*.npz.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "outputs" / "unet_predictions"
RESUNET_DIR = PROJECT_ROOT / "outputs" / "resunet_predictions"
ATTENTION_DIR = PROJECT_ROOT / "outputs" / "attention_unet_predictions"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same threshold used throughout downstream_tracking.py / run_downstream_eval.py
# -- keeps "cloud" defined identically everywhere in the project.
BT_THRESHOLD = 0.35

# ResUNet / Attention U-Net are optional -- only included if their output
# dirs exist (i.e. the corresponding train_*.py + eval/evaluate_*.py have
# already been run). Both are paired 1:1 by index with outputs/unet_predictions
# -- all three were sampled with the identical seed from the identical test
# file list (see eval/compare_unet_vs_resunet.py for the same pairing pattern).
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


def confusion_counts(true_mask, pred_mask):
    """Pixel-level TP/FP/FN/TN, both boolean arrays of the same shape."""
    tp = int(np.sum(pred_mask & true_mask))
    fp = int(np.sum(pred_mask & ~true_mask))
    fn = int(np.sum(~pred_mask & true_mask))
    tn = int(np.sum(~pred_mask & ~true_mask))
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return precision, recall, f1, accuracy


def main():
    files = sorted(PRED_DIR.glob("pred_*.npz"))
    print("Files found:", len(files))
    if not files:
        raise FileNotFoundError(f"No prediction files in {PRED_DIR}. Run eval/evaluate_model.py first.")

    resunet_files = sorted(RESUNET_DIR.glob("pred_*.npz")) if INCLUDE_RESUNET else []
    if INCLUDE_RESUNET:
        print("ResUNet files found:", len(resunet_files), "-- including as a method")
        if len(resunet_files) != len(files):
            print(f"WARNING: count mismatch (U-Net {len(files)} vs ResUNet {len(resunet_files)}) "
                  "-- pairing by index may be unreliable beyond the shorter list.")
    else:
        print("outputs/resunet_predictions not found -- run train_resunet.py + "
              "eval/evaluate_resunet.py to include ResUNet in this comparison.")

    attention_files = sorted(ATTENTION_DIR.glob("pred_*.npz")) if INCLUDE_ATTENTION else []
    if INCLUDE_ATTENTION:
        print("Attention U-Net files found:", len(attention_files), "-- including as a method")
        if len(attention_files) != len(files):
            print(f"WARNING: count mismatch (U-Net {len(files)} vs Attention U-Net "
                  f"{len(attention_files)}) -- pairing by index may be unreliable beyond the shorter list.")
    else:
        print("outputs/attention_unet_predictions not found -- run train_attention_unet.py + "
              "eval/evaluate_attention_unet.py to include it in this comparison.")

    # Accumulate pixel counts across the ENTIRE test set (not averaged
    # per-sample) -- this is the standard way to build one confusion matrix
    # over a whole dataset rather than one per image.
    totals = {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for m in METHODS}
    n_pairing_checked = 0

    for i, file in enumerate(files):
        data = np.load(file)
        target = np.clip(data["target"], 0, 1)
        frame0 = np.clip(data["frame0"], 0, 1)
        frame2 = np.clip(data["frame2"], 0, 1)
        pred = np.clip(data["pred"], 0, 1)
        tvl1_warp = np.clip(data["warped"], 0, 1) if "warped" in data.files else (frame0 + frame2) / 2.0

        true_mask = target[0] < BT_THRESHOLD

        candidates = {
            "average": ((frame0[0] + frame2[0]) / 2.0) < BT_THRESHOLD,
            "tvl1_warp": tvl1_warp[0] < BT_THRESHOLD,
            "unet": pred[0] < BT_THRESHOLD,
        }

        if INCLUDE_RESUNET and i < len(resunet_files):
            r = np.load(resunet_files[i])
            r_target = np.clip(r["target"], 0, 1)
            # Same pairing-safety check as compare_unet_vs_resunet.py -- don't
            # trust the index correspondence blindly, verify it on every file.
            if not np.allclose(target, r_target, atol=1e-5):
                raise RuntimeError(
                    f"pred_{i:05d}.npz targets differ between U-Net and ResUNet dirs -- "
                    "sampling seeds don't line up, pairing assumption is broken."
                )
            n_pairing_checked += 1
            r_pred = np.clip(r["pred"], 0, 1)
            candidates["resunet"] = r_pred[0] < BT_THRESHOLD

        if INCLUDE_ATTENTION and i < len(attention_files):
            a = np.load(attention_files[i])
            a_target = np.clip(a["target"], 0, 1)
            if not np.allclose(target, a_target, atol=1e-5):
                raise RuntimeError(
                    f"pred_{i:05d}.npz targets differ between U-Net and Attention U-Net dirs -- "
                    "sampling seeds don't line up, pairing assumption is broken."
                )
            a_pred = np.clip(a["pred"], 0, 1)
            candidates["attention_unet"] = a_pred[0] < BT_THRESHOLD

        for method, pred_mask in candidates.items():
            tp, fp, fn, tn = confusion_counts(true_mask, pred_mask)
            totals[method]["tp"] += tp
            totals[method]["fp"] += fp
            totals[method]["fn"] += fn
            totals[method]["tn"] += tn

    print()
    print(f"Cloud-Top Pixel Detection Confusion Matrix (n={len(files)} test frames, "
          f"threshold={BT_THRESHOLD} normalized ~229K)")
    print("=" * 78)

    results = {}
    for method in METHODS:
        c = totals[method]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        precision, recall, f1, accuracy = metrics_from_counts(tp, fp, fn, tn)
        results[method] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                                precision=precision, recall=recall, f1=f1, accuracy=accuracy)

        total_px = tp + fp + fn + tn
        print()
        print(METHOD_LABELS[method])
        print("-" * len(METHOD_LABELS[method]))
        print(f"{'':>18s}{'Pred: Cloud':>14s}{'Pred: Clear':>14s}")
        print(f"{'True: Cloud':>18s}{tp:>14,d}{fn:>14,d}")
        print(f"{'True: Clear':>18s}{fp:>14,d}{tn:>14,d}")
        print(f"  (total pixels scored: {total_px:,})")
        print(f"  Precision: {precision:.4f}   Recall: {recall:.4f}   "
              f"F1: {f1:.4f}   Accuracy: {accuracy:.4f}")

    # --- chart: N heatmaps side by side, row-normalized for readability ------
    n_methods = len(METHODS)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    # Sequential single-hue ramp (dataviz skill: sequential = one hue,
    # light->dark, for magnitude data) -- default blue.
    cmap = plt.get_cmap("Blues")

    for ax, method in zip(axes, METHODS):
        c = results[method]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        # row-normalize: top row (true=cloud) sums to 1, bottom row (true=clear) sums to 1
        cloud_row_total = tp + fn
        clear_row_total = fp + tn
        norm_matrix = np.array([
            [tp / cloud_row_total, fn / cloud_row_total],
            [fp / clear_row_total, tn / clear_row_total],
        ])
        count_matrix = np.array([[tp, fn], [fp, tn]])

        im = ax.imshow(norm_matrix, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred: Cloud", "Pred: Clear"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True: Cloud", "True: Clear"])
        ax.set_title(f"{METHOD_LABELS[method]}\nF1={c['f1']:.3f}  Acc={c['accuracy']:.3f}")

        for i in range(2):
            for j in range(2):
                # text color follows the cell's own lightness, not the series
                # (dataviz non-negotiable: labels stay in ink tokens, not
                # arbitrary color -- here the ink IS the readability choice)
                text_color = "white" if norm_matrix[i, j] > 0.6 else "black"
                ax.text(j, i, f"{norm_matrix[i, j]*100:.1f}%\n({count_matrix[i, j]:,})",
                        ha="center", va="center", color=text_color, fontsize=10)

    fig.suptitle(f"Cloud-Top Pixel Detection Confusion Matrix (n={len(files)} frames, "
                 f"row-normalized)")
    fig.tight_layout()
    save_path = OUT_DIR / "confusion_matrix_cloud_detection.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print()
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
