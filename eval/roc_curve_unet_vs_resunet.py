"""
ROC curve for the actual trained models in this project -- U-Net, ResUNet,
and Attention U-Net (Average and TV-L1 aren't learned classifiers, so a ROC
curve isn't a meaningful frame for them the way it is for a model's
continuous output).

The project already fixes ONE decision threshold (BT_THRESHOLD=0.35) to
turn a continuous predicted brightness-temperature into a binary
cloud/clear call -- see confusion_matrix_cloud_detection.py. A ROC curve
generalizes that: instead of one fixed threshold, sweep the decision
threshold across the full range and trace out (false-positive rate,
true-positive rate) at every point. This answers "how good is the ranking
the model produces," independent of which single cutoff you happen to pick
-- and the single-threshold confusion matrix numbers should sit exactly on
this curve at BT_THRESHOLD, which is a useful sanity check.

Ground-truth labels are still defined by the SAME fixed threshold
(BT_THRESHOLD applied to the true frame) -- only each model's OWN decision
threshold is swept.

ResUNet and Attention U-Net are optional -- only plotted if their prediction
dirs exist (i.e. their train_*.py + eval/evaluate_*.py have already run).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BT_THRESHOLD = 0.35  # same ground-truth cloud/clear cutoff used throughout the project

MODEL_DIRS = {
    "unet": PROJECT_ROOT / "outputs" / "unet_predictions",
    "resunet": PROJECT_ROOT / "outputs" / "resunet_predictions",
    "attention_unet": PROJECT_ROOT / "outputs" / "attention_unet_predictions",
}
MODEL_LABELS = {
    "unet": "U-Net",
    "resunet": "ResUNet",
    "attention_unet": "Attention U-Net",
}
# Validated 3-slot categorical palette (dataviz skill) -- blue/orange/aqua,
# same slot order used across compare_unet_vs_resunet.py and friends.
MODEL_COLORS = {"unet": "#2a78d6", "resunet": "#eb6834", "attention_unet": "#1baf7a"}


def operating_point(labels, scores, threshold):
    """FPR/TPR at the project's actual fixed decision threshold -- should
    reproduce confusion_matrix_cloud_detection.py's numbers exactly."""
    pred = scores > -threshold  # score=-BT, so BT<threshold <=> score>-threshold
    tp = np.sum(pred & labels)
    fp = np.sum(pred & ~labels)
    fn = np.sum(~pred & labels)
    tn = np.sum(~pred & ~labels)
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return fpr, tpr


def main():
    if not MODEL_DIRS["unet"].exists():
        raise FileNotFoundError(
            f"No files in {MODEL_DIRS['unet']}. Run eval/evaluate_model.py first."
        )

    available = {
        name: sorted(d.glob("pred_*.npz"))
        for name, d in MODEL_DIRS.items()
        if d.exists() and any(d.glob("pred_*.npz"))
    }
    print("Models found:", ", ".join(f"{MODEL_LABELS[m]} ({len(f)})" for m, f in available.items()))

    n = min(len(f) for f in available.values())
    unet_files = available["unet"]

    print(f"Building ROC curves from {n} paired test samples")

    results = {}
    for model_name in available:
        files = available[model_name]
        labels_list, scores_list = [], []

        for i in range(n):
            # ground truth always comes from the U-Net dir's target -- every
            # model's own predictions file also carries "target" (same
            # underlying triplet), used here only for the pairing safety check.
            u = np.load(unet_files[i])
            u_target = np.clip(u["target"], 0, 1)

            data = np.load(files[i])
            m_target = np.clip(data["target"], 0, 1)
            if not np.allclose(u_target, m_target, atol=1e-5):
                raise RuntimeError(
                    f"pred_{i:05d}.npz targets differ between U-Net and {model_name} dirs -- "
                    "sampling seeds don't line up, pairing assumption is broken."
                )

            true_mask = (u_target[0] < BT_THRESHOLD).ravel()
            m_pred = np.clip(data["pred"], 0, 1)[0].ravel()

            labels_list.append(true_mask)
            # score = -predicted_BT, since LOWER brightness temp = MORE
            # cloud-like -- sklearn's roc_curve expects "higher score = more
            # positive," so negate.
            scores_list.append(-m_pred)

        labels = np.concatenate(labels_list)
        scores = np.concatenate(scores_list)

        fpr, tpr, _ = roc_curve(labels, scores)
        model_auc = auc(fpr, tpr)
        op = operating_point(labels, scores, BT_THRESHOLD)

        results[model_name] = dict(fpr=fpr, tpr=tpr, auc=model_auc, op=op)

    print(f"Total pixels scored per model: {labels.size:,}")
    print()
    print("ROC AUC (cloud/clear pixel classification)")
    print("---------------------------------------------")
    for model_name, r in results.items():
        print(f"{MODEL_LABELS[model_name]:<16s} AUC: {r['auc']:.5f}")

    print()
    print(f"Operating point at project's BT_THRESHOLD={BT_THRESHOLD} "
          "(should match confusion_matrix_cloud_detection.py):")
    for model_name, r in results.items():
        print(f"{MODEL_LABELS[model_name]:<16s} FPR={r['op'][0]:.4f}  TPR(Recall)={r['op'][1]:.4f}")

    # --- chart --------------------------------------------------------------
    # All three models are near-perfect classifiers (AUC > 0.999), so their
    # curves overlap almost exactly near the top-left corner on a full 0-1
    # axis -- the difference between models is real but invisible at that
    # scale. Panel 2 zooms into where the actual operating points sit so the
    # separation is actually visible.
    all_ops = np.array([r["op"] for r in results.values()])
    zoom_x = min(0.02, all_ops[:, 0].max() * 3)
    zoom_y_lo = max(0.0, all_ops[:, 1].min() - 0.03)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, is_zoom in [(ax_full, False), (ax_zoom, True)]:
        for model_name, r in results.items():
            color = MODEL_COLORS[model_name]
            ax.plot(r["fpr"], r["tpr"], color=color, lw=2,
                    label=f"{MODEL_LABELS[model_name]} (AUC={r['auc']:.4f})" if not is_zoom else None)
            ax.scatter(*r["op"], color=color, zorder=5, s=60, edgecolor="white",
                       label=f"{MODEL_LABELS[model_name]} @ threshold={BT_THRESHOLD}" if not is_zoom else None)
        ax.plot([0, 1], [0, 1], color="#8a8a86", lw=1, linestyle="--",
                label="Random guess" if not is_zoom else None)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")

    ax_full.set_xlim(0, 1)
    ax_full.set_ylim(0, 1)
    ax_full.set_title("Full ROC curve")
    ax_full.legend(loc="lower right", fontsize=8)

    ax_zoom.set_xlim(0, zoom_x)
    ax_zoom.set_ylim(zoom_y_lo, 1.001)
    ax_zoom.set_title(f"Zoomed into the operating region\n(FPR 0-{zoom_x:.3f}, TPR {zoom_y_lo:.2f}-1.0)"
                       " -- this is where the models actually differ")

    fig.suptitle(f"ROC Curve: Cloud-Top Pixel Classification (n={n} frames)", fontsize=13)
    fig.tight_layout()

    save_path = OUT_DIR / "roc_curve_unet_vs_resunet.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print()
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
