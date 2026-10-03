"""
Visual side-by-side comparison of ALL THREE trained models (U-Net, ResUNet,
Attention U-Net) on the SAME real test samples -- for presentation/demo use.

Relies on the pairing guarantee already established by compare_unet_vs_resunet.py
and confusion_matrix_cloud_detection.py: outputs/unet_predictions,
outputs/resunet_predictions, and outputs/attention_unet_predictions were all
sampled with the identical seed from the identical test file list, so
pred_XXXXX.npz at the same index refers to the SAME underlying triplet in
every directory. No new inference needed -- just reads what's already saved.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def bare_axis(ax, xlabel=None):
    """Strip ticks/spines but keep the frame usable for an xlabel --
    ax.axis('off') suppresses the xlabel too, not just ticks/spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRS = {
    "unet": PROJECT_ROOT / "outputs" / "unet_predictions",
    "resunet": PROJECT_ROOT / "outputs" / "resunet_predictions",
    "attention_unet": PROJECT_ROOT / "outputs" / "attention_unet_predictions",
}
MODEL_LABELS = {"unet": "U-Net", "resunet": "ResUNet", "attention_unet": "Attention U-Net"}
OUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def normalize(img):
    return np.clip(img, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Side-by-side visual comparison of all trained models.")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2],
                         help="which paired sample indices to show (default: 0 1 2)")
    args = parser.parse_args()

    available = {name: d for name, d in MODEL_DIRS.items() if d.exists() and any(d.glob("pred_*.npz"))}
    print("Models found:", ", ".join(MODEL_LABELS[m] for m in available))
    if "unet" not in available:
        raise FileNotFoundError("No U-Net predictions found -- run eval/evaluate_model.py first.")

    model_names = list(available.keys())
    n_rows = len(args.indices)
    n_model_cols = len(model_names)
    n_cols = 2 + n_model_cols + 1  # Frame0, Frame2, [models...], Ground Truth

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, idx in enumerate(args.indices):
        fname = f"pred_{idx:05d}.npz"
        u = np.load(available["unet"] / fname)
        f0 = normalize(u["frame0"][0])
        f2 = normalize(u["frame2"][0])
        true_b13 = normalize(u["target"][0])

        col = 0
        axes[row, col].imshow(f0, cmap="gray"); bare_axis(axes[row, col])
        if row == 0:
            axes[row, col].set_title("Frame 0 (real)", fontsize=10)
        col += 1
        axes[row, col].imshow(f2, cmap="gray"); bare_axis(axes[row, col])
        if row == 0:
            axes[row, col].set_title("Frame 2 (real)", fontsize=10)
        col += 1

        for model_name in model_names:
            data = np.load(available[model_name] / fname)
            m_target = normalize(data["target"][0])
            if not np.allclose(true_b13, m_target, atol=1e-5):
                raise RuntimeError(
                    f"{fname} target mismatch between U-Net and {model_name} dirs -- "
                    "pairing assumption broken."
                )
            pred_b13 = normalize(data["pred"][0])
            p = psnr(true_b13, pred_b13, data_range=1.0)
            s = ssim(true_b13, pred_b13, data_range=1.0)

            axes[row, col].imshow(pred_b13, cmap="gray")
            bare_axis(axes[row, col], xlabel=f"PSNR={p:.1f}  SSIM={s:.3f}")
            if row == 0:
                axes[row, col].set_title(f"{MODEL_LABELS[model_name]}\nPrediction", fontsize=10)
            col += 1

        axes[row, col].imshow(true_b13, cmap="gray"); bare_axis(axes[row, col])
        if row == 0:
            axes[row, col].set_title("Ground Truth\n(what really happened)", fontsize=10)

    fig.suptitle("All Trained Models: Same Real Test Frames, Side by Side (Band 13)", fontsize=13)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / "compare_all_models_visual.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
