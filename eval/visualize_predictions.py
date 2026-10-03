from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PRED_DIR = Path("outputs/unet_predictions")
OUT_DIR = Path("outputs/figures")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(img):
    img = np.clip(img, 0, 1)
    return img


def save_comparison(sample_id):
    file_path = PRED_DIR / f"pred_{sample_id:05d}.npz"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {file_path}")

    data = np.load(file_path)

    pred = data["pred"]       # [2, H, W]
    target = data["target"]   # [2, H, W]
    frame0 = data["frame0"]   # [2, H, W]
    frame2 = data["frame2"]   # [2, H, W]

    # Band 13 only for visualization
    f0 = normalize(frame0[0])
    f2 = normalize(frame2[0])
    gt = normalize(target[0])
    pr = normalize(pred[0])

    avg = (f0 + f2) / 2.0
    tvl1 = normalize(data["warped"][0]) if "warped" in data.files else avg
    error = np.abs(gt - pr)

    # Panel order tells the actual story: the two REAL inputs you have
    # (Frame 0, Frame 2) on the left, the three candidate guesses at the
    # REAL middle frame in the center, and Ground Truth + Error on the
    # right so you can judge them.
    plt.figure(figsize=(24, 4))

    plt.subplot(1, 7, 1)
    plt.imshow(f0, cmap="gray")
    plt.title("Frame 0 (t)")
    plt.axis("off")

    plt.subplot(1, 7, 2)
    plt.imshow(f2, cmap="gray")
    plt.title("Frame 2 (t+2)")
    plt.axis("off")

    plt.subplot(1, 7, 3)
    plt.imshow(avg, cmap="gray")
    plt.title("Average Input")
    plt.axis("off")

    plt.subplot(1, 7, 4)
    plt.imshow(tvl1, cmap="gray")
    plt.title("TV-L1 Warp")
    plt.axis("off")

    plt.subplot(1, 7, 5)
    plt.imshow(pr, cmap="gray")
    plt.title("U-Net Prediction\n(predicted t+1)")
    plt.axis("off")

    plt.subplot(1, 7, 6)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth (t+1)")
    plt.axis("off")

    plt.subplot(1, 7, 7)
    plt.imshow(error, cmap="hot")
    plt.title("Error Map")
    plt.axis("off")

    plt.tight_layout()

    save_path = OUT_DIR / f"comparison_{sample_id:05d}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Saved:", save_path)


if __name__ == "__main__":
    sample_ids = [0, 1, 2, 5, 10]

    for sid in sample_ids:
        save_comparison(sid)