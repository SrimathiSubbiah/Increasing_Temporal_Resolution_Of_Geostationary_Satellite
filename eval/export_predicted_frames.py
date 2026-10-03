from pathlib import Path
import numpy as np
from PIL import Image


PRED_DIR = Path("outputs/unet_predictions")

OUT_PRED_B13 = Path("outputs/predicted_midframes/band13")
OUT_PRED_B09 = Path("outputs/predicted_midframes/band09")
OUT_GT_B13 = Path("outputs/ground_truth_midframes/band13")
OUT_AVG_B13 = Path("outputs/average_midframes/band13")
OUT_TVL1_B13 = Path("outputs/tvl1_midframes/band13")

OUT_PRED_B13.mkdir(parents=True, exist_ok=True)
OUT_PRED_B09.mkdir(parents=True, exist_ok=True)
OUT_GT_B13.mkdir(parents=True, exist_ok=True)
OUT_AVG_B13.mkdir(parents=True, exist_ok=True)
OUT_TVL1_B13.mkdir(parents=True, exist_ok=True)


def save_png(arr, path):
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


files = sorted(PRED_DIR.glob("pred_*.npz"))

print("Prediction files found:", len(files))

if len(files) == 0:
    raise FileNotFoundError("No prediction files found in outputs/unet_predictions")

for file in files:
    data = np.load(file)

    pred = np.clip(data["pred"], 0, 1)       # [2, H, W]
    target = np.clip(data["target"], 0, 1)   # [2, H, W]
    frame0 = np.clip(data["frame0"], 0, 1)   # [2, H, W]
    frame2 = np.clip(data["frame2"], 0, 1)   # [2, H, W]

    name = file.stem.replace("pred_", "mid_") + ".png"

    # Channel 0 = Band 13
    pred_b13 = pred[0]
    target_b13 = target[0]
    avg_b13 = (frame0[0] + frame2[0]) / 2.0
    tvl1_b13 = np.clip(data["warped"], 0, 1)[0] if "warped" in data.files else avg_b13

    # Channel 1 = Band 9
    pred_b09 = pred[1]

    save_png(pred_b13, OUT_PRED_B13 / name)
    save_png(pred_b09, OUT_PRED_B09 / name)
    save_png(target_b13, OUT_GT_B13 / name)
    save_png(avg_b13, OUT_AVG_B13 / name)
    save_png(tvl1_b13, OUT_TVL1_B13 / name)

print("Export completed.")
print("Predicted Band 13 frames saved to:", OUT_PRED_B13)
print("Predicted Band 9 frames saved to :", OUT_PRED_B09)
print("Ground truth Band 13 saved to    :", OUT_GT_B13)
print("Average baseline Band 13 saved to:", OUT_AVG_B13)
print("TV-L1 warp Band 13 saved to      :", OUT_TVL1_B13)