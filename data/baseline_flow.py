"""
TV-L1 optical flow interpolation baseline (per WR-Net's approach).
Computes flow on Band 13, applies the same flow field to warp both channels.
"""

import glob
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_tvl1_flow(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """frame_a, frame_b: single-channel float32 in [0,1]. Returns flow [H, W, 2]."""
    a8 = (frame_a * 255).astype(np.uint8)
    b8 = (frame_b * 255).astype(np.uint8)
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(a8, b8, None)
    return flow


def warp(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = frame.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def interpolate_middle_frame(frame0: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    frame0, frame2: [2, H, W] two-channel arrays (Band13, Band9).
    Flow computed on Band 13 (channel 0), applied to both channels.
    """
    band13_0, band13_2 = frame0[0], frame2[0]

    flow_fwd = compute_tvl1_flow(band13_0, band13_2) * 0.5
    flow_bwd = compute_tvl1_flow(band13_2, band13_0) * 0.5

    warped = np.zeros_like(frame0)
    for c in range(2):
        warped_from_0 = warp(frame0[c], flow_fwd)
        warped_from_2 = warp(frame2[c], flow_bwd)
        warped[c] = 0.5 * warped_from_0 + 0.5 * warped_from_2

    return warped


def evaluate_test_set(test_dir: str = "triplets/test", max_samples: int = 200):
    """
    max_samples caps evaluation to keep runtime sane -- TV-L1 is slow per pair,
    running all 3207 test triplets isn't necessary to get a solid metric estimate.
    """
    files = sorted(glob.glob(f"{test_dir}/*.npz"))[:max_samples]
    print(f"Evaluating baseline on {len(files)} test triplets...")

    psnr_scores, ssim_scores = [], []

    for i, f in enumerate(files):
        data = np.load(f)
        frame0, frame1_target, frame2 = data["frame0"], data["frame1"], data["frame2"]

        pred = interpolate_middle_frame(frame0, frame2)

        # evaluate on Band 13 channel only for now -- keeps metric interpretation simple
        p = psnr(frame1_target[0], pred[0], data_range=1.0)
        s = ssim(frame1_target[0], pred[0], data_range=1.0)
        psnr_scores.append(p)
        ssim_scores.append(s)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} done | running PSNR: {np.mean(psnr_scores):.3f}")

    print(f"\nBaseline results (n={len(files)}):")
    print(f"  PSNR: {np.mean(psnr_scores):.3f} +/- {np.std(psnr_scores):.3f}")
    print(f"  SSIM: {np.mean(ssim_scores):.3f} +/- {np.std(ssim_scores):.3f}")

    return psnr_scores, ssim_scores


if __name__ == "__main__":
    evaluate_test_set()