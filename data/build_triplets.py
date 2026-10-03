"""
Build (frame_t, frame_t+1, frame_t+2) triplets from matched Band 13 + Band 9 files.
Each "frame" is a 2-channel array: [Band13_BT, Band9_BT], normalized to [0,1].
"""

import re
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
from tqdm import tqdm

from radiance_utils import radiance_to_brightness_temp

BAND13_DIR = "raw_data/band13/noaa-goes16"
BAND9_DIR = "raw_data/band9/noaa-goes16"
OUT_DIR = "triplets"

PATCH_SIZE = 256
STRIDE = 192

BT13_MIN, BT13_MAX = 180.0, 320.0
BT9_MIN, BT9_MAX = 190.0, 265.0

VAL_FRAC = 0.15
TEST_FRAC = 0.15


def parse_timestamp(filename: str) -> datetime:
    """
    GOES ABI filenames encode scan start time like:
    OR_ABI-L1b-RadC-M6C13_G16_s20241520001173_e20241520003546_c20241520004008.nc
    's' segment = start time: YYYYDDDHHMMSSt (year, day-of-year, time, tenths)
    """
    match = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", filename)
    if not match:
        raise ValueError(f"Could not parse timestamp from filename: {filename}")
    year, doy, hh, mm, ss = match.groups()
    dt = datetime.strptime(f"{year}{doy}", "%Y%j")
    return dt.replace(hour=int(hh), minute=int(mm), second=int(ss))


def match_band_files(dir13: str, dir9: str, max_gap_seconds: int = 90):
    """
    Match each Band 13 file to its nearest-timestamp Band 9 file.
    Skip pairs where the gap exceeds max_gap_seconds -- a large gap means
    something is misaligned and that frame pair shouldn't be trusted.
    """
    files13 = sorted(glob.glob(f"{dir13}/**/*.nc", recursive=True))
    files9 = sorted(glob.glob(f"{dir9}/**/*.nc", recursive=True))

    ts13 = [(f, parse_timestamp(f)) for f in files13]
    ts9 = [(f, parse_timestamp(f)) for f in files9]

    matched = []
    for f13, t13 in ts13:
        best_f9, best_gap = None, None
        for f9, t9 in ts9:
            gap = abs((t13 - t9).total_seconds())
            if best_gap is None or gap < best_gap:
                best_gap, best_f9 = gap, f9
        if best_gap is not None and best_gap <= max_gap_seconds:
            matched.append((f13, best_f9, t13))

    matched.sort(key=lambda x: x[2])  # sort by time
    print(f"Matched {len(matched)} / {len(files13)} Band13 files to Band9 within {max_gap_seconds}s")
    return matched


def load_frame(f13: str, f9: str) -> np.ndarray:
    """Load, convert to BT, normalize, stack as [2, H, W]."""
    ds13 = xr.open_dataset(f13)
    ds9 = xr.open_dataset(f9)

    bt13 = radiance_to_brightness_temp(ds13["Rad"].values, ds13)
    bt9 = radiance_to_brightness_temp(ds9["Rad"].values, ds9)

    ds13.close()
    ds9.close()

    norm13 = np.clip((bt13 - BT13_MIN) / (BT13_MAX - BT13_MIN), 0.0, 1.0)
    norm9 = np.clip((bt9 - BT9_MIN) / (BT9_MAX - BT9_MIN), 0.0, 1.0)

    return np.stack([norm13, norm9], axis=0).astype(np.float32)  # [2, H, W]


def extract_patches(frame: np.ndarray, patch_size: int, stride: int):
    """frame: [2, H, W]. Returns list of [2, patch_size, patch_size] patches."""
    _, h, w = frame.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(frame[:, y:y + patch_size, x:x + patch_size])
    return patches


def build_triplets(matched_files, out_dir: str, patch_size: int, val_frac: float, test_frac: float):
    n = len(matched_files)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": matched_files[: n - n_val - n_test],
        "val": matched_files[n - n_val - n_test: n - n_test],
        "test": matched_files[n - n_test:],
    }

    for split_name, files in splits.items():
        split_dir = Path(out_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        triplet_idx = 0

        print(f"Building {split_name} triplets from {len(files)} frames...")
        for i in tqdm(range(len(files) - 2)):
            f0 = load_frame(files[i][0], files[i][1])
            f1 = load_frame(files[i + 1][0], files[i + 1][1])
            f2 = load_frame(files[i + 2][0], files[i + 2][1])

            p0s = extract_patches(f0, patch_size, STRIDE)
            p1s = extract_patches(f1, patch_size, STRIDE)
            p2s = extract_patches(f2, patch_size, STRIDE)

            for p0, p1, p2 in zip(p0s, p1s, p2s):
                # skip patches with any NaN (off-disk edge pixels)
                if np.isnan(p0).any() or np.isnan(p1).any() or np.isnan(p2).any():
                    continue
                # skip near-uniform patches (empty ocean/clear sky, teaches nothing)
                if p1[0].std() < 0.02:
                    continue

                np.savez(
                    split_dir / f"triplet_{triplet_idx:06d}.npz",
                    frame0=p0, frame1=p1, frame2=p2,
                )
                triplet_idx += 1

        print(f"  -> {triplet_idx} triplets saved to {split_dir}")


if __name__ == "__main__":
    matched = match_band_files(BAND13_DIR, BAND9_DIR)
    build_triplets(matched, OUT_DIR, PATCH_SIZE, VAL_FRAC, TEST_FRAC)