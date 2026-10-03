"""
Precompute TV-L1 warp for every triplet once, save alongside the original
data. Training then loads the cached warp instead of recomputing it every
single epoch -- this is the fix for the multi-hour training stall.

Parallelized across processes (TV-L1 is CPU-bound and embarrassingly
parallel across files) -- run this once per split. Safe to interrupt and
resume -- it skips files that already have a cached warp.
"""

import glob
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import cv2
from tqdm import tqdm

N_WORKERS = max(1, (os.cpu_count() or 4) - 2)
SAMPLE_SEED = 42  # must match train.py's TripletDataset sampling exactly


def compute_tvl1_flow_and_warp(frame0: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    def flow(a, b):
        a8 = (a * 255).astype(np.uint8)
        b8 = (b * 255).astype(np.uint8)
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        return tvl1.calc(a8, b8, None)

    def warp(frame, f):
        h, w = frame.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (gx + f[..., 0]).astype(np.float32)
        map_y = (gy + f[..., 1]).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    flow_fwd = flow(frame0[0], frame2[0]) * 0.5
    flow_bwd = flow(frame2[0], frame0[0]) * 0.5

    warped = np.zeros_like(frame0)
    for c in range(2):
        warped[c] = 0.5 * warp(frame0[c], flow_fwd) + 0.5 * warp(frame2[c], flow_bwd)
    return warped


def _process_one(f: str):
    """Worker fn: returns 'skipped' if already cached, else 'done'."""
    data = np.load(f)
    if "warped" in data.files:
        return "skipped"

    f0, f1, f2 = data["frame0"], data["frame1"], data["frame2"]
    warped = compute_tvl1_flow_and_warp(f0, f2)

    # overwrite the file with the warp included
    np.savez(f, frame0=f0, frame1=f1, frame2=f2, warped=warped)
    return "done"


def precompute_split(triplet_dir: str, max_samples: int = None, n_workers: int = N_WORKERS):
    files = sorted(glob.glob(f"{triplet_dir}/*.npz"))
    if max_samples:
        # Must select the SAME subset train.py's TripletDataset will train
        # on -- train.py takes a random.seed(42) + random.sample() draw
        # from the full sorted file list, not the first N. Precomputing a
        # different subset (e.g. files[:max_samples]) silently caches warps
        # for triplets that never get trained on, leaving the actual
        # training sample without cached warps.
        random.seed(SAMPLE_SEED)
        files = random.sample(files, min(max_samples, len(files)))

    print(f"Precomputing warps for {len(files)} files in {triplet_dir} "
          f"using {n_workers} worker processes...")
    t0 = time.time()
    skipped, done = 0, 0

    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap_unordered(_process_one, files, chunksize=8), total=len(files)):
            if result == "skipped":
                skipped += 1
            else:
                done += 1

    elapsed = time.time() - t0
    print(f"Done: {done} computed, {skipped} already cached, {elapsed:.1f}s total"
          f" ({elapsed/max(done,1):.2f}s/sample)")


if __name__ == "__main__":
    # cap train/val precompute to match what train.py actually uses --
    # no point precomputing warps for samples you won't train on.
    # Test gets full coverage since eval needs a warped input for every
    # test triplet (and there's no "unused" test data to skip); it's
    # already fully cached so it's a fast no-op skip pass here.
    precompute_split("data/triplets/train", max_samples=8000)
    precompute_split("data/triplets/val", max_samples=1500)
    precompute_split("data/triplets/test", max_samples=None)
