"""
Display/save the RAW Band 13 (IR window) and Band 9 (water vapor) satellite
imagery straight from the downloaded NetCDF files -- before any of the
project's preprocessing (no cropping to patches, no [0,1] normalization).

Converts radiance -> brightness temperature (Kelvin) using each file's own
Planck calibration constants (same conversion build_triplets.py uses), then
renders the FULL 1500x2500 frame for each band, plus a difference view.

Usage:
    python data/visualize_raw_bands.py
    python data/visualize_raw_bands.py --index 5   # a later timestamp
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from radiance_utils import radiance_to_brightness_temp

BAND13_DIR = "raw_data/band13/noaa-goes16"
BAND9_DIR = "raw_data/band9/noaa-goes16"
OUT_DIR = Path("../outputs/figures")


def main():
    parser = argparse.ArgumentParser(description="Visualize raw Band13/Band9 GOES-16 imagery.")
    parser.add_argument("--index", type=int, default=0, help="which raw file (by time-sorted index) to display")
    args = parser.parse_args()

    files13 = sorted(glob.glob(f"{BAND13_DIR}/**/*.nc", recursive=True))
    files9 = sorted(glob.glob(f"{BAND9_DIR}/**/*.nc", recursive=True))

    if not files13 or not files9:
        raise FileNotFoundError("No raw .nc files found -- run data/download_goes.py first.")

    f13 = files13[args.index]
    f9 = files9[args.index]
    print("Band 13 file:", f13)
    print("Band 9  file:", f9)

    ds13 = xr.open_dataset(f13)
    ds9 = xr.open_dataset(f9)

    bt13 = radiance_to_brightness_temp(ds13["Rad"].values, ds13)
    bt9 = radiance_to_brightness_temp(ds9["Rad"].values, ds9)

    print(f"Band 13: shape={bt13.shape}  BT range={np.nanmin(bt13):.1f}K - {np.nanmax(bt13):.1f}K")
    print(f"Band 9 : shape={bt9.shape}  BT range={np.nanmin(bt9):.1f}K - {np.nanmax(bt9):.1f}K")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # IR window: convention is COLDER=BRIGHTER (cmap reversed) so cold cloud
    # tops read as bright/white, matching how meteorologists view this band.
    im0 = axes[0].imshow(bt13, cmap="gray_r", vmin=180, vmax=320)
    axes[0].set_title(f"Band 13 -- Clean IR Window (10.3μm)\nRaw brightness temperature, Kelvin")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], label="Brightness Temp (K)", fraction=0.04)

    im1 = axes[1].imshow(bt9, cmap="gray_r", vmin=190, vmax=265)
    axes[1].set_title(f"Band 9 -- Water Vapor (6.9μm)\nRaw brightness temperature, Kelvin")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], label="Brightness Temp (K)", fraction=0.04)

    fig.suptitle(f"Raw GOES-16 Imagery (full CONUS frame, index={args.index})", fontsize=13)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / f"raw_bands_{args.index:03d}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("Saved:", save_path.resolve())

    ds13.close()
    ds9.close()


if __name__ == "__main__":
    main()
