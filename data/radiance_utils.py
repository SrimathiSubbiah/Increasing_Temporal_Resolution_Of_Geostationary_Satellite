import numpy as np

def radiance_to_brightness_temp(rad: np.ndarray, ds) -> np.ndarray:
    """
    Convert ABI radiance to brightness temperature (Kelvin) using
    the file's own Planck calibration constants — these differ per band,
    so always read them from the dataset, never hardcode.
    """
    fk1 = float(ds["planck_fk1"].values)
    fk2 = float(ds["planck_fk2"].values)
    bc1 = float(ds["planck_bc1"].values)
    bc2 = float(ds["planck_bc2"].values)

    rad = np.clip(rad, 1e-6, None)  # avoid log(0)/negative radiance
    bt = (fk2 / np.log((fk1 / rad) + 1) - bc1) / bc2
    return bt