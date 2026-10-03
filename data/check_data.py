import xarray as xr
import glob
import numpy as np
from radiance_utils import radiance_to_brightness_temp

f13 = sorted(glob.glob("raw_data/band13/noaa-goes16/**/*.nc", recursive=True))[0]
f9 = sorted(glob.glob("raw_data/band9/noaa-goes16/**/*.nc", recursive=True))[0]

ds13 = xr.open_dataset(f13)
ds9 = xr.open_dataset(f9)

print("Band 13 Rad shape:", ds13["Rad"].shape)
print("Band 9 Rad shape:", ds9["Rad"].shape)

bt13 = radiance_to_brightness_temp(ds13["Rad"].values, ds13)
bt9 = radiance_to_brightness_temp(ds9["Rad"].values, ds9)

print("Band 13 BT range:", bt13.min(), bt13.max())
print("Band 9 BT range:", bt9.min(), bt9.max())

print("Band 13 Rad NaN count:", np.isnan(ds13["Rad"].values).sum(), "/", ds13["Rad"].values.size)
print("Band 13 Rad min/max (ignoring NaN):", np.nanmin(ds13["Rad"].values), np.nanmax(ds13["Rad"].values))

print("Band 13 BT range:", np.nanmin(bt13), np.nanmax(bt13))
print("Band 9 BT range:", np.nanmin(bt9), np.nanmax(bt9))