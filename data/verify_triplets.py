import numpy as np
import glob

test_files = glob.glob("triplets/test/*.npz")
print(f"Test files: {len(test_files)}")

sample = np.load(test_files[0])
print(sample["frame0"].shape, sample["frame1"].shape, sample["frame2"].shape)
print(sample["frame0"].min(), sample["frame0"].max())