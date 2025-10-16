# Script to convert E65_struct.mat to a .npz file with ROI and behavioral variables.
# Worked as of 2025-09-10
# Written by: Kathleen Higgins


import h5py
import numpy as np

path = "../mat_E65_data/E65_struct.mat"
f = h5py.File(path, "r")

bv = f["/nic_output/behavioralVariables"]

behavioral = {}
for k in bv.keys():
    try:
        arr = bv[k][()]
        # flatten (1, N) to (N,)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        # if this is an object array of references, try to dereference
        if arr.dtype == np.dtype("O"):
            flat = []
            for elem in arr:
                if isinstance(elem, h5py.Reference):
                    try:
                        ref_obj = f[elem][()]
                        # if that object is numeric, append mean or scalar
                        if np.issubdtype(ref_obj.dtype, np.number):
                            flat.append(np.array(ref_obj).squeeze())
                        else:
                            # fallback: append 0 or nan if not numeric
                            flat.append(np.nan)
                    except Exception:
                        flat.append(np.nan)
                else:
                    # already a number
                    flat.append(elem)
            arr = np.array(flat)
        behavioral[k] = np.array(arr).squeeze()
        print(f"Loaded {k}, shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print(f"Skipping {k}: {e}")

# load roi
roi = f["/nic_output/ROIactivities"][()]
print("ROI shape:", roi.shape)

# only keep numeric arrays
safe_behavioral = {
    k: v for k, v in behavioral.items()
    if np.issubdtype(v.dtype, np.number)
}

np.savez("../npz_e65_data/E65_data.npz", roi=roi, **safe_behavioral)
print("Saved numeric dataset to E65_data.npz")

# reopen the saved npz
data = np.load("E65_data.npz")
print("\n NPZ loaded back successfully.")
print(f"Contains {len(data.files)} arrays: {data.files}")

# compare array lengths to ensure no truncation
lengths = {k: data[k].shape for k in data.files}
for k, v in lengths.items():
    print(f"{k:<20} shape={v}")

# check for NaNs or weird constant arrays
for k in data.files:
    arr = data[k]
    if np.isnan(arr).any():
        print(f"{k} contains NaNs ({np.isnan(arr).sum()} total)")
    elif np.all(arr == arr[0]):
        print(f"{k} is constant (possible flat signal)")
    else:
        print(f"{k} OK: mean={np.mean(arr):.3f}, std={np.std(arr):.3f}")