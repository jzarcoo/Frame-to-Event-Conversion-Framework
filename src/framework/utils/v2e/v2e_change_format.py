# uv run python -m framework.utils.v2e.v2e_change_format
import h5py
import numpy as np
import hdf5plugin

dsec_path = "./data/dsec/test/zurich_city_13_a/events/left/events.h5"
v2e_path = "./v2e-output/events.h5"
output_path = "./v2e-output/events_v2e.h5"

# Read DSEC offset
with h5py.File(dsec_path, "r") as f:
    t_offset = int(f["t_offset"][()])

# Read v2e events
with h5py.File(v2e_path, "r") as f:
    events = f["events"]

    t = events[:, 0].astype(np.uint32)
    x = events[:, 1].astype(np.uint16)
    y = events[:, 2].astype(np.uint16)
    p = events[:, 3].astype(np.uint8)

# Build ms_to_idx
ms = np.arange(int(t[-1] // 1000) + 1, dtype=np.uint32) * 1000
ms_to_idx = np.searchsorted(t, ms).astype(np.uint64)

# Write DSEC-compatible file
with h5py.File(output_path, "w") as f:
    g = f.create_group("events")
    g.create_dataset("t", data=t)
    g.create_dataset("x", data=x)
    g.create_dataset("y", data=y)
    g.create_dataset("p", data=p)

    f.create_dataset("ms_to_idx", data=ms_to_idx)
    f.create_dataset("t_offset", data=t_offset)

print(f"Created: {output_path}")
print(f"t_offset = {t_offset}")