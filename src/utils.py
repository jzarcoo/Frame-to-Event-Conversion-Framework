import h5py
import numpy as np

def save_h5(events_array, output_path):
    """Save events in HDF5 format compatible with DSEC."""
    x = events_array[:, 0].astype(np.uint16)
    y = events_array[:, 1].astype(np.uint16)
    t = events_array[:, 2].astype(np.uint32)

    # convert -1/+1 -> 0/1
    p = (events_array[:, 3] > 0).astype(np.uint8)

    with h5py.File(output_path, "w") as f:
        events = f.create_group("events")

        events.create_dataset("x", data=x)
        events.create_dataset("y", data=y)
        events.create_dataset("t", data=t)
        events.create_dataset("p", data=p)

        f.create_dataset("t_offset", data=0)