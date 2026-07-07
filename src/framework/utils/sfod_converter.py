import os
import h5py
import hdf5plugin
import numpy as np

from framework.utils.dsec.eventslicer import EventSlicer
from framework.utils.prophesee.dat_events_tools import (
    write_header,
    write_event_buffer,
)

# Framework events
# H5_FILE = "results/thun_00_a_events.h5"
# Ground truth events
H5_FILE = "data/thun_00_a/thun_00_a_events_left/events.h5"

OUTPUT_DIR = "gt_dat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The window size in microseconds for slicing events into separate .dat files
WINDOW_US = 100_000 # 100 ms

# Dimension of the source events, from the HDF5 file
SRC_WIDTH = 640
SRC_HEIGHT = 480

# SFOD uses 304x240
DST_WIDTH = 304
DST_HEIGHT = 240

with h5py.File(H5_FILE, "r") as h5:
    slicer = EventSlicer(h5)
    t0 = slicer.get_start_time_us()
    tf = slicer.get_final_time_us()
    print(f"Start: {t0}")
    print(f"End  : {tf}")
    print(f"Duration: {(tf-t0)/1e6:.2f} s")
    file_id = 0
    for start in range(t0, tf, WINDOW_US):
        end = min(start + WINDOW_US, tf)
        ev = slicer.get_events(start, end)
        if ev is None:
            continue
        if len(ev["t"]) == 0:
            continue
        buffer = np.empty(
            len(ev["t"]),
            dtype=[
                ("t", np.uint32),
                ("x", np.uint16),
                ("y", np.uint16),
                ("p", np.uint8),
            ],
        )
        # Resize events to match SFOD dimensions
        x = ev["x"].astype(np.float32)
        y = ev["y"].astype(np.float32)
        x = np.round(x * (DST_WIDTH / SRC_WIDTH))
        y = np.round(y * (DST_HEIGHT / SRC_HEIGHT))
        buffer["x"] = np.clip(x, 0, DST_WIDTH - 1).astype(np.uint16)
        buffer["y"] = np.clip(y, 0, DST_HEIGHT - 1).astype(np.uint16)
        # timestamps start at zero, like N-CARS
        buffer["t"] = (ev["t"] - ev["t"][0]).astype(np.uint32)
        buffer["p"] = ev["p"]
        filename = os.path.join(OUTPUT_DIR, f"obj_{file_id:06d}_td.dat")
        f = write_header(filename, height=DST_HEIGHT, width=DST_WIDTH)
        write_event_buffer(f, buffer)
        f.close()
        print(f"Saved {filename} ({len(buffer)} events)")
        file_id += 1
print(f"\nGenerated {file_id} .dat files")
