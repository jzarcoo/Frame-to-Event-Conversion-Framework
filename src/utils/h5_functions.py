import h5py
import numpy as np

class EventH5Writer:
    """Utility class to write events to an HDF5 file in a format compatible with DSEC."""
    def __init__(self, output_path):
        self.output_path = output_path
        self.file = h5py.File(output_path, "w")
        self.events_group = self.file.create_group("events")
        self.x_dataset = self.events_group.create_dataset(
            "x",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint16,
            chunks=True,
        )
        self.y_dataset = self.events_group.create_dataset(
            "y",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint16,
            chunks=True,
        )
        self.t_dataset = self.events_group.create_dataset(
            "t",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint32,
            chunks=True,
        )
        self.p_dataset = self.events_group.create_dataset(
            "p",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=True,
        )
        self.total_events = 0
        
        self.last_timestamp_us = None
        self.next_ms = 0
        self.ms_to_idx = []

    def add_events(self, events_array):
        """Add a batch of events to the HDF5 file. Events should be a numpy array of shape (N, 4) with columns [x, y, t, p]."""
        if len(events_array) == 0:
            return

        assert np.all(events_array[1:, 2] >= events_array[:-1, 2]), (
            "Events must be sorted by timestamp"
        )

        x = events_array[:, 0].astype(np.uint16)
        y = events_array[:, 1].astype(np.uint16)
        t = events_array[:, 2].astype(np.uint32)
        p = (events_array[:, 3] > 0).astype(np.uint8)

        x_end = self.total_events + len(events_array)
        self.x_dataset.resize((x_end,))
        self.y_dataset.resize((x_end,))
        self.t_dataset.resize((x_end,))
        self.p_dataset.resize((x_end,))

        self.x_dataset[self.total_events:x_end] = x
        self.y_dataset[self.total_events:x_end] = y
        self.t_dataset[self.total_events:x_end] = t
        self.p_dataset[self.total_events:x_end] = p

        chunk_last_timestamp = int(t[-1])
        while self.next_ms * 1000 <= chunk_last_timestamp:
            self.ms_to_idx.append(self.total_events)
            self.next_ms += 1

        self.total_events = x_end
        self.last_timestamp_us = chunk_last_timestamp

    def finalize(self):
        """Finalize the HDF5 file by writing the ms_to_idx mapping and closing the file."""
        if self.last_timestamp_us is not None:
            max_ms = int(np.ceil(self.last_timestamp_us / 1000))
            while self.next_ms <= max_ms:
                self.ms_to_idx.append(self.total_events)
                self.next_ms += 1

        if len(self.ms_to_idx) == 0:
            self.ms_to_idx.append(0)

        self.file.create_dataset(
            "ms_to_idx",
            data=np.array(self.ms_to_idx, dtype=np.uint32),
        )
        self.file.create_dataset("t_offset", data=0)
        self.file.flush()
        self.file.close()


def save_h5(events_array, output_path):
    """Save events in HDF5 format compatible with DSEC."""
    writer = EventH5Writer(output_path)
    writer.add_events(events_array)
    writer.finalize()

def set_offset(h5_file, offset):
    """Set t_offset in an existing HDF5 file."""
    with h5py.File(h5_file, "a") as f:
        if "t_offset" in f:
            f["t_offset"][()] = offset
        else:
            f.create_dataset("t_offset", data=offset)
    