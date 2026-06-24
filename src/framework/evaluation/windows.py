from pathlib import Path


def load_windows(path: str | Path):
    """
    Load DSEC timestamp windows.

    Format:
    start_us,end_us
    """
    windows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            start_us, end_us = line.split(",")
            windows.append((
                    int(start_us),
                    int(end_us)
                ))
    return windows