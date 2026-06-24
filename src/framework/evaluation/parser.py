import argparse 
def build_parser():
    """
    Build the argument parser for the event evaluation CLI.
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated events "
            "against DSEC ground truth."
        )
    )

    parser.add_argument(
        "--gt",
        required=True,
        help="Ground truth events.h5",
    )

    parser.add_argument(
        "--pred",
        required=True,
        help="Predicted events.h5",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory",
    )

    # MODE1: Evaluate a specific time window
    parser.add_argument(
        "--timewindow",
        "--tw",
        nargs=2,
        type=int,
        metavar=("START_US", "END_US"),
    )

    # MODE2: Evaluate using a windows file
    parser.add_argument(
        "--windows",
        help=(
            "Timestamp windows file"
        ),
    )

    # MODE3: Evaluate frame by frame using a specified time window
    parser.add_argument(
        "--frame-time",
        type=int,
        help=(
            "Evaluate frame by frame using this "
            "time window (e.g., 50000 for 50ms). "
            "Defaults to 50000 if no other mode is selected."
        )
    )

    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[640, 480],
        help="Target resolution as WIDTH HEIGHT (default: 640 480)",
    )

    parser.add_argument(
        "--seconds",
        nargs=2,
        type=float,
        metavar=("START_SEC", "END_SEC"),
    )

    return parser