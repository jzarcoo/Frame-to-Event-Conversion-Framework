import argparse

def build_parser():
    """Build the command-line argument parser for the event generation script."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate event streams from RGB videos."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.2,
        help=(
            "Log intensity threshold "
            "(default: 0.2)"
        ),
    )

    parser.add_argument(
        "-m",
        "--min-pixels",
        type=int,
        default=500,
        help=(
            "Minimum motion pixels "
            "(default: 500)"
        ),
    )

    parser.add_argument(
        "-n",
        "--name",
        help=(
            "Base output name. "
            "Defaults to input filename."
        ),
    )

    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        "--video",
        dest="generate_video",
        action="store_true",
        help="Generate the output video (default).",
    )
    video_group.add_argument(
        "--no-video",
        dest="generate_video",
        action="store_false",
        help="Skip output video generation.",
    )
    parser.set_defaults(generate_video=True)

    parser.add_argument(
        "--timestamps",
        help="Optional timestamp file (microseconds)"
    )

    parser.add_argument(
        "--resize",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        type=int,
        help="Resize output frames to WIDTH HEIGHT."
    )

    return parser