import argparse 

from .images import dsec_events_to_video

def build_parser():
    """
    Build the argument parser for the DSEC to video converter.
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Convert DSEC events to video.")
    parser.add_argument(
        "--h5_path", type=str, required=True, help="Path to the input HDF5 file containing DSEC events."
    )
    parser.add_argument(
        "--output_video_path", type=str, required=True, help="Path to the output video file."
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Width of the output video frames."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Height of the output video frames."
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Frames per second for the output video."
    )
    return parser

# uv run python -m framework.utils.dsec_to_video_converter --h5_path data/zurich_city_13_a/zurich_city_13_a_events_left/events.h5 --output_video_path results/zurich_city_13_a_DSEC_events.mp4 
if __name__ == "__main__":
    """
    Main function to convert DSEC events to video using command-line arguments.
    """
    parser = build_parser()
    args = parser.parse_args()
    dsec_events_to_video(
        h5_path=args.h5_path,
        output_video_path=args.output_video_path,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
