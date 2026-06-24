import os

from framework.core.event_generator import EventGenerator
from framework.utils.parser import build_parser

def main():
    """Entry point for the command-line interface to generate events from RGB videos."""
    args = build_parser().parse_args()

    # Ensure video file exists
    if not os.path.exists(args.input):
        print(f"Input file does not exist: {args.input}")
        return
    
    # Output paths setup
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    output_h5_path = os.path.join(args.output_dir, f"{base_name}_events.h5")
    output_video_path = os.path.join(args.output_dir, f"{base_name}_video.mp4") if args.generate_video else None

    event_generator = EventGenerator(
        threshold=args.threshold,
        min_pixels=args.min_pixels,
        resize_width=args.resize[0],
        resize_height=args.resize[1], 
    )

    event_generator.generate(
        input_video=args.input,
        output_h5_path=output_h5_path,
        output_video_path=output_video_path,
        timestamps_path=args.timestamps,
    )

if __name__ == "__main__":
    main()