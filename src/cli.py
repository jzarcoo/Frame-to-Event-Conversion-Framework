import argparse
import cv2
import numpy as np
import os

try:
    from utils.images import process_frame, build_event_image
except ImportError:
    print("[Error] Could not import 'images' module.")
    exit(1)

try:
    from utils.h5_functions import EventH5Writer
except ImportError:
    print("[Error] Could not import 'h5_functions' module.")
    exit(1)


def generate_events_from_video(
    input_video,
    output_video_path,
    output_h5_path,
    threshold,
    min_pixels,
    generate_video,
):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_video}"
        )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = None
    if generate_video:
        # Output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height),
        )
    # Initialization
    ret, frame = cap.read()
    if not ret:
        cap.release()
        if out is not None:
            out.release()
        raise RuntimeError("Video is empty or first frame cannot be read.")
    last_event_image = process_frame(frame)
    last_event_image = cv2.GaussianBlur(last_event_image, (3, 3), 0)
    rendered_frames = 0
    event_writer = EventH5Writer(output_h5_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_image = process_frame(frame)
        # Noise reduction
        current_image_blurred = cv2.GaussianBlur(
            current_image,
            (3, 3),
            0,
        )
        diff = current_image_blurred.astype(np.float32) - last_event_image.astype(np.float32)
        motion_mask = np.abs(diff) > threshold
        motion_pixels = np.argwhere(motion_mask)

        if len(motion_pixels) <= min_pixels:
            last_event_image = current_image_blurred
            rendered_frames += 1
            continue

        timestamp_us = int((rendered_frames / fps) * 1e6)
        rows = motion_pixels[:, 0]
        cols = motion_pixels[:, 1]
        polarities = np.where(diff[rows, cols] > 0, 1, -1)
        events = np.column_stack((
            cols,
            rows,
            np.full(len(rows), timestamp_us, dtype=np.uint64),
            polarities,
        ))
        event_writer.add_events(events)

        last_event_image = current_image_blurred

        if out is not None:
            event_img = build_event_image(events, height, width)
            out.write(event_img)

        rendered_frames += 1

    # Save files
    cap.release()
    if out is not None:
        out.release()
    event_writer.finalize()

    if event_writer.total_events == 0:
        print("[WARNING] No events were generated.")
        return

    print(f"[INFO] Rendered event frames: {rendered_frames}")
    print(f"[INFO] Total events generated: {event_writer.total_events}")


def main():
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

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(
            f"[ERROR] Input file does not exist: "
            f"{args.input}"
        )
        return

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    if args.name:
        base_name = args.name
    else:
        base_name = os.path.splitext(
            os.path.basename(args.input)
        )[0]

    output_video_path = os.path.join(
        args.output_dir,
        f"{base_name}_generated_video.mp4",
    )

    output_h5_path = os.path.join(
        args.output_dir,
        f"{base_name}_generated_events.h5",
    )

    generate_events_from_video(
        input_video=args.input,
        output_video_path=output_video_path,
        output_h5_path=output_h5_path,
        threshold=args.threshold,
        min_pixels=args.min_pixels,
        generate_video=args.generate_video,
    )

if __name__ == "__main__":
    main()