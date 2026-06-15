import argparse
import cv2
import numpy as np
import os

try:
    from images import process_frame, build_event_image
except ImportError:
    print("[Error] Could not import 'images' module.")
    exit(1)

try:
    from utils import save_h5
except ImportError:
    print("[Error] Could not import 'hdf5' module.")
    exit(1)


def generate_events_from_video(
    input_video,
    output_video_path,
    output_npy_path,
    output_h5_path,
    threshold,
    min_pixels,
):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_video}"
        )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
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
        out.release()
        raise RuntimeError("Video is empty or first frame cannot be read.")
    last_event_image = process_frame(frame)
    generated_events = []
    rendered_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_image = process_frame(frame)
        # Noise reduction
        current_image_blurred = cv2.GaussianBlur(
            current_image,
            (7, 7),
            0,
        )
        diff = current_image_blurred.astype(np.float32) - last_event_image.astype(np.float32)
        motion_mask = np.abs(diff) > threshold
        motion_pixels = np.argwhere(motion_mask)

        if len(motion_pixels) <= min_pixels:
            continue

        timestamp_us = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)
        rows = motion_pixels[:, 0]
        cols = motion_pixels[:, 1]
        polarities = np.where(
            diff[rows, cols] > 0,
            1,
            -1,
        )
        events = np.column_stack((
                cols,
                rows,
                np.full(
                    len(rows),
                    timestamp_us,
                    dtype=np.uint64,
                ),
                polarities,
            ))
        generated_events.extend(events)

        last_event_image[rows, cols] = current_image[rows, cols]

        event_img = build_event_image(
            events,
            height,
            width,
        )
        out.write(event_img)
        rendered_frames += 1

        # if rendered_frames > 230:
        #     print("force to stop after 230 frames for testing")
        #     break

    # Save files
    cap.release()
    out.release()

    if len(generated_events) == 0:
        print("[WARNING] No events were generated.")
        return

    generated_events = np.array(generated_events)
    # np.save(output_npy_path, generated_events)
    save_h5(generated_events, output_h5_path)

    print(f"[INFO] Rendered event frames: {rendered_frames}")
    print(f"[INFO] Total events generated: {len(generated_events)}")


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

    output_npy_path = os.path.join(
        args.output_dir,
        f"{base_name}_generated_events.npy",
    )

    output_h5_path = os.path.join(
        args.output_dir,
        f"{base_name}_generated_events.h5",
    )

    generate_events_from_video(
        input_video=args.input,
        output_video_path=output_video_path,
        output_npy_path=output_npy_path,
        output_h5_path=output_h5_path,
        threshold=args.threshold,
        min_pixels=args.min_pixels,
    )

if __name__ == "__main__":
    main()