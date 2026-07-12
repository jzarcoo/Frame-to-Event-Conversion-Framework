import cv2
import numpy as np
import os
import sys

from framework.utils.h5_functions import EventH5Writer
from framework.utils.images import (
    process_frame,
    build_event_image,
)

class EventGenerator:
    """Generates events from RGB videos based on frame differences and saves them in HDF5 format (compatible with DSEC)."""
    def __init__(
        self,
        threshold=0.2,
        min_pixels=500,
        resize_width=None,
        resize_height=None,
    ):
        """Initialize the EventGenerator with the specified parameters.
        Args:
            threshold (float): Minimum intensity change required to generate an event.
            min_pixels (int): Minimum number of pixels that must change to generate events for a frame.
            resize_width (int, optional): If specified, resize frames to this width before processing.
            resize_height (int, optional): If specified, resize frames to this height before processing.
        """
        self.threshold = threshold
        self.min_pixels = min_pixels
        self.resize_width = resize_width
        self.resize_height = resize_height

    def _resize(self, frame):
        """Resize the input frame if resize dimensions are specified."""
        if (
            self.resize_width is not None
            and self.resize_height is not None
        ):
            return cv2.resize(
                frame,
                (self.resize_width, self.resize_height),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _compute_events(
        self,
        previous,
        current,
        timestamp_us,
    ):
        """
        Compute events based on the difference between the current and previous frames.
        Args:
            previous (numpy.ndarray): Previous frame (grayscale, processed).
            current (numpy.ndarray): Current frame (grayscale, processed).
            timestamp_us (int): Timestamp in microseconds for the current frame.
        Returns:
            numpy.ndarray: Array of events with shape (N, 4) where each event is represented as [x, y, t, p].
        """
        diff = current.astype(np.float32) - previous.astype(np.float32)
        mask = np.abs(diff) > self.threshold
        pixels = np.argwhere(mask)

        if len(pixels) <= self.min_pixels:
            return None

        rows = pixels[:, 0]
        cols = pixels[:, 1]
        polarities = np.where(
            diff[rows, cols] > 0,
            1,
            -1,
        )

        return np.column_stack((
            cols,
            rows,
            np.full(
                len(rows),
                timestamp_us,
                dtype=np.uint32,
            ),
            polarities,  
        ))

    def _timestamp_us(
        self,
        frame_idx,
        fps,
        timestamps,
    ):
        """Get the timestamp in microseconds for the given frame index."""
        if timestamps is not None:
            return int(timestamps[frame_idx] - timestamps[0])
        return int((frame_idx / fps) * 1e6)

    def generate(
        self,
        input_video,
        output_h5_path,
        output_video_path=None,
        timestamps_path=None,
    ):
        """
        Generate events from the input video and save them in HDF5 format. Optionally, also save a video visualizing the events.
        Args:
            input_video (str): Path to the input RGB video.
            output_h5_path (str): Path to save the output HDF5 file containing events.
            output_video_path (str, optional): Path to save the output video visualizing events. If None, no video will be saved.
            timestamps_path (str, optional): Path to a text file containing timestamps for each frame in microseconds. If None, timestamps will be generated based on frame index and FPS.
        Returns:
            int: Total number of events generated.
        """
        kernel_size = 5 # adjustable parameter for Gaussian blur
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Video is empty.")
        frame = self._resize(frame)
        height, width = frame.shape[:2]
        timestamps = None
        if timestamps_path:
            if not os.path.isfile(timestamps_path):
                raise FileNotFoundError(timestamps_path)
            timestamps = np.loadtxt(timestamps_path, dtype=np.uint64)
        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (width, height),
            )
        # Grayscale conversion
        previous = process_frame(frame)
        previous = cv2.GaussianBlur(
            previous,
            (kernel_size, kernel_size),
            0,
        )
        frame_idx = 0
        writer = EventH5Writer(output_h5_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._resize(frame)
            current = process_frame(frame)
            current = cv2.GaussianBlur(
                current,
                (kernel_size, kernel_size),
                0,
            )
            timestamp_us = self._timestamp_us(frame_idx, fps, timestamps)
            events = self._compute_events(previous, current, timestamp_us)
            if events is None:
                previous = current
                frame_idx += 1
                continue
            writer.add_events(events)
            if out is not None:
                event_frame = build_event_image(events, height, width)
                out.write(event_frame)
            previous = current
            frame_idx += 1
            # Progress display
            progress = (frame_idx + 1) / total_frames * 100
            print(
                f"\rGenerating events: {progress:.1f}% "
                f"[{frame_idx + 1}/{total_frames}]",
                end="",
                flush=True
            )
        print()
        cap.release()
        if out is not None:
            out.release()
        writer.finalize()
        return writer.total_events