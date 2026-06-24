import cv2
import h5py
import hdf5plugin
import numpy as np
import sys

from framework.utils.eventslicer import EventSlicer

# BGR color format for OpenCV
BLUE = (255, 0, 0)
RED = (0, 0, 255)

def process_frame(frame):
    """
    Convert the input frame to grayscale.
    Args:        
        frame (numpy.ndarray): Input color image.
    Returns:        
        numpy.ndarray: Grayscale image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # log_image = np.log1p(gray)
    return gray

def build_event_image(events, height, width):
    """
    Build an event image from a list of events (BGR color format)
    Args:
        events (numpy.ndarray): Array of events, where each event is represented as [x, y, timestamp, polarity].
        height (int): Height of the output image.
        width (int): Width of the output image.
    Returns:
        numpy.ndarray: Event image with positive events in red and negative events in blue.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if events is None or len(events) == 0:
        return img
    events = np.atleast_2d(events)
    for event in events:
        x, y, _, p = event
        x = int(x)
        y = int(y)
        if 0 <= x < width and 0 <= y < height:
            if p > 0:
                img[y, x] = RED   
            else:
                img[y, x] = BLUE  
    return img

def dsec_events_to_video(h5_path, output_video_path, width=640, height=480, fps=20):
    """
    Convert DSEC events from an HDF5 file to a video file.
    Args:
        h5_path (str): Path to the input HDF5 file containing DSEC events.
        output_video_path (str): Path to the output video file.
        width (int): Width of the output video frames.
        height (int): Height of the output video frames.
        fps (int): Frames per second for the output video.
    """
    with h5py.File(h5_path, 'r') as f:
        slicer = EventSlicer(f)
        t_start = slicer.get_start_time_us()
        t_end = slicer.get_final_time_us()
        dt_us = int(1e6 / fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))        
        total_frames = int((t_end - t_start) / dt_us)        
        current_time_us = t_start        
        print(f"Generating event video: {total_frames} frames at {fps} FPS...")
        for frame_idx in range(total_frames):
            next_time_us = current_time_us + dt_us
            evs = slicer.get_events(current_time_us, next_time_us)
            if evs is None or evs['x'].size == 0:
                events_matrix = np.empty((0, 4))
            else:
                events_matrix = np.stack((evs['x'], evs['y'], evs['t'], evs['p']), axis=-1)
            frame = build_event_image(events_matrix, height, width)
            video_writer.write(frame)
            current_time_us = next_time_us
            # Progress display
            progress = (frame_idx + 1) / total_frames * 100
            #sys.stdout.write(f"\rProcesando: {progress:.1f}% [{frame_idx + 1}/{total_frames}]")
            sys.stdout.write(f"\rGenerating video: {progress:.1f}% [{frame_idx + 1}/{total_frames}]")
            sys.stdout.flush()
        video_writer.release()
        sys.stdout.write("\n")
        print(f"Video saved successfully at: {output_video_path}")
        video_writer.release()