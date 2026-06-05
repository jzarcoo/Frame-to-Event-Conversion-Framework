import cv2
import numpy as np

# BGR color format for OpenCV
BLUE = (255, 0, 0)
RED = (0, 0, 255)

def process_frame(frame):
    """
    Convert the input frame to grayscale and apply logarithmic transformation.
    Args:        
        frame (numpy.ndarray): Input color image.
    Returns:        
        numpy.ndarray: Logarithmically transformed grayscale image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    log_image = np.log(gray + 1.0) 
    return log_image

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