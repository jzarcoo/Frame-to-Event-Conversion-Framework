# uv run python -m framework.utils.dsec.visualize_detection

from pathlib import Path
import cv2
import numpy as np

working_directory = Path.cwd()
sequence = working_directory / "data" / "dsec" / "test" / "zurich_city_13_a"

timestamps = np.loadtxt(sequence / "images/timestamps.txt", dtype=np.int64)
images = sorted((sequence / "images/left/distorted").glob("*.png"))
tracks = np.load(sequence / "object_detections/left/tracks.npy")

frame_idx = 361

timestamp = timestamps[frame_idx]

image = cv2.imread(str(images[frame_idx]))

boxes = tracks[tracks["t"] == timestamp]

for box in boxes:
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["w"])
    h = int(box["h"])

    cv2.rectangle(image,
                  (x, y),
                  (x + w, y + h),
                  (0,255,0),
                  2)

    cls = int(box["class_id"])
    tid = int(box["track_id"])

    cv2.putText(image,
                f"{cls}:{tid}",
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1)

cv2.imshow("Bounding Boxes", image)
cv2.waitKey(0)