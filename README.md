# Frame-to-Event Conversion Framework

Develop a lightweight, classical (non-learning-based) framework that converts RGB video into an event stream $(x, y, t, p)$.

* [RGB Video](https://drive.google.com/file/d/1msXFjfhcXd0mQUxFB1uFG95AYDQHdoL0/view?usp=sharing)

* [Generated Event Video](https://drive.google.com/file/d/1-FwmC2GNNtcpFwTAkSFBYAakHmfJC1Fb/view?usp=sharing)

* [DSEC](https://drive.google.com/file/d/1CYu6uEyCvd8mNtiE88KFAfW66kLx6H4W/view?usp=sharing)

## Dataset

Use the [DSEC](https://dsec.ifi.uzh.ch/dsec-datasets/download/) public dataset that provides synchronized RGB frames and ground-truth events.

* For simplicity, we use the `thun_00_a` sequence
* Download the dataset and extract the RGB frames from the left camera.

```bash
.
├── data
│   └── thun_00_a
│       ├── thun_00_a_events_left
│       │   ├── events.h5
│       │   └── rectify_map.h5
│       ├── thun_00_a_images_rectified_left
│       │   ├── 000000.png
│       │   ├── 000001.png
│       │   ├── ...
│       │   └── 000238.png
```

The extracted image sequence can be converted into a video using

```sh
ffmpeg -framerate 20 -i %06d.png -c:v mpeg4 -q:v 1 thun_00_a.mp4
```

```bash
.
├── data
│   └── thun_00_a
│       ├── thun_00_a_images_rectified_left
│       │   ├── ...
│       │   └── thun_00_a.mp4
```

## How it works

Each RGB frame is converted to grayscale. The processed frame is compared with the last event frame, and pixels whose intensity change exceeds a predefined threshold are marked as events. If enough pixels have changed, an event stream \((x, y, t, p)\) is generated, where \(x\) and \(y\) are the pixel coordinates, \(t\) is the timestamp, and \(p\) is the polarity indicating whether the intensity increased or decreased. The reference frame is updated with the detected changes, and the process repeats for the next frame. The resulting events are rendered and saved as a video for visualization.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jzarcoo/Frame-to-Event-Conversion-Framework.git
```
2. Navigate to the project directory:
```bash
cd Frame-to-Event-Conversion-Framework
```
3. Install the required dependencies:
```bash
uv sync
```

## CLI

The framework provides a command-line interface for converting RGB videos into event streams.

### Arguments

| Argument | Description |
|-----------|-------------|
| `-i`, `--input` | Input RGB video path |
| `-o`, `--output-dir` | Output directory where generated files will be stored |
| `-t`, `--threshold` | Intensity-change threshold used to trigger events |
| `-m`, `--min-pixels` | Minimum number of motion pixels required to generate an event frame |
| `-n`, `--name` | Base name for generated output files |
| `--video` / `--no-video` | Enable or disable output video generation. |
| `--timestamps`           | Optional timestamp file containing frame timestamps in microseconds                  |
| `--resize WIDTH HEIGHT`  | Resize output frames to the specified width and height before processing             |

### Example

```bash
uv run python -m framework.cli -i data/thun_00_a/thun_00_a_images_rectified_left/thun_00_a.mp4 -o results -t 3 -m 0 --resize 640 480 --video
```

This command generates:

```text
results/
├── thun_00_a_events.h5
└── thun_00_a_video.mp4
```
