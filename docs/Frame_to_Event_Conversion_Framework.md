# Frame-to-Event Conversion Framework

**Author:** Zarco Romero, José Antonio  
**Repository:** https://github.com/jzarcoo/Frame-to-Event-Conversion-Framework.git

---

# 1. Background

Spiking Neural Networks (SNNs) perform best when processing sparse, asynchronous events from event cameras (DVS/DAVIS), rather than dense RGB pixels.

However, we face a data limitation: event camera datasets are limited. Traditional RGB video is abundant.

The solution: Convert RGB video sequences into event streams.

This allows SNNs to be trained using existing RGB data. Subsequently, the same trained model can be reused with a real event sensor, resulting in a single model compatible with heterogeneous sensors (RGB + events).

Public datasets such as MVSEC and DSEC provide RGB frames synchronized with real-world events (ground-truth), allowing for direct validation of any conversion framework.

---

# 2. Project Objective

Develop a lightweight, classic framework (not based on deep learning) that converts RGB video into an event stream in the format `(x,y,t,p)`.

## Core Concept

Compare the current frame to the previous one at the pixel level and emit an event where the intensity change exceeds a configurable threshold:

- **Positive polarity (`p = 1`)**: Increase in brightness.
- **Negative polarity (`p = -1`)**: Decrease in brightness.

---

# 3. Project Phases

## Phase 1: Literature Review

Header to document the state of the art and justify the chosen approach.

### ESIM

ESIM simulates event cameras by operating in the log-intensity domain, where an asynchronous event is triggered whenever the per-pixel brightness change exceeds a specified contrast threshold.

To accurately approximate this continuous level-crossing process without resorting to inefficient, fixed high-rate rendering, the framework tightly couples a 3D rendering engine with the event simulator. By evaluating the continuous camera trajectory and generating dense motion fields at each step, the system dynamically predicts the optimal timestamp for the next frame. This allows the simulator to sample heavily during rapid transient states and conserve computational resources when the scene is static.

This adaptive sampling is governed by two core mathematical strategies:

1. The first calculates the maximum expected rate of brightness change using a first-order Taylor expansion of the brightness constancy assumption.
2. A simplified alternative directly bounds the maximum pixel displacement.

To bridge the sim-to-real gap, ESIM incorporates physical sensor non-idealities. It models the contrast threshold as a Gaussian-distributed variable rather than a static value and supports independent, asymmetric thresholds for positive and negative events to faithfully replicate real-world hardware electronic biases.

The experiments in the article demonstrate that the adaptive method based on optical flow reduces simulation time and the number of frames needed to achieve the same level of accuracy (RMSE) as uniform sampling by up to 60%.

### v2e

v2e is a video-to-event conversion framework that generates synthetic DVS events directly from conventional RGB videos.

The pipeline first converts RGB frames into luma images and optionally increases their temporal resolution using Super-SloMo interpolation, producing intermediate frames that better approximate the continuous evolution of the scene.

The resulting intensities are then mapped to the logarithmic domain, since event cameras respond to relative brightness changes rather than absolute intensity values.

Events are generated whenever the accumulated change in log-intensity exceeds a positive or negative contrast threshold, closely mimicking the operation of a real DVS pixel.

To improve realism, v2e incorporates several sensor non-idealities:

- It models the finite bandwidth of photoreceptors through an intensity-dependent low-pass filter.
- It introduces pixel-to-pixel threshold variations using Gaussian-distributed contrast thresholds.
- It simulates hot pixels, leak events, and temporal shot noise through stochastic processes.

Unlike ESIM, which relies on a 3D rendering engine and adaptive rendering strategies, v2e operates directly on recorded video sequences, making it a practical and computationally efficient approach for converting existing image datasets into event-based data.

### Selected Approach

Based on this review, we chose a lightweight classical approach inspired by v2e.

Our goal is not to maximize physical realism, but to provide an efficient RGB-to-event conversion framework that captures the core behavior of event cameras while remaining computationally inexpensive and easy to apply to existing video datasets.

---

## Phase 2: Framework Implementation

Programming Language: Python

Each RGB frame is converted to grayscale.

The processed frame is compared with the last event frame, and pixels whose intensity change exceeds a predefined threshold are marked as events.

If enough pixels have changed, an event stream `(x, y, t, p)` is generated, where:

- `x` and `y` are the pixel coordinates.
- `t` is the timestamp.
- `p` is the polarity indicating whether the intensity increased or decreased.

The reference frame is updated with the detected changes, and the process repeats for the next frame.

The resulting events are rendered and saved as a video for visualization.

---

## Phase 3: Evaluation

---

## Phase 4: Documentation

---

# 4. Additional Notes

## References

* Delbruck, T., Hu, Y., & Liu, S. (2021). *v2e: From Video Frames to Realistic DVS Events*. Institute of Neuroinformatics, University of Zürich and ETH Zürich, Switzerland. 

* Gehrig, D., Rebecq, H., & Scaramuzza, D. (2018). *ESIM: an Open Event Camera Simulator*. Robotics and Perception Group, Depts. Informatics and Neuroinformatics, University of Zurich and ETH Zurich.