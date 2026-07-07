import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from framework.src.framework.utils.dsec.eventslicer import EventSlicer
from framework.utils.images import (
    build_event_image, 
    # dsec_events_to_video,
)

from .metrics import compute_metrics
from .visualization import (
    plot_metrics,
    show_pair,
)
from .windows import load_windows

def events_to_frame(
    slicer,
    start_us,
    end_us,
    width,
    height,
):
    """
    Convert events within a specified time window to a frame image.
    Args:
        slicer (EventSlicer): An instance of EventSlicer to extract events.
        start_us (int): Start timestamp in microseconds.
        end_us (int): End timestamp in microseconds.
        width (int): Width of the output frame.
        height (int): Height of the output frame.
    Returns:
        np.ndarray: Frame image constructed from events, or False if no events are found.
    """
    evs = slicer.get_events(start_us, end_us)
    if evs is None or len(evs["x"]) == 0:
        return False
    events = np.column_stack((
            evs["x"],
            evs["y"],
            evs["t"],
            evs["p"],
    ))
    return build_event_image(events, height, width)


def get_common_time_range(
    gt_slicer,
    pred_slicer,
):
    """
    Get the common time range between ground truth and predicted events.
    Args:
        gt_slicer (EventSlicer): Slicer for ground truth events.
        pred_slicer (EventSlicer): Slicer for predicted events.
    Returns:
        tuple: (common_start_us, common_end_us) representing the common time range in microseconds.
    """
    return (
        max(
            gt_slicer.get_start_time_us(),
            pred_slicer.get_start_time_us(),
        ),
        min(
            gt_slicer.get_final_time_us(),
            pred_slicer.get_final_time_us(),
        )
    )


class EventEvaluator:
    """
    Class to evaluate predicted events against ground truth events.
    """
    def __init__(
        self,
        gt_h5,
        pred_h5,
        output_dir,
        width=640,
        height=480,
    ):
        """
        Initialize the EventEvaluator.
        Args:
            gt_h5 (str): Path to the ground truth HDF5 file.
            pred_h5 (str): Path to the predicted events HDF5 file.
            output_dir (str): Directory to save evaluation results.
            width (int): Width of the output frames.
            height (int): Height of the output frames.
        """
        self.gt_h5 = gt_h5
        self.pred_h5 = pred_h5
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.width = width
        self.height = height

    def _progress(self, current, total, prefix="Progress", start_us=None, end_us=None):
        """
        Display progress information.
        Args:
            current (int): Current progress count.
            total (int): Total count for completion.
            prefix (str): Prefix message for the progress display.
            start_us (int, optional): Start timestamp in microseconds for the current window/frame.
            end_us (int, optional): End timestamp in microseconds for the current window/frame.
        """
        pct = 100.0 * current / total
        if start_us is not None and end_us is not None:
            print(f"\r{prefix}: {current}/{total} ({pct:.2f}%) [{start_us}-{end_us} us]", end="")
        else:
            print(f"\r{prefix}: {current}/{total} ({pct:.2f}%)", end="")

    def _save_summary(
        self,
        results,
        skipped,
        prefix="window",
    ):
        """
        Save a summary of the evaluation results, including metrics and visualizations.
        Args:
            results (list): List of dictionaries containing evaluation metrics for each window.
            skipped (int): Number of skipped windows due to errors or invalid data.
            prefix (str): Prefix for the output files.
        """
        if len(results) == 0:
            print("No valid evaluations.")
            return
        plot_metrics(
            results,
            save_path=os.path.join(
                self.output_dir,
                f"{prefix}_metrics.png"
            )
        )
        metrics_list = ["mse", "mae", "ssim", "psnr", "lpips"]
        summary_dir = os.path.join(self.output_dir, f"{prefix}_summaries")
        os.makedirs(summary_dir, exist_ok=True)
        print()
        print(f"Valid evaluations : {len(results)}")
        print(f"Skipped           : {skipped}")
        print("-" * 50)
        print(f"{'METRIC':<8} | {'BEST VALUE':<12} | {'WORST VALUE':<12}")
        print("-" * 50)
        for metric in metrics_list:
            lower_is_better = metric in ["mse", "mae", "lpips"]
            if lower_is_better:
                best_res = min(results, key=lambda r: r[metric])
                worst_res = max(results, key=lambda r: r[metric])
            else:
                best_res = max(results, key=lambda r: r[metric])
                worst_res = min(results, key=lambda r: r[metric])
            print(f"{metric.upper():<8} | {best_res[metric]:<12.4f} | {worst_res[metric]:<12.4f}")
            save_path = os.path.join(summary_dir, f"{metric}_summary_quad.png")
            self._save_metric_quad(best_res, worst_res, metric, save_path)

    def _save_metric_quad(
        self,
        best_res,
        worst_res,
        metric_name,
        save_path,
    ):
        """
        Save a 2x2 grid of images showing the best and worst frames for a specific metric.
        Args:
            best_res (dict): Dictionary containing the best evaluation result for the metric.
            worst_res (dict): Dictionary containing the worst evaluation result for the metric.
            metric_name (str): Name of the metric being evaluated.
            save_path (str): Path to save the resulting image.
        """
        with h5py.File(self.gt_h5, "r") as gt_file, \
             h5py.File(self.pred_h5, "r") as pred_file:
            gt_slicer = EventSlicer(gt_file)
            pred_slicer = EventSlicer(pred_file)
            best_gt_img = events_to_frame(
                gt_slicer, best_res["start_us"], best_res["end_us"], self.width, self.height
            )
            best_pred_img = events_to_frame(
                pred_slicer, best_res["start_us"], best_res["end_us"], self.width, self.height
            )
            worst_gt_img = events_to_frame(
                gt_slicer, worst_res["start_us"], worst_res["end_us"], self.width, self.height
            )
            worst_pred_img = events_to_frame(
                pred_slicer, worst_res["start_us"], worst_res["end_us"], self.width, self.height
            )

            fallback_shape = (self.height, self.width, 3)
            if best_gt_img is False: best_gt_img = np.zeros(fallback_shape, dtype=np.uint8)
            if best_pred_img is False: best_pred_img = np.zeros(fallback_shape, dtype=np.uint8)
            if worst_gt_img is False: worst_gt_img = np.zeros(fallback_shape, dtype=np.uint8)
            if worst_pred_img is False: worst_pred_img = np.zeros(fallback_shape, dtype=np.uint8)

            fig, axs = plt.subplots(2, 2, figsize=(14, 12))

            # Best Frame
            axs[0, 0].imshow(cv2.cvtColor(best_gt_img, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title(f"Best GT\nTime: {best_res['start_us']}-{best_res['end_us']} us", fontsize=10)
            axs[0, 0].axis("off")

            axs[0, 1].imshow(cv2.cvtColor(best_pred_img, cv2.COLOR_BGR2RGB))
            axs[0, 1].set_title(f"Best Prediction\n{metric_name.upper()}: {best_res[metric_name]:.4f}", fontsize=10)
            axs[0, 1].axis("off")
            
            # Worst Frame
            axs[1, 0].imshow(cv2.cvtColor(worst_gt_img, cv2.COLOR_BGR2RGB))
            axs[1, 0].set_title(f"Worst GT\nTime: {worst_res['start_us']}-{worst_res['end_us']} us", fontsize=10)
            axs[1, 0].axis("off")

            axs[1, 1].imshow(cv2.cvtColor(worst_pred_img, cv2.COLOR_BGR2RGB))
            axs[1, 1].set_title(f"Worst Prediction\n{metric_name.upper()}: {worst_res[metric_name]:.4f}", fontsize=10)
            axs[1, 1].axis("off")

            plt.suptitle(
                f"Metric Summary: {metric_name.upper()}\n"
                f"Best: {best_res[metric_name]:.4f} vs Worst: {worst_res[metric_name]:.4f}",
                fontsize=16,
                fontweight="bold"
            )
            
            plt.tight_layout()
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

    def evaluate_window(
        self,
        start_us,
        end_us,
        visualize=True,
        save_path=None,
    ):
        """
        Evaluate the predicted events against ground truth events for a specific time window.
        Args:
            start_us (int): Start timestamp in microseconds.
            end_us (int): End timestamp in microseconds.
            visualize (bool): Whether to visualize the results.
            save_path (str, optional): Path to save the visualization. If None, the visualization is not saved.
        Returns:
            dict: Dictionary containing evaluation metrics (MSE, MAE, SSIM, PSNR, LPIPS) for the specified time window.
        """
        with h5py.File(self.gt_h5, "r") as gt_file, \
             h5py.File(self.pred_h5, "r") as pred_file:
            gt_slicer = EventSlicer(gt_file)
            pred_slicer = EventSlicer(pred_file)
            common_start, common_end = get_common_time_range(gt_slicer, pred_slicer)
            if start_us < common_start or end_us > common_end:
                print(f"Window {start_us}-{end_us} us is out of common range {common_start}-{common_end} us. Skipping.")
                return None
            gt_img = events_to_frame(
                gt_slicer,
                start_us,
                end_us,
                self.width,
                self.height,
            )
            pred_img = events_to_frame(
                pred_slicer,
                start_us,
                end_us,
                self.width,
                self.height,
            )
            if gt_img is False or pred_img is False:
                print(f"No events found in window {start_us}-{end_us} us in {'GT' if gt_img is False else 'Prediction'}. Skipping.")
                return None
            metrics = compute_metrics(gt_img, pred_img)
            if visualize:
                title = (
                    f"Time: {start_us} - {end_us} us\n"
                    f"SSIM: {metrics['ssim']:.4f} | PSNR: {metrics['psnr']:.2f} | LPIPS: {metrics['lpips']:.4f}\n"
                    f"MSE: {metrics['mse']:.2f} | MAE: {metrics['mae']:.2f}"
                )
                show_pair(
                    gt_img,
                    pred_img,
                    title=title,
                    save_path=save_path
                )
            return metrics

    def evaluate_windows_file(
        self,
        windows_file,
    ):
        """
        Evaluate the predicted events against ground truth events using a windows file.
        Args:
            windows_file (str): Path to the windows file containing start and end timestamps in microseconds.
                                The format of the file should be: start_us,end_us (one window per line).
        Returns:
            list: List of dictionaries containing evaluation metrics for each window.
        """
        windows = load_windows(windows_file)
        results = []
        skipped = 0
        with h5py.File(self.gt_h5, "r") as gt_file, \
             h5py.File(self.pred_h5, "r") as pred_file:
            gt_slicer = EventSlicer(gt_file)
            pred_slicer = EventSlicer(pred_file)
            common_start, common_end = get_common_time_range(gt_slicer, pred_slicer)
        windows_dir = os.path.join(self.output_dir, "windows")
        os.makedirs(windows_dir, exist_ok=True)
        total_windows = len(windows)
        for idx, (start_us, end_us) in enumerate(windows):
            if start_us < common_start or end_us > common_end or end_us <= start_us:
                skipped += 1
                continue
            try:
                window_save_path = os.path.join(
                    windows_dir,
                    f"window_{idx:04d}_{start_us}_{end_us}.png"
                )                
                metrics = self.evaluate_window(
                    start_us,
                    end_us,
                    visualize=True,
                    save_path=window_save_path
                )
                if metrics is None:
                    skipped += 1
                    continue
            except AssertionError:
                skipped += 1
                continue
            results.append({
                "start_us": start_us,
                "end_us": end_us,
                **metrics
            })
            self._progress(idx + 1, total_windows, "Windows", start_us, end_us)
        print()
        self._save_summary(results, skipped, prefix="window")
        return results

    def evaluate_frame_by_frame(
        self,
        frame_time_us=50000,
        start_us=None,
        end_us=None,
    ):
        """
        Evaluate the predicted events against ground truth events in a frame-by-frame manner.
        Args:
            frame_time_us (int): Time window for each frame in microseconds.
        Returns:
            list: List of dictionaries containing evaluation metrics for each frame.
        """
        # results stores the metrics for each frame
        results = []
        skipped = 0
        with h5py.File(self.gt_h5, "r") as gt_file, \
             h5py.File(self.pred_h5, "r") as pred_file:
            gt_slicer = EventSlicer(gt_file)
            pred_slicer = EventSlicer(pred_file)
            common_start, common_end = get_common_time_range(gt_slicer, pred_slicer)
        current = common_start + (start_us if start_us is not None else 0)
        limit = common_end if end_us is None else min(common_end, common_start + end_us)
        frame_idx = 0
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        total_frames = max(0, (limit - current) // frame_time_us)
        while current + frame_time_us <= limit:
            try:
                frame_save_path = os.path.join(
                    frames_dir,
                    f"frame_{frame_idx:04d}_{current}_{current + frame_time_us}.png"
                )
                metrics = self.evaluate_window(
                    current,
                    current + frame_time_us,
                    visualize=True,
                    save_path=frame_save_path
                )
                if metrics is None:
                    skipped += 1
                else:
                    results.append({
                        "start_us": current,
                        "end_us": current + frame_time_us,
                        **metrics
                    })
            except Exception:
                skipped += 1
            frame_idx += 1
            current += frame_time_us
            self._progress(frame_idx, total_frames, "Frames", current, current + frame_time_us)
        print()
        self._save_summary(results, skipped, prefix="frame")
        return results