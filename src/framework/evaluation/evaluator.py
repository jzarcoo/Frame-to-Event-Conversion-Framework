import gc
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback

from framework.utils.dsec.eventslicer import EventSlicer
from framework.utils.images import (
    build_event_image, 
)

from .metrics import compute_metrics
from .visualization import (
    plot_histograms_count,
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

def events_to_frame(
    events, 
    width,
    height,
):
    """
    Convert a list of events to a frame image.
    Args:
        events
        width (int): Width of the output frame.
        height (int): Height of the output frame.
    Returns:
        np.ndarray: Frame image constructed from events, or False if no events are found.
    """
    if events is None or len(events) == 0:
        return False
    events = np.column_stack((
        events["x"],
        events["y"],
        events["t"],
        events["p"],
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
    common_start = max(
        gt_slicer.get_start_time_us(),
        pred_slicer.get_start_time_us(),
    )
    common_end = min(
        gt_slicer.get_final_time_us(),
        pred_slicer.get_final_time_us(),
    )
    return common_start, common_end

def filter_polarity(evs, polarity_value):
    """
    Filters events based on their polarity.

    Args:
        evs (dict): Dictionary containing event data with keys 'x', 'y', 't', 'p'.
        polarity_value (int): Polarity value to filter by (1 for ON, 0 for OFF).
    """
    if evs is None or len(evs.get('x', [])) == 0:
        return {'x': np.array([]), 'y': np.array([]), 't': np.array([]), 'p': np.array([])}
    mask = evs['p'] == polarity_value
    return {k: v[mask] for k, v in evs.items()}

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
        self.width = width
        self.height = height

        os.makedirs(output_dir, exist_ok=True)
        self.gt_file = h5py.File(gt_h5, "r")
        self.pred_file = h5py.File(pred_h5, "r")
        self.gt_slicer = EventSlicer(self.gt_file)
        self.pred_slicer = EventSlicer(self.pred_file)

        grid_percent = 0.05
        self.grid_size = max(1, round(min(width, height) * grid_percent))
        print("Grid size for coverage evaluation:", self.grid_size)
        self.coverage_threshold = 0
        self.x_bins = np.linspace(0, width, width//self.grid_size + 1)
        self.y_bins = np.linspace(0, height, height//self.grid_size + 1)

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
        def json_converter(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        """
        Save raw analysis results for later processing.
        """
        analysis = {
            "valid": len(results),
            "skipped": skipped,
            "results": results,
        }
        save_path = os.path.join(
            self.output_dir,
            "analysis.json",
        )
        with open(save_path, "w") as f:
            json.dump(
                analysis, 
                f, 
                indent=4,
                default=json_converter
            )
        print(f"Analysis saved to {save_path}")
        print(f"Valid evaluations : {len(results)}")
        print(f"Skipped           : {skipped}")

        

    # def _save_summary(
    #     self,
    #     results,
    #     skipped,
    #     prefix="window",
    # ):
    #     """
    #     Save a summary of the evaluation results, including metrics and visualizations.
    #     Args:
    #         results (list): List of dictionaries containing evaluation metrics for each window.
    #         skipped (int): Number of skipped windows due to errors or invalid data.
    #         prefix (str): Prefix for the output files.
    #     """
    #     if len(results) == 0:
    #         print("No valid evaluations.")
    #         return
    #     plot_metrics(
    #         results,
    #         save_path=os.path.join(
    #             self.output_dir,
    #             "metrics.png"
    #         )
    #     )
    #     metrics_list = ["mse", "mae", "ssim", "psnr", "lpips"]
    #     summary_dir = os.path.join(self.output_dir, "summaries")
    #     os.makedirs(summary_dir, exist_ok=True)
    #     print()
    #     print(f"Valid evaluations : {len(results)}")
    #     print(f"Skipped           : {skipped}")
    #     print("-" * 50)
    #     print(f"{'METRIC':<8} | {'BEST VALUE':<12} | {'WORST VALUE':<12}")
    #     print("-" * 50)
    #     for metric in metrics_list:
    #         lower_is_better = metric in ["mse", "mae", "lpips"]
    #         if lower_is_better:
    #             best_res = min(results, key=lambda r: r[metric])
    #             worst_res = max(results, key=lambda r: r[metric])
    #         else:
    #             best_res = max(results, key=lambda r: r[metric])
    #             worst_res = min(results, key=lambda r: r[metric])
    #         print(f"{metric.upper():<8} | {best_res[metric]:<12.4f} | {worst_res[metric]:<12.4f}")
    #         save_path = os.path.join(summary_dir, f"{metric}_summary_quad.png")
    #         self._save_metric_quad(best_res, worst_res, metric, save_path)

    # def _save_metric_quad(
    #     self,
    #     best_res,
    #     worst_res,
    #     metric_name,
    #     save_path,
    # ):
    #     """
    #     Save a 2x2 grid of images showing the best and worst frames for a specific metric.
    #     Args:
    #         best_res (dict): Dictionary containing the best evaluation result for the metric.
    #         worst_res (dict): Dictionary containing the worst evaluation result for the metric.
    #         metric_name (str): Name of the metric being evaluated.
    #         save_path (str): Path to save the resulting image.
    #     """
    #     best_gt_img = events_to_frame(
    #         self.gt_slicer, best_res["start_us"], best_res["end_us"], self.width, self.height
    #     )
    #     best_pred_img = events_to_frame(
    #         self.pred_slicer, best_res["start_us"], best_res["end_us"], self.width, self.height
    #     )
    #     worst_gt_img = events_to_frame(
    #         self.gt_slicer, worst_res["start_us"], worst_res["end_us"], self.width, self.height
    #     )
    #     worst_pred_img = events_to_frame(
    #         self.pred_slicer, worst_res["start_us"], worst_res["end_us"], self.width, self.height
    #     )

    #     fallback_shape = (self.height, self.width, 3)
    #     if best_gt_img is False: best_gt_img = np.zeros(fallback_shape, dtype=np.uint8)
    #     if best_pred_img is False: best_pred_img = np.zeros(fallback_shape, dtype=np.uint8)
    #     if worst_gt_img is False: worst_gt_img = np.zeros(fallback_shape, dtype=np.uint8)
    #     if worst_pred_img is False: worst_pred_img = np.zeros(fallback_shape, dtype=np.uint8)

    #     fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    #     # Best Frame
    #     axs[0, 0].imshow(cv2.cvtColor(best_gt_img, cv2.COLOR_BGR2RGB))
    #     axs[0, 0].set_title(f"Best GT\nTime: {best_res['start_us']}-{best_res['end_us']} us", fontsize=10)
    #     axs[0, 0].axis("off")

    #     axs[0, 1].imshow(cv2.cvtColor(best_pred_img, cv2.COLOR_BGR2RGB))
    #     axs[0, 1].set_title(f"Best Prediction\n{metric_name.upper()}: {best_res[metric_name]:.4f}", fontsize=10)
    #     axs[0, 1].axis("off")
        
    #     # Worst Frame
    #     axs[1, 0].imshow(cv2.cvtColor(worst_gt_img, cv2.COLOR_BGR2RGB))
    #     axs[1, 0].set_title(f"Worst GT\nTime: {worst_res['start_us']}-{worst_res['end_us']} us", fontsize=10)
    #     axs[1, 0].axis("off")

    #     axs[1, 1].imshow(cv2.cvtColor(worst_pred_img, cv2.COLOR_BGR2RGB))
    #     axs[1, 1].set_title(f"Worst Prediction\n{metric_name.upper()}: {worst_res[metric_name]:.4f}", fontsize=10)
    #     axs[1, 1].axis("off")

    #     plt.suptitle(
    #         f"Metric Summary: {metric_name.upper()}\n"
    #         f"Best: {best_res[metric_name]:.4f} vs Worst: {worst_res[metric_name]:.4f}",
    #         fontsize=16,
    #         fontweight="bold"
    #     )
        
    #     plt.tight_layout()
    #     fig.savefig(save_path, bbox_inches="tight")
    #     plt.close(fig)

    def _compute_histogram(self, evs):
        """
        Compute a 2D histogram of event counts for the given events.
        Args:
            evs (dict): Dictionary containing event data with keys 'x', 'y', 't', 'p'.
        Returns:
            tuple: (H, count) where H is the 2D histogram and count is the total number of events.
        """
        count = len(evs["x"]) if evs is not None else 0
        H, _, _ = np.histogram2d(
            evs["x"] if count else [],
            evs["y"] if count else [],
            bins=[self.x_bins, self.y_bins],
        )
        return H.astype(np.float32, copy=False), count

    def _coverage_statistics(self, coverage):
        """
        Compute coverage statistics for the given coverage array.
        Args:
            coverage (np.ndarray): Array of coverage values.
        """
        if len(coverage) == 0:
            return {
                "mean_cov": 0, "median_cov": 0, "std_cov": 0,
                "p25_cov": 0, "p75_cov": 0,
                "cov_gt_05": 0, "cov_gt_07": 0, "cov_gt_10": 0,
            }
        return {
            "mean_cov": np.mean(coverage),
            "median_cov": np.median(coverage),
            "std_cov": np.std(coverage),
            "p25_cov": np.percentile(coverage, 25),
            "p75_cov": np.percentile(coverage, 75),
            "cov_gt_05": np.mean(coverage >= 0.5),
            "cov_gt_07": np.mean(coverage >= 0.7),
            "cov_gt_10": np.mean(coverage >= 1.0),
        }

    def _classification_metrics(self, tp, fp, fn, tn):
        """
        Compute classification metrics based on true positives, false positives, false negatives, and true negatives.
        Args:
            tp (int): True positives count.
            fp (int): False positives count.
            fn (int): False negatives count.
            tn (int): True negatives count.
        Returns:
            tuple: (accuracy, precision, recall, f1) metrics.
        """
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return accuracy, precision, recall, f1

    def _compute_event_metrics(self, name, gt_evs, pred_evs, start_us, end_us):
        """
        Calculate detailed spatial coverage and compression metrics.
        Returns a dictionary with the numbers and coverage array for the histogram.

        Args:
            name (str): Name of the evaluation (e.g., "ON", "OFF", "ALL").
            gt_evs (dict): Ground truth events with keys 'x', 'y', 't', 'p'.
            pred_evs (dict): Predicted events with keys 'x', 'y', 't', 'p'.
            start_us (int): Start timestamp in microseconds for the evaluation window.
            end_us (int): End timestamp in microseconds for the evaluation window.
        """
        H_gt, gt_count = self._compute_histogram(gt_evs)
        H_pred, pred_count = self._compute_histogram(pred_evs)
        
        global_ratio = (pred_count / gt_count) if gt_count > 0 else 0.0
        normalized_coverage = H_pred / (H_gt * global_ratio + 1e-8)
        absolute_coverage = H_pred / (H_gt + 1e-8)

        normalized_coverage_gt = normalized_coverage[H_gt > 0]
        absolute_coverage_gt = absolute_coverage[H_gt > 0]

        TP_mask = (H_gt > 0) & (H_pred > 0)
        FN_mask = (H_gt > 0) & (H_pred == 0)
        FP_mask = (H_gt == 0) & (H_pred > 0)
        TN_mask = (H_gt == 0) & (H_pred == 0)

        occupied_gt = np.sum(H_gt > 0)
        occupied_pred = np.sum(H_pred > 0)
        
        tp = np.sum(TP_mask)
        fp = np.sum(FP_mask)
        fn = np.sum(FN_mask)
        tn = np.sum(TN_mask)

        del TP_mask
        del FP_mask
        del FN_mask
        del TN_mask

        accuracy, precision, recall, f1 = self._classification_metrics(tp, fp, fn, tn)
        
        metrics = {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "event_ratio": global_ratio,
            "occupied_gt": occupied_gt,
            "occupied_pred": occupied_pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        normalized_metrics = self._coverage_statistics(normalized_coverage_gt)
        absolute_metrics = self._coverage_statistics(absolute_coverage_gt)

        metrics.update({
            "normalized_coverage": normalized_metrics,
            "absolute_coverage": absolute_metrics,
        })

        # Plot the histograms side by side
        histograms_dir = os.path.join(self.output_dir, "histograms")
        os.makedirs(histograms_dir, exist_ok=True)
        name_dir = os.path.join(histograms_dir, name)
        os.makedirs(name_dir, exist_ok=True)
        plot_histograms_count(H_gt.T, H_pred.T, name, self.width, self.height, metrics, save_path=os.path.join(name_dir, f"histogram_{start_us}_{end_us}.png"))
        del H_pred
        del H_gt

        return metrics, normalized_coverage_gt, absolute_coverage_gt

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
        common_start, common_end = get_common_time_range(self.gt_slicer, self.pred_slicer)
        if start_us < common_start or end_us > common_end:
            print(f"Window {start_us}-{end_us} us is out of common range {common_start}-{common_end} us. Skipping.")
            return None
        # histograms
        gt_raw_evs = self.gt_slicer.get_events(start_us, end_us)
        pred_raw_evs = self.pred_slicer.get_events(start_us, end_us)
        # image metrics
        gt_img = events_to_frame(
            gt_raw_evs,
            self.width,
            self.height,
        )
        pred_img = events_to_frame(
            pred_raw_evs,
            self.width,
            self.height,
        )
        if gt_img is False or pred_img is False:
            print(f"No events found in window {start_us}-{end_us} us in {'GT' if gt_img is False else 'Prediction'}. Skipping.")
            return None

        image_metrics = compute_metrics(gt_img, pred_img)
        
        polarities = {
            "all": (gt_raw_evs, pred_raw_evs),
            "on": (
                filter_polarity(gt_raw_evs, 1),
                filter_polarity(pred_raw_evs, 1),
            ),
            "off": (
                filter_polarity(gt_raw_evs, 0),
                filter_polarity(pred_raw_evs, 0),
            ),
        }
        event_metrics = {}
        coverage = {}

        for name, (gt, pred) in polarities.items():
            (
                event_metrics[name],
                normalized_coverage_gt, 
                absolute_coverage_gt
            ) = self._compute_event_metrics(
                name,
                gt,
                pred,
                start_us,
                end_us,
            )

            # coverage[name] = {
            #     "normalized": normalized_coverage_gt,
            #     "absolute": absolute_coverage_gt,
            # }

        if visualize:
            # histograms
            # coverage_histograms_dir = os.path.join(self.output_dir, "coverage_histograms")
            # os.makedirs(coverage_histograms_dir, exist_ok=True)
            
            # plot_coverage_histograms(
            #     coverage,
            #     save_path=os.path.join(
            #         coverage_histograms_dir,
            #         f"coverage_histograms_{start_us}_{end_us}.png",
            #     ),
            # )

            # image metrics
            title = (
                f"Time: {start_us} - {end_us} us\n"
                f"SSIM: {image_metrics['ssim']:.4f} | PSNR: {image_metrics['psnr']:.2f} | LPIPS: {image_metrics['lpips']:.4f}\n"
                f"MSE: {image_metrics['mse']:.2f} | MAE: {image_metrics['mae']:.2f}"
            )
            show_pair(
                gt_img,
                pred_img,
                title=title,
                save_path=save_path
            )
        

        metrics = {
            "images": image_metrics,
            "events": event_metrics,
        }

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
        common_start, common_end = get_common_time_range(self.gt_slicer, self.pred_slicer)
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
                    print("Window skipped\n")
                    skipped += 1
                    continue
            except AssertionError:
                print("Error\n")
                skipped += 1
                continue
            results.append({
                "start_us": start_us,
                "end_us": end_us,
                **metrics
            })
            self._progress(idx + 1, total_windows, "Windows", start_us, end_us)
            gc.collect()
            plt.close("all")
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
        common_start, common_end = get_common_time_range(self.gt_slicer, self.pred_slicer)
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
                    print()
                else:
                    results.append({
                        "start_us": current,
                        "end_us": current + frame_time_us,
                        **metrics
                    })
            except Exception:
                print("Error occurred while evaluating frame.")
                traceback.print_exc()
                skipped += 1
            frame_idx += 1
            current += frame_time_us
            self._progress(frame_idx, total_frames, "Frames", current, current + frame_time_us)
            gc.collect()
            plt.close("all")
        print()
        self._save_summary(results, skipped, prefix="frame")
        return results