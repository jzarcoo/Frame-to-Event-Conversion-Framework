import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def _save_or_show(fig, save_path=None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def show_pair(
    gt_img,
    pred_img,
    title,
    save_path=None,
):
    """
    Display or save a pair of images (ground truth and prediction) side by side.
    Args:
        gt_img (np.ndarray): Ground truth image.
        pred_img (np.ndarray): Predicted image.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Ground Truth")
    ax[1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Prediction")
    plt.suptitle(title)
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    _save_or_show(fig, save_path)

def plot_metrics(
    results,
    save_path=None,
):
    """
    Plot evaluation metrics for a list of results.
    Args:
        results (list): List of dictionaries containing metric values.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    frames = np.arange(len(results))
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    metrics = [
        "mse",
        "mae",
        "ssim",
        "psnr",
        "lpips",
    ]
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        axs[i].plot(frames, values)
        axs[i].set_title(metric.upper())
        axs[i].set_xlabel("Window")
        
    axs[-1].axis("off")
    plt.tight_layout()    
    _save_or_show(fig, save_path)

def plot_histograms_count(
    gt_hist,
    pred_hist,
    name,
    width,
    height,
    metrics=None,
    save_path=None,
):
    """
    Plot and save histograms of event counts for ground truth and predicted events.

    Args:
        gt_hist (np.ndarray): Ground truth histogram.
        pred_hist (np.ndarray): Predicted histogram.
        name (str): Name of the evaluation ("ALL", "ON", "OFF").
        width (int): Image width.
        height (int): Image height.
        metrics (dict, optional): Dictionary of evaluation metrics.
        save_path (str, optional): Output path.
    """

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    extent = [0, width, 0, height]

    images = [
        (axs[0], gt_hist, "Ground Truth", "Greens"),
        (axs[1], pred_hist, "Prediction", "Blues"),
    ]

    for ax, hist, title, cmap in images:
        im = ax.imshow(
            hist,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if metrics is not None:
        fig.suptitle(
            f"{name}\n"
            f"F1={metrics['f1']:.3f} | "
            f"Precision={metrics['precision']:.3f} | "
            f"Recall={metrics['recall']:.3f} | "
            f"Accuracy={metrics['accuracy']:.3f}\n"
            f"Mean Absolute Coverage={metrics['absolute_coverage']['mean_cov']:.3f} | "
            f"Mean Normalized Coverage={metrics['normalized_coverage']['mean_cov']:.3f} | "
            f"Ratio={metrics['event_ratio']:.3f} | "
            f"GT={metrics['gt_count']} | "
            f"Pred={metrics['pred_count']}",
            fontsize=12,
        )
    else:
        fig.suptitle(name, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    _save_or_show(fig, save_path)