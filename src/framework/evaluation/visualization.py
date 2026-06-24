import cv2
import matplotlib.pyplot as plt
import numpy as np


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
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


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
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()