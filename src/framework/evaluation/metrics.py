import cv2
import lpips
import numpy as np
import torch
import warnings

from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    _lpips_model = lpips.LPIPS(net="alex")

def compute_metrics(
    gt_img,
    pred_img,
):
    """
    Compute evaluation metrics between ground truth and predicted images.
    Args:
        gt_img (np.ndarray): Ground truth image.
        pred_img (np.ndarray): Predicted image.
    Returns:
        dict: Dictionary containing MSE, MAE, SSIM, PSNR, and LPIPS scores.
    """
    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
    mse = np.mean(
        (
            gt_gray.astype(np.float32) - pred_gray.astype(np.float32)
        ) ** 2
    )
    mae = np.mean(
        np.abs(
            gt_gray.astype(np.float32)
            - pred_gray.astype(np.float32)
        )
    )
    ssim_score = structural_similarity(gt_gray, pred_gray, data_range=255)
    psnr_score = peak_signal_noise_ratio(gt_gray, pred_gray, data_range=255)
    gt_tensor = (
        torch.from_numpy(gt_img)
        .permute(2, 0, 1)
        .float()
        / 127.5
        - 1
    ).unsqueeze(0)
    pred_tensor = (
        torch.from_numpy(pred_img)
        .permute(2, 0, 1)
        .float()
        / 127.5
        - 1
    ).unsqueeze(0)
    lpips_score = _lpips_model(gt_tensor, pred_tensor).item()
    return {
        "mse": mse,
        "mae": mae,
        "ssim": ssim_score,
        "psnr": psnr_score,
        "lpips": lpips_score,
    }