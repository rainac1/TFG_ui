"""
Image/Video Quality Metrics Calculator
Implements NIQE, PSNR, and SSIM metrics

Provides callable Python function interface for backend integration
"""

import argparse
import cv2
import imageio
import numpy as np
import os
from pathlib import Path
from typing import List
from tqdm import tqdm


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: int = 255) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    Higher is better. Typical range: 20-50 dB.
    
    Args:
        img1: First image array
        img2: Second image array
        data_range: Pixel value range (default: 255 for uint8)
    
    Returns:
        psnr: PSNR value in dB
    """
    assert img1.shape == img2.shape, f"Image shapes don't match: {img1.shape} vs {img2.shape}"
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((data_range ** 2) / mse)
    
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: int = 255,
                  win_size: int = 11, K1: float = 0.01, K2: float = 0.03) -> float:
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    Range: [-1, 1], closer to 1 is better.
    
    Args:
        img1: First image array
        img2: Second image array
        data_range: Pixel value range (default: 255)
        win_size: Sliding window size (default: 11)
        K1, K2: Stability constants
    
    Returns:
        ssim: SSIM value
    """
    assert img1.shape == img2.shape, f"Image shapes don't match: {img1.shape} vs {img2.shape}"
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    sigma = 1.5
    window = cv2.getGaussianKernel(win_size, sigma)
    window = window @ window.T
    
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def calculate_niqe(img: np.ndarray) -> float:
    """
    Calculate NIQE (Natural Image Quality Evaluator) score.
    No-reference metric. Lower is better. Typical range: 2-10.
    
    Args:
        img: Input image array
    
    Returns:
        niqe: NIQE score
    """
    try:
        import piq
        import torch
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        niqe_score = piq.niqe(img_tensor, data_range=1.0)
        return float(niqe_score.item())
    
    except ImportError:
        return _calculate_niqe_simplified(img)


def _calculate_niqe_simplified(img: np.ndarray) -> float:
    """Simplified NIQE calculation based on image statistics."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    gray = gray.astype(np.float64)
    
    kernel_size = 7
    mu = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 7/6)
    mu_sq = mu ** 2
    sigma = cv2.GaussianBlur(gray ** 2, (kernel_size, kernel_size), 7/6) - mu_sq
    sigma = np.sqrt(np.maximum(sigma, 0))
    
    struct = (gray - mu) / (sigma + 1)
    
    features = []
    features.append(np.mean(struct))
    features.append(np.var(struct))
    features.append(np.mean((struct - np.mean(struct)) ** 3) / (np.std(struct) ** 3 + 1e-6))
    features.append(np.mean((struct - np.mean(struct)) ** 4) / (np.std(struct) ** 4 + 1e-6))
    
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(dx**2 + dy**2)
    
    features.append(np.mean(gradient_mag))
    features.append(np.std(gradient_mag))
    
    niqe_approx = np.linalg.norm(features) / 10.0
    
    return float(niqe_approx)


def calculate_video_metrics(video_path: str, reference_video_path: str = None,
                            metrics: List[str] = ['psnr', 'ssim', 'niqe']) -> dict:
    """
    Calculate quality metrics for a video.
    
    Args:
        video_path: Path to video to evaluate
        reference_video_path: Path to reference video (for PSNR/SSIM)
        metrics: List of metrics to calculate
    
    Returns:
        results: Dictionary with metric results
    """
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()
    
    ref_frames = None
    if reference_video_path and ('psnr' in metrics or 'ssim' in metrics):
        ref_reader = imageio.get_reader(reference_video_path)
        ref_frames = [frame for frame in ref_reader]
        ref_reader.close()
        
        min_frames = min(len(frames), len(ref_frames))
        frames = frames[:min_frames]
        ref_frames = ref_frames[:min_frames]
    
    results = {}
    
    if 'psnr' in metrics and ref_frames is not None:
        psnr_values = []
        for frame, ref_frame in tqdm(zip(frames, ref_frames), total=len(frames), desc="Calculating PSNR"):
            psnr = calculate_psnr(frame, ref_frame)
            psnr_values.append(psnr)
        
        results['psnr'] = {
            'mean': float(np.mean(psnr_values)),
            'std': float(np.std(psnr_values)),
            'per_frame': psnr_values
        }
    
    if 'ssim' in metrics and ref_frames is not None:
        ssim_values = []
        for frame, ref_frame in tqdm(zip(frames, ref_frames), total=len(frames), desc="Calculating SSIM"):
            ssim = calculate_ssim(frame, ref_frame)
            ssim_values.append(ssim)
        
        results['ssim'] = {
            'mean': float(np.mean(ssim_values)),
            'std': float(np.std(ssim_values)),
            'per_frame': ssim_values
        }
    
    if 'niqe' in metrics:
        niqe_values = []
        for frame in tqdm(frames, desc="Calculating NIQE"):
            niqe = calculate_niqe(frame)
            niqe_values.append(niqe)
        
        results['niqe'] = {
            'mean': float(np.mean(niqe_values)),
            'std': float(np.std(niqe_values)),
            'per_frame': niqe_values
        }
    
    # 添加帧数信息
    results['n_frames'] = len(frames)
    
    return results


def calculate_image_metrics(image_path: str, reference_image_path: str = None,
                           metrics: List[str] = ['psnr', 'ssim', 'niqe']) -> dict:
    """
    Calculate quality metrics for an image.
    
    Args:
        image_path: Path to image to evaluate
        reference_image_path: Path to reference image (for PSNR/SSIM)
        metrics: List of metrics to calculate
    
    Returns:
        results: Dictionary with metric results
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    results = {}
    
    if reference_image_path and ('psnr' in metrics or 'ssim' in metrics):
        ref_img = cv2.imread(reference_image_path)
        if ref_img is None:
            raise FileNotFoundError(f"Cannot read reference image: {reference_image_path}")
        
        if 'psnr' in metrics:
            results['psnr'] = calculate_psnr(img, ref_img)
        
        if 'ssim' in metrics:
            results['ssim'] = calculate_ssim(img, ref_img)
    
    if 'niqe' in metrics:
        results['niqe'] = calculate_niqe(img)
    
    return results


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='Calculate image/video quality metrics (NIQE, PSNR, SSIM)')
    
    parser.add_argument('--input', type=str, required=True, help='Input video or image file path')
    parser.add_argument('--reference', type=str, default=None, help='Reference file path (required for PSNR/SSIM)')
    parser.add_argument('--type', type=str, default='video', choices=['video', 'image'], help='Input type')
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'niqe'],
                       choices=['psnr', 'ssim', 'niqe'], help='Metrics to calculate')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if args.reference and not os.path.exists(args.reference):
        print(f"Error: Reference file not found: {args.reference}")
        return 1
    
    if ('psnr' in args.metrics or 'ssim' in args.metrics) and not args.reference:
        print("Warning: PSNR and SSIM require a reference file. Only NIQE will be calculated.")
        args.metrics = ['niqe']
    
    print("="*70)
    print("Image/Video Quality Metrics Calculation")
    print("="*70)
    print(f"Input: {args.input}")
    if args.reference:
        print(f"Reference: {args.reference}")
    print(f"Metrics: {', '.join(args.metrics).upper()}")
    print("-"*70)
    
    try:
        if args.type == 'video':
            results = calculate_video_metrics(args.input, args.reference, args.metrics)
        else:
            results = calculate_image_metrics(args.input, args.reference, args.metrics)
        
        print("\nResults:")
        print("-"*70)
        
        if args.type == 'video':
            for metric, values in results.items():
                if metric == 'n_frames':
                    print(f"Total frames: {values}")
                    continue
                print(f"{metric.upper()}:")
                print(f"  Mean:  {values['mean']:.4f}")
                print(f"  Std:   {values['std']:.4f}")
                print(f"  Min:   {min(values['per_frame']):.4f}")
                print(f"  Max:   {max(values['per_frame']):.4f}")
                print()
        else:
            for metric, value in results.items():
                print(f"{metric.upper()}: {value:.4f}")
        
        print("="*70)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
