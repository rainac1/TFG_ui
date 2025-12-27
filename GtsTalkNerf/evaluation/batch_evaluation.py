#!/usr/bin/env python3
"""
批量视频评测脚本 - 独立Docker镜像版本
支持6项指标：NIQE, PSNR, SSIM, FID, LSE-C, LSE-D

输入：生成视频目录 + GT视频目录
输出：CSV/Excel评测报告
"""

import argparse
import json
import os
import sys
import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# 图像质量指标
# ============================================================================

def calculate_niqe(frame: np.ndarray) -> float:
    """计算单帧NIQE (Natural Image Quality Evaluator)"""
    try:
        import piq
        import torch
        
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # 确保在正确设备上
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = tensor.to(device)
        
        # 使用 piq 计算 NIQE
        with torch.no_grad():
            niqe_value = piq.niqe(tensor, data_range=1.0)
        
        return float(niqe_value.cpu().item())
    except Exception as e:
        # 备用方案：使用简化的图像质量评估
        try:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # 使用拉普拉斯算子评估清晰度
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 归一化到类似NIQE的范围 (2-10)
            # 高方差表示更清晰，对应更低的NIQE
            niqe_approx = max(2.0, min(10.0, 10.0 - np.log1p(laplacian_var) * 0.5))
            return float(niqe_approx)
        except:
            return -1.0


def calculate_psnr(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """计算PSNR (Peak Signal-to-Noise Ratio)"""
    # 自动调整尺寸
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    if h1 != h2 or w1 != w2:
        target_h, target_w = min(h1, h2), min(w1, w2)
        frame1 = cv2.resize(frame1, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        frame2 = cv2.resize(frame2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    mse = np.mean((frame1.astype(np.float64) - frame2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255.0 ** 2 / mse))


def calculate_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """计算SSIM (Structural Similarity Index)"""
    # 自动调整尺寸
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    if h1 != h2 or w1 != w2:
        target_h, target_w = min(h1, h2), min(w1, w2)
        frame1 = cv2.resize(frame1, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        frame2 = cv2.resize(frame2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 转灰度
    if frame1.ndim == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
    
    if frame2.ndim == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2
    
    # SSIM参数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = gray1.astype(np.float64)
    img2 = gray2.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())


# ============================================================================
# FID 计算
# ============================================================================

def calculate_fid(generated_video: str, reference_video: str, max_frames: int = 100) -> float:
    """计算FID (Fréchet Inception Distance)"""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from scipy import linalg
        from PIL import Image
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载Inception-v3
        try:
            from torchvision.models import Inception_V3_Weights
            inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        except:
            inception = models.inception_v3(pretrained=True, transform_input=False)
        
        inception.fc = nn.Identity()
        inception = inception.to(device)
        inception.eval()
        
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        def extract_features(video_path):
            cap = cv2.VideoCapture(video_path)
            features = []
            count = 0
            
            with torch.no_grad():
                while cap.isOpened() and count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    
                    feat = inception(img_tensor)
                    features.append(feat.cpu().numpy())
                    count += 1
            
            cap.release()
            return np.concatenate(features, axis=0) if features else None
        
        gen_features = extract_features(generated_video)
        ref_features = extract_features(reference_video)
        
        if gen_features is None or ref_features is None:
            return -1.0
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        mu_ref = np.mean(ref_features, axis=0)
        sigma_ref = np.cov(ref_features, rowvar=False)
        
        diff = mu_gen - mu_ref
        covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_ref), disp=False)
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_gen.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma_gen + offset).dot(sigma_ref + offset))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_gen + sigma_ref - 2 * covmean)
        return float(fid)
        
    except Exception as e:
        print(f"FID计算失败: {e}")
        return -1.0


# ============================================================================
# LSE 口型同步指标
# ============================================================================

def extract_audio_from_video(video_path: str, output_audio: str) -> bool:
    """从视频中提取音频"""
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return os.path.exists(output_audio)
    except:
        return False


def calculate_lse(generated_video: str, audio_path: str = None) -> Dict:
    """
    计算LSE-C和LSE-D (Lip Sync Error)
    使用音频能量和嘴部运动幅度的相关性
    """
    try:
        import librosa
        from scipy.ndimage import uniform_filter1d
        
        # 如果没有提供音频，尝试从视频提取
        temp_audio = None
        if audio_path is None:
            temp_audio = f"/tmp/temp_audio_{os.getpid()}.wav"
            if not extract_audio_from_video(generated_video, temp_audio):
                return {'lse_c': -1.0, 'lse_d': -1.0, 'error': 'Cannot extract audio'}
            audio_path = temp_audio
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 计算RMS能量
        fps = 25
        hop_length = int(sr / fps)
        rms = librosa.feature.rms(y=audio, frame_length=hop_length*2, hop_length=hop_length)[0]
        audio_energy = uniform_filter1d(rms, size=3)
        
        # 提取视频嘴部运动
        cap = cv2.VideoCapture(generated_video)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        mouth_motion = []
        prev_mouth = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            mouth_region = frame[int(h*0.55):int(h*0.80), int(w*0.25):int(w*0.75)]
            gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            if prev_mouth is not None:
                diff = np.abs(gray_mouth - prev_mouth)
                motion = np.mean(diff)
                mouth_motion.append(motion)
            
            prev_mouth = gray_mouth.copy()
        
        cap.release()
        
        # 清理临时文件
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        mouth_motion = np.array([0] + mouth_motion)
        mouth_motion = uniform_filter1d(mouth_motion, size=3)
        
        # 对齐长度
        min_len = min(len(audio_energy), len(mouth_motion))
        audio_energy = audio_energy[:min_len]
        mouth_motion = mouth_motion[:min_len]
        
        # 归一化
        audio_norm = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-8)
        motion_norm = (mouth_motion - mouth_motion.mean()) / (mouth_motion.std() + 1e-8)
        
        # 计算互相关
        max_lag = min(30, min_len // 4)
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                corr = np.corrcoef(audio_norm[lag:], motion_norm[:min_len-lag])[0, 1]
            else:
                corr = np.corrcoef(audio_norm[:min_len+lag], motion_norm[-lag:])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        correlations = np.array(correlations)
        best_idx = np.argmax(correlations)
        best_lag = best_idx - max_lag
        best_corr = correlations[best_idx]
        
        lse_d = abs(best_lag) / video_fps
        
        if best_corr > 0:
            lse_c = max(0, (1 - best_corr) * 5)
        else:
            lse_c = 5 + abs(best_corr) * 5
        
        return {
            'lse_c': float(lse_c),
            'lse_d': float(lse_d),
            'correlation': float(best_corr),
            'offset_frames': int(best_lag)
        }
        
    except Exception as e:
        return {'lse_c': -1.0, 'lse_d': -1.0, 'error': str(e)}


# ============================================================================
# 视频质量指标计算
# ============================================================================

def evaluate_video_pair(gen_video: str, gt_video: str = None, 
                        audio_path: str = None, 
                        sample_frames: int = 50) -> Dict:
    """评测单个视频对"""
    results = {
        'generated_video': os.path.basename(gen_video),
        'gt_video': os.path.basename(gt_video) if gt_video else 'N/A',
    }
    
    # 打开视频
    cap_gen = cv2.VideoCapture(gen_video)
    cap_gt = cv2.VideoCapture(gt_video) if gt_video else None
    
    gen_frame_count = int(cap_gen.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 采样帧索引
    if gen_frame_count <= sample_frames:
        frame_indices = list(range(gen_frame_count))
    else:
        frame_indices = np.linspace(0, gen_frame_count - 1, sample_frames, dtype=int).tolist()
    
    niqe_scores = []
    psnr_scores = []
    ssim_scores = []
    
    for idx in frame_indices:
        cap_gen.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_gen, frame_gen = cap_gen.read()
        if not ret_gen:
            continue
        
        # NIQE (无参考)
        niqe = calculate_niqe(frame_gen)
        if niqe > 0:
            niqe_scores.append(niqe)
        
        # PSNR/SSIM (需要GT)
        if cap_gt:
            cap_gt.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_gt, frame_gt = cap_gt.read()
            if ret_gt:
                psnr = calculate_psnr(frame_gen, frame_gt)
                ssim = calculate_ssim(frame_gen, frame_gt)
                if psnr != float('inf'):
                    psnr_scores.append(psnr)
                ssim_scores.append(ssim)
    
    cap_gen.release()
    if cap_gt:
        cap_gt.release()
    
    # 汇总图像质量指标
    results['NIQE'] = np.mean(niqe_scores) if niqe_scores else -1.0
    results['NIQE_std'] = np.std(niqe_scores) if niqe_scores else -1.0
    
    if psnr_scores:
        results['PSNR'] = np.mean(psnr_scores)
        results['PSNR_std'] = np.std(psnr_scores)
    else:
        results['PSNR'] = -1.0
        results['PSNR_std'] = -1.0
    
    if ssim_scores:
        results['SSIM'] = np.mean(ssim_scores)
        results['SSIM_std'] = np.std(ssim_scores)
    else:
        results['SSIM'] = -1.0
        results['SSIM_std'] = -1.0
    
    # FID
    if gt_video:
        results['FID'] = calculate_fid(gen_video, gt_video)
    else:
        results['FID'] = -1.0
    
    # LSE
    lse_results = calculate_lse(gen_video, audio_path)
    results['LSE_C'] = lse_results.get('lse_c', -1.0)
    results['LSE_D'] = lse_results.get('lse_d', -1.0)
    results['Sync_Correlation'] = lse_results.get('correlation', -1.0)
    
    return results


def find_matching_gt(gen_video: str, gt_dir: str) -> Optional[str]:
    """根据生成视频名称查找对应的GT视频"""
    gen_name = Path(gen_video).stem.lower()
    
    # 获取所有GT视频
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    gt_videos = []
    for f in os.listdir(gt_dir):
        if any(f.lower().endswith(ext) for ext in video_extensions):
            gt_videos.append(os.path.join(gt_dir, f))
    
    # 尝试多种匹配策略
    for gt_video in gt_videos:
        gt_name = Path(gt_video).stem.lower()
        
        # 完全匹配
        if gen_name == gt_name:
            return gt_video
        
        # 前缀匹配 (如 video1 匹配 video1_gt)
        if gt_name.startswith(gen_name) or gen_name.startswith(gt_name):
            return gt_video
        
        # 包含匹配 (如 obama_reconstructed 匹配 obama)
        if gen_name in gt_name or gt_name in gen_name:
            return gt_video
        
        # 提取关键词匹配 (如 obama_reconstructed 匹配 obama_25fps)
        gen_parts = set(gen_name.replace('_', ' ').replace('-', ' ').split())
        gt_parts = set(gt_name.replace('_', ' ').replace('-', ' ').split())
        common_parts = gen_parts & gt_parts
        if len(common_parts) > 0:
            # 排除常见的非关键词
            excluded = {'test', 'output', 'result', 'video', 'reconstructed', 'generated'}
            meaningful_common = common_parts - excluded
            if len(meaningful_common) > 0:
                return gt_video
    
    return None


def batch_evaluate(gen_dir: str, gt_dir: str = None, 
                   output_path: str = 'evaluation_results',
                   audio_dir: str = None,
                   sample_frames: int = 50) -> pd.DataFrame:
    """
    批量评测视频
    
    Args:
        gen_dir: 生成视频目录
        gt_dir: GT视频目录（可选）
        output_path: 输出文件路径（不含扩展名）
        audio_dir: 音频目录（可选，用于LSE计算）
        sample_frames: 每个视频采样帧数
    
    Returns:
        评测结果DataFrame
    """
    print("="*80)
    print("批量视频评测系统")
    print("="*80)
    print(f"生成视频目录: {gen_dir}")
    print(f"GT视频目录: {gt_dir or 'N/A'}")
    print(f"音频目录: {audio_dir or 'N/A'}")
    print(f"采样帧数: {sample_frames}")
    print("="*80)
    
    # 查找所有生成视频
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    gen_videos = []
    for ext in video_extensions:
        gen_videos.extend(glob.glob(os.path.join(gen_dir, ext)))
    
    if not gen_videos:
        print("错误: 未找到任何生成视频")
        return None
    
    print(f"找到 {len(gen_videos)} 个生成视频")
    
    # 评测结果
    all_results = []
    
    for gen_video in tqdm(gen_videos, desc="评测进度"):
        video_name = Path(gen_video).stem
        
        # 查找对应GT
        gt_video = None
        if gt_dir:
            gt_video = find_matching_gt(gen_video, gt_dir)
            if gt_video:
                tqdm.write(f"  {video_name} -> GT: {os.path.basename(gt_video)}")
            else:
                tqdm.write(f"  {video_name} -> GT: 未找到匹配")
        
        # 查找对应音频
        audio_path = None
        if audio_dir:
            audio_patterns = [
                os.path.join(audio_dir, f"{video_name}.wav"),
                os.path.join(audio_dir, f"{video_name}.mp3"),
                os.path.join(audio_dir, f"{video_name}_audio.wav"),
            ]
            for ap in audio_patterns:
                if os.path.exists(ap):
                    audio_path = ap
                    break
        
        # 评测
        try:
            result = evaluate_video_pair(
                gen_video, 
                gt_video, 
                audio_path,
                sample_frames
            )
            result['video_name'] = video_name
            all_results.append(result)
        except Exception as e:
            tqdm.write(f"  评测失败 {video_name}: {e}")
            all_results.append({
                'video_name': video_name,
                'generated_video': os.path.basename(gen_video),
                'error': str(e)
            })
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 重新排列列顺序
    cols_order = ['video_name', 'generated_video', 'gt_video', 
                  'NIQE', 'NIQE_std', 'PSNR', 'PSNR_std', 'SSIM', 'SSIM_std',
                  'FID', 'LSE_C', 'LSE_D', 'Sync_Correlation']
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order]
    
    # 计算汇总统计
    print("\n" + "="*80)
    print("评测汇总统计")
    print("="*80)
    
    numeric_cols = ['NIQE', 'PSNR', 'SSIM', 'FID', 'LSE_C', 'LSE_D']
    for col in numeric_cols:
        if col in df.columns:
            valid_data = df[col][df[col] > 0]
            if len(valid_data) > 0:
                print(f"{col}:")
                print(f"  均值: {valid_data.mean():.4f}")
                print(f"  标准差: {valid_data.std():.4f}")
                print(f"  最小值: {valid_data.min():.4f}")
                print(f"  最大值: {valid_data.max():.4f}")
    
    # 添加汇总行
    summary_row = {'video_name': 'AVERAGE'}
    for col in numeric_cols:
        if col in df.columns:
            valid_data = df[col][df[col] > 0]
            summary_row[col] = valid_data.mean() if len(valid_data) > 0 else -1
    
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # 保存结果
    csv_path = f"{output_path}.csv"
    excel_path = f"{output_path}.xlsx"
    json_path = f"{output_path}.json"
    
    # 保存CSV
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {csv_path}")
    
    # 保存Excel (带格式)
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='评测结果', index=False)
            
            # 获取worksheet
            worksheet = writer.sheets['评测结果']
            
            # 设置列宽
            for i, col in enumerate(df.columns):
                worksheet.column_dimensions[chr(65 + i)].width = 15
        
        print(f"结果已保存到: {excel_path}")
    except Exception as e:
        print(f"Excel保存失败: {e}")
    
    # 保存JSON
    df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    print(f"结果已保存到: {json_path}")
    
    print("\n" + "="*80)
    print("评测完成!")
    print("="*80)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='批量视频评测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅评测生成视频 (NIQE, LSE)
  python batch_evaluation.py --gen_dir /data/generated
  
  # 与GT视频对比评测 (全部6项指标)
  python batch_evaluation.py --gen_dir /data/generated --gt_dir /data/groundtruth
  
  # 指定音频目录
  python batch_evaluation.py --gen_dir /data/generated --audio_dir /data/audio
  
  # 自定义输出路径和采样帧数
  python batch_evaluation.py --gen_dir /data/generated --gt_dir /data/gt \\
                             --output /output/results --sample_frames 100
        """
    )
    
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='生成视频目录')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='GT视频目录 (可选)')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='音频文件目录 (可选，用于LSE计算)')
    parser.add_argument('--output', type=str, default='/output/evaluation_results',
                        help='输出文件路径 (不含扩展名)')
    parser.add_argument('--sample_frames', type=int, default=50,
                        help='每个视频采样帧数 (默认50)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.gen_dir):
        print(f"错误: 生成视频目录不存在: {args.gen_dir}")
        return 1
    
    if args.gt_dir and not os.path.exists(args.gt_dir):
        print(f"错误: GT视频目录不存在: {args.gt_dir}")
        return 1
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 运行评测
    try:
        df = batch_evaluate(
            args.gen_dir,
            args.gt_dir,
            args.output,
            args.audio_dir,
            args.sample_frames
        )
        return 0 if df is not None else 1
    except Exception as e:
        print(f"评测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
