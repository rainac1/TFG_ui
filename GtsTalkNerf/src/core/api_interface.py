"""
GtsTalkNeRF Backend API Interface

 Python 函数接口供后端调用，支持"加载一次模型，多次推理"

Features:
    - 统一的推理接口
    - 异常处理和状态管理

Usage:
    >>> from api_interface import GtsTalkService
    >>>
    >>> # 初始化服务（只需一次）
    >>> service = GtsTalkService("results/obama")
    >>>
    >>> # 多次推理
    >>> result = service.process("audio1.wav", "output1.mp4")
    >>> result = service.process("audio2.wav", "output2.mp4")
"""

import sys
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

# 确保引用路径正确
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# StyleUNet 路径
_styleunet_path = os.path.join(_current_dir, "styleunet")
if _styleunet_path not in sys.path:
    sys.path.insert(0, _styleunet_path)

from core.inference import TalkingHeadGenerator

class GtsTalkService:
    """
    GtsTalkNeRF 后端服务类

    支持模型预加载和多次推理，提 API 接口

    Attributes:
        model_path (str): 模型目录路径
        device (str): 计算设备 ('cuda' or 'cpu')
        generator (TalkingHeadGenerator): 视频生成器实例
        is_ready (bool): 模型是否已加载完成

    Example:
        >>> service = GtsTalkService("results/obama", device="cuda")
        >>>
        >>> # 生成视频
        >>> result = service.process("speech.wav", "output.mp4")
        >>> if result["status"] == "success":
        ...     f"(: {result['video_path']}")
    """

    def __init__(self, model_path: str, device: str = "cuda", preload: bool = True):
        """
        初始化服务：预加载模型，避免每次请求都重新加载

        Args:
            model_path (str): 训练
                - 例如: "results/obama"
                - 需要包含 transforms_train.json, logs/ 等

            device (str): 计算设备
                - "cuda": 使用 GPU (默认，推荐)
                - "cpu": 使用 CPU

            preload (bool): 是否预加载所有模型
                - True: 初始化时加载所 (默认，推荐)
                - False: 延迟加载，首次推理时加载

        Raises:
            FileNotFoundError: 模型路径不存在
            RuntimeError: CUDA 不可用
        """
        self.model_path = str(model_path)
        self.device = device
        self.is_ready = False

        print(f"[GtsTalkService] Initializing...")
        print(f"[GtsTalkService] Model path: {self.model_path}")
        print(f"[GtsTalkService] Device: {self.device}")

        # 检查模型路径
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        # 创建生成器实例
        self.generator = TalkingHeadGenerator(self.model_path, device=self.device)

        # 预加载模型
        if preload:
            self._preload_models()

        print("[GtsTalkService] Initialization complete.")

    def _preload_models(self):
        """预加载所有子模型"""
        print("[GtsTalkService] Preloading models...")

        try:
            # 加载 AudioFace (Stage 1)
            print("[GtsTalkService]   Loading AudioFace...")
            self.generator.load_audioface()

            # 加载 NeRF (Stage 2)
            print("[GtsTalkService]   Loading NeRF...")
            self.generator.load_nerf()

            #  StyleUNet (Stage 3)
            print("[GtsTalkService]   Loading StyleUNet...")
            self.generator.load_styleunet()

            self.is_ready = True
            print("[GtsTalkService] All models loaded successfully.")

        except Exception as e:
            print(f"[GtsTalkService] Error loading models: {e}")
            raise

    def process(
        self, audio_path: str, output_path: str, use_styleunet: bool = True
    ) -> Dict[str, Any]:
        """
        #


                Args:
                    audio_path (str): 输入音频路径
                        - 支持格式: .wav, .mp3 等
                        - 建议: 16kHz 采样率的 WAV 文

                    output_path (str): 输出视频
                        - 格式: .mp4

                    use_styleunet (bool): 是否使用 StyleUNet 增强
                        - False: 跳过增强，速度更快

                Returns:
                    Dict[str, Any]: 结果字
                        成功时:
                            {
                                "status": "success",
                                "video_path": "/path/to/output.mp4",
                                "resolution": "512x512",
                                "fps": 25
                            }
                        失败时:
                            {
                                "status": "error",
                                "message": "错误信息"
                            }

                Example:
                    >>> result = service.process("speech.wav", "output.mp4")
                    >>> if result["status"] == "success":
                    ...     print(f"生成完成: {result['video_path']}")
                    ... else:
                    ...     print(f"生成失败: {result['message']}")
        """
        try:
            # 验证输入
            if not Path(audio_path).exists():
                return {
                    "status": "error",
                    "message": f"Audio file not found: {audio_path}",
                }

            # 生成视频
            result_path = self.generator.generate_video(
                audio_path, output_path, use_styleunet=use_styleunet
            )

            # 获取视频信息
            return {
                "status": "success",
                "video_path": result_path,
                "resolution": f"{self.generator.W}x{self.generator.H}",
                "fps": 25,
            }

        except Exception as e:
            import traceback

            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

    def get_status(self) -> Dict[str, Any]:
        """
        获取服务状态

        Returns:
            Dict[str, Any]:
                {
                    "is_ready": True,
                    "model_path": "results/obama",
                    "device": "cuda",
                    "resolution": "512x512",
                    "models_loaded": {
                        "audioface": True,
                        "nerf": True,
                        "styleunet": True
                    }
                }
        """
        return {
            "is_ready": self.is_ready,
            "model_path": self.model_path,
            "device": self.device,
            "resolution": f"{self.generator.W}x{self.generator.H}",
            "models_loaded": {
                "audioface": self.generator.audioface is not None,
                "nerf": self.generator.nerf_model is not None,
                "styleunet": self.generator.styleunet is not None,
            },
        }

    def release(self):
        """
        释放模型资源
        """
        import torch
        import gc

        self.generator.audioface = None
        self.generator.nerf_model = None
        self.generator.styleunet = None
        self.is_ready = False

        # 清理 GPU 内存
        if self.device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()
        print("[GtsTalkService] Resources released.")


def quick_inference(
    model_path: str, audio_path: str, output_path: str, device: str = "cuda"
) -> Dict[str, Any]:
    """
    快速推理函数：单次使用，

    适用于只需要单次推理的场景。如需多次推理，请使用 GtsTalkService 类。

    Args:
        model_path: 模型目录路
        audio_path: 输入音频路径
        output_path:
        device: 计算设备

    Returns:
        Dict[str, Any]: 结果字典
    """
    service = GtsTalkService(model_path, device, preload=True)
    result = service.process(audio_path, output_path)
    service.release()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GtsTalkNeRF API Interface Demo")
    parser.add_argument(
        "--model_path", type=str, default="results/obama", help="Model directory path"
    )
    parser.add_argument(
        "--audio", type=str, default=None, help="Input audio file for testing"
    )
    parser.add_argument(
        "--output", type=str, default="output_api.mp4", help="Output video path"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    print("=" * 70)
    print("GtsTalkNeRF API Interface Demo")
    print("=" * 70)

    # 初始化服务
    service = GtsTalkService(args.model_path, device=args.device)

    # 打印状态
    print("\n[Status]")
    status = service.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 如果提供了音频，执行推理
    if args.audio:
        print(f"\n[Processing] {args.audio} -> {args.output}")
        result = service.process(args.audio, args.output)
        print(f"[Result] {result}")
    else:
        print("\n[Info] No audio provided. Use --audio to test inference.")
        print("Example: python api_interface.py --audio test.wav --output result.mp4")

    print("\n" + "=" * 70)
