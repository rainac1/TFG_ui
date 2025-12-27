"""
GtsTalkNeRF Inference Script

Video generation pipeline:
  Audio -> AudioFace -> Landmarks [3, 68]
       |
  Landmarks -> NeRF -> Coarse Rendering [512x512]
       |
  Coarse -> StyleUNet -> Enhanced [512x512]
       |
      Final Video @25fps

Usage:
    from inference import inference
    inference(model_path="results/obama", audio_path="test.wav", output_path="output.mp4")
    
Command line:
    python inference.py --model_path results/obama --audio test.wav --output output.mp4
"""
import argparse
import imageio
import os
import torch
import torchaudio
import json
import numpy as np
from pathlib import Path
import sys
import cv2

# StyleUNet import
_styleunet_path = Path(__file__).parent / "styleunet"
if str(_styleunet_path) not in sys.path:
    sys.path.insert(0, str(_styleunet_path))

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.styleunet.networks.generator import StyleUNet
from tqdm import tqdm
from core.trans_aud import AudioFace, FormantFeatureExporter
from core.nerf_lmk.network import NeRFNetwork
from core.nerf_lmk.utils import seed_everything, linear_to_srgb, get_rays


def nerf_matrix_to_ngp(pose, scale=1.0, offset=[0, 0, 0]):
    """Convert NeRF camera pose to NGP format"""
    new_pose = pose.astype(np.float32).copy()
    new_pose[:3, 3] = new_pose[:3, 3] * scale + np.array(offset)
    return new_pose


def mat2angle(mat):
    """Convert rotation matrix to euler angles and translation"""
    trans = mat[:3, 3]
    rot = mat[:3, :3]
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rot[2, 1], rot[2, 2])
        y = np.arctan2(-rot[2, 0], sy)
        z = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        x = np.arctan2(-rot[1, 2], rot[1, 1])
        y = np.arctan2(-rot[2, 0], sy)
        z = 0
    angle = np.array([x, y, z])
    return np.concatenate([angle, trans])



class InferenceConfig:
    """Inference configuration"""
    def __init__(self, model_path):
        self.path = str(model_path)
        self.workspace = str(Path(model_path) / "logs" / "stage2")
        self.ckpt = "latest"
        self.O = True
        self.fp16 = True
        self.cuda_ray = True
        self.seed = 0
        self.color_space = "srgb"
        self.preload = 0
        self.bound = 1
        self.dt_gamma = 1 / 256
        self.min_near = 0.05
        self.density_thresh = 10
        self.bg_img = str(Path(model_path) / "bg.png")
        self.ind_dim = 4
        self.ind_num = 10000
        self.amb_dim = 2
        self.max_steps = 128
        self.update_extra_interval = 16
        self.lambda_amb = 0.1
        self.smooth_path = True
        self.smooth_eye = True
        self.smooth_path_window = 3
        self.patch_size = 1
        self.finetune_lips = True
        self.finetune_eyes = False
        self.head_ckpt = ""
        self.train_camera = False
        self.part = False
        self.part2 = False
        self.test = True
        self.test_train = False
        self.aud = ""
        self.num_rays = -1


class TalkingHeadGenerator:
    """Talking head video generator"""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = Path(model_path)
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self._load_transforms()
        
        self.audioface = None
        self.nerf_model = None
        self.styleunet = None
        
    def _load_transforms(self):
        """Load training data transforms"""
        transforms_path = self.model_path / "transforms_train.json"
        if not transforms_path.exists():
            raise FileNotFoundError(f"transforms_train.json not found")
        
        with open(transforms_path, "r") as f:
            self.transforms = json.load(f)
        
        self.H = int(self.transforms.get("h", 512))
        self.W = int(self.transforms.get("w", 512))
        
        self.fl_x = float(self.transforms.get("fl_x", 500))
        self.fl_y = float(self.transforms.get("fl_y", 500))
        self.cx = float(self.transforms.get("cx", self.W / 2))
        self.cy = float(self.transforms.get("cy", self.H / 2))
        
        bound = np.array(self.transforms["bound"])
        aabb = 1.5 * (bound - bound.mean(0)) + bound.mean(0)
        self.bound = 1.0
        self.scale = self.bound / (aabb[1] - aabb[0]).max()
        self.offset = -self.scale * aabb[1] + [0.5 * self.bound, 0.5 * self.bound, 0.5 * self.bound]
        self.aabb = aabb * self.scale + np.array(self.offset)[None]
        
        pose0 = np.array(self.transforms["base_transform_matrix"], dtype=np.float32).reshape(4, 4)
        self.pose0 = nerf_matrix_to_ngp(pose0, self.scale, self.offset)
        self.R0 = self.pose0[:3, :3]
        
        lmk0 = np.array(self.transforms["base_landmark"], dtype=np.float32)
        if lmk0.shape == (68, 3):
            lmk0 = lmk0.T
        self.lmk0 = lmk0 * self.scale + np.array(self.offset)[:, None]
        
        # Compute pose angle for StyleUNet
        self.pose0_angle = mat2angle(self.pose0)  # [6] - euler angles + translation
        
        print(f"[Transforms] H={self.H}, W={self.W}")
        print(f"[Transforms] scale={self.scale:.6f}, offset={self.offset}")
        
    def load_audioface(self):
        """Load Stage 1 AudioFace model"""
        if self.audioface is not None:
            return self.audioface
        
        stage1_dir = self.model_path / "logs" / "stage1"
        model_weights = sorted(stage1_dir.glob("*.tar"), reverse=True)
        if not model_weights:
            raise FileNotFoundError(f"No model weights found in {stage1_dir}")
        
        print(f"[AudioFace] Loading: {model_weights[0]}")
        checkpoint = torch.load(model_weights[0], map_location=self.device)
        
        self.audioface = AudioFace(**checkpoint["config"]).to(self.device)
        self.audioface.load_state_dict(checkpoint["model"])
        self.audioface.eval()
        self.audioface.requires_grad_(False)
        
        lmk0_raw = np.array(self.transforms["base_landmark"], dtype=np.float32)
        if lmk0_raw.shape == (68, 3):
            lmk0_raw = lmk0_raw.T
        self.audioface_lmk0 = torch.from_numpy(lmk0_raw).float().reshape(1, 1, 3, 68).to(self.device)
        
        self.max_audio_length = checkpoint["config"].get("max_length", 64)
        
        return self.audioface
    
    def load_nerf(self):
        """Load Stage 2 NeRF model"""
        if self.nerf_model is not None:
            return self.nerf_model
        
        stage2_dir = self.model_path / "logs" / "stage2"
        opt = InferenceConfig(self.model_path)
        seed_everything(opt.seed)
        
        self.nerf_model = NeRFNetwork(opt).to(self.device)
        
        lmk0_tensor = torch.from_numpy(self.lmk0.T).float().to(self.device)
        R0_tensor = torch.from_numpy(self.R0).float().to(self.device)
        a0_tensor = torch.zeros(64).to(self.device)
        aabb_tensor = torch.from_numpy(self.aabb.flatten()).float().to(self.device)
        
        self.nerf_model.init_state(lmk0_tensor, R0_tensor, a0_tensor, aabb_tensor)
        
        ckpt_dir = stage2_dir / "checkpoints"
        ckpt_list = sorted(ckpt_dir.glob("ngp_ep*.pth"), reverse=True)
        
        if ckpt_list:
            ckpt_path = ckpt_list[0]
        elif (ckpt_dir / "ngp.pth").exists():
            ckpt_path = ckpt_dir / "ngp.pth"
        else:
            raise FileNotFoundError(f"No NeRF checkpoint found in {ckpt_dir}")
        
        print(f"[NeRF] Loading: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        if "model" in checkpoint:
            self.nerf_model.load_state_dict(checkpoint["model"], strict=False)
        else:
            self.nerf_model.load_state_dict(checkpoint, strict=False)
        
        self.nerf_model.eval()
        self.nerf_opt = opt
        
        return self.nerf_model
    
    def load_styleunet(self):
        """Load StyleUNet enhancement model"""
        if self.styleunet is not None:
            return self.styleunet
        
        checkpoint_path = self.model_path / "logs" / "checkpoint" / "ch_final.pt"
        if not checkpoint_path.exists():
            print(f"[StyleUNet] Checkpoint not found, skipping")
            return None
        
        print(f"[StyleUNet] Loading: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = checkpoint["args"]
        
        self.styleunet = StyleUNet(
            input_size=args.input_size,
            output_size=args.output_size,
            style_dim=args.style_dim,
            mlp_num=args.n_mlp,
            channel_multiplier=args.channel_multiplier
        )
        self.styleunet.load_state_dict(checkpoint["g"])
        self.styleunet = self.styleunet.to(self.device)
        self.styleunet.eval()
        
        self.styleunet_input_size = args.input_size
        self.styleunet_output_size = args.output_size
        
        print(f"[StyleUNet] input={args.input_size}, output={args.output_size}")
        
        return self.styleunet
    
    def process_audio(self, audio_path):
        """Process audio file"""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        wav, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        if wav.size(0) >= 2:
            wav = wav.mean(0, keepdim=True)
        wav = wav[0].float()
        
        ffe = FormantFeatureExporter(str(audio_path), sr=16000)
        fmt_frq, fmt_bw = ffe.formant()
        formant = torch.from_numpy(np.concatenate([fmt_frq, fmt_bw], 0)).float()
        
        num_frames = (wav.shape[0] // 80 - 1) // 8
        
        formant = torch.nn.functional.interpolate(
            formant[None], size=num_frames, mode="linear", align_corners=False
        )[0].t().contiguous()
        
        return wav, formant, num_frames
    
    def generate_landmarks(self, wav, formant):
        """Generate landmarks from audio"""
        self.load_audioface()
        
        num_frames = formant.shape[0]
        pid = torch.tensor([0])
        
        all_landmarks = []
        all_audio_features = []
        
        with torch.inference_mode():
            for i in tqdm(range(0, num_frames, self.max_audio_length), desc="Generating landmarks"):
                end_idx = min(i + self.max_audio_length, num_frames)
                batch_formant = formant[i:end_idx][None].to(self.device)
                
                wav_start = i * 640
                wav_end = min(end_idx * 640 + 720, wav.shape[0])
                batch_wav = wav[wav_start:wav_end][None].to(self.device)
                
                lms_pred = self.audioface.inference_forward(
                    wav=batch_wav,
                    lmk0=self.audioface_lmk0,
                    formant=batch_formant,
                    pid=pid[None].to(self.device)
                )
                
                aud_feat = self.audioface.audenc(batch_wav, batch_formant)
                
                all_landmarks.append(lms_pred[0].cpu())
                all_audio_features.append(aud_feat[0].cpu())
        
        landmarks = torch.cat(all_landmarks, dim=0)  # [L, 3, 68]
        audio_features = torch.cat(all_audio_features, dim=0)  # [L, 64]
        
        # Convert from [L, 3, 68] to [L, 68, 3] to match NeRF format
        landmarks = landmarks.permute(0, 2, 1)  # [L, 68, 3]
        
        return landmarks, audio_features
    
    def _prepare_camera(self, frame_idx=0):
        """Prepare camera parameters (centered)"""
        K = torch.tensor([
            [self.fl_x, 0, self.cx],
            [0, self.fl_y, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        # Use base pose to keep subject centered
        pose = torch.from_numpy(self.pose0).float().to(self.device)
        
        # Convert c2w to w2c for get_rays
        R = pose[:3, :3]
        T = pose[:3, 3]
        
        pose_w2c = torch.eye(4, device=self.device)
        pose_w2c[:3, :3] = R.t()
        pose_w2c[:3, 3] = -R.t() @ T
        
        return K, pose_w2c
    
    def render_frame(self, landmarks, audio_features, frame_idx, bg_color):
        """Render single frame with NeRF"""
        self.load_nerf()
        
        K, pose = self._prepare_camera(frame_idx)
        cams = torch.cat([K, pose[:3]], dim=-1).unsqueeze(0)
        
        rays = get_rays(cams, self.H, self.W)
        
        # landmarks is [68, 3], apply scale and offset
        lmk = landmarks.numpy()  # [68, 3]
        lmk_transformed = lmk * self.scale + np.array(self.offset)  # [68, 3]
        lmk_tensor = torch.from_numpy(lmk_transformed).float().to(self.device)  # [68, 3]
        
        aud_tensor = audio_features.to(self.device)
        
        R = torch.from_numpy(self.R0).float().to(self.device)
        
        with torch.inference_mode():
            outputs = self.nerf_model.render(
                rays["rays_o"],
                rays["rays_d"],
                lmk_tensor.unsqueeze(0),
                R.unsqueeze(0),
                aud_tensor.unsqueeze(0),
                index=torch.tensor([frame_idx % self.nerf_opt.ind_num], device=self.device),
                bg_color=bg_color,
                perturb=False,
                dt_gamma=self.nerf_opt.dt_gamma,
                max_steps=self.nerf_opt.max_steps,
            )
        
        pred_rgb = outputs["image"].reshape(self.H, self.W, 3)
        pred_rgb = pred_rgb.clamp(0, 1)
        
        if self.nerf_opt.color_space == "linear":
            pred_rgb = linear_to_srgb(pred_rgb)
        
        return pred_rgb
    
    def enhance_frame(self, frame):
        """Enhance frame with StyleUNet"""
        styleunet = self.load_styleunet()
        if styleunet is None:
            return frame
        
        with torch.inference_mode():
            inp = frame.permute(2, 0, 1).unsqueeze(0)
            
            if inp.shape[-1] != self.styleunet_input_size:
                inp = torch.nn.functional.interpolate(
                    inp, size=(self.styleunet_input_size, self.styleunet_input_size),
                    mode="bilinear", align_corners=False
                )
            
            inp_norm = (inp - 0.5) * 2.0
            
            # Use base pose angle for StyleUNet
            style_pose = torch.from_numpy(self.pose0_angle).float().unsqueeze(0).to(self.device)
            out = styleunet(inp_norm, style_pose)
            
            out = (out * 0.5 + 0.5).clamp(0, 1)
            
            if out.shape[-1] != self.H:
                out = torch.nn.functional.interpolate(
                    out, size=(self.H, self.W),
                    mode="bilinear", align_corners=False
                )
            
            enhanced = out[0].permute(1, 2, 0)
        
        return enhanced
    
    def generate_video(self, audio_path, output_path, use_styleunet=True):
        """Generate complete video"""
        print("=" * 70)
        print("GtsTalkNeRF Video Generation Pipeline")
        print("=" * 70)
        
        print("\n[Stage 1] Processing audio...")
        wav, formant, num_frames = self.process_audio(audio_path)
        print(f"  Audio: {num_frames} frames at 25 fps")
        
        print("\n[Stage 2] Generating landmarks...")
        landmarks, audio_features = self.generate_landmarks(wav, formant)
        print(f"  Generated {len(landmarks)} landmark frames")
        
        print("\n[Stage 3] Loading NeRF model...")
        self.load_nerf()
        
        if use_styleunet:
            print("\n[Stage 4] Loading StyleUNet...")
            self.load_styleunet()
        
        bg_path = self.model_path / "bg.png"
        if bg_path.exists():
            bg_img = cv2.imread(str(bg_path))
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (self.W, self.H))
            bg_img = torch.from_numpy(bg_img).float().to(self.device) / 255.0
        else:
            bg_img = torch.ones(self.H, self.W, 3, device=self.device)
        
        bg_flat = bg_img.reshape(-1, 3)
        
        print("\n[Stage 5] Rendering frames...")
        frames = []
        
        for i in tqdm(range(len(landmarks)), desc="Rendering"):
            frame = self.render_frame(landmarks[i], audio_features[i], i, bg_flat)
            
            if use_styleunet and self.styleunet is not None:
                frame = self.enhance_frame(frame)
            
            frame_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            frames.append(frame_np)
        
        print(f"\n[Stage 6] Saving video: {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        imageio.mimwrite(str(output_path), frames, fps=25, quality=9, macro_block_size=1)
        
        print("[Stage 7] Adding audio...")
        self._add_audio(str(output_path), str(audio_path))
        
        print("=" * 70)
        print(f"Video generated: {output_path}")
        print(f"Resolution: {self.W}x{self.H} @25fps")
        print(f"Frames: {len(frames)}")
        print("=" * 70)
        
        return str(output_path)
    
    def _add_audio(self, video_path, audio_path):
        """Add audio using ffmpeg"""
        temp_video = video_path.replace(".mp4", "_no_audio.mp4")
        os.rename(video_path, temp_video)
        
        cmd = f'ffmpeg -y -i "{temp_video}" -i "{audio_path}" -c:v copy -c:a aac -shortest "{video_path}" -loglevel error'
        ret = os.system(cmd)
        
        if ret == 0 and os.path.exists(video_path):
            os.remove(temp_video)
            print(f"  Audio added successfully")
        else:
            os.rename(temp_video, video_path)
            print("  Warning: Failed to add audio")


def inference(model_path: str, 
              audio_path: str, 
              output_path: str, 
              device: str = "cuda", 
              use_styleunet: bool = True) -> str:
    """
    GtsTalkNeRF 视频生成主函数 - 后端调用接口
    
    输入"模型路径+音频"，输出"视频"
    
    Pipeline: Audio -> AudioFace -> Landmarks -> NeRF -> StyleUNet -> Video
    
    Args:
        model_path (str): 训练好的模型目录路径
            - 例如: "results/obama"
            - 目录结构要求:
                ├── transforms_train.json   # 相机参数和基础姿态
                ├── bg.png                  # 背景图片
 logs/                └─
                    ├── stage1/*.tar        # AudioFace 模型
                    ├── stage2/checkpoints/ # NeRF 模型
                    └── checkpoint/         # StyleUNet 模型
        
        audio_path (str): 输入音频文件路径
            - 支持格式: .wav, .mp3 等 torchaudio 支持的格式
            - 建议: 16kHz 采样 WAV 文件
            - 例如: "input.wav"
        
        output_path (str): 输出视频文件路径
            - 例如: "output.mp4"
            - 目录不存在会自动创建
        
        device (str, optional): 计算设备
            - "cuda": 使用 GPU (默认，推荐)
            - "cpu": 使用 CPU (较慢)
        
        use_styleunet (bool, optional): 是否使用 StyleUNet 增强
            - True: 启用增强，输出更清晰 (默认)
            - False: 跳过增强，速度更快
    
    Returns:
        str: 生成的视频 (与 output_path 相同)
    
    Raises:
        FileNotFoundError: 模型路径或音频文件不存在
        RuntimeError: CUDA 不可用但指定了 cuda 设备
    
    Example:
        >>> from inference import inference
        >>> 
        >>> # 基础用法
        >>> video_path = inference(
        ...     model_path="results/obama",
        ...     audio_path="speech.wav", 
        ...     output_path="talking_head.mp4"
        ... )
        >>> print(f"视频已生成: {video_path}")
        
        >>> # 快速模式 (不使用 StyleUNet)
        >>> video_path = inference(
        ...     model_path="results/obama",
        ...     audio_path="speech.wav",
        ...     output_path="output_fast.mp4",
        ...     use_styleunet=False
        ... )
    
    Note:
        - 输出视频分辨率: 512x512 @ 25fps
        - 视频会自动添加音频轨道
        - 人物在画面中居中显示
    """
    generator = TalkingHeadGenerator(model_path, device)
    return generator.generate_video(audio_path, output_path, use_styleunet)


def main():
    """Command line entry"""
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except AttributeError:
        pass
    
    parser = argparse.ArgumentParser(description="GtsTalkNeRF Inference")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--audio", type=str, required=True,
                       help="Input audio file (.wav)")
    parser.add_argument("--output", type=str, default="output.mp4",
                       help="Output video file")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--no_styleunet", action="store_true",
                       help="Disable StyleUNet enhancement")
    
    args = parser.parse_args()
    
    try:
        output = inference(
            model_path=args.model_path,
            audio_path=args.audio,
            output_path=args.output,
            device=args.device,
            use_styleunet=not args.no_styleunet
        )
        print(f"\nSuccess! Video saved to: {output}")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
