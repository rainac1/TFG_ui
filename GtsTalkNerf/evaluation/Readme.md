# 视频评测 Docker 镜像

## 📋 概述

独立的视频质量评测 Docker 镜像，支持批量评测并输出 CSV/Excel 格式的评测报告。

### 支持的评测指标

| 指标 | 类型 | 说明 | 最优值 |
| --- | --- | --- | --- |
| **NIQE** | 无参考 | 自然图像质量 | 越低越好 (2-10) |
| **PSNR** | 全参考 | 峰值信噪比 | 越高越好 (>30dB) |
| **SSIM** | 全参考 | 结构相似度 | 越高越好 (0-1) |
| **FID** | 分布距离 | 特征分布距离 | 越低越好 (<50) |
| **LSE-C** | 口型同步 | 同步置信度 | 越低越好 (0-10) |
| **LSE-D** | 口型同步 | 同步延迟(秒) | 越低越好 (<0.2s) |

## 🚀 快速开始

### 1. 构建镜像

```bash
cd evaluation
docker build -t video-eval .
```

### 2. 运行评测

#### 基本用法（仅NIQE和LSE）

```bash
docker run --gpus all \
    -v /path/to/generated_videos:/input/generated \
    -v /path/to/output:/output \
    video-eval
```

#### 完整评测（全部6项指标）

```bash
docker run --gpus all \
    -v /path/to/generated_videos:/input/generated \
    -v /path/to/groundtruth_videos:/input/groundtruth \
    -v /path/to/output:/output \
    video-eval
```

#### 指定音频目录

```bash
docker run --gpus all \
    -v /path/to/generated_videos:/input/generated \
    -v /path/to/groundtruth_videos:/input/groundtruth \
    -v /path/to/audio_files:/input/audio \
    -v /path/to/output:/output \
    video-eval --sample_frames 100
```

## 📁 目录结构

### 输入目录

```file_structure
/input/
├── generated/          # 生成的视频 (必需)
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── groundtruth/        # GT视频 (可选，用于PSNR/SSIM/FID)
│   ├── video1.mp4      # 文件名需与生成视频匹配
│   ├── video2.mp4
│   └── ...
└── audio/              # 音频文件 (可选，用于LSE)
    ├── video1.wav
    ├── video2.wav
    └── ...
```

### 输出目录

```file_structure
/output/
├── evaluation_results.csv    # CSV格式结果
├── evaluation_results.xlsx   # Excel格式结果
└── evaluation_results.json   # JSON格式结果
```

## ⚙️ 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--gen_dir` | 生成视频目录 | `/input/generated` |
| `--gt_dir` | GT视频目录 | `/input/groundtruth` |
| `--audio_dir` | 音频文件目录 | `/input/audio` |
| `--output` | 输出文件路径(不含扩展名) | `/output/evaluation_results` |
| `--sample_frames` | 每个视频采样帧数 | `50` |
| `--help` | 显示帮助信息 | - |

## 📊 输出示例

### CSV/Excel 格式

| video_name | NIQE | PSNR | SSIM | FID | LSE_C | LSE_D |
| --- | --- | --- | --- | --- | --- | --- |
| video1 | 3.72 | 28.5 | 0.89 | 45.2 | 2.1 | 0.08 |
| video2 | 3.85 | 27.8 | 0.87 | 48.7 | 2.4 | 0.12 |
| AVERAGE | 3.78 | 28.2 | 0.88 | 47.0 | 2.3 | 0.10 |

### JSON 格式

```json
[
  {
    "video_name": "video1",
    "NIQE": 3.72,
    "PSNR": 28.5,
    "SSIM": 0.89,
    "FID": 45.2,
    "LSE_C": 2.1,
    "LSE_D": 0.08
  }
]
```

## 🔧 文件匹配规则

GT视频和音频文件通过文件名与生成视频匹配：

- **完全匹配**: `video1.mp4` ↔ `video1.mp4`
- **前缀匹配**: `video1.mp4` ↔ `video1_gt.mp4`
- **包含匹配**: `video1.mp4` ↔ `gt_video1_final.mp4`

## 🐳 Docker Compose 示例

```yaml
services:
  video-eval:
    image: video-eval
    volumes:
      - ./generated:/input/generated:ro
      - ./groundtruth:/input/groundtruth:ro
      - ./audio:/input/audio:ro
      - ./results:/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [ gpu ]
    command: ["--sample_frames", "100"]
```

运行：

```bash
docker-compose run video-eval
```

## 📝 注意事项

1. **GPU支持**: 使用 `--gpus all` 启用GPU加速FID和NIQE计算
2. **内存需求**: 建议至少8GB内存用于FID计算
3. **视频格式**: 支持 `.mp4`, `.avi`, `.mov`, `.mkv`
4. **音频格式**: 支持 `.wav`, `.mp3`
5. **PSNR/SSIM**: 需要逐帧对应的GT视频，否则结果无意义

## 🔍 指标解读

### NIQE (Natural Image Quality Evaluator)

- 无参考图像质量评估
- 范围: 通常 2-10
- 越低越好，<4 表示高质量

### PSNR (Peak Signal-to-Noise Ratio)  

- 峰值信噪比，衡量像素级差异
- 单位: dB
- 越高越好，>30dB 表示高质量

### SSIM (Structural Similarity Index)

- 结构相似度
- 范围: 0-1
- 越高越好，>0.9 表示高相似度

### FID (Fréchet Inception Distance)

- 特征分布距离
- 越低越好，<50 表示高质量

### LSE-C (Lip Sync Error - Confidence)

- 口型同步置信度误差
- 范围: 0-10
- 越低越好，<3 表示良好同步

### LSE-D (Lip Sync Error - Distance)

- 口型同步时间偏移
- 单位: 秒
- 越低越好，<0.2秒 在人类感知阈值内
