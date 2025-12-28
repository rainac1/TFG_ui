# GtsTalkNeRF

本项目是一个基于 NeRF 的数字人生成系统。

## 🚀 快速开始

### 1. 环境准备

在构建镜像之前，请确保已拉取所有子模块：

```bash
git submodule update --init --recursive
```

### 2. 配置文件修改

由于路径配置需求，需要将 `GtsTalkNeRF/src/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml` 文件的前 8 行替换为以下内容：

```yaml
custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cuda
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1v3.ckpt
  version: v2Pro
  vits_weights_path: GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
```

### 3. 构建镜像

主应用位于 `GtsTalkNeRF` 目录下，使用 `docker/Dockerfile.app` 进行构建。

> **注意**：构建过程中需要从 GitHub、Anaconda 和 PyPI 下载大量资源，请务必配置好网络代理。

```bash
# 在 GtsTalkNeRF 根目录下执行
docker build -f docker/Dockerfile.app -t gtstalknerf-app:latest .
```

### 4. 运行应用

使用根目录下的 `docker-compose.yml` 启动服务。

> **注意**：模型运行时可能需要访问外网下载预训练模型等，请确保容器环境的网络代理已适当配置。

```bash
docker-compose up -d
```

启动后，可以通过浏览器访问 `http://localhost:7860` 进入前端界面。

## 数据准备

在运行应用之前，您需要手动下载以下必要的数据和模型：

1. **Upstream 数据**：包含基础模型和必要的数据文件。
    * 下载地址：[Google Drive](https://drive.google.com/drive/folders/1uo1sYMFwVzTfmSYtSBb0TSCnyC2LbZW_?usp=sharing)
    * 下载后请解压按正确的目录结构放置在 `data/upstream/` 目录下。

2. **GPT-SoVITS 预训练模型**：
    * 下载地址：[Hugging Face](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)
    * 下载后请按正确的目录结构放置在 `data/gpt-sovits/pretrained_models/` 目录下。

## 数据目录结构

为了使应用正常运行，请确保 `data` 目录结构如下（部分目录在运行过程中会自动生成）：

```text
GtsTalkNeRF/data/
├── backend_data
├── checkpoints
├── gpt-sovits
│   └── pretrained_models
│       ├── chinese-hubert-base
│       │   ├── config.json
│       │   ├── preprocessor_config.json
│       │   └── pytorch_model.bin
│       ├── chinese-roberta-wwm-ext-large
│       │   ├── config.json
│       │   ├── pytorch_model.bin
│       │   └── tokenizer.json
│       ├── fast_langdetect
│       │   └── lid.176.bin
│       ├── s1v3.ckpt
│       ├── sv
│       │   └── pretrained_eres2netv2w24s4ep4.ckpt
│       └── v2pro
│           └── s2Gv2Pro.pth
└── upstream
    ├── data
    │   ├── FLAME2020
    │   │   ├── FLAME_masks
    │   │   │   ├── FLAME_masks.gif
    │   │   │   ├── FLAME_masks.pkl
    │   │   │   └── readme
    │   │   ├── FLAME_texture.npz
    │   │   ├── female_model.pkl
    │   │   ├── generic_model.pkl
    │   │   ├── head_template_color.obj
    │   │   ├── head_template_mesh.obj
    │   │   ├── landmark_embedding.npy
    │   │   ├── male_model.pkl
    │   │   └── uv_mask_eyes.png
    │   ├── pretrained
    │   │   ├── face_parsing
    │   │   │   └── 79999_iter.pth
    │   │   ├── mica.tar
    │   │   └── u2net_human_seg.onnx
    │   └── voca
    │       ├── raw_audio_fixed.pkl
    │       ├── subj_seq_to_idx.pkl
    │       └── templates.pkl
    ├── insightface
    │   └── models
    │       ├── antelopev2
    │       │   ├── 1k3d68.onnx
    │       │   ├── 2d106det.onnx
    │       │   ├── genderage.onnx
    │       │   ├── glintr100.onnx
    │       │   └── scrfd_10g_bnkps.onnx
    │       ├── antelopev2.zip
    │       ├── buffalo_l
    │       │   ├── 1k3d68.onnx
    │       │   ├── 2d106det.onnx
    │       │   ├── det_10g.onnx
    │       │   ├── genderage.onnx
    │       │   └── w600k_r50.onnx
    │       └── buffalo_l.zip
    └── torch-cache
        └── hub
            └── ...
```

## 指标计算

如果您需要对生成的视频进行质量评测（如 NIQE, PSNR, SSIM, FID, LSE-C, LSE-D 等），请参考 `evaluation` 目录下的说明文档：

[evaluation/Readme.md](evaluation/Readme.md)

## 网络代理配置

如果您的服务器无法直接访问外网，可以在构建或运行时通过以下方式配置代理：

**构建时：**

```bash
docker build --build-arg http_proxy=http://your-proxy:port --build-arg https_proxy=http://your-proxy:port -f docker/Dockerfile.app -t gtstalknerf-app:latest .
```

**运行时（修改 docker-compose.yml）：**

```yaml
services:
  gtstalknerf-app:
    environment:
      - http_proxy=http://your-proxy:port
      - https_proxy=http://your-proxy:port
```
