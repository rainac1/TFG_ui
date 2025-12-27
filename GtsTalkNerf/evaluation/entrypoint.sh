#!/bin/bash
# 评测Docker入口脚本

set -e

# 显示帮助信息
show_help() {
    echo "============================================================"
    echo "视频评测 Docker 工具"
    echo "============================================================"
    echo ""
    echo "支持指标: NIQE, PSNR, SSIM, FID, LSE-C, LSE-D"
    echo ""
    echo "用法:"
    echo "  docker run -v <生成视频目录>:/input/generated \\"
    echo "             -v <GT视频目录>:/input/groundtruth \\"
    echo "             -v <输出目录>:/output \\"
    echo "             video-eval [选项]"
    echo ""
    echo "选项:"
    echo "  --gen_dir     生成视频目录 (默认: /input/generated)"
    echo "  --gt_dir      GT视频目录 (默认: /input/groundtruth)"
    echo "  --audio_dir   音频目录 (默认: /input/audio)"
    echo "  --output      输出文件路径 (默认: /output/evaluation_results)"
    echo "  --sample_frames 采样帧数 (默认: 50)"
    echo "  --help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 基本使用 (仅NIQE和LSE)"
    echo "  docker run --gpus all \\"
    echo "      -v /path/to/generated:/input/generated \\"
    echo "      -v /path/to/output:/output \\"
    echo "      video-eval"
    echo ""
    echo "  # 完整评测 (全部6项指标)"
    echo "  docker run --gpus all \\"
    echo "      -v /path/to/generated:/input/generated \\"
    echo "      -v /path/to/groundtruth:/input/groundtruth \\"
    echo "      -v /path/to/output:/output \\"
    echo "      video-eval --sample_frames 100"
    echo ""
    echo "输出文件:"
    echo "  - evaluation_results.csv   (CSV格式)"
    echo "  - evaluation_results.xlsx  (Excel格式)"
    echo "  - evaluation_results.json  (JSON格式)"
    echo "============================================================"
}

# 检查是否请求帮助
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ -z "$1" ]; then
    show_help
    exit 0
fi

# 设置默认值
GEN_DIR=${GEN_DIR:-/input/generated}
GT_DIR=${GT_DIR:-/input/groundtruth}
AUDIO_DIR=${AUDIO_DIR:-/input/audio}
OUTPUT=${OUTPUT:-/output/evaluation_results}
SAMPLE_FRAMES=${SAMPLE_FRAMES:-50}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gen_dir)
            GEN_DIR="$2"
            shift 2
            ;;
        --gt_dir)
            GT_DIR="$2"
            shift 2
            ;;
        --audio_dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --sample_frames)
            SAMPLE_FRAMES="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 检查生成视频目录
if [ ! -d "$GEN_DIR" ]; then
    echo "错误: 生成视频目录不存在: $GEN_DIR"
    echo "请使用 -v 挂载目录，例如: -v /your/path:/input/generated"
    exit 1
fi

# 检查是否有视频文件
VIDEO_COUNT=$(find "$GEN_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "错误: 生成视频目录中没有找到视频文件: $GEN_DIR"
    exit 1
fi

echo "============================================================"
echo "开始视频评测"
echo "============================================================"
echo "生成视频目录: $GEN_DIR (共 $VIDEO_COUNT 个视频)"

# 构建命令
CMD="python /app/batch_evaluation.py --gen_dir $GEN_DIR --output $OUTPUT --sample_frames $SAMPLE_FRAMES"

# 检查GT目录
if [ -d "$GT_DIR" ]; then
    GT_COUNT=$(find "$GT_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) 2>/dev/null | wc -l)
    if [ "$GT_COUNT" -gt 0 ]; then
        echo "GT视频目录: $GT_DIR (共 $GT_COUNT 个视频)"
        CMD="$CMD --gt_dir $GT_DIR"
    else
        echo "GT视频目录: 未提供或为空 (跳过PSNR/SSIM/FID)"
    fi
else
    echo "GT视频目录: 未提供 (跳过PSNR/SSIM/FID)"
fi

# 检查音频目录
if [ -d "$AUDIO_DIR" ]; then
    AUDIO_COUNT=$(find "$AUDIO_DIR" -maxdepth 1 -type f \( -name "*.wav" -o -name "*.mp3" \) 2>/dev/null | wc -l)
    if [ "$AUDIO_COUNT" -gt 0 ]; then
        echo "音频目录: $AUDIO_DIR (共 $AUDIO_COUNT 个音频)"
        CMD="$CMD --audio_dir $AUDIO_DIR"
    fi
fi

echo "采样帧数: $SAMPLE_FRAMES"
echo "输出路径: $OUTPUT"
echo "============================================================"

# 执行评测
exec $CMD
