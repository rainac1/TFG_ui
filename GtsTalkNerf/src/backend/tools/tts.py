import requests
import os

def text_to_speech(text: str, ref_wav: str, ref_text: str, output_path: str, api_url: str = "http://127.0.0.1:9880/tts"):
    """
    调用 GPT-SoVITS API 进行 Zero-shot 声音克隆
    
    Args:
        text: 想要合成的目标文本
        ref_wav: 参考音频在容器内的绝对路径
        ref_text: 参考音频对应的文字内容
        output_path: 输出音频路径
        api_url: API 接口地址
    """
    payload = {
        "text": text,                   # 想要合成的目标文本
        "text_lang": "auto",              # 目标文本语种: zh, en, ja, ko, yue, auto
        "ref_audio_path": ref_wav,      # 参考音频在容器内的绝对路径
        "prompt_text": ref_text,        # 参考音频对应的文字内容
        "prompt_lang": "auto"             # 参考音频的语种
    }

    try:
        # 发送 POST 请求
        response = requests.post(api_url, json=payload, timeout=60)

        # 检查响应状态
        if response.status_code == 200:
            # 保存音频文件
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"❌ 合成失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False

    except Exception as e:
        print(f"发生异常: {e}")
        return False

'''
调用示例：
import requests
import json
import argparse

def test_zero_shot_tts(api_url, text, ref_wav, ref_text, ref_lang, target_lang, output_path):
    """
    调用 GPT-SoVITS API 进行 Zero-shot 声音克隆测试
    """
    payload = {
        "text": text,                   # 想要合成的目标文本
        "text_lang": target_lang,   # 目标文本语种: zh, en, ja, ko, yue, auto
        "ref_audio_path": ref_wav,      # 参考音频在容器内的绝对路径
        "prompt_text": ref_text,        # 参考音频对应的文字内容
        "prompt_lang": ref_lang     # 参考音频的语种
    }

    print(f"正在发送请求到: {api_url}")
    print(f"合成文本: {text}")
    print(f"使用参考音频: {ref_wav}")

    try:
        # 发送 POST 请求
        response = requests.post(api_url, json=payload, timeout=60)

        # 检查响应状态
        if response.status_code == 200:
            # 保存音频文件
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ 合成成功！音频已保存至: {output_path}")
        else:
            print(f"❌ 合成失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"发生异常: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS API 测试脚本")

    # 基础配置
    parser.add_argument("--url", type=str, default="http://127.0.0.1:9880/tts", help="API 接口地址")
    parser.add_argument("--output", type=str, default="test_output.wav", help="输出音频路径")

    # 推理参数
    parser.add_argument("--text", type=str, default="你好，这是一段使用零样本克隆技术生成的测试音频。", help="目标文本")
    parser.add_argument("--target_lang", type=str, default="zh", help="目标语种")

    # 参考音频参数 (必须提供)
    parser.add_argument("--ref_wav", type=str, required=True, help="容器内参考音频的绝对路径")
    parser.add_argument("--ref_text", type=str, required=True, help="参考音频的文本内容")
    parser.add_argument("--ref_lang", type=str, default="zh", help="参考音频语种")

    args = parser.parse_args()

    test_zero_shot_tts(
        args.url,
        args.text,
        args.ref_wav,
        args.ref_text,
        args.ref_lang,
        args.target_lang,
        args.output
    )
'''