import requests
import os

# 智谱 AI API Key (建议通过环境变量设置)
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "fefd1a9d89454eb7b1d390d2bbc71e2d.4ifJ5T0T0EC632MP")

def speech_to_text(audio_path: str) -> str:
    """
    将音频文件转换为文字 (使用智谱 AI glm-asr-2512 模型)
    
    Args:
        audio_path: 音频文件的绝对路径
        
    Returns:
        str: 识别出的文字内容
    """
    url = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {ZAI_API_KEY}"
    }
    
    data = {
        "model": "glm-asr-2512",
        "stream": "false"
    }
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件未找到: {audio_path}")
        
    try:
        with open(audio_path, "rb") as f:
            files = {
                "file": f
            }
            response = requests.post(url, headers=headers, data=data, files=files)
            
        response.raise_for_status()
        result = response.json()
        
        # 根据智谱 API 文档，结果通常在 'text' 字段中
        return result.get("text", "")
        
    except Exception as e:
        print(f"[ASR Error] {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # 测试代码
    test_audio = "test.mp3"
    print(speech_to_text(test_audio))