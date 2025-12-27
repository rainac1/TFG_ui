import os
from zai import ZhipuAiClient
from zai.types.chat.chat_completion import Completion

# 智谱 AI API Key
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "fefd1a9d89454eb7b1d390d2bbc71e2d.4ifJ5T0T0EC632MP")

# 初始化客户端
client = ZhipuAiClient(api_key=ZAI_API_KEY)

def get_llm_response(prompt: str) -> str:
    """
    调用大模型获取回答
    
    Args:
        prompt: 用户输入的文字
        model: 模型名称，默认为 glm-4.5-flash
        
    Returns:
        str: 大模型生成的回答
    """
    try:
        model = "glm-4.5-flash"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个聊天助手，你的任务是回应用户的任何话语，回复长度应当少于50字，语言风格为日常对话。"},
                {"role": "user", "content": prompt}
            ],
            thinking={
                "type": "disabled",
            },
            stream=False,
            max_tokens=256,
            temperature=1
        )
        
        # stream=False 时，直接获取内容
        if isinstance(response, Completion):
            if response.choices:
                content = response.choices[0].message.content
                if content is not None:
                    return str(content)
        return "无回答内容"
        
    except Exception as e:
        print(f"[LLM Error] {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # 测试代码
    print(get_llm_response("今天天气不错。"))
