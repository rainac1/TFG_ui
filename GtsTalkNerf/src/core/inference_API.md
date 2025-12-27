# 后端 Python API 接口文档

## 文件概览

api_interface.py 提供了后端可调用的 Python 接口：

| 类/函数 | 说明 |
| -------- | ------ |
| `GtsTalkService` | 主服务类，支持模型预加载 |
| `process()` | 推理接口：音频 → 视频 |
| `evaluate()` | 指标计算接口 |
| `get_status()` | 获取服务状态 |
| `release()` | 释放资源 |
| `quick_inference()` | 便捷函数（单次使用） |

## 后端集成示例

```python
from api_interface import GtsTalkService

# 1. 服务启动时：一次加载模型
service = GtsTalkService("results/obama", device="cuda")

# 2. 处理请求（可多次调用）
result = service.process("audio.wav", "output.mp4")
# 返回: {"status": "success", "video_path": "...", "resolution": "512x512", "fps": 25}

# 3. 评估质量
metrics = service.evaluate("output.mp4", metrics=['niqe'])
# 返回: {"status": "success", "data": {"niqe": {"mean": 3.68, ...}}}

# 4. 获取状态
status = service.get_status()

# 5. 释放资源
service.release()
```
