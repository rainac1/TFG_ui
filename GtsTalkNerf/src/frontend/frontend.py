import os
import time
import re
import requests
import gradio as gr
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

API_BASE = os.environ.get("TFG_API_BASE", "http://127.0.0.1:5001").rstrip("/")
DEFAULT_MODELS = ["obama", "trump", "custom_model_01"]
DEFAULT_DEVICES = ["cuda:0", "cpu"]

# --- API Helpers ---
def api_get(path: str, **kwargs):
    try:
        return requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)
    except Exception as e:
        print(f"API Error: {e}")
        return None

def api_post_multipart(path: str, files: dict, data: dict):
    try:
        return requests.post(f"{API_BASE}{path}", files=files, data=data, timeout=60)
    except Exception as e:
        print(f"API Error: {e}")
        return None

def get_models():
    r = api_get("/api/models")
    if r and r.status_code == 200:
        body = r.json() or {}
        models = body.get("models")
        if isinstance(models, list):
            return [str(m) for m in models]
        
        # Legacy support
        data = body.get("data") or {}
        synctalk_models = data.get("synctalk_models")
        if isinstance(synctalk_models, list):
            return [str(m) for m in synctalk_models]
            
    return DEFAULT_MODELS

# --- Helper Functions ---
def ansi_to_html(text):
    # Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # ANSI color codes map
    colors = {
        '30': 'black', '31': 'red', '32': 'green', '33': '#e6db74', # yellow
        '34': 'blue', '35': 'magenta', '36': 'cyan', '37': 'white',
        '90': 'gray', '91': 'lightcoral', '92': 'lightgreen', '93': 'lightyellow',
        '94': 'lightblue', '95': 'lightpink', '96': 'lightcyan', '97': 'white',
    }
    
    # Reset code
    text = re.sub(r'\x1b\[0?m', '</span>', text)
    
    # Color codes
    for code, color in colors.items():
        # Match \x1b[31m or \x1b[0;31m
        text = re.sub(r'\x1b\[(?:0;)?' + code + 'm', f'<span style="color:{color}">', text)
        
    # Remove remaining ANSI codes
    text = re.sub(r'\x1b\[[\d;]*m', '', text)
    
    return text.replace("\n", "<br>")

def get_devices():
    r = api_get("/api/hardware_info")
    if r and r.status_code == 200:
        body = r.json() or {}
        devices = body.get("devices")
        if isinstance(devices, list) and devices:
             return [str(d["id"]) for d in devices]
    return DEFAULT_DEVICES

def fetch_status(task_id):
    if not task_id:
        return "无任务", None, ""
    
    # Strip any whitespace
    task_id = task_id.strip()
    
    r = api_get(f"/api/status/{task_id}")
    status_info = {}
    
    if r and r.status_code == 200:
        status_info = r.json() or {}
    else:
        # Try legacy endpoint
        rr = api_get(f"/api/tasks/{task_id}")
        if rr and rr.status_code == 200:
            body = rr.json() or {}
            data = body.get("data") or {}
            status_info = {
                "status": data.get("state"),
                "result": data.get("result"),
                "error": data.get("error"),
                "log_tail": data.get("log_tail") or []
            }
    
    if not status_info:
        return "查询失败或任务不存在", None, ""

    status = status_info.get("status")
    result = status_info.get("result")
    error = status_info.get("error")
    log_tail = status_info.get("log_tail") or []
    
    msg = f"状态: {status}"
    if error:
        msg += f"\n错误: {error}"
    
    video_url = None
    if status in ["completed", "done"] and result:
        if isinstance(result, str):
            # If it's a local static path, try to use the absolute local path to avoid SSRF issues with 127.0.0.1
            if result.startswith("/static/"):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                local_path = os.path.join(base_dir, "backend", result.lstrip("/"))
                if os.path.exists(local_path):
                    video_url = local_path
                else:
                    video_url = f"{API_BASE}{result}"
            else:
                video_url = f"{API_BASE}{result}" if result.startswith("/") else result
    
    # Show last 500 lines
    logs = "\n".join(log_tail[-500:]) if isinstance(log_tail, list) else str(log_tail)
    logs_html = f"""
    <div id="log-container" style="height: 450px; overflow-y: scroll; background-color: #1e1e1e; color: #d4d4d4; padding: 10px; font-family: monospace; white-space: pre-wrap; border-radius: 5px; border: 1px solid #333; font-size: 12px; line-height: 1.4;">
    {ansi_to_html(logs)}
    <div id="log-end"></div>
    </div>
    <script>
    var container = document.getElementById('log-container');
    if (container) {{
        container.scrollTop = container.scrollHeight;
    }}
    </script>
    """
    return msg, video_url, logs_html

# --- Action Functions ---
def train_action(video, stage1_epochs, stage2_iters, eye_iters, lip_iters, styleunet_iters, device):
    if not video:
        return "请上传视频", None
    
    filename = os.path.basename(video)
    with open(video, "rb") as f:
        video_bytes = f.read()
        
    files = {"video": (filename, video_bytes)}
    data = {
        "device": device,
        "stage1_epochs": str(int(stage1_epochs)),
        "stage2_iters": str(int(stage2_iters)),
        "eye_iters": str(int(eye_iters)),
        "lip_iters": str(int(lip_iters)),
        "styleunet_iters": str(int(styleunet_iters))
    }
    
    r = api_post_multipart("/api/train", files=files, data=data)
    if r and r.status_code == 200:
        task_id = r.json().get("task_id")
        return f"任务已提交成功! Task ID: {task_id}\n请到'任务中心'查看进度。", task_id
    return "提交失败，请检查后端连接。", None

def infer_action(model_name, audio, device):
    if not audio:
        yield "请上传音频", None, None
        return
    if not model_name:
        yield "请选择模型", None, None
        return
        
    filename = os.path.basename(audio)
    with open(audio, "rb") as f:
        audio_bytes = f.read()
        
    files = {"audio": (filename, audio_bytes)}
    data = {"model_name": model_name, "device": device}
    
    r = api_post_multipart("/api/inference", files=files, data=data)
    if r and r.status_code == 200:
        task_id = r.json().get("task_id")
        yield "任务已提交，正在处理...", task_id, gr.update(label="生成结果 (思考中，请稍等...)")
        
        # Polling for status
        while True:
            time.sleep(2)
            msg, video_url, logs = fetch_status(task_id)
            if "completed" in msg or "done" in msg:
                yield f"处理完成！\n{msg}", task_id, gr.update(value=video_url, label="生成结果")
                break
            elif "failed" in msg or "error" in msg or "失败" in msg:
                yield f"处理失败！\n{msg}", task_id, gr.update(value=None, label="生成结果 (处理失败)")
                break
            else:
                yield f"{msg}", task_id, gr.update(value=None, label="生成结果 (思考中，请稍等...)")
    else:
        yield "提交失败，请检查后端连接。", None, None

def chat_action(model_name, ref_audio, user_audio, device):
    if not model_name:
        yield "请选择模型", None
        return
    if not ref_audio:
        yield "请上传参考音频", None
        return
    if not user_audio:
        yield "请提供用户语音", None
        return

    try:
        files = [
            ('ref_audio', (os.path.basename(ref_audio), open(ref_audio, 'rb'), 'audio/wav')),
            ('user_audio', (os.path.basename(user_audio), open(user_audio, 'rb'), 'audio/wav'))
        ]
        
        data = {
            "model_name": model_name,
            "device": device
        }
        
        # Submit chat task
        r = requests.post(f"{API_BASE}/api/chat", files=files, data=data, timeout=120)
        
        for _, (_, f, _) in files:
            f.close()
            
        if r.status_code == 200:
            task_id = r.json().get("task_id")
            yield "对话任务已提交，正在处理...", gr.update(label="数字人回复视频 (思考中，请稍等...)")
            
            # Polling for status
            while True:
                time.sleep(2)
                msg, video_url, logs = fetch_status(task_id)
                if "completed" in msg or "done" in msg:
                    yield f"对话生成成功！\n{msg}", gr.update(value=video_url, label="数字人回复视频")
                    break
                elif "failed" in msg or "error" in msg or "失败" in msg:
                    yield f"对话生成失败！\n{msg}", gr.update(value=None, label="数字人回复视频 (生成失败)")
                    break
                else:
                    yield f"{msg}", gr.update(value=None, label="数字人回复视频 (思考中，请稍等...)")
        else:
            yield f"请求失败: {r.status_code} - {r.text}", None
            
    except Exception as e:
        yield f"发生错误: {str(e)}", None

def update_model_list():
    return gr.Dropdown(choices=get_models())

def update_device_list():
    return gr.Dropdown(choices=get_devices())

with gr.Blocks(title="GtsTalk") as app:
    gr.Markdown("# GtsTalkNeRF 数字人生成与实时对话系统")
    
    with gr.Tabs():
        # --- Train Tab ---
        with gr.Tab("训练 (Train)"):
            gr.Markdown("### 上传参考视频，启动训练任务，产出模型")
            with gr.Row():
                with gr.Column():
                    train_video = gr.Video(label="上传训练视频", sources=["upload"])
                    with gr.Accordion("高级训练参数", open=False):
                        train_stage1_epochs = gr.Slider(1, 100, value=30, step=1, label="Stage 1 Epochs (Landmark)")
                        train_stage2_iters = gr.Slider(1000, 200000, value=50000, step=1000, label="Stage 2 Base Iters")
                        train_eye_iters = gr.Slider(1000, 100000, value=20000, step=1000, label="Eye Finetune Iters")
                        train_lip_iters = gr.Slider(1000, 100000, value=40000, step=1000, label="Lip Finetune Iters")
                        train_styleunet_iters = gr.Slider(1000, 100000, value=30000, step=1000, label="StyleUNet Iters")
                    train_device = gr.Dropdown(choices=get_devices(), value=get_devices()[0] if get_devices() else None, label="设备")
                    train_btn = gr.Button("开始训练", variant="primary")
                with gr.Column():
                    train_status_msg = gr.Textbox(label="提交状态", interactive=False, lines=3)
                    train_task_id = gr.Textbox(label="Task ID (可复制到任务中心查询)", interactive=True)
        
        # --- Inference Tab ---
        with gr.Tab("推理 (Inference)"):
            gr.Markdown("### 选择模型 + 上传音频，一键生成视频")
            with gr.Row():
                with gr.Column():
                    infer_model = gr.Dropdown(choices=get_models(), label="选择模型", interactive=True)
                    infer_audio = gr.Audio(type="filepath", label="上传驱动音频", sources=["upload", "microphone"])
                    infer_device = gr.Dropdown(choices=get_devices(), value=get_devices()[0] if get_devices() else None, label="设备")
                    infer_refresh_btn = gr.Button("刷新模型列表", size="sm")
                    infer_btn = gr.Button("开始推理", variant="primary")
                with gr.Column():
                    infer_status_msg = gr.Textbox(label="提交状态", interactive=False, lines=3)
                    infer_task_id = gr.Textbox(label="Task ID (可复制到任务中心查询)", interactive=True)
                    infer_video_output = gr.Video(label="生成结果")

        # --- Chat Tab ---
        with gr.Tab("对话 (Chat)"):
            gr.Markdown("### 实时对话交互 (Demo)")
            with gr.Row():
                with gr.Column():
                    chat_model = gr.Dropdown(choices=get_models(), label="选择模型")
                    chat_ref_audio = gr.Audio(type="filepath", label="参考音频 (用于TTS声音克隆)", sources=["upload", "microphone"])
                    chat_device = gr.Dropdown(choices=get_devices(), value=get_devices()[0] if get_devices() else None, label="设备")
                    chat_user_audio = gr.Audio(type="filepath", label="用户语音输入 (提问)", sources=["microphone", "upload"])
                    chat_btn = gr.Button("开始对话", variant="primary")
                with gr.Column():
                    chat_status = gr.Textbox(label="状态", lines=3)
                    chat_video_output = gr.Video(label="数字人回复视频")

        # --- Task Center Tab ---
        with gr.Tab("任务中心 (Task Center)"):
            gr.Markdown("### 查询任务状态与结果")
            with gr.Row():
                with gr.Column(scale=1):
                    task_id_input = gr.Textbox(label="输入 Task ID")
                    check_btn = gr.Button("查询状态", variant="primary")
                with gr.Column(scale=2):
                    task_status_output = gr.Textbox(label="状态详情", lines=5)
                    task_video_output = gr.Video(label="结果视频")
                    task_log_output = gr.HTML(label="日志 (Tail)")

    # --- Event Wiring ---
    
    # Train
    train_btn.click(
        train_action, 
        inputs=[
            train_video, 
            train_stage1_epochs, 
            train_stage2_iters, 
            train_eye_iters, 
            train_lip_iters, 
            train_styleunet_iters, 
            train_device
        ], 
        outputs=[train_status_msg, train_task_id]
    )
    
    # Inference
    infer_btn.click(
        infer_action, 
        inputs=[infer_model, infer_audio, infer_device], 
        outputs=[infer_status_msg, infer_task_id, infer_video_output]
    )
    infer_refresh_btn.click(update_model_list, outputs=infer_model)
    
    # Chat
    chat_btn.click(
        chat_action,
        inputs=[chat_model, chat_ref_audio, chat_user_audio, chat_device],
        outputs=[chat_status, chat_video_output]
    )
    
    # Task Center
    check_btn.click(
        fetch_status, 
        inputs=[task_id_input], 
        outputs=[task_status_output, task_video_output, task_log_output]
    )

if __name__ == "__main__":
    # Allow external access
    app.launch(server_name="0.0.0.0", server_port=7860, theme="soft")
