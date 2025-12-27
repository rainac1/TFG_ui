import os
import sys
import uuid
import shutil
import logging
import asyncio
import subprocess
import pty
import select
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GtsTalkBackend")

# --- Path Configuration ---
class Config:
    # /app/backend/backend.py -> BASE_DIR = /app/backend
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # APP_ROOT = /app
    APP_ROOT = os.path.dirname(BASE_DIR)
    
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
    
    # External Tools Paths
    CORE_DIR = os.path.join(APP_ROOT, 'core')
    PROCESS_DATA_DIR = os.path.join(CORE_DIR, 'process_data')
    MODEL_ROOT = os.path.join(CORE_DIR, 'results')
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.RESULT_FOLDER, exist_ok=True)
        os.makedirs(cls.MODEL_ROOT, exist_ok=True)
        
        # Add APP_ROOT to sys.path to allow imports from core/
        if cls.APP_ROOT not in sys.path:
            sys.path.append(cls.APP_ROOT)

Config.setup()

# --- Core Imports ---
try:
    from core.api_interface import GtsTalkService
except ImportError as e:
    logger.warning(f"Could not import core.api_interface: {e}. Inference will be disabled.")
    GtsTalkService = None

# --- Service Layer ---
class ModelService:
    """Manages the lifecycle of the inference model."""
    def __init__(self):
        self._loaded_services: Dict[str, Any] = {}
        self._current_model_name: Optional[str] = None

    def get_instance(self, model_name: str, device: str = 'cuda:0'):
        if not GtsTalkService:
            raise HTTPException(status_code=500, detail="Inference core not available")
            
        # Return cached if matches
        if self._current_model_name == model_name and model_name in self._loaded_services:
            return self._loaded_services[model_name]
            
        logger.info(f"Switching model to: {model_name} (Device: {device})")
        
        # Unload previous
        if self._current_model_name and self._current_model_name in self._loaded_services:
            self._unload_model(self._current_model_name)
                
        # Load new
        model_path = os.path.join(Config.MODEL_ROOT, model_name)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found at {model_path}")
            
        try:
            instance = GtsTalkService(model_path, device=device)
            self._loaded_services[model_name] = instance
            self._current_model_name = model_name
            return instance
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    def _unload_model(self, model_name: str):
        if model_name in self._loaded_services:
            old = self._loaded_services.pop(model_name)
            if hasattr(old, 'release'):
                old.release()
            
            # Force GC
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def list_models(self) -> List[str]:
        if not os.path.exists(Config.MODEL_ROOT):
            return []
        return [d for d in os.listdir(Config.MODEL_ROOT) if os.path.isdir(os.path.join(Config.MODEL_ROOT, d))]

model_service = ModelService()

# --- Task Management ---
class TaskManager:
    """Simple in-memory task state management."""
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, task_type: str) -> str:
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "queued",
            "result": None,
            "error": None,
            "log_tail": []
        }
        return task_id

    def update(self, task_id: str, status: str, result: Any = None, error: Optional[str] = None):
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = status
            if result is not None:
                self._tasks[task_id]["result"] = result
            if error is not None:
                self._tasks[task_id]["error"] = error
            logger.info(f"Task {task_id} updated: {status}")

    def get(self, task_id: str):
        return self._tasks.get(task_id)

    def append_log(self, task_id: str, message: str):
        if task_id in self._tasks:
            self._tasks[task_id]["log_tail"].append(message)
            # Keep only last 1000 lines to save memory
            if len(self._tasks[task_id]["log_tail"]) > 1000:
                self._tasks[task_id]["log_tail"] = self._tasks[task_id]["log_tail"][-1000:]

task_manager = TaskManager()

# --- Background Workers ---

def run_command_streaming(command: List[str], task_id: str, cwd: str, env: Dict[str, str]):
    logger.info(f"[{task_id}] Executing: {' '.join(command)}")
    master_fd, slave_fd = pty.openpty()
    
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True
        )
        os.close(slave_fd) # Close slave in parent
        
        buffer = ""
        while True:
            try:
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in r:
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    text = data.decode('utf-8', errors='replace')
                    buffer += text
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        # Clean up carriage returns which are common in terminal output (progress bars)
                        line = line.replace('\r', '').strip()
                        if line:
                            task_manager.append_log(task_id, line)
                elif process.poll() is not None:
                    break
            except OSError:
                break
        
        if buffer:
            line = buffer.replace('\r', '').strip()
            if line:
                task_manager.append_log(task_id, line)
                
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with return code {process.returncode}")
            
    except Exception as e:
        raise e
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

def worker_inference(task_id: str, model_name: str, audio_path: str, output_filename: str, device: str):
    try:
        task_manager.update(task_id, "processing")
        output_path = os.path.join(Config.RESULT_FOLDER, output_filename)
        
        service = model_service.get_instance(model_name, device)
        
        # Blocking call to inference
        result = service.process(audio_path, output_path)
        
        if isinstance(result, dict) and result.get('status') == 'success':
            # Return web-accessible path
            web_path = f"/static/results/{output_filename}"
            task_manager.update(task_id, "completed", result=web_path)
        else:
            msg = result.get('message', 'Unknown error') if isinstance(result, dict) else str(result)
            task_manager.update(task_id, "failed", error=msg)
            
    except Exception as e:
        logger.error(f"Inference task {task_id} failed: {e}", exc_info=True)
        task_manager.update(task_id, "failed", error=str(e))

def worker_training(task_id: str, video_path: str, device: str, 
                    stage1_epochs: int, stage2_iters: int, 
                    eye_iters: int, lip_iters: int, styleunet_iters: int):
    try:
        task_manager.update(task_id, "preprocessing")
        
        # Environment setup
        env = os.environ.copy()
        if "cuda" in device:
            gpu_id = device.split(":")[-1] if ":" in device else "0"
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        else:
            env["CUDA_VISIBLE_DEVICES"] = "-1"

        # --- Step 1: Preprocessing ---
        process_script = os.path.join(Config.PROCESS_DATA_DIR, 'process.py')
        if not os.path.exists(process_script):
            raise FileNotFoundError(f"Process script not found at {process_script}")

        cmd_process = [
            sys.executable, process_script,
            '-f', video_path,
            '-s', '1,2,3,4',
            '--nworkers', '2'
        ]
        
        run_command_streaming(cmd_process, task_id, Config.PROCESS_DATA_DIR, env)

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        data_path = os.path.join(Config.CORE_DIR, 'results', video_stem)

        # --- Step 2: Stage 1 Finetune ---
        task_manager.update(task_id, "stage1_finetune")
        stage1_script = os.path.join(Config.CORE_DIR, 'stage1_finetune.py')
        cmd_stage1 = [
            sys.executable, stage1_script,
            '--train', '--test',
            '-w', data_path,
            '-e', str(stage1_epochs)
        ]
        run_command_streaming(cmd_stage1, task_id, Config.CORE_DIR, env)

        # --- Step 3: Stage 2 Base ---
        task_manager.update(task_id, "stage2_base")
        train_script = os.path.join(Config.CORE_DIR, 'main.py')
        workspace_path = os.path.join(data_path, 'logs', 'stage2')
        
        cmd_s2_base = [
            sys.executable, train_script,
            data_path,
            '--workspace', workspace_path,
            '--iters', str(stage2_iters),
            '-O'
        ]
        run_command_streaming(cmd_s2_base, task_id, Config.CORE_DIR, env)

        # --- Step 4: Stage 2 Eyes ---
        task_manager.update(task_id, "stage2_eyes")
        cmd_s2_eyes = [
            sys.executable, train_script,
            data_path,
            '--workspace', workspace_path,
            '--iters', str(eye_iters),
            '-O', '--finetune_eyes'
        ]
        run_command_streaming(cmd_s2_eyes, task_id, Config.CORE_DIR, env)

        # --- Step 5: Stage 2 Lips ---
        task_manager.update(task_id, "stage2_lips")
        cmd_s2_lips = [
            sys.executable, train_script,
            data_path,
            '--workspace', workspace_path,
            '--iters', str(lip_iters),
            '-O', '--finetune_lips'
        ]
        run_command_streaming(cmd_s2_lips, task_id, Config.CORE_DIR, env)

        # --- Step 6: StyleUNet ---
        task_manager.update(task_id, "styleunet")
        styleunet_script = os.path.join(Config.CORE_DIR, 'styleunet', 'stage.py')
        cmd_styleunet = [
            sys.executable, styleunet_script,
            data_path,
            '--iter', str(styleunet_iters)
        ]
        run_command_streaming(cmd_styleunet, task_id, Config.CORE_DIR, env)

        task_manager.update(task_id, "completed", result={
            "model_name": video_stem,
            "message": "Full training pipeline finished successfully"
        })

    except Exception as e:
        logger.error(f"Training task {task_id} failed: {e}", exc_info=True)
        task_manager.update(task_id, "failed", error=str(e))

def worker_chat(task_id: str, model_name: str, ref_audio_path: str, user_audio_path: str, output_filename: str, device: str):
    try:
        task_manager.update(task_id, "asr")
        from tools.asr import speech_to_text
        from tools.llm import get_llm_response
        from tools.tts import text_to_speech
        
        # 1. ASR
        task_manager.append_log(task_id, "Starting ASR for user audio...")
        user_text = speech_to_text(user_audio_path)
        task_manager.append_log(task_id, "Starting ASR for reference audio...")
        ref_text = speech_to_text(ref_audio_path)
        
        if user_text.startswith("Error:") or ref_text.startswith("Error:"):
            raise RuntimeError(f"ASR failed: user={user_text}, ref={ref_text}")
            
        task_manager.append_log(task_id, f"User text: {user_text}")
        task_manager.append_log(task_id, f"Ref text: {ref_text}")
        
        # 2. LLM
        task_manager.update(task_id, "llm")
        task_manager.append_log(task_id, "Requesting LLM response...")
        llm_response = get_llm_response(user_text)
        if llm_response.startswith("Error:"):
            raise RuntimeError(f"LLM failed: {llm_response}")
            
        task_manager.append_log(task_id, f"LLM response: {llm_response}")
        
        # 3. TTS
        task_manager.update(task_id, "tts")
        task_manager.append_log(task_id, "Generating TTS audio...")
        tts_output_path = os.path.join(Config.UPLOAD_FOLDER, f"{task_id}_tts.wav")
        success = text_to_speech(llm_response, ref_audio_path, ref_text, tts_output_path)
        if not success:
            raise RuntimeError("TTS failed")
            
        # 4. Inference
        task_manager.update(task_id, "inference")
        task_manager.append_log(task_id, "Starting TFG inference...")
        output_path = os.path.join(Config.RESULT_FOLDER, output_filename)
        service = model_service.get_instance(model_name, device)
        result = service.process(tts_output_path, output_path)
        
        if isinstance(result, dict) and result.get('status') == 'success':
            web_path = f"/static/results/{output_filename}"
            task_manager.update(task_id, "completed", result=web_path)
        else:
            msg = result.get('message', 'Unknown error') if isinstance(result, dict) else str(result)
            task_manager.update(task_id, "failed", error=msg)
            
    except Exception as e:
        logger.error(f"Chat task {task_id} failed: {e}", exc_info=True)
        task_manager.update(task_id, "failed", error=str(e))

# --- API Router ---
router = APIRouter(prefix="/api")

@router.get("/models")
def list_models():
    return {"status": "success", "models": model_service.list_models()}

@router.get("/hardware_info")
def get_hardware_info():
    import torch
    devices = [{"id": "cpu", "name": "CPU (Slow)"}]
    has_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if has_cuda else 0
    
    if has_cuda:
        for i in range(gpu_count):
            devices.append({
                "id": f"cuda:{i}",
                "name": f"{torch.cuda.get_device_name(i)} (GPU {i})"
            })
            
    return {
        "has_cuda": has_cuda,
        "gpu_count": gpu_count,
        "suggested_device": "cuda:0" if has_cuda else "cpu",
        "devices": devices
    }

@router.get("/status/{task_id}")
def get_task_status(task_id: str):
    task = task_manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.post("/inference")
async def start_inference(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    audio: UploadFile = File(...),
    device: str = Form("cuda:0")
):
    # Validate
    if model_name not in model_service.list_models():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    task_id = task_manager.create_task("inference")
    
    # Save Input
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Audio filename is missing")
    ext = os.path.splitext(audio.filename)[1]
    filename = f"{task_id}_input{ext}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(audio.file, f)
        
    output_filename = f"{task_id}_output.mp4"
    
    background_tasks.add_task(
        worker_inference,
        task_id, model_name, filepath, output_filename, device
    )
    
    return {"status": "success", "task_id": task_id, "message": "Inference queued"}

@router.post("/train")
async def start_training(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    device: str = Form("cuda:0"),
    stage1_epochs: int = Form(30),
    stage2_iters: int = Form(50000),
    eye_iters: int = Form(20000),
    lip_iters: int = Form(40000),
    styleunet_iters: int = Form(30000)
):
    task_id = task_manager.create_task("training")
    
    if not video.filename:
        raise HTTPException(status_code=400, detail="Video filename is missing")
    filename = f"train_{task_id}_{video.filename}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(video.file, f)
        
    background_tasks.add_task(
        worker_training,
        task_id, filepath, device,
        stage1_epochs, stage2_iters, eye_iters, lip_iters, styleunet_iters
    )
    
    return {"status": "success", "task_id": task_id, "message": "Training queued"}

@router.post("/chat")
async def chat_pipeline(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    device: str = Form("cuda:0"),
    ref_audio: UploadFile = File(...),
    user_audio: UploadFile = File(...)
):
    """
    Chat Pipeline: ASR -> LLM -> TTS -> TFG
    """
    # 1. 接收数据：模型名称/路径、参考音频、用户音频
    if model_name not in model_service.list_models():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    task_id = task_manager.create_task("chat")
    
    # Save Reference Audio
    if not ref_audio.filename:
        raise HTTPException(status_code=400, detail="Reference audio filename is missing")
    ref_ext = os.path.splitext(ref_audio.filename)[1]
    ref_filename = f"chat_{task_id}_ref{ref_ext}"
    ref_filepath = os.path.join(Config.UPLOAD_FOLDER, ref_filename)
    
    with open(ref_filepath, "wb") as f:
        shutil.copyfileobj(ref_audio.file, f)

    # Save User Audio
    if not user_audio.filename:
        raise HTTPException(status_code=400, detail="User audio filename is missing")
    user_ext = os.path.splitext(user_audio.filename)[1]
    user_filename = f"chat_{task_id}_user{user_ext}"
    user_filepath = os.path.join(Config.UPLOAD_FOLDER, user_filename)
    with open(user_filepath, "wb") as f:
        shutil.copyfileobj(user_audio.file, f)

    output_filename = f"chat_{task_id}_output.mp4"
    
    background_tasks.add_task(
        worker_chat,
        task_id, model_name, ref_filepath, user_filepath, output_filename, device
    )
    
    return {
        "status": "success",
        "task_id": task_id,
        "message": "Chat pipeline queued."
    }

# --- App Factory ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Backend V2 Service Starting...")
    yield
    logger.info("Backend V2 Service Shutting Down...")

app = FastAPI(title="GtsTalk Backend V2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(Config.BASE_DIR, 'static')), name="static")

# Include Routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
