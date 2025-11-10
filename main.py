import os
import shutil
import subprocess
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Video Upscaler Pro – FFmpeg + Real-ESRGAN")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def upscale_video(input_path: Path, output_path: Path):
    # 1. Extraer audio
    audio_path = UPLOAD_DIR / "temp_audio.aac"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-c:a", "copy", str(audio_path)
    ], check=True)

    # 2. Upscale con Real-ESRGAN (4x)
    temp_upscaled = UPLOAD_DIR / "temp_upscaled.mp4"
    subprocess.run([
        "realesrgan-ncnn-vulkan",
        "-i", str(input_path),
        "-o", str(temp_upscaled),
        "-n", "realesr-animevideov3",  # modelo ligero
        "-s", "4"  # 4x upscale
    ], check=True)

    # 3. Reemplazar audio
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(temp_upscaled),
        "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        str(output_path)
    ], check=True)

    # Limpieza
    audio_path.unlink()
    temp_upscaled.unlink()

def process_video(video_path: str, task_id: str):
    orig = Path(video_path)
    output = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    upscale_video(orig, output)
    shutil.move(orig, OUTPUT_DIR / f"{task_id}_orig.mp4")

@app.post("/upload/")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        return {"error": "Solo videos"}
    task_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    background_tasks.add_task(process_video, str(path), task_id)
    return {"task_id": task_id, "status": f"/status/{task_id}"}

@app.get("/status/{task_id}")
def status(task_id: str):
    if (OUTPUT_DIR / f"{task_id}_upscaled.mp4").exists():
        return {"status": "ready", "download": f"/download/{task_id}"}
    return {"status": "processing"}

@app.get("/download/{task_id}")
def download(task_id: str):
    file = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    if file.exists():
        return FileResponse(file, media_type="video/mp4", filename=f"upscaled_{task_id}.mp4")
    return {"error": "No listo"}

@app.get("/")
def home():
    return {"message": "Video Upscaler Pro – Real-ESRGAN", "upload": "/upload/"}
