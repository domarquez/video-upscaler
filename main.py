import os
import shutil
import subprocess
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
from glob import glob

app = FastAPI(title="Video Upscaler Pro - Railway (Grok Imagine Compatible)")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_frames_ffmpeg(video_path: Path, frames_dir: Path):
    """Extrae frames con FFmpeg – perfecto para H.264 High L2.2 de Grok"""
    frames_dir.mkdir(exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "fps=30",  # Fuerza 30 FPS desde el inicio
        str(frames_dir / "frame_%08d.png")  # PNG para colores lossless
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Extract failed: {result.stderr}")

def upscale_and_effects(frame_path: Path, output_path: Path):
    """Upscale + efectos frame por frame – colores perfectos"""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Failed to read {frame_path}")
    
    # Asegurar BGR
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Efectos
    blurred = cv2.GaussianBlur(frame, (21, 21), 3)
    blended = cv2.addWeighted(frame, 0.6, blurred, 0.4, 0)
    denoised = cv2.fastNlMeansDenoisingColored(blended, None, 5, 5, 7, 21)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
    frame = cv2.filter2D(denoised, -1, kernel)
    
    # Upscale (Grok: 640x496 → 1920x1080)
    h, w = frame.shape[:2]
    if w <= 640:
        new_w = 1920
        new_h = int(h * 1920 / w)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # RGB para JPG
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_path), frame_rgb)

def interpolate_frames(enhanced_dir: Path):
    """Agrega frames intermedios para 30 FPS suave"""
    frames = sorted(enhanced_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        return
    
    for i in range(len(frames) - 1):
        prev = cv2.imread(str(frames[i]))
        curr = cv2.imread(str(frames[i+1]))
        if prev is None or curr is None:
            continue
        prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
        # Intermedio (suavidad x1.5 aprox)
        inter = cv2.addWeighted(prev_rgb, 0.5, curr_rgb, 0.5, 0)
        cv2.imwrite(str(enhanced_dir / f"inter_{i:04d}.jpg"), inter)

def make_video(enhanced_dir: Path, output_path: Path, audio_path: Path):
    """Recombinar con FFmpeg – yuv420p para compatibilidad"""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-pattern_type", "glob",
        "-i", str(enhanced_dir / "frame_*.jpg"),
        "-i", str(audio_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a", "copy",  # Copia audio sin reencode para sync Grok
        "-r", "30", "-shortest",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Recombine failed: {result.stderr}")

def process_video(video_path: str, task_id: str):
    orig = Path(video_path)
    
    # 1. Extraer frames con FFmpeg (evita corrupción de OpenCV)
    raw_frames_dir = UPLOAD_DIR / f"{task_id}_raw"
    extract_frames_ffmpeg(orig, raw_frames_dir)
    
    # 2. Upscale + efectos frame por frame
    enhanced_dir = UPLOAD_DIR / f"{task_id}_enhanced"
    enhanced_dir.mkdir(exist_ok=True)
    raw_frames = sorted(raw_frames_dir.glob("frame_*.png"))
    for frame_path in raw_frames:
        out_path = enhanced_dir / frame_path.name.replace('.png', '.jpg')
        upscale_and_effects(frame_path, out_path)
    
    # 3. Interpolación para 30 FPS
    interpolate_frames(enhanced_dir)
    
    # 4. Recombinar
    output = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    make_video(enhanced_dir, output, orig)
    
    # Limpieza
    shutil.rmtree(raw_frames_dir)
    shutil.rmtree(enhanced_dir)
    shutil.move(orig, OUTPUT_DIR / f"{task_id}_orig.mp4")

# Endpoints (igual)
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
    return {"message": "Video Upscaler Pro - Railway (Grok Imagine Compatible)", "upload": "/upload/"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
