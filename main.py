import os
import shutil
import subprocess
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Video Upscaler Pro - Railway (Grok Imagine Fix)")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_frames_ffmpeg(video_path: Path, frames_dir: Path):
    """Extrae frames con FFmpeg – maneja códecs Grok Imagine"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "fps=30",  # Fuerza 30 FPS para interpolación
        str(frames_dir / "frame_%08d.png")  # PNG para colores perfectos
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extract failed: {result.stderr}")

def apply_effects(frame_path: Path) -> np.ndarray:
    """Aplica efectos y upscale – maneja frame por frame"""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")
    
    # Fuerza BGR si necesario
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
    
    # Upscale si bajo res (Grok: 640x496 → 1080p)
    h, w = frame.shape[:2]
    if w <= 640:
        scale = 1920 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # RGB para JPG output
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def interpolate_frames(frames_dir: Path, output_frames_dir: Path):
    """Interpola frames a 30 FPS exactos"""
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) < 2:
        return
    
    for i in range(len(frames) - 1):
        prev = cv2.imread(str(frames[i]))
        curr = cv2.imread(str(frames[i+1]))
        if prev is None or curr is None:
            continue
        
        prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
        
        # Interpolación simple (x2 para 30 FPS si original ~15)
        for alpha in [0.5]:  # Solo medio frame para suavidad
            inter = cv2.addWeighted(prev_rgb, 1 - alpha, curr_rgb, alpha, 0)
            cv2.imwrite(str(output_frames_dir / f"inter_{i:04d}.jpg"), inter)

def process_video(video_path: str, task_id: str):
    orig = Path(video_path)
    
    # Extraer frames con FFmpeg (maneja Grok códecs)
    frames_dir = UPLOAD_DIR / f"{task_id}_raw_frames"
    frames_dir.mkdir(exist_ok=True)
    extract_frames_ffmpeg(orig, frames_dir)
    
    # Aplicar efectos y upscale frame por frame
    enhanced_frames_dir = UPLOAD_DIR / f"{task_id}_enhanced_frames"
    enhanced_frames_dir.mkdir(exist_ok=True)
    
    raw_frames = sorted(frames_dir.glob("frame_*.png"))
    for frame_path in raw_frames:
        try:
            enhanced_frame = apply_effects(frame_path)
            out_path = enhanced_frames_dir / frame_path.name.replace('.png', '.jpg')
            cv2.imwrite(str(out_path), enhanced_frame)
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")
    
    # Interpolación para 30 FPS
    interpolate_frames(enhanced_frames_dir, enhanced_frames_dir)
    
    # Recombinar con audio original
    output = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    make_video(enhanced_frames_dir, output, orig)
    
    # Limpieza
    shutil.rmtree(frames_dir)
    shutil.rmtree(enhanced_frames_dir)
    shutil.move(orig, OUTPUT_DIR / f"{task_id}_orig.mp4")

def make_video(frames_dir: Path, output_path: Path, audio_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-pattern_type", "glob",
        "-i", str(frames_dir / "*.jpg"),
        "-i", str(audio_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a", "aac", "-b:a", "128k", "-r", "30", "-shortest",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

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
