import os
import shutil
import subprocess
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Video Upscaler Pro – Railway")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_frames_ffmpeg(video_path: Path, frames_dir: Path):
    frames_dir.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", "fps=30", str(frames_dir / "frame_%08d.png")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Extract failed: {r.stderr}")

def apply_effects(frame_path: Path, out_path: Path):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Cannot read {frame_path}")
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Blur suave
    blurred = cv2.GaussianBlur(frame, (9, 9), 1.5)
    blended = cv2.addWeighted(frame, 0.8, blurred, 0.2, 0)

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(blended, None, 3, 3, 7, 15)

    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.05
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Upscale
    h, w = sharpened.shape[:2]
    if w <= 540:
        new_w = 1920
        new_h = int(h * 1920 / w)
        sharpened = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # GUARDAR EN BGR (NO RGB) → COLORES CORRECTOS
    cv2.imwrite(str(out_path), sharpened)

def interpolate_frames(enhanced_dir: Path):
    frames = sorted(enhanced_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        return
    for i in range(len(frames) - 1):
        prev = cv2.imread(str(frames[i]))
        curr = cv2.imread(str(frames[i + 1]))
        if prev is None or curr is None:
            continue
        inter = cv2.addWeighted(prev, 0.5, curr, 0.5, 0)
        cv2.imwrite(str(enhanced_dir / f"inter_{i:04d}.jpg"), inter)

def make_video(enhanced_dir: Path, out_path: Path, audio_src: Path):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-pattern_type", "glob",
        "-i", f"{enhanced_dir}/*.jpg",
        "-i", str(audio_src),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a", "copy",
        "-r", "30", "-shortest",
        str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {r.stderr}")

def process_video(video_path: str, task_id: str):
    orig = Path(video_path)
    raw_dir = UPLOAD_DIR / f"{task_id}_raw"
    enhanced_dir = UPLOAD_DIR / f"{task_id}_enhanced"
    enhanced_dir.mkdir(exist_ok=True)

    # NOMBRE SIMPLE PARA AUDIO
    safe_audio = UPLOAD_DIR / f"{task_id}_audio.mp4"
    shutil.copy2(orig, safe_audio)

    extract_frames_ffmpeg(safe_audio, raw_dir)

    for png in sorted(raw_dir.glob("frame_*.png")):
        jpg = enhanced_dir / png.name.replace(".png", ".jpg")
        apply_effects(png, jpg)

    interpolate_frames(enhanced_dir)

    output = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    make_video(enhanced_dir, output, safe_audio)

    shutil.rmtree(raw_dir)
    shutil.rmtree(enhanced_dir)
    safe_audio.unlink()
    shutil.move(orig, OUTPUT_DIR / f"{task_id}_orig.mp4")

# === API ===
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
    return {"message": "Video Upscaler Pro – Railway", "upload": "/upload/"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
