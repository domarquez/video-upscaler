import os
import shutil
import subprocess
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Upscaler Pro – FFmpeg (FIX ALL)")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def upscale_with_ffmpeg(input_path: Path, output_path: Path):
    logger.info(f"Iniciando upscale de {input_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", (
            "scale=-1:1080:flags=lanczos,"  # UPSCAL E A ALTURA 1080p, MANTEN PROPORCIÓN
            "unsharp=9:9:1.5:7:7:1.0,"
            "eq=contrast=1.2:brightness=0.03:gamma=1.1,"
            "boxblur=1:1,"
            "fps=30"
        ),
        "-c:v", "libx264",
        "-crf", "12",
        "-preset", "veryslow",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    logger.info(f"Ejecutando: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"FFmpeg error: {r.stderr}")
        raise RuntimeError(r.stderr)
    logger.info(f"Upscale completado: {output_path}")

def process_video(video_path: str, task_id: str):
    orig = Path(video_path)
    output = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    upscale_with_ffmpeg(orig, output)
    shutil.move(orig, OUTPUT_DIR / f"{task_id}_orig.mp4")

@app.post("/upload/")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        return {"error": "Solo videos"}
    task_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Subido: {file.filename} – ID: {task_id}")
    background_tasks.add_task(process_video, str(path), task_id)
    return {"task_id": task_id, "status": f"/status/{task_id}"}

@app.get("/status/{task_id}")
def status(task_id: str):
    file = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    if file.exists():
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
    return {"message": "Video Upscaler Pro – FFmpeg", "upload": "/upload/"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
