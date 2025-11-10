import os
import shutil
import subprocess
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Video Upscaler Pro – FFmpeg (Nitidez PRO)")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def upscale_with_ffmpeg(input_path: Path, output_path: Path):
    """
    - Mantiene proporción original (letterbox si es necesario)
    - Upscale a 1920x1080 con Lanczos (alta calidad)
    - Nitidez extrema: unsharp + hqdn3d
    - Color boost: vibrance + eq
    - Blur cinematográfico sutil
    - 30 FPS suave
    - Audio AAC 192 kbps
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", (
            "scale='min(1920,iw*min(1,(1080/ih)))':"
            "min(1080,ih*min(1,(1920/iw)))':flags=lanczos,"
            "unsharp=7:7:1.2:5:5:0.8,"        # ← Nitidez AGUDA
            "hqdn3d=3:3:4:4,"                  # ← Denoise sin perder detalle
            "vibrance=1.3,"                    # ← Colores más vivos
            "eq=contrast=1.15:brightness=0.03,"# ← Contraste + brillo
            "boxblur=1.5:1.5,"                 # ← Blur cinematográfico
            "fps=30,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black"
        ),
        "-c:v", "libx264",
        "-crf", "15",          # ← Calidad excelente
        "-preset", "veryslow", # ← Máxima compresión eficiente
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {r.stderr}")


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
        return FileResponse(file, media_type="video/mp4",
                            filename=f"upscaled_{task_id}.mp4")
    return {"error": "No listo"}


@app.get("/")
def home():
    return {"message": "Video Upscaler Pro – Nitidez PRO", "upload": "/upload/"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
