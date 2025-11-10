import os
import shutil
import subprocess
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

# Usa /tmp (Fly.io lo limpia entre deploys, pero OK para videos pequeños)
UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Efecto cinematic: blur + sharpen
def apply_effects(frame):
    blurred = cv2.GaussianBlur(frame, (21, 21), 3)
    blended = cv2.addWeighted(frame, 0.6, blurred, 0.4, 0)
    denoised = cv2.fastNlMeansDenoisingColored(blended, None, 5, 5, 7, 21)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
    return cv2.filter2D(denoised, -1, kernel)

# Extrae, upscalea, interpola a 30 FPS
def process_frames(video_path: Path, frames_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    factor = max(1, int(30 // fps))
    frame_idx = 0

    ret, prev = cap.read()
    if not ret:
        return 0
    prev = apply_effects(prev)
    h, w = prev.shape[:2]
    if w <= 540:
        prev = cv2.resize(prev, (1920, int(h * 1920 / w)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(frames_dir / f"frame_{frame_idx:08d}.jpg"), prev)
    frame_idx += 1

    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr = apply_effects(curr)
        h, w = curr.shape[:2]
        if w <= 540:
            curr = cv2.resize(curr, (1920, int(h * 1920 / w)), interpolation=cv2.INTER_CUBIC)

        # Interpolación simple: blend entre frames
        for i in range(1, factor):
            alpha = i / factor
            inter = cv2.addWeighted(prev, 1 - alpha, curr, alpha, 0)
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:08d}.jpg"), inter)
            frame_idx += 1

        cv2.imwrite(str(frames_dir / f"frame_{frame_idx:08d}.jpg"), curr)
        prev = curr
        frame_idx += 1

    cap.release()
    return frame_idx

# Reconstruye con FFmpeg a 30 FPS
def make_video(frames_dir: Path, output_path: Path, audio_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", str(frames_dir / "frame_%08d.jpg"),
        "-i", str(audio_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-r", "30", "-shortest",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def process_video(video_path: str, task_id: str):
    orig = Path(video_path)
    frames_dir = UPLOAD_DIR / f"{task_id}_frames"
    frames_dir.mkdir(exist_ok=True)

    process_frames(orig, frames_dir)
    output = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    make_video(frames_dir, output, orig)

    shutil.rmtree(frames_dir)
    shutil.move(orig, OUTPUT_DIR / f"{task_id}_orig.mp4")

@app.post("/upload/")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
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
    return {"message": "Sube un video en /upload/"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
