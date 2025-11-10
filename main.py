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

app = FastAPI(title="Video Upscaler Pro – Railway (Grok Imagine)")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# 1. Extraer frames con FFmpeg (compatible con Grok)
# -------------------------------------------------
def extract_frames_ffmpeg(video_path: Path, frames_dir: Path):
    frames_dir.mkdir(exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "fps=30",                     # fuerza 30 FPS desde el inicio
        str(frames_dir / "frame_%08d.png")   # PNG → colores sin pérdida
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg extract failed: {r.stderr}")


# -------------------------------------------------
# 2. Upscale + efectos (blur suave, no negro)
# -------------------------------------------------
def apply_effects(frame_path: Path, out_path: Path):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Cannot read {frame_path}")

    # Asegurar BGR
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ---- BLUR CINEMÁTICO SUAVE ----
    blurred = cv2.GaussianBlur(frame, (9, 9), 1.5)          # kernel pequeño
    blended = cv2.addWeighted(frame, 0.8, blurred, 0.2, 0)   # 80 % original

    # ---- DENOISE LIGERO ----
    denoised = cv2.fastNlMeansDenoisingColored(
        blended, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=15
    )

    # ---- SHARPEN SUTIL ----
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) * 0.05
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # ---- UPSCALE (540p → 1080p) ----
    h, w = sharpened.shape[:2]
    if w <= 540:
        new_w = 1920
        new_h = int(h * 1920 / w)
        sharpened = cv2.resize(
            sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )

    # ---- Guardar en RGB (FFmpeg espera RGB) ----
    rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(out_path), rgb)


# -------------------------------------------------
# 3. Interpolación simple (suaviza 30 FPS)
# -------------------------------------------------
def interpolate_frames(enhanced_dir: Path):
    frames = sorted(enhanced_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        return
    for i in range(len(frames) - 1):
        prev = cv2.imread(str(frames[i]))
        curr = cv2.imread(str(frames[i + 1]))
        if prev is None or curr is None:
            continue
        prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
        inter = cv2.addWeighted(prev_rgb, 0.5, curr_rgb, 0.5, 0)
        cv2.imwrite(str(enhanced_dir / f"inter_{i:04d}.jpg"), inter)


# -------------------------------------------------
# 4. Recombinar con audio original
# -------------------------------------------------
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
        "-c:a", "copy",          # mantiene audio Grok sin re‑encode
        "-r", "30", "-shortest",
        str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg combine failed: {r.stderr}")


# -------------------------------------------------
# 5. Proceso completo (background)
# -------------------------------------------------
def process_video(video_path: str, task_id: str):
    orig = Path(video_path)

    # ---- 1. Extraer frames ----
    raw_dir = UPLOAD_DIR / f"{task_id}_raw"
    extract_frames_ffmpeg(orig, raw_dir)

    # ---- 2. Aplicar upscale + efectos ----
    enhanced_dir = UPLOAD_DIR / f"{task_id}_enhanced"
    enhanced_dir.mkdir(exist_ok=True)

    for png in sorted(raw_dir.glob("frame_*.png")):
        jpg = enhanced_dir / png.name.replace(".png", ".jpg")
        apply_effects(png, jpg)

    # ---- 3. Interpolación ----
    interpolate_frames(enhanced_dir)

    # ---- 4. Recombinar ----
    output = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    make_video(enhanced_dir, output, orig)

    # ---- Limpieza ----
    shutil.rmtree(raw_dir)
    shutil.rmtree(enhanced_dir)
    shutil.move(orig, OUTPUT_DIR / f"{task_id}_orig.mp4")


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
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
    return {"message": "Video Upscaler Pro – Railway", "upload": "/upload/"}


# -------------------------------------------------
# Entrypoint (Railway usa $PORT)
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
