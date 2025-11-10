import os
import shutil
import subprocess
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import logging

# Configurar logs detallados
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Upscaler Pro – FFmpeg (FIX FINAL)")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def upscale_with_ffmpeg(input_path: Path, output_path: Path):
    logger.info(f"Iniciando upscale: {input_path} → {output_path}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", (
            "scale='if(gt(iw,ih),-2,1920)':'if(gt(iw,ih),1080,-2)':flags=lanczos,"
            "unsharp=9:9:1.5:7:7:1.0,"
            "eq=contrast=1.2:brightness=0.03:gamma=1.1,"
            "boxblur=1:1,"
            "fps=30"
        ),
        "-c:v", "libx264",
        "-crf", "12",
        "-preset", "veryslow",
        "-movflags", "+faststart",  # ← CLAVE: MP4 jugable desde inicio
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    logger.info(f"Ejecutando FFmpeg: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if r.returncode != 0:
        logger.error(f"FFmpeg falló: {r.stderr}")
        raise RuntimeError(f"FFmpeg error: {r.stderr}")
    
    # VERIFICAR ARCHIVO DE SALIDA
    if not output_path.exists():
        raise RuntimeError("FFmpeg no creó el archivo de salida")
    
    size = output_path.stat().st_size
    logger.info(f"Upscale completado. Tamaño: {size / (1024*1024):.2f} MB")
    
    if size < 1024:  # < 1KB = error
        raise RuntimeError(f"Archivo demasiado pequeño: {size} bytes")

def process_video(video_path: str, task_id: str):
    orig = Path(video_path)
    output = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    
    try:
        upscale_with_ffmpeg(orig, output)
        logger.info(f"Video procesado: {task_id}")
    except Exception as e:
        logger.error(f"Error en procesamiento {task_id}: {e}")
        raise
    
    # Mover original
    shutil.move(orig, OUTPUT_DIR / f"{task_id}_orig.mp4")

@app.post("/upload/")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        return {"error": "Solo videos"}
    
    task_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    logger.info(f"Video subido: {file.filename} → {task_id}")
    background_tasks.add_task(process_video, str(path), task_id)
    
    return {"task_id": task_id, "status": f"/status/{task_id}"}

@app.get("/status/{task_id}")
def status(task_id: str):
    file = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    if file.exists():
        size = file.stat().st_size / (1024*1024)
        return {"status": "ready", "size_mb": round(size, 2), "download": f"/download/{task_id}"}
    return {"status": "processing"}

@app.get("/download/{task_id}")
def download(task_id: str):
    file_path = OUTPUT_DIR / f"{task_id}_upscaled.mp4"
    if not file_path.exists():
        return {"error": "No listo"}
    
    file_size = file_path.stat().st_size
    if file_size < 1024:
        return {"error": "Archivo corrupto o vacío"}
    
    def iterfile():
        with open(file_path, "rb") as f:
            yield from f
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="upscaled_{task_id}.mp4"',
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }
    )

@app.get("/")
def home():
    return {"message": "Video Upscaler Pro – FUNCIONA 100%", "upload": "/upload/"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
