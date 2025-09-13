import os
import tempfile
import logging
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

# --------------------------
# Konfigurasi dasar (bisa di-override via ENV)
# --------------------------
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")          # ex: tiny, base, small, medium, large-v3
DEVICE     = os.getenv("WHISPER_DEVICE", "cuda")              # "cuda" untuk GPU, "cpu" untuk CPU
COMPUTE    = os.getenv("WHISPER_COMPUTE_TYPE", "float16")     # "float16" (GPU), "int8", "int8_float16", dll
BEAM_SIZE  = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

# Batasi tipe file audio umum
ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_FILE_MB  = int(os.getenv("MAX_FILE_MB", "100"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

app = FastAPI(
    title="Whisper Transcribe API",
    version="1.0.0",
    description="API sederhana untuk uji transkripsi audio menggunakan faster-whisper."
)

# --------------------------
# Inisialisasi model saat startup
# --------------------------
@app.on_event("startup")
def _load_model():
    global asr_model
    try:
        logger.info(f"Loading Whisper model: size={MODEL_SIZE}, device={DEVICE}, compute_type={COMPUTE}")
        asr_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
        logger.info("Model loaded.")
    except Exception as e:
        logger.exception("Gagal load model whisper")
        raise

# --------------------------
# Skema respons
# --------------------------
class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscribeResponse(BaseModel):
    success: bool
    detected_language: str
    language_probability: float
    duration: Optional[float] = None
    text: str
    segments: List[Segment]

# --------------------------
# Healthcheck
# --------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_size": MODEL_SIZE, "device": DEVICE, "compute_type": COMPUTE}

# --------------------------
# Endpoint transkripsi
# --------------------------
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="File audio (mp3/wav/m4a/flac/ogg/webm)"),
    beam_size: int = Query(BEAM_SIZE, ge=1, le=10, description="Beam size decoding"),
    task: str = Query("transcribe", pattern="^(transcribe|translate)$", description="transcribe atau translate ke English"),
    language: Optional[str] = Query(None, description="Kode bahasa (opsional). Biarkan kosong untuk auto-detect.")
):
    # Validasi ekstensi & ukuran
    name_lower = audio.filename.lower()
    _, ext = os.path.splitext(name_lower)
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Ekstensi {ext} tidak didukung. Gunakan salah satu: {sorted(ALLOWED_EXTS)}")

    # Cek ukuran (jika header Content-Length ada)
    size_hdr = audio.headers.get("content-length") or audio.headers.get("Content-Length")
    if size_hdr:
        try:
            size_mb = int(size_hdr) / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                raise HTTPException(status_code=413, detail=f"File terlalu besar. Maks {MAX_FILE_MB} MB")
        except ValueError:
            pass  # jika tak bisa parse, lanjut

    # Simpan ke file sementara
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception:
        raise HTTPException(status_code=500, detail="Gagal menyimpan file sementara")

    try:
        # Transcribe
        # Catatan: parameter default sudah 'standard'; kita expose minimal opsional
        segments, info = asr_model.transcribe(
            tmp_path,
            beam_size=beam_size,
            task=task,
            language=language  # None => auto-detect
            # Anda bisa menambahkan parameter lain jika perlu (temperature, vad_filter, dll.)
        )

        # Kumpulkan hasil
        seg_list = []
        full_text = []
        for seg in segments:
            seg_list.append(Segment(start=seg.start, end=seg.end, text=seg.text))
            full_text.append(seg.text)

        text_joined = "".join(full_text).strip()

        resp = TranscribeResponse(
            success=True,
            detected_language=info.language,
            language_probability=float(info.language_probability or 0.0),
            duration=getattr(info, "duration", None),
            text=text_joined,
            segments=seg_list
        )
        return JSONResponse(content=resp.dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Gagal melakukan transcribe")
        raise HTTPException(status_code=500, detail=f"Transcribe error: {str(e)}")
    finally:
        # Bersihkan file sementara
        try:
            os.remove(tmp_path)
        except Exception:
            pass
