import os
import tempfile
import logging
from typing import Optional, List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from faster_whisper import WhisperModel

# =========================
# Konfigurasi
# =========================
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE     = os.getenv("WHISPER_DEVICE", "cuda")         # "cuda"|"cpu"
COMPUTE    = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
MAX_FILE_MB  = int(os.getenv("MAX_FILE_MB", "100"))
# mode pilihan bahasa: "force_id", "force_en", "auto_id_en"
LANGUAGE_MODE = os.getenv("LANGUAGE_MODE", "auto_id_en")

# kecepatan/akurasi (default cepat)
DEFAULT_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))   # 1=greedy
DEFAULT_CHUNK_LEN = int(os.getenv("WHISPER_CHUNK_LENGTH", "30"))
DEFAULT_BATCH     = int(os.getenv("WHISPER_BATCH_SIZE", "16"))

ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
ALLOWED_LANGS = {"id", "en"}  # HANYA dua ini

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

app = FastAPI(
    title="Whisper (ID/EN only) - Fast API",
    version="1.2.0",
    description="Transkripsi cepat hanya Bahasa Indonesia & Inggris (faster-whisper)."
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# =========================
# Model
# =========================
@app.on_event("startup")
def _load_model():
    global asr
    logger.info(f"Loading model: size={MODEL_SIZE}, device={DEVICE}, compute={COMPUTE}")
    asr = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE
    )
    logger.info("Model loaded.")

# =========================
# Schemas
# =========================
class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscribeResponse(BaseModel):
    success: bool
    final_language: str
    text: str
    segments: List[Segment]
    mode: str

# =========================
# Util
# =========================
def _quick_text(path: str, lang: str) -> Tuple[str, int]:
    """
    Transkripsi cepat (tanpa timestamp) utk scoring pemilihan ID vs EN.
    Mengembalikan (text, length).
    """
    segs, _ = asr.transcribe(
        path,
        language=lang,
        task="transcribe",
        beam_size=1,                  # greedy
        vad_filter=True,
        without_timestamps=True,
        chunk_length=15,              # lebih pendek buat cepat
        batch_size=DEFAULT_BATCH,
        temperature=0.0,
        condition_on_previous_text=False
    )
    txt = "".join(s.text for s in segs).strip()
    return txt, len(txt)

def _full_transcribe(path: str, lang: str, beam_size: int) -> Tuple[List[Segment], str]:
    segs, _ = asr.transcribe(
        path,
        language=lang,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=True,
        chunk_length=DEFAULT_CHUNK_LEN,
        batch_size=DEFAULT_BATCH,
        temperature=0.0,
        condition_on_previous_text=False
    )
    out = [Segment(start=s.start, end=s.end, text=s.text) for s in segs]
    return out, "".join(s.text for s in out).strip()

# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE,
        "language_mode": LANGUAGE_MODE,
        "allowed_langs": sorted(ALLOWED_LANGS),
        "beam_size": DEFAULT_BEAM_SIZE,
        "chunk_length": DEFAULT_CHUNK_LEN,
        "batch_size": DEFAULT_BATCH
    }

# =========================
# Transcribe (ID/EN only)
# =========================
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio mp3/wav/m4a/flac/ogg/webm"),
    # override per-request (opsional)
    language_mode: str = Query(
        None,
        pattern="^(force_id|force_en|auto_id_en)$",
        description="force_id | force_en | auto_id_en (default dari ENV)"
    ),
    beam_size: int = Query(None, ge=1, le=10, description="Beam size (1=greedy tercepat)")
):
    mode = language_mode or LANGUAGE_MODE
    beam = beam_size or DEFAULT_BEAM_SIZE

    # Validasi file
    name_low = audio.filename.lower()
    _, ext = os.path.splitext(name_low)
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Ekstensi {ext} tidak didukung. {sorted(ALLOWED_EXTS)}")

    hdr = audio.headers.get("content-length") or audio.headers.get("Content-Length")
    if hdr:
        try:
            if int(hdr) / (1024*1024) > MAX_FILE_MB:
                raise HTTPException(status_code=413, detail=f"File > {MAX_FILE_MB} MB")
        except ValueError:
            pass

    # Simpan sementara
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await audio.read())
            path = tmp.name
    except Exception:
        raise HTTPException(status_code=500, detail="Gagal menyimpan file sementara")

    try:
        # Tentukan bahasa final:
        if mode == "force_id":
            final_lang = "id"
        elif mode == "force_en":
            final_lang = "en"
        else:
            # auto_id_en: bandingkan hasil cepat antara 'id' vs 'en'
            _, len_id = _quick_text(path, "id")
            _, len_en = _quick_text(path, "en")
            final_lang = "id" if len_id >= len_en else "en"

        segments, text = _full_transcribe(path, final_lang, beam)
        return JSONResponse(content=TranscribeResponse(
            success=True,
            final_language=final_lang,
            text=text,
            segments=segments,
            mode=mode
        ).dict())
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Transcribe error")
        raise HTTPException(status_code=500, detail=f"Transcribe error: {str(e)}")
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
