import os
import time
import tempfile
import logging
from typing import Optional, List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from faster_whisper import WhisperModel

# =========================
# Konfigurasi via ENV
# =========================
MODEL_SIZE   = os.getenv("WHISPER_MODEL", "large-v3")   # large-v3 | medium | ...
DEVICE       = os.getenv("WHISPER_DEVICE", "cuda")      # cuda | cpu
COMPUTE      = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # float16 | int8_float16 | int8
MAX_FILE_MB  = int(os.getenv("MAX_FILE_MB", "200"))

# Dorong pemakaian GPU: lebih banyak thread/worker
CPU_THREADS  = int(os.getenv("WHISPER_CPU_THREADS", str(os.cpu_count() or 8)))
NUM_WORKERS  = int(os.getenv("WHISPER_NUM_WORKERS", "2"))  # 2–4 bagus untuk overlap pipeline

# Default decoding (balanced cepat)
DEFAULT_BEAM        = int(os.getenv("WHISPER_BEAM_SIZE", "1"))  # 1=greedy (cepat)
DEFAULT_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
DEFAULT_CHUNK_SEC   = int(os.getenv("WHISPER_CHUNK_LENGTH", "30"))  # chunk panjang -> kerja GPU lebih padat

ALLOWED_EXTS  = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

# Opsional (kadang menambah performa CTranslate2 di Ampere/Turing)
os.environ.setdefault("CT2_USE_EXPERIMENTAL_PACKED_GEMM", "1")
# os.environ.setdefault("CT2_CUDA_ALLOCATOR", "cuda_malloc_async")  # opsional


app = FastAPI(
    title="Whisper Transcribe API (ID only, Optimized)",
    version="3.0.0",
    description=(
        "Transkripsi Bahasa Indonesia saja. Timestamps per segmen. "
        "Optimasi untuk mendorong util GPU (float16, num_workers, cpu_threads, chunk panjang)."
    ),
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json"
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# =========================
# Startup: load model
# =========================
@app.on_event("startup")
def _load_model():
    global ASR
    logger.info(
        f"Loading Whisper: model={MODEL_SIZE}, device={DEVICE}, compute={COMPUTE}, "
        f"cpu_threads={CPU_THREADS}, num_workers={NUM_WORKERS}"
    )
    try:
        ASR = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE,
            cpu_threads=CPU_THREADS,
            num_workers=NUM_WORKERS
        )
        logger.info("Model loaded.")
    except Exception:
        logger.exception("Gagal load model Whisper")
        raise
    
# =========================
# Helper functions
# =========================
def _fmt_mmss(sec: float) -> str:
    s = int(round(sec))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


# =========================
# Schemas
# =========================
class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscribeResponse(BaseModel):
    success: bool
    language: str  # selalu "id"
    mode: str      # balanced | aggressive
    text: str
    segments: List[Segment]

    # Audio info
    audio_duration_sec: Optional[float] = None

    # Timing
    processing_ms_transcribe: float
    processing_ms_total: float

    # Throughput / RTF
    realtime_factor_transcribe: Optional[float] = None
    realtime_factor_total: Optional[float] = None

    # Runtime info
    model: str
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int
    
class SegOnlyItem(BaseModel):
    range_mmss: str
    text: str

class SegOnlyResponse(BaseModel):
    language: str  # selalu "id"
    segments: List[SegOnlyItem]


# =========================
# Transcribe helper
# =========================
def _full_transcribe_with_timestamps(
    path: str,
    beam_size: int,
    chunk_length: Optional[int],
    temperature: float
) -> Tuple[List[Segment], str, Optional[float], float]:
    """
    Transkripsi penuh (dengan timestamps segmen).
    Return: (segments, full_text, audio_duration_sec, elapsed_ms)
    """
    t0 = time.perf_counter()
    segs, info = ASR.transcribe(
        path,
        language="id",                # KUNCI: Bahasa Indonesia saja
        task="transcribe",
        beam_size=beam_size,
        temperature=temperature,
        condition_on_previous_text=False,
        # NOTE: gunakan chunk_length untuk mendorong kerja GPU lebih padat
        chunk_length=chunk_length if chunk_length and chunk_length > 0 else None,
        # without_timestamps=False  # default menghasilkan segmen dg timestamps
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_segments = [Segment(start=s.start, end=s.end, text=s.text) for s in segs]
    full_text = "".join(s.text for s in out_segments).strip()
    duration = getattr(info, "duration", None)
    return out_segments, full_text, duration, elapsed_ms


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
        "cpu_threads": CPU_THREADS,
        "num_workers": NUM_WORKERS,
        "default_beam": DEFAULT_BEAM,
        "default_chunk_length": DEFAULT_CHUNK_SEC,
        "temperature": DEFAULT_TEMPERATURE
    }


# =========================
# Endpoint: Transcribe (ID only)
# =========================
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio mp3/wav/m4a/flac/ogg/webm"),
    mode: str = Query(
        "balanced",
        pattern="^(balanced|aggressive)$",
        description="balanced (cepat) | aggressive (dorong GPU lebih tinggi, bisa sedikit lebih lama)"
    ),
    beam_size: Optional[int] = Query(
        None, ge=1, le=10,
        description="Override beam size. Default: balanced=1, aggressive=3"
    ),
    chunk_length: Optional[int] = Query(
        None, ge=0, le=240,
        description="Override chunk length (detik). Default: balanced=60, aggressive=90"
    ),
):
    # Validasi
    name_lower = (audio.filename or "").lower()
    _, ext = os.path.splitext(name_lower)
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Ekstensi {ext} tidak didukung. Gunakan salah satu: {sorted(ALLOWED_EXTS)}")

    hdr = audio.headers.get("content-length") or audio.headers.get("Content-Length")
    if hdr:
        try:
            size_mb = int(hdr) / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                raise HTTPException(status_code=413, detail=f"File terlalu besar. Maks {MAX_FILE_MB} MB")
        except ValueError:
            pass

    # Preset performa
    if mode == "balanced":
        beam = beam_size if beam_size else DEFAULT_BEAM          # 1
        chunk = chunk_length if chunk_length else DEFAULT_CHUNK_SEC  # 60
        temp  = DEFAULT_TEMPERATURE                               # 0.0
    else:  # aggressive: dorong GPU lebih berat
        beam = beam_size if beam_size else 3
        chunk = chunk_length if chunk_length else 90
        temp  = DEFAULT_TEMPERATURE  # tetap 0.0 (akurasi stabil)

    # Simpan sementara
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
    except Exception:
        raise HTTPException(status_code=500, detail="Gagal menyimpan file sementara")

    t_total = time.perf_counter()
    try:
        segments, text, audio_sec, trans_ms = _full_transcribe_with_timestamps(
            tmp_path, beam_size=beam, chunk_length=chunk, temperature=temp
        )

        total_ms = (time.perf_counter() - t_total) * 1000.0
        rtf_trans = (trans_ms / 1000.0) / float(audio_sec) if audio_sec else None
        rtf_total = (total_ms / 1000.0) / float(audio_sec) if audio_sec else None

        resp = TranscribeResponse(
            success=True,
            language="id",
            mode=mode,
            text=text,
            segments=segments,
            audio_duration_sec=None if audio_sec is None else round(float(audio_sec), 3),
            processing_ms_transcribe=round(trans_ms, 2),
            processing_ms_total=round(total_ms, 2),
            realtime_factor_transcribe=None if rtf_trans is None else round(rtf_trans, 4),
            realtime_factor_total=None if rtf_total is None else round(rtf_total, 4),
            model=MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE,
            cpu_threads=CPU_THREADS,
            num_workers=NUM_WORKERS
        )
        logger.info(
            f"[TIMING] mode={mode} beam={beam} chunk={chunk}s "
            f"trans_ms={resp.processing_ms_transcribe} total_ms={resp.processing_ms_total} "
            f"audio_sec={resp.audio_duration_sec} rtf_trans={resp.realtime_factor_transcribe} "
            f"rtf_total={resp.realtime_factor_total}"
        )
        return JSONResponse(content=resp.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcribe error")
        raise HTTPException(status_code=500, detail=f"Transcribe error: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        
@app.post("/transcribe_segments", response_model=SegOnlyResponse)
async def transcribe_segments(
    audio: UploadFile = File(..., description="Audio mp3/wav/m4a/flac/ogg/webm"),
    mode: str = Query(
        "balanced",
        pattern="^(balanced|aggressive)$",
        description="balanced (cepat) | aggressive (dorong GPU)"
    ),
    beam_size: int | None = Query(None, ge=1, le=10, description="Default balanced=1, aggressive=3"),
    chunk_length: int | None = Query(None, ge=0, le=30, description="Maks 30s (batas Whisper)"),
):
    # --- validasi & simpan sementara (sama seperti /transcribe) ---
    name_lower = (audio.filename or "").lower()
    _, ext = os.path.splitext(name_lower)
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Ekstensi {ext} tidak didukung. {sorted(ALLOWED_EXTS)}")

    hdr = audio.headers.get("content-length") or audio.headers.get("Content-Length")
    if hdr:
        try:
            if int(hdr) / (1024 * 1024) > MAX_FILE_MB:
                raise HTTPException(status_code=413, detail=f"File > {MAX_FILE_MB} MB")
        except ValueError:
            pass

    # preset performa
    if mode == "balanced":
        beam = beam_size if beam_size else 1
        chunk = chunk_length if chunk_length is not None else 30
    else:
        beam = beam_size if beam_size else 3
        chunk = chunk_length if chunk_length is not None else 30

    # simpan file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
    except Exception:
        raise HTTPException(status_code=500, detail="Gagal menyimpan file sementara")

    # transcribe penuh (pakai helper yang sudah ada)
    try:
        segments, _text_full, _audio_sec, _ms = _full_transcribe_with_timestamps(
            tmp_path, beam_size=beam, chunk_length=chunk, temperature=DEFAULT_TEMPERATURE
        )

        # konversi ke list ringkas: range_mmss + text
        items = []
        for s in segments:
            sm = _fmt_mmss(s.start)
            em = _fmt_mmss(s.end)
            items.append(SegOnlyItem(range_mmss=f"{sm}–{em}", text=s.text))

        return SegOnlyResponse(language="id", segments=items)
    except Exception as e:
        logging.exception("Transcribe error")
        raise HTTPException(status_code=500, detail=f"Transcribe error: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

