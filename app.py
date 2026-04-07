# app.py — Diarization (embeddings locaux) · unifié
# -------------------------------------------------
import asyncio, datetime as dt, json, os, uuid, logging, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from filelock import FileLock

# Audio utils (robuste: ogg/webm/wav/… -> wav 16k mono)
from pydub import AudioSegment

# Embedding model (local)
import torch
from pyannote.audio import Model, Inference

log = logging.getLogger("diarization")
logging.basicConfig(level=logging.INFO)

# ============================== Config ==============================
VALID_TOKENS = {t.strip() for t in os.getenv("API_TOKENS", "").split(",") if t.strip()}

MODEL_ID = os.getenv("EMB_MODEL_ID", "pyannote/wespeaker-voxceleb-resnet34-LM")
DEVICE = os.getenv("EMB_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
WINDOW = os.getenv("EMB_WINDOW", "whole")  # "whole" recommandé pour voix courtes

EMBEDDING_STORE = Path(os.getenv("EMBEDDING_STORE", "embeddings.json"))
NOISE_STORE = Path(os.getenv("NOISE_STORE", "noise_embeddings.json"))
EMBEDDING_LOCK = FileLock(str(EMBEDDING_STORE) + ".lock")
NOISE_LOCK = FileLock(str(NOISE_STORE) + ".lock")

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
NOISE_THRESHOLD = float(os.getenv("NOISE_THRESHOLD", "0.5"))
RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "1"))
PURGE_INTERVAL = int(os.getenv("PURGE_INTERVAL", "3600"))

# ============================== App ==============================
app = FastAPI(title="Optimised Diarisation Service (unified)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ============================== Auth ==============================
def _verify_token(authorization: Optional[str] = Header(None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization[7:] not in VALID_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================== Stores helpers ==============================
def _load(path: Path, lock: FileLock) -> Dict[str, dict]:
    with lock:
        if not path.exists():
            return {}
        try:
            content = path.read_text()
            if not content.strip():
                return {}
            return json.loads(content)
        except json.JSONDecodeError:
            log.warning("Invalid JSON in %s; ignoring corrupted store.", path)
            return {}

def _save(data: Dict[str, dict], path: Path, lock: FileLock) -> None:
    with lock:
        path.write_text(json.dumps(data, separators=(",", ":")))

def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n

def _purge() -> None:
    cutoff = (dt.datetime.utcnow() - dt.timedelta(hours=RETENTION_HOURS)).isoformat()
    data = _load(EMBEDDING_STORE, EMBEDDING_LOCK)
    data = {k: v for k, v in data.items() if v["date"] > cutoff}
    _save(data, EMBEDDING_STORE, EMBEDDING_LOCK)

def _add(emb: List[float]) -> str:
    vec = _l2(np.asarray(emb, dtype=np.float32))
    data = _load(EMBEDDING_STORE, EMBEDDING_LOCK)
    guid = str(uuid.uuid4())
    data[guid] = {"embedding": vec.tolist(), "date": dt.datetime.utcnow().isoformat()}
    _save(data, EMBEDDING_STORE, EMBEDDING_LOCK)
    return guid

def _update(guid: str, emb: List[float]) -> None:
    vec = _l2(np.asarray(emb, dtype=np.float32))
    data = _load(EMBEDDING_STORE, EMBEDDING_LOCK)
    if guid not in data:
        raise KeyError(f"Unknown speaker id: {guid}")
    old = np.asarray(data[guid]["embedding"], dtype=np.float32)
    data[guid]["embedding"] = _l2((old + vec) / 2).tolist()
    data[guid]["date"] = dt.datetime.utcnow().isoformat()
    _save(data, EMBEDDING_STORE, EMBEDDING_LOCK)

def _best_match(emb: List[float]) -> Tuple[Optional[str], float]:
    vec = _l2(np.asarray(emb, dtype=np.float32))
    data = _load(EMBEDDING_STORE, EMBEDDING_LOCK)
    if not data:
        return None, -1.0

    keys = list(data.keys())
    embeddings = np.asarray([data[k]["embedding"] for k in keys], dtype=np.float32)
    scores = embeddings @ vec
    best_idx = int(np.argmax(scores))
    return keys[best_idx], float(scores[best_idx])

def _is_noise(emb: List[float]) -> bool:
    vec = _l2(np.asarray(emb, dtype=np.float32))
    noise_embeddings = _load(NOISE_STORE, NOISE_LOCK)
    if not noise_embeddings:
        return False

    matrix = np.asarray([v["embedding"] for v in noise_embeddings.values()], dtype=np.float32)
    scores = matrix @ vec
    return bool(np.any(scores > NOISE_THRESHOLD))

# ============================== Audio I/O ==============================
def _load_audio_any(src_path: str) -> AudioSegment:
    """
    Essaie successivement : auto (ffmpeg), ogg, webm.
    """
    last_err = None
    for fmt in (None, "ogg", "webm"):
        try:
            return (AudioSegment.from_file(src_path, format=fmt)
                    if fmt else AudioSegment.from_file(src_path))
        except Exception as e:
            last_err = e
    raise last_err

def _to_wav_16k_mono_bytes(src_tmpfile_path: str) -> bytes:
    audio = _load_audio_any(src_tmpfile_path)
    audio = audio.set_frame_rate(16_000).set_channels(1)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        audio.export(wav_path, format="wav")
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

# ============================== Embedding model ==============================
# Charge le modèle au démarrage, pousse sur DEVICE, prépare Inference
log.info(f"Loading embedding model: {MODEL_ID} on {DEVICE}")
_model = Model.from_pretrained(MODEL_ID)
_inference = Inference(_model, window=WINDOW)
_inference.to(torch.device(DEVICE))

def _embedding_from_wav_bytes(wav_bytes: bytes) -> List[float]:
    # pyannote.Inference attend un chemin; on écrit un tmp
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(wav_bytes)
        emb = _inference(path)
        return emb.tolist()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# ============================== Background purge ==============================
@app.on_event("startup")
async def _startup() -> None:
    async def _purge_task():
        while True:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _purge)
            await asyncio.sleep(PURGE_INTERVAL)
    asyncio.create_task(_purge_task())

# ============================== Routes ==============================
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...), _=Depends(_verify_token)) -> dict:
    # On écrit d’abord le flux uploadé sur disque (nécessaire pour pydub/ffmpeg)
    suffix = Path(file.filename or "").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        src_path = tmp.name

    wav_bytes: Optional[bytes] = None
    try:
        wav_bytes = _to_wav_16k_mono_bytes(src_path)

        # Durée (approx) depuis taille/PCM mono 16kHz int16 => 2 bytes/sample
        duration = len(wav_bytes) / (2 * 16_000)

        # Embedding local (non bloquant: threadpool)
        loop = asyncio.get_running_loop()
        emb = await loop.run_in_executor(None, _embedding_from_wav_bytes, wav_bytes)

        # Vérifie le bruit pour toutes les durées (pas uniquement les voix courtes)
        if _is_noise(emb):
            return {
                "identifier": f"noise-{uuid.uuid4()}",
                "noise": True,
                "duration": duration,
            }

        if duration < 3:
            best, score = _best_match(emb)
            if best is None:
                return {"identifier": "noise-0", "noise": True, "duration": duration}
            return {
                "identifier": best,
                "noise": False,
                "duration": duration,
                "similarity": score,
                "added": False,
            }

        best, score = _best_match(emb)
        if best and score >= SIMILARITY_THRESHOLD:
            await loop.run_in_executor(None, _update, best, emb)
            return {
                "identifier": best,
                "noise": False,
                "duration": duration,
                "similarity": score,
                "updated": True,
            }

        guid = await loop.run_in_executor(None, _add, emb)
        return {"identifier": guid, "noise": False, "duration": duration, "added": True}

    except Exception as e:
        log.exception("Unhandled error while processing %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(src_path)
        except Exception:
            pass

@app.get("/health")
async def health(_=Depends(_verify_token)):
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}

# ============================== Run ==============================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
