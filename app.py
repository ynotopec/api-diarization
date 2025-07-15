import asyncio, datetime as dt, json, os, uuid, logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import httpx, uvicorn, torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from filelock import FileLock

import io

log = logging.getLogger("diarization")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------ config
VALID_TOKENS = {t.strip() for t in os.getenv("API_TOKENS", "").split(",") if t.strip()}
API_URL = "https://api-audio2embedding.c0.cloud-pi-native.com/get_embedding/"
EMBEDDING_STORE = Path("embeddings.json")
NOISE_STORE = Path("noise_embeddings.json")
EMBEDDING_LOCK = FileLock(str(EMBEDDING_STORE) + ".lock")
NOISE_LOCK = FileLock(str(NOISE_STORE) + ".lock")

SIMILARITY_THRESHOLD = 0.6
NOISE_THRESHOLD = 0.5
RETENTION_HOURS = 1
PURGE_INTERVAL = 3600

# ------------------------------------------------------------------ app
app = FastAPI(title="Optimised Diarisation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ helpers
def _verify_token(authorization: str = Header(...)) -> None:
    if not authorization.startswith("Bearer ") or authorization[7:] not in VALID_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid token")

def _load(path: Path, lock: FileLock) -> Dict[str, dict]:
    with lock:
        return json.loads(path.read_text()) if path.exists() else {}

def _save(data: Dict[str, dict], path: Path, lock: FileLock) -> None:
    with lock:
        path.write_text(json.dumps(data, separators=(",", ":")))

def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)

# ------------------------------------------------------------------ embedding mgmt
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
    old = np.asarray(data[guid]["embedding"], dtype=np.float32)
    data[guid]["embedding"] = _l2((old + vec) / 2).tolist()
    data[guid]["date"] = dt.datetime.utcnow().isoformat()
    _save(data, EMBEDDING_STORE, EMBEDDING_LOCK)

def _best_match(emb: List[float]) -> tuple[str | None, float]:
    vec = _l2(np.asarray(emb, dtype=np.float32))
    data = _load(EMBEDDING_STORE, EMBEDDING_LOCK)
    best, score = None, -1.0
    for k, v in data.items():
        s = _cosine(vec, np.asarray(v["embedding"], dtype=np.float32))
        if s > score:
            best, score = k, s
    return best, score

def _is_noise(emb: List[float]) -> bool:
    vec = _l2(np.asarray(emb, dtype=np.float32))
    noise_vecs = [
        np.asarray(v["embedding"], dtype=np.float32)
        for v in _load(NOISE_STORE, NOISE_LOCK).values()
    ]
    return any(_cosine(vec, nv) > NOISE_THRESHOLD for nv in noise_vecs)

# ------------------------------------------------------------------ audio
async def _wav_bytes(file: UploadFile) -> tuple[bytes, float]:
    loop = asyncio.get_event_loop()
    buf = await file.read()
    waveform, sr = await loop.run_in_executor(None, torchaudio.load, buf)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != 16_000:
        resampler = torchaudio.transforms.Resample(sr, 16_000)
        waveform = await loop.run_in_executor(None, resampler, waveform)
    out = io.BytesIO()
    await loop.run_in_executor(
        None,
        lambda: torchaudio.save(out, waveform, 16_000, format="wav"),
    )

    duration = waveform.shape[-1] / 16_000
    return out.getvalue(), duration

async def _embedding(wav_bytes: bytes) -> List[float]:
    async with httpx.AsyncClient(timeout=60) as client:
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        r = await client.post(API_URL, files=files, headers={"accept": "application/json"})
        r.raise_for_status()
        return r.json()["embedding"]

# ------------------------------------------------------------------ routes

@app.on_event("startup")
async def _startup() -> None:
    async def _purge_task():
        while True:
            await asyncio.get_event_loop().run_in_executor(None, _purge)
            await asyncio.sleep(PURGE_INTERVAL)
    asyncio.create_task(_purge_task())

async def _purge_loop():
    while True:
        await asyncio.sleep(PURGE_INTERVAL)
        await asyncio.get_event_loop().run_in_executor(None, _purge)

@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...), _=Depends(_verify_token)
) -> dict:
    wav_bytes, duration = await _wav_bytes(file)
    emb = await _embedding(wav_bytes)

    if duration < 3:
        if _is_noise(emb):
            return {"identifier": f"noise-{uuid.uuid4()}", "noise": True, "duration": duration}
        best, score = _best_match(emb)
        if best is None:
            return {"identifier": "noise-0", "noise": True, "duration": duration}
        return {"identifier": best, "noise": False, "duration": duration, "similarity": score, "added": False}

    best, score = _best_match(emb)
    if best and score >= SIMILARITY_THRESHOLD:
        await asyncio.get_event_loop().run_in_executor(None, _update, best, emb)
        return {"identifier": best, "noise": False, "duration": duration, "similarity": score, "updated": True}

    guid = await asyncio.get_event_loop().run_in_executor(None, _add, emb)
    return {"identifier": guid, "noise": False, "duration": duration, "added": True}

@app.get("/health")
async def health(_=Depends(_verify_token)):
    return {"status": "ok"}

# ------------------------------------------------------------------ run
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
