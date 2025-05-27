# Optimised diarisation service
# ---------------------------------
# Principales améliorations :
# 1. Appels HTTP asynchrones (httpx) – aucun blocage d’I/O.
# 2. FileLock pour un JSON thread‑safe.
# 3. **Sélection du locuteur = vecteur le plus proche** : on parcourt tous les embeddings, on garde celui dont la similarité cosinus est la plus élevée ; on ne s’arrête plus au premier ≥ SEUIL.
#    ‑ Si cette similarité dépasse SIMILARITY_THRESHOLD, on fusionne par moyenne simple.
#    ‑ Sinon, on crée un nouveau locuteur.
# 4. Purge automatique via tâche asyncio.
# 5. Typage strict, endpoint `/health`, normalisation L2 immédiate.
# 6. Code prêt pour un Vector Store externe (Faiss/Qdrant).

import asyncio
import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Dict, List

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from filelock import FileLock
import uvicorn

# ---------------------- Configuration ----------------------
API_URL = "https://api-audio2embedding.c0.cloud-pi-native.com/get_embedding/"
EMBEDDING_STORE = Path("embeddings.json")
EMBEDDING_LOCK = FileLock("embeddings.json.lock")
SIMILARITY_THRESHOLD = 0.6
RETENTION_HOURS = 0.25
DELETION_INTERVAL_SECONDS = 900  # 1 heure

app = FastAPI(title="Optimised Diarisation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Utilitaires embeddings ----------------------

def l2_norm(vec: np.ndarray) -> np.ndarray:
    """Normalise un vecteur (L2)."""
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Produit scalaire parce que les vecteurs sont normalisés."""
    return float(a @ b)

# ---------------------- Stockage ----------------------

def _load_embeddings() -> Dict[str, dict]:
    if not EMBEDDING_STORE.exists():
        return {}
    with EMBEDDING_LOCK, EMBEDDING_STORE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_embeddings(data: Dict[str, dict]) -> None:
    with EMBEDDING_LOCK, EMBEDDING_STORE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ---------------------- Logic métier ----------------------

async def fetch_embedding(file_path: Path) -> List[float]:
    """Appelle l'API externe pour obtenir l'embedding (asynchrone)."""
    async with httpx.AsyncClient(timeout=60) as client:
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, "audio/opus")}
            r = await client.post(API_URL, files=files, headers={"accept": "application/json"})
    try:
        r.raise_for_status()
        return r.json()["embedding"]
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Erreur API externe : {exc}") from exc


def add_or_update_embedding(new_embedding: List[float]) -> str:
    """Trouve le locuteur le PLUS PROCHE ; fusionne si similarité ≥ seuil sinon ajoute."""
    emb_new = l2_norm(np.asarray(new_embedding, dtype=np.float32))
    data = _load_embeddings()
    now = dt.datetime.utcnow().isoformat()

    best_guid = None
    best_sim = -1.0

    # Chercher le meilleur match
    for guid, payload in data.items():
        stored_vec = np.asarray(payload["embedding"], dtype=np.float32)
        sim = cosine_similarity(emb_new, stored_vec)
        if sim > best_sim:
            best_sim = sim
            best_guid = guid

    # Si un locuteur « assez proche » existe, on le met à jour
    if best_guid is not None and best_sim >= SIMILARITY_THRESHOLD:
        stored_vec = np.asarray(data[best_guid]["embedding"], dtype=np.float32)
        updated_vec = l2_norm((stored_vec + emb_new) / 2)
        data[best_guid].update({
            "embedding": updated_vec.tolist(),
            "date": now,
        })
        _save_embeddings(data)
        return best_guid

    # Sinon, on crée un nouveau locuteur
    guid = str(uuid.uuid4())
    data[guid] = {
        "embedding": emb_new.tolist(),
        "date": now,
    }
    _save_embeddings(data)
    return guid


def purge_old_embeddings() -> None:
    data = _load_embeddings()
    cutoff = dt.datetime.utcnow() - dt.timedelta(hours=RETENTION_HOURS)
    data = {k: v for k, v in data.items() if v["date"] > cutoff.isoformat()}
    _save_embeddings(data)

# ---------------------- FastAPI ----------------------

@app.on_event("startup")
async def _startup() -> None:
    async def _purge_task():
        while True:
            purge_old_embeddings()
            await asyncio.sleep(DELETION_INTERVAL_SECONDS)

    asyncio.create_task(_purge_task())


@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)) -> dict:
    tmp_path = Path(f"tmp_{uuid.uuid4()}_{file.filename}")
    try:
        tmp_path.write_bytes(await file.read())
        try:
            embedding = await fetch_embedding(tmp_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        speaker_id = add_or_update_embedding(embedding)
        return {"identifier": speaker_id}
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

# ---------------------- Entrée principale ----------------------
if __name__ == "__main__":
    uvicorn.run("optimized_diarization:app", host="0.0.0.0", port=8000, reload=False)
