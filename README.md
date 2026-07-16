# API Diarization

Minimal FastAPI service for local speaker embedding and identification. It accepts an audio file, converts it to 16 kHz mono WAV with ffmpeg/pydub, computes a local `pyannote.audio` embedding, and returns an existing or new speaker identifier.

## Requirements

- Linux with Python 3.10+
- `uv` (installed automatically by `install.sh` when `curl` is available)
- `ffmpeg`
- NVIDIA GPU recommended for production, including H100 and DGX Spark class systems. CPU works for testing but is slower.

## Install or compatible upgrade

`install.sh` is idempotent and safe to rerun for compatible upgrades. It creates the virtual environment in `~/venv/<basename project dir>` by default, so this repository installs to `~/venv/api-diarization` unless you override `VENV_DIR`.

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
./install.sh
cp .env.example .env
nano .env
```

Set at least `API_TOKENS` in `.env` before production use. For a CUDA-specific PyTorch wheel on H100/DGX Spark hosts, set `TORCH_INDEX_URL` before install, for example:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./install.sh
```

## Start

```bash
source ./run.sh 0.0.0.0 8000
```

`run.sh` accepts `source ./run.sh [IP] [PORT]`, auto-runs `install.sh` if the venv is missing, loads `.env`, and avoids `exec` so your shell survives `CTRL+C`. It is also compatible with systemd:

```ini
[Service]
WorkingDirectory=/workspace/api-diarization
ExecStart=/bin/bash -lc 'source /workspace/api-diarization/run.sh 0.0.0.0 8000'
Restart=always
```

## Configuration

Important variables are shown in `.env.example`. Optional/default variables are commented with `#`.

- `API_TOKENS`: comma-separated API tokens.
- `EMB_DEVICE`: `auto`, `cuda`, `cuda:0`, or `cpu`. Default `auto` uses CUDA when available and falls back to CPU if CUDA model initialization runs out of memory.
- `TORCH_INDEX_URL`: optional PyTorch wheel index for CUDA-specific installs during `install.sh`.
- `EMBEDDING_STORE`: speaker embedding JSON store.
- `NOISE_STORE`: noise embedding JSON store.

## API

### Health

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/health
```

### Diarize audio

Popular token styles are supported: `Authorization: Bearer ...` and `X-API-Key: ...`.

```bash
curl -X POST http://127.0.0.1:8000/v1/diarize \
  -H 'Authorization: Bearer change-me' \
  -F 'file=@sample.wav'
```

```bash
curl -X POST http://127.0.0.1:8000/v1/diarize \
  -H 'X-API-Key: change-me' \
  -F 'file=@sample.wav'
```

Legacy endpoint remains available for compatibility:

```text
POST /upload-audio/
```

Example response:

```json
{
  "identifier": "d5bc7c3f-5c83-4d39-8bfa-8f3b3d6d4aa7",
  "noise": false,
  "duration": 5.23,
  "similarity": 0.83,
  "updated": true
}
```
