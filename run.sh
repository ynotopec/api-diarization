#!/usr/bin/env bash
# Usage:
#   source ./run.sh [IP] [PORT]
#   ./run.sh [IP] [PORT]
# Compatible with systemd: ExecStart=/bin/bash -lc 'source /path/to/run.sh 0.0.0.0 8000'
set -Eeuo pipefail

SERVER_ADDRESS="${1:-${HOST:-${SERVER_NAME:-0.0.0.0}}}"
PORT_NUMBER="${2:-${PORT:-${SERVER_PORT:-8000}}}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
UVICORN_RELOAD="${UVICORN_RELOAD:-0}"

cd "${PROJECT_DIR}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HOST="${SERVER_ADDRESS}"
export PORT="${PORT_NUMBER}"
export SERVER_NAME="${SERVER_ADDRESS}"
export SERVER_PORT="${PORT_NUMBER}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Virtualenv not found at ${VENV_DIR}; running install.sh" >&2
  "${PROJECT_DIR}/install.sh"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

reload_args=()
if [ "${UVICORN_RELOAD}" = "1" ]; then
  reload_args+=(--reload)
fi

exec python -m uvicorn app:app --host "${SERVER_ADDRESS}" --port "${PORT_NUMBER}" "${reload_args[@]}"
