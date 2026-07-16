#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-uv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

cd "${PROJECT_DIR}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  else
    echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "${VENV_DIR}")"
"${UV_BIN}" venv --python "${PYTHON_BIN}" "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
"${UV_BIN}" pip install --upgrade pip setuptools wheel

install_args=(--upgrade -r requirements.txt)
if [ -n "${TORCH_INDEX_URL}" ]; then
  install_args+=(--extra-index-url "${TORCH_INDEX_URL}")
fi
"${UV_BIN}" pip install "${install_args[@]}"

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit API_TOKENS before production use."
fi

echo "Installed ${PROJECT_NAME} in ${VENV_DIR}"
