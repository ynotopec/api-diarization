#!/usr/bin/env bash
# Usage:
#   source ./run.sh [IP] [PORT]
#   ./run.sh [IP] [PORT]
# Compatible with systemd: ExecStart=/bin/bash -lc 'source /path/to/run.sh 0.0.0.0 8000'
set -Eeuo pipefail

_run_diarization_api() {
  local server_address="${1:-${HOST:-${SERVER_NAME:-0.0.0.0}}}"
  local port_number="${2:-${PORT:-${SERVER_PORT:-8000}}}"
  local project_dir project_name venv_dir uvicorn_reload

  project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  project_name="$(basename "${project_dir}")"
  venv_dir="${VENV_DIR:-${HOME}/venv/${project_name}}"
  uvicorn_reload="${UVICORN_RELOAD:-0}"

  cd "${project_dir}"

  if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi

  export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
  export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:invalid escape sequence:SyntaxWarning}"
  export HOST="${server_address}"
  export PORT="${port_number}"
  export SERVER_NAME="${server_address}"
  export SERVER_PORT="${port_number}"

  if [ ! -x "${venv_dir}/bin/python" ]; then
    echo "Virtualenv not found at ${venv_dir}; running install.sh" >&2
    "${project_dir}/install.sh"
  fi

  # shellcheck disable=SC1091
  source "${venv_dir}/bin/activate"

  local reload_args=()
  if [ "${uvicorn_reload}" = "1" ]; then
    reload_args+=(--reload)
  fi

  python -m uvicorn app:app --host "${server_address}" --port "${port_number}" "${reload_args[@]}"
}

set +e
_run_diarization_api "$@"
_run_status=$?
set -e
unset -f _run_diarization_api

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return "${_run_status}"
fi
exit "${_run_status}"
