#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-seed}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "${DATASET}" in seed|seediv|seedv) ;; *) echo "Unknown dataset: ${DATASET}" >&2; exit 1 ;; esac
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_ROOT}/outputs/${DATASET}/${RUN_ID}}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd -- "${RUN_ROOT}" && pwd)"
LAUNCH_LOG="${RUN_ROOT}/launcher.log"
PID_FILE="${RUN_ROOT}/run.pid"
if [[ "${DETACH:-1}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
    nohup env DETACH=0 RUN_ID="${RUN_ID}" RUN_ROOT="${RUN_ROOT}" \
        bash "${EXPERIMENT_ROOT}/run.sh" "${DATASET}" "$@" >>"${LAUNCH_LOG}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "Started ${DATASET}; pid=$!; log=${LAUNCH_LOG}"
    exit 0
fi
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    VENV_ACTIVATE="${VENV_ACTIVATE-/mnt/sdc/sdc1/yangli/yangli/EEG/bk1/ZTC/workspace/bin/activate}"
    if [[ -n "${VENV_ACTIVATE}" ]]; then source "${VENV_ACTIVATE}"; fi
fi
args=(--dataset "${DATASET}" --run-dir "${RUN_ROOT}")
if [[ -n "${DATASET_PATH:-}" ]]; then args+=(--dataset-path "${DATASET_PATH}"); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then args+=(--dry-run); fi
if [[ -n "${DEVICE:-}" ]]; then args+=(--device "${DEVICE}"); fi
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python3}" "${EXPERIMENT_ROOT}/run.py" "${args[@]}" "$@"
