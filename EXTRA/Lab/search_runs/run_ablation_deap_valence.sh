#!/usr/bin/env bash
# BAPCN ablation on DEAP-V (binarized VALENCE), subject-independent TVT.
#
# Identical protocol and hyper-parameters to run_ablation_deap_arousal.sh; only
# the label axis differs (ablation_config.py, protocol "deap_v"). Keeping the two
# as separate entry points matches how the main table reports DEAP-A and DEAP-V
# as two tasks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIBEER_DIR="${REPO_ROOT}/LibEER/LibEER"
RUNNER="${LIBEER_DIR}/ablation/run_ablation.py"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/sdc/sdc1/yangli/yangli/EEG/bk1/ZTC/workspace/bin/activate}"
DATASET_PATH="${DEAP_DATASET_PATH:-/mnt/sdc/sdc1/yangli/yangli/EEG/EEG_Dataset/DEAP/data_preprocessed_python}"

SEARCH_NAME="${SEARCH_NAME:-tvt_deap_valence_ablation}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/${SEARCH_NAME}/${RUN_ID}}"
LAUNCH_LOG="${SCRIPT_DIR}/${SEARCH_NAME}.launcher.log"
PID_FILE="${SCRIPT_DIR}/${SEARCH_NAME}.pid"

GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"
# a dry run is meant to be read on the terminal, so never detach it
if [[ "${DRY_RUN}" == "1" ]]; then
  DETACH=0
fi
RUN_VARIANTS="${RUN_VARIANTS:-default}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv activate file not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
if [[ ! -f "${RUNNER}" ]]; then
  echo "Ablation runner not found: ${RUNNER}" >&2
  exit 1
fi

if [[ "${DETACH}" == "1" && "${DEAP_VALENCE_CHILD:-0}" != "1" ]]; then
  nohup env \
    DEAP_VALENCE_CHILD=1 \
    DETACH=0 \
    RUN_ID="${RUN_ID}" \
    RUN_ROOT="${RUN_ROOT}" \
    SEARCH_NAME="${SEARCH_NAME}" \
    DEAP_DATASET_PATH="${DATASET_PATH}" \
    GPU_ID="${GPU_ID}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    DRY_RUN="${DRY_RUN}" \
    RUN_VARIANTS="${RUN_VARIANTS}" \
    VENV_ACTIVATE="${VENV_ACTIVATE}" \
    bash "${BASH_SOURCE[0]}" "$@" \
    > "${LAUNCH_LOG}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "Started ${SEARCH_NAME}; pid=$(cat "${PID_FILE}"); launcher log=${LAUNCH_LOG}"
  exit 0
fi

source "${VENV_ACTIVATE}"
mkdir -p "${RUN_ROOT}"

if [[ "${DRY_RUN}" != "1" && ! -d "${DATASET_PATH}" ]]; then
  echo "DEAP dataset not found: ${DATASET_PATH}" >&2
  echo "set DEAP_DATASET_PATH, or run with DRY_RUN=1 to inspect commands only." >&2
  exit 1
fi

RUN_ARGS=( --protocol deap_v --libeer-dir "${LIBEER_DIR}" --gpu "${GPU_ID}"
           --num-workers "${NUM_WORKERS}" --run-root "${RUN_ROOT}"
           --dataset-path "${DATASET_PATH}" --variants "${RUN_VARIANTS}" )
[[ "${DRY_RUN}" == "1" ]] && RUN_ARGS+=( --dry-run )

echo "Run root : ${RUN_ROOT}"
echo "Dataset  : DEAP-V (valence), subject-independent TVT"
echo "GPU      : ${GPU_ID}; variants: ${RUN_VARIANTS}"
python3 "${RUNNER}" "${RUN_ARGS[@]}"
