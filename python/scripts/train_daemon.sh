#!/usr/bin/env bash
# Start training in tmux with a persistent log file.
#
# Usage:
#   bash python/scripts/train_daemon.sh hmap-barcode-qroi-v2
#   bash python/scripts/train_daemon.sh hmap-barcode-qroi-v2 --wandb
#
# Attach:
#   tmux attach -t hmap-hmap-barcode-qroi-v2
#
# Tail log:
#   tail -f python/runs/<run_name>/train.log

set -euo pipefail

EXP="${1:-hmap-barcode-qroi-v2}"
shift || true

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_NAME="${HMAP_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_${EXP}}"
RUN_DIR="${REPO}/python/runs/${RUN_NAME}"
SESSION="hmap-${EXP}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-hmap}"

mkdir -p "${RUN_DIR}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

GIT_BRANCH="$(git -C "${REPO}" branch --show-current 2>/dev/null || true)"
GIT_COMMIT="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || true)"
GIT_STATUS_COUNT="$(git -C "${REPO}" status --short 2>/dev/null | wc -l | tr -d ' ')"

{
  echo "run_name=${RUN_NAME}"
  echo "exp=${EXP}"
  echo "started_at=$(date -Is)"
  echo "repo=${REPO}"
  echo "git_branch=${GIT_BRANCH}"
  echo "git_commit=${GIT_COMMIT}"
  echo "git_status=${GIT_STATUS_COUNT} changed files"
  echo "conda_env=${CONDA_ENV}"
  echo "args=$*"
} > "${RUN_DIR}/run.env"

if [[ -f "${REPO}/python/configs/${EXP}.yaml" ]]; then
  cp "${REPO}/python/configs/${EXP}.yaml" "${RUN_DIR}/config.yaml"
fi

CMD="export HMAP_RUN_NAME='${RUN_NAME}' && source '${CONDA_SH}' && conda activate '${CONDA_ENV}' && cd '${REPO}' && python python/train.py --exp '${EXP}' $* 2>&1 | tee '${RUN_DIR}/train.log'"

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" "${CMD}"
  echo "Started training in tmux session: ${SESSION}"
  echo "Run directory: ${RUN_DIR}"
  echo "Live log: ${RUN_DIR}/train.log"
  echo "Attach: tmux attach -t ${SESSION}"
else
  nohup bash -lc "${CMD}" >/dev/null 2>&1 &
  echo "tmux not found; started with nohup."
  echo "Run directory: ${RUN_DIR}"
  echo "Live log: ${RUN_DIR}/train.log"
fi
