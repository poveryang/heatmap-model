#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MQBENCH_REPO="${MQBENCH_REPO:-https://github.com/ModelTC/MQBench.git}"
MQBENCH_REF="${MQBENCH_REF:-main}"
PIP_TIMEOUT="${PIP_TIMEOUT:-600}"

usage() {
  cat <<'USAGE'
Usage: python/scripts/install_mqbench.sh [options]

Install MQBench from GitHub. MQBench is not published on PyPI.

Run this after installing python/requirements.txt inside your Python
environment (for example conda env hmap).

Options:
  --ref VALUE   Git branch, tag, or commit. Default: main
  --repo URL    Git repository URL. Default: https://github.com/ModelTC/MQBench.git
  -h, --help    Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      MQBENCH_REF="$2"
      shift 2
      ;;
    --repo)
      MQBENCH_REPO="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH." >&2
  exit 1
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch is not installed. Run: pip install -r python/requirements.txt" >&2
  exit 1
fi

echo "Installing MQBench from ${MQBENCH_REPO}@${MQBENCH_REF} ..."
pip install --default-timeout="${PIP_TIMEOUT}" --no-deps \
  "git+${MQBENCH_REPO}@${MQBENCH_REF}"

python - <<'PY'
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import BackendType, prepare_by_platform
from mqbench.utils.state import enable_calibration, enable_quantization

if not hasattr(BackendType, "Tengine_u8"):
    raise SystemExit("Installed MQBench is missing BackendType.Tengine_u8.")
print(f"MQBench OK (Tengine backend: {BackendType.Tengine_u8.value})")
PY
