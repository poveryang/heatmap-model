"""Heatmap model Python package."""

from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_ROOT.parent
CONFIGS_DIR = PYTHON_ROOT / "configs"
RUNS_DIR = PYTHON_ROOT / "runs"
