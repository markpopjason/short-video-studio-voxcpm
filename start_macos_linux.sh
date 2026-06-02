#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
else
  PYTHON_BIN="python3"
fi

if [ ! -x ".venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

".venv/bin/python" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' || {
  echo "VoxCPM requires Python 3.10, 3.11, or 3.12."
  exit 1
}

".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt || {
  echo "Dependency installation failed."
  echo "Please make sure your platform has a PyTorch 2.5+ wheel."
  exit 1
}

".venv/bin/python" -m voxcpm_studio serve --host 127.0.0.1 --port 8808 --device auto
