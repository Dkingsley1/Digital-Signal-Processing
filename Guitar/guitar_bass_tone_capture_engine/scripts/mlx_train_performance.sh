#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

perf_cores="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"
if ! [[ "$perf_cores" =~ ^[0-9]+$ ]] || [ "$perf_cores" -lt 1 ]; then
  perf_cores="$(sysctl -n hw.physicalcpu 2>/dev/null || true)"
fi
if ! [[ "$perf_cores" =~ ^[0-9]+$ ]] || [ "$perf_cores" -lt 1 ]; then
  perf_cores=4
fi

export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$perf_cores}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$perf_cores}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$perf_cores}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$perf_cores}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

python_bin="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python3}"
if [ ! -x "$python_bin" ]; then
  python_bin="python3"
fi

echo "Performance training launcher"
echo "Project: $PROJECT_DIR"
echo "Python: $python_bin"
echo "Math threads: $perf_cores"
echo "Note: macOS schedules P/E cores automatically; this favors performance but is not hard CPU pinning."
echo

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu "$python_bin" "$PROJECT_DIR/tone_capture_engine.py" "$@"
fi

exec "$python_bin" "$PROJECT_DIR/tone_capture_engine.py" "$@"
