#!/bin/zsh

set -eu

SCRIPT_DIR=${0:A:h}
PROJECT_DIR=${SCRIPT_DIR:h}
IMAGE="/Volumes/VIDEO/ToneCaptureResearch/ToneCaptureResearch.sparsebundle"
MOUNT="/Volumes/ToneCaptureResearch"
PYTHON="$MOUNT/venv/bin/python"
MAX_USED_KB=$((5 * 1024 * 1024))

case "$PROJECT_DIR:${PYTHONPATH:-}:${VIRTUAL_ENV:-}" in
  *schwab_trading_bot*)
    print -u2 "Refusing to run: the tone research stack cannot use a Schwab project path or environment."
    print -u2 "Deactivate that environment, open the guitar tone project, and run this command again."
    exit 64
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  if [[ ! -d "$IMAGE" ]]; then
    print -u2 "Tone research image not found: $IMAGE"
    print -u2 "Connect the VIDEO drive. The production MLX system still works without this research stack."
    exit 66
  fi
  /usr/bin/hdiutil attach "$IMAGE" -mountpoint "$MOUNT" -nobrowse >/dev/null
fi

if [[ ! -x "$PYTHON" ]]; then
  print -u2 "Research Python is unavailable after mounting $IMAGE"
  exit 69
fi

USED_KB=$(/usr/bin/du -sk "$MOUNT" | /usr/bin/awk '{print $1}')
if (( USED_KB > MAX_USED_KB )); then
  print -u2 "Research image has exceeded the 5 GB working limit."
  print -u2 "Run the stack check before adding packages or caches."
  exit 70
fi

mkdir -p \
  "$MOUNT/tmp" \
  "$MOUNT/cache/matplotlib" \
  "$MOUNT/cache/numba" \
  "$MOUNT/cache/torch" \
  "$MOUNT/cache/huggingface"

export TMPDIR="$MOUNT/tmp"
export MPLCONFIGDIR="$MOUNT/cache/matplotlib"
export NUMBA_CACHE_DIR="$MOUNT/cache/numba"
export TORCHINDUCTOR_CACHE_DIR="$MOUNT/cache/torch"
export HF_HOME="$MOUNT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PIP_NO_CACHE_DIR=1
export PYTHONNOUSERSITE=1

cd "$PROJECT_DIR"
if [[ "${1:-}" == "--nam-full" ]]; then
  shift
  exec "$MOUNT/venv/bin/nam-full" "$@"
fi
exec "$PYTHON" "$@"
