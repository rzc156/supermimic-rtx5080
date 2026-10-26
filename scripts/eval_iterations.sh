#!/usr/bin/env bash
# Evaluate selected checkpoints of a run on all 40 LAFAN1 clips.
#
# Usage:
#   bash scripts/eval_iterations.sh <run-name> 24000 48000
#   bash scripts/eval_iterations.sh <run-name> latest
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:?usage: bash scripts/eval_iterations.sh <run-name> <iter|latest> [...]}"
shift

for IT in "$@"; do
  if [ "$IT" = "latest" ]; then
    CKPT="logs/${RUN}/ckpt_latest.pt"
  else
    CKPT=$(printf "logs/%s/ckpt_%06d.pt" "$RUN" "$IT")
  fi
  if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not found"
    continue
  fi
  echo "==================== $CKPT ===================="
  bash scripts/eval_lafan1.sh "$CKPT" 1024
done
