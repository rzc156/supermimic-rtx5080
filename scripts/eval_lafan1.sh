#!/usr/bin/env bash
# Evaluate a checkpoint on all 40 LAFAN1 clips (paper Appendix F protocol).
#
# Usage:
#   bash scripts/eval_lafan1.sh <checkpoint> [num-envs]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?usage: bash scripts/eval_lafan1.sh <checkpoint> [num-envs]}"
NUM_ENVS="${2:-1024}"

export PYTHONPATH=
exec .venv/bin/python scripts/eval_benchmark.py \
  --ckpt "$CKPT" \
  --npz-dir data/npz/lafan1 \
  --num-envs "$NUM_ENVS"
