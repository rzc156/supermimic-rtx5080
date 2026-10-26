#!/usr/bin/env bash
# Train SuperMimic on the 40 LAFAN1 clips on a 16 GB GPU.
#
# Usage:
#   bash scripts/run_lafan1.sh <run-name> <anchor:0|1> [num-envs] [batch] [max-hours] [iterations]
#
# Examples:
#   bash scripts/run_lafan1.sh lafan1_anchor 1 2048 2048 3.5 320000
#   bash scripts/run_lafan1.sh lafan1_paper  0 2048 2048 3.0 320000
#
# Keep total-iterations at 320000 so the cosine LR schedule matches the paper;
# max-hours stops the run early (the reference repo does the same).
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${1:-lafan1_anchor}"
ANCHOR="${2:-1}"
NUM_ENVS="${3:-2048}"
BATCH="${4:-2048}"
MAX_HOURS="${5:-6}"
ITERATIONS="${6:-320000}"  # keep the paper schedule; use max-hours to stop early

if [ "$ANCHOR" = "1" ]; then
  ANCHOR_ARG="--env.anchor-obs"
else
  ANCHOR_ARG="--env.no-anchor-obs"
fi

export PYTHONPATH=
exec .venv/bin/python -m supermimic.scripts.train \
  --run-name "$RUN_NAME" \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs "$NUM_ENVS" \
  "$ANCHOR_ARG" \
  --policy-trainer.batch-size "$BATCH" \
  --wm-trainer.batch-size "$BATCH" \
  --policy-trainer.compile default \
  --wm-trainer.compile default \
  --schedule.total-iterations "$ITERATIONS" \
  --schedule.max-hours "$MAX_HOURS" \
  --schedule.log-interval 50 \
  --schedule.checkpoint-interval 4000
