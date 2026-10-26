#!/usr/bin/env bash
# Train SuperMimic on the 40 LAFAN1 clips on a 16 GB GPU.
#
# Usage:
#   bash scripts/run_lafan1.sh <run-name> <anchor:0|1> [num-envs] [batch] [max-hours] [iterations] [seed]
#
# Examples:
#   bash scripts/run_lafan1.sh lafan1_anchor_s0 1 2048 2048 3.5 48000 0
#   bash scripts/run_lafan1.sh lafan1_paper_s0  0 2048 2048 3.5 48000 0
#
# The LR at every iteration matches the original 320k schedule, including short runs.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${1:-lafan1_anchor}"
ANCHOR="${2:-1}"
NUM_ENVS="${3:-2048}"
BATCH="${4:-2048}"
MAX_HOURS="${5:-6}"
ITERATIONS="${6:-320000}"
SEED="${7:-0}"
LR_DECAY_FRACTION="$(.venv/bin/python -c "print(250000 / $ITERATIONS)")"

if [ "$ANCHOR" = "1" ]; then
  ANCHOR_ARG="--env.anchor-obs"
else
  ANCHOR_ARG="--env.no-anchor-obs"
fi

export PYTHONPATH=
exec .venv/bin/python -m supermimic.scripts.train \
  --run-name "$RUN_NAME" \
  --seed "$SEED" \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs "$NUM_ENVS" \
  "$ANCHOR_ARG" \
  --policy-trainer.batch-size "$BATCH" \
  --wm-trainer.batch-size "$BATCH" \
  --policy-trainer.compile default \
  --wm-trainer.compile default \
  --schedule.total-iterations "$ITERATIONS" \
  --schedule.lr-decay-fraction "$LR_DECAY_FRACTION" \
  --schedule.max-hours "$MAX_HOURS" \
  --schedule.log-interval 50 \
  --schedule.checkpoint-interval 4000
