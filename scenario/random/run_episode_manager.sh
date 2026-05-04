#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$ROOT_DIR/runner" && pwd)"

PLANNER="${PLANNER:-rda_traj}"
FOLLOW_POSITION="${FOLLOW_POSITION:-back}"
DESIRED_DISTANCE="${DESIRED_DISTANCE:-1.5}"

pkill -f "run_episode_manager.py" >/dev/null 2>&1 || true
python "$BASE_DIR/run_episode_manager.py" \
  --load-map \
  --map-name Town10HD_Opt \
  --grid-npz "$ROOT_DIR/assets/gridmap_roi.npz" \
  --flow-points-json "$ROOT_DIR/assets/gridmap_roi_flow_points.json" \
  --robot-spawn-mode near_target \
  --robot-spawn-min-dist 0.8 \
  --robot-spawn-max-dist 2.0 \
  --num-walkers 15 \
  --target-track-id N01 \
  --visibility-threshold 400 \
  --robot-rescue-lift-z 0.25 \
  --sensor-image-w 800 \
  --sensor-image-h 600 \
  --sensor-fov-deg 90 \
  --duration-sec 300 \
  --draw-debug \
  --draw-laser \
  --planner "$PLANNER" \
  --follow-position "$FOLLOW_POSITION" \
  --desired-distance "$DESIRED_DISTANCE" \
  --eval-scenario-type random \
  --output-dir "$ROOT_DIR/runs" \
  "$@"
