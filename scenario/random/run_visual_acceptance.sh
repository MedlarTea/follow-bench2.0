#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$ROOT_DIR/carla_roi_crowd_runner.py"
ASSET_DIR="$ROOT_DIR/assets"

python "$SCRIPT_PATH" \
  --grid-npz "$ASSET_DIR/gridmap_roi.npz" \
  --roi-polygon-json "$ASSET_DIR/gridmap_roi_polygon.json" \
  --flow-points-json "$ASSET_DIR/gridmap_roi_flow_points.json" \
  --debug-draw-roi \
  --debug-roi-every-sec 1.2 \
  --num-walkers 35 \
  --duration-sec 240 \
  --motion-mode bidirectional \
  --flow-anchor-radius 9 \
  --min-speed 1.0 \
  --max-speed 2.0 \
  --retarget-sec 4.8 \
  --stuck-window-sec 3.0 \
  --stuck-dist-threshold 0.35 \
  --avoid-radius 2.0 \
  --avoid-gain 1.5 \
  --lateral-bias 0.6 \
  --max-neighbors 12 \
  --crowd-slowdown-gain 0.5 \
  --enable-vertical-guidance \
  --enable-curb-assist \
  --spawn-min-sep 1.0 \
  --stat-every-sec 3
