#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


RUN_SH_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$ROOT_DIR/../random" && pwd)"

pkill -f "run_episode_manager.py" >/dev/null 2>&1 || true

python "$BASE_DIR/run_episode_manager.py" \\
  --load-map \\
  --map-name {map_name} \\
  --grid-npz "$ROOT_DIR/assets/gridmap_roi.npz" \\
  --flow-points-json "$ROOT_DIR/assets/gridmap_roi_flow_points.json" \\
  --robot-spawn-mode near_target \\
  --robot-spawn-min-dist 0.8 \\
  --robot-spawn-max-dist 2.0 \\
  --num-walkers {num_walkers} \\
  --npc-min-speed {npc_min_speed} \\
  --npc-max-speed {npc_max_speed} \\
  --target-track-id N01 \\
  --visibility-threshold 400 \\
  --robot-rescue-lift-z 0.25 \\
  --sensor-image-w 800 \\
  --sensor-image-h 600 \\
  --sensor-fov-deg 90 \\
  --duration-sec {duration_sec} \\
  --draw-debug \\
  --draw-laser \\
  --output-dir "$ROOT_DIR/runs"
"""


README_TEMPLATE = """# {scenario_name} 场景

本场景通过 `scenario/random/run_episode_manager.py` 复用通用 episode 管线。

## 最小资产要求

请先在 `assets/` 下准备：

- `gridmap_roi.npz`
- `gridmap_roi_flow_points.json`

建议流程（标准化）：

1. 用 `scenario/map_annotator/carla_walk_roi_annotator.py` 在 CARLA 里采集 ROI。
2. 用 `scenario/map_annotator/convert_roi_to_minimal_assets.py` 生成最小可用集到本目录 `assets/`。
3. 运行本目录 `run_episode_manager.sh`。

## 启动

```bash
cd <repo-root>
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate followbench
cd scenario/{scenario_name}
./run_episode_manager.sh
```

## 备注

- 当前默认地图：`{map_name}`
- 你可以按场景需要调整 `run_episode_manager.sh` 里的 NPC 数量、速度、时长等参数。
"""


GITIGNORE_CONTENT = """runs/
assets/*.tmp
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create standardized scenario scaffold.")
    parser.add_argument("--scenario-name", required=True, help="new scenario folder name, e.g. clutter")
    parser.add_argument(
        "--scenario-root",
        default=os.environ.get("FOLLOWBENCH_SCENARIO_ROOT", os.getcwd()),
        help="path to scenario root",
    )
    parser.add_argument("--map-name", default="Town10HD_Opt")
    parser.add_argument("--num-walkers", type=int, default=28)
    parser.add_argument("--npc-min-speed", type=float, default=0.7)
    parser.add_argument("--npc-max-speed", type=float, default=1.2)
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_text(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    scenario_root = Path(args.scenario_root).expanduser().resolve()
    scenario_name = args.scenario_name.strip()
    if not scenario_name:
        raise SystemExit("scenario-name cannot be empty")
    if "/" in scenario_name or "\\" in scenario_name:
        raise SystemExit("scenario-name must be a single folder name")

    scenario_dir = scenario_root / scenario_name
    assets_dir = scenario_dir / "assets"
    runs_dir = scenario_dir / "runs"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_sh = scenario_dir / "run_episode_manager.sh"
    readme = scenario_dir / "README.md"
    gitignore = scenario_dir / ".gitignore"

    run_content = RUN_SH_TEMPLATE.format(
        map_name=args.map_name,
        num_walkers=args.num_walkers,
        npc_min_speed=args.npc_min_speed,
        npc_max_speed=args.npc_max_speed,
        duration_sec=args.duration_sec,
    )
    readme_content = README_TEMPLATE.format(
        scenario_name=scenario_name,
        map_name=args.map_name,
    )

    write_text(run_sh, run_content, overwrite=args.overwrite)
    write_text(readme, readme_content, overwrite=args.overwrite)
    write_text(gitignore, GITIGNORE_CONTENT, overwrite=args.overwrite)

    run_sh.chmod(0o775)

    print(f"Created scenario scaffold: {scenario_dir}")
    print(f"Assets dir: {assets_dir}")
    print("Next step:")
    print(
        f"  python3 {scenario_root / 'map_annotator' / 'convert_roi_to_minimal_assets.py'} "
        f"--annotated-npz <your_raw_npz> --base-npz <base_npz> --output-dir {assets_dir} --overwrite"
    )


if __name__ == "__main__":
    main()

