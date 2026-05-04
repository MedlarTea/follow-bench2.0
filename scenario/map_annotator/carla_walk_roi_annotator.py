#!/usr/bin/env python3
from __future__ import annotations

"""
Scenario 侧 map annotator 入口。

为了保持两套目录（`senario/` 与 `scenario/`）行为一致，这个入口直接调用
已验证的实现文件：
`follow-bench-v2/senario/map_builder/carla_walk_roi_annotator.py`
"""

import runpy
from pathlib import Path


def main() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    impl = repo_root / "senario" / "map_builder" / "carla_walk_roi_annotator.py"
    if not impl.exists():
        raise SystemExit(f"annotator implementation not found: {impl}")
    runpy.run_path(str(impl), run_name="__main__")


if __name__ == "__main__":
    main()

