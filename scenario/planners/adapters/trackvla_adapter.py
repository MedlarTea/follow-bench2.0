"""End-to-end TrackVLA follower adapter.

Wraps `OfflineUnifiedAgent` (vendored under
``planners/learning_based/trackvla/socialRPF/``). On every tick, takes the
front-camera RGB, runs ViT (DINOv3 + SigLIP) → Qwen3-4B VLM → planner head, and
turns the second predicted waypoint into a (v, w) command (dt=0.1 s).

Single-view "forward" only. Side cameras are ignored.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

_PLANNERS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEARNING_DIR = os.path.join(_PLANNERS_DIR, "learning_based", "trackvla")
if _LEARNING_DIR not in sys.path:
    sys.path.insert(0, _LEARNING_DIR)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from core_types import FollowAction, FollowObservation
from follow_policy_adapter import FollowerPolicyAdapter

try:
    from socialRPF.baseline.offline_agent_unified import OfflineUnifiedAgent
    from socialRPF.constants import DATA_ROOT, IMAGE_SIZE

    _HAS_TRACKVLA = True
    _TRACKVLA_IMPORT_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_TRACKVLA = False
    _TRACKVLA_IMPORT_ERR = str(_e)


_DEFAULT_CKPT_REL = (
    "ckpts/260417-qwen3-4b-unified-nav-qa-view-fwd-ws16-bs16-acc1-glb256-"
    "qabs16-ep5-lrb2e-5-lrh2e-4-bn10.0-bq1.0/model_epoch04_step024000.pt"
)


def _resize_with_padding(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(side / w, side / h)
    rw, rh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = side - rh, side - rw
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=0,
    )


_TRAIN_TIME_DT = 0.1  # Waypoint spacing the checkpoint was trained with.


class TrackVlaFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        instruction: str = "Follow the person",
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        render_overlay: bool = False,
    ) -> None:
        if not _HAS_TRACKVLA:
            raise ImportError(
                f"TrackVLA imports failed: {_TRACKVLA_IMPORT_ERR}. Make sure "
                "scenario/planners/learning_based/trackvla/socialRPF/ is present "
                "and transformers/peft are installed in the active env."
            )

        if ckpt_path is None:
            ckpt_path = os.path.join(DATA_ROOT, _DEFAULT_CKPT_REL)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"TrackVLA ckpt not found: {ckpt_path}. Did the rsync to "
                f"{DATA_ROOT} finish?"
            )

        self._dt_carla = float(dt)
        self._instruction = str(instruction)
        self._render_overlay = bool(render_overlay)
        self._image_size = int(IMAGE_SIZE)

        # Two dt-related decisions when sim != training rate:
        #   1) waypoint→velocity uses _TRAIN_TIME_DT so command magnitudes match
        #      what the planner head was trained to produce.
        #   2) inference is throttled to ~10 Hz; intermediate sim ticks reuse
        #      the previous command. Otherwise the 31-step coarse history would
        #      cover 31*dt_carla seconds (1.55 s at 20 Hz) instead of the 3.1 s
        #      training window — the model would perceive the world as moving
        #      at 2× speed.
        self._inference_stride = max(1, int(round(_TRAIN_TIME_DT / self._dt_carla)))
        self._tick_counter = 0
        self._last_action_cached = (0.0, 0.0, 0.0)
        if self._inference_stride != 1 or abs(self._dt_carla - _TRAIN_TIME_DT) > 1e-3:
            print(
                f"[TRACKVLA] sim dt={self._dt_carla:.3f}s, model dt={_TRAIN_TIME_DT}s "
                f"→ inference_stride={self._inference_stride} (skip {self._inference_stride-1} ticks between calls).",
                flush=True,
            )

        print(f"[TRACKVLA] loading ckpt {ckpt_path}", flush=True)
        t0 = time.perf_counter()
        self._agent = OfflineUnifiedAgent(
            ckpt_path=ckpt_path, device=device, dt=_TRAIN_TIME_DT,
        )
        print(f"[TRACKVLA] ready ({(time.perf_counter()-t0):.1f}s)", flush=True)

        self._last_traj_world: List[List[float]] = []
        self._last_rendered: Optional[np.ndarray] = None
        self._last_action = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._agent.reset()
        self._last_traj_world = []
        self._last_rendered = None
        self._last_action = (0.0, 0.0, 0.0)
        self._tick_counter = 0
        self._last_action_cached = (0.0, 0.0, 0.0)

    def get_debug_info(self) -> dict:
        info = {
            "obstacles": [],
            "traj_points": list(self._last_traj_world),
            "instruction": self._instruction,
            "trackvla_action": list(self._last_action),
        }
        if self._last_rendered is not None:
            info["rendered_front"] = self._last_rendered
        return info

    def act(self, obs: FollowObservation) -> FollowAction:
        if obs.rgb_image is None:
            print("[TRACKVLA] no rgb_image — emitting zero action", flush=True)
            return FollowAction(v_mps=0.0, w_radps=0.0)

        # Hold previous command on intermediate ticks to keep model query rate
        # close to the 10 Hz training rate.
        if self._tick_counter % self._inference_stride != 0:
            self._tick_counter += 1
            v_mps, _, w_radps = self._last_action_cached
            return FollowAction(v_mps=v_mps, w_radps=w_radps)

        rgb = _resize_with_padding(obs.rgb_image, self._image_size)
        t0 = time.perf_counter()
        action, traj, rendered = self._agent.predict_frame(
            rgb, self._instruction, render=self._render_overlay,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self._tick_counter += 1

        vx = float(action[0])
        vy = float(action[1])
        wz = float(action[2])
        # Handedness flip: model is trained in a right-handed frame (+y=left,
        # +theta=CCW). CARLA's world is left-handed (+yaw_deg=CW from above),
        # and apply_velocity_command does yaw_deg += w*dt directly. So +theta
        # from the model means physical CCW, which CARLA reaches with negative
        # w_radps. Without this flip the robot turns toward the wrong side.
        v_mps = vx
        w_radps = -wz
        self._last_action = (vx, vy, wz)
        self._last_action_cached = (v_mps, vy, w_radps)

        # Transform robot-frame traj → CARLA world. Model is right-handed
        # (+x forward, +y left); CARLA world is left-handed (+yaw_deg=CW). The
        # forward axis aligns; the lateral axis is mirrored, so ty contributes
        # with the opposite sign of the standard rotation matrix.
        traj_world: List[List[float]] = []
        if traj is not None and traj.shape[0] > 0:
            cy = math.cos(float(obs.robot.yaw_rad))
            sy = math.sin(float(obs.robot.yaw_rad))
            rx, ry = float(obs.robot.x), float(obs.robot.y)
            for i in range(traj.shape[0]):
                tx = float(traj[i, 0])
                ty = float(traj[i, 1]) if traj.shape[1] >= 2 else 0.0
                wx = rx + tx * cy + ty * sy
                wy = ry + tx * sy - ty * cy
                traj_world.append([wx, wy])
        self._last_traj_world = traj_world
        self._last_rendered = rendered

        print(
            f"[TRACKVLA] tick={obs.tick}  v={v_mps:+.3f} w={w_radps:+.3f}"
            f"  vy_drop={vy:+.3f}  infer={infer_ms:.1f}ms"
            f"  instr={self._instruction!r}",
            flush=True,
        )
        return FollowAction(v_mps=v_mps, w_radps=w_radps)


__all__ = ["TrackVlaFollowerPolicy"]
