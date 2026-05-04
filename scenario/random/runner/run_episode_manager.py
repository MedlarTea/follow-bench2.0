#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import carla
import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RANDOM_DIR = os.path.dirname(THIS_DIR)
if RANDOM_DIR not in sys.path:
    sys.path.insert(0, RANDOM_DIR)

from core_types import FollowObservation, NpcState, RobotState

PLANNERS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "planners"))
if PLANNERS_DIR not in sys.path:
    sys.path.insert(0, PLANNERS_DIR)

DEBUG_VIS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if DEBUG_VIS_DIR not in sys.path:
    sys.path.insert(0, DEBUG_VIS_DIR)
from npc_runtime import NpcRuntime
from robot_runtime import ScoutRobotRuntime
from visibility_instance import InstanceVisibilityChecker
from scenario.evaluation.core.collision import HumanCollisionMonitor, actor_track_lookup
from scenario.evaluation.core.logging import (
    EvaluationLogger,
    enrich_sensor_collision_events,
    make_geometry_collision_event,
)
from scenario.evaluation.core.schemas import add_evaluation_args

try:
    import pygame

    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False


# ── Manual control constants ──────────────────────────────────────────────────
MANUAL_WALKER_SPEED_MPS = 1.8
MANUAL_ROBOT_V_MPS      = 2.0
MANUAL_ROBOT_W_RADPS    = 2.2
MANUAL_SPRINT_MULT      = 1.8
MANUAL_CAM_DIST         = 4.5   # metres behind actor
MANUAL_CAM_HEIGHT       = 2.0   # metres above actor
MANUAL_MOUSE_SENS       = 0.20  # degrees per pixel


class TargetIdentityMonitor:
    """Evaluation-only watchdog for perception target identity switches."""

    def __init__(
        self,
        *,
        enabled: bool,
        associate_gate_m: float,
        wrong_dist_m: float,
        startup_grace_sec: float,
        confirm_sec: float,
        fail_sec: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.associate_gate_m = max(float(associate_gate_m), 0.0)
        self.wrong_dist_m = max(float(wrong_dist_m), 0.0)
        self.startup_grace_sec = max(float(startup_grace_sec), 0.0)
        self.confirm_sec = max(float(confirm_sec), 0.0)
        self.fail_sec = max(float(fail_sec), self.confirm_sec)
        self.mismatch_duration_s = 0.0
        self.max_mismatch_duration_s = 0.0
        self.switch_count = 0
        self._was_mismatching = False
        self.failure_track_id: Optional[str] = None

    def reset(self) -> None:
        self.mismatch_duration_s = 0.0
        self.max_mismatch_duration_s = 0.0
        self.switch_count = 0
        self._was_mismatching = False
        self.failure_track_id = None

    def update(
        self,
        *,
        dt: float,
        eval_time_s: float,
        target_track_id: str,
        target_state,
        npc_states,
        policy_debug_info: Optional[dict],
    ) -> dict:
        payload = {
            "enabled": bool(self.enabled),
            "status": "disabled",
            "identity_state": "disabled",
            "perception_has_target": None,
            "perception_track_id": None,
            "tracker_xy": None,
            "gt_target_track_id": str(target_track_id),
            "gt_target_xy": None,
            "err_to_gt_target_m": None,
            "associated_gt_track_id": None,
            "associated_gt_dist_m": None,
            "mismatch_active": False,
            "recovery_active": False,
            "recovery_reason": None,
            "mismatch_duration_s": float(self.mismatch_duration_s),
            "max_mismatch_duration_s": float(self.max_mismatch_duration_s),
            "switch_count": int(self.switch_count),
            "failure_track_id": self.failure_track_id,
            "termination_triggered": False,
            "termination_reason": None,
        }
        if not self.enabled:
            return payload

        payload["status"] = "no_perception"
        debug = policy_debug_info if isinstance(policy_debug_info, dict) else {}
        perception_has_target = bool(debug.get("perception_has_target", False))
        target_lost = bool(debug.get("target_lost", False))
        payload["perception_has_target"] = perception_has_target
        payload["perception_track_id"] = debug.get("active_track_id", debug.get("target_track_id"))
        recovery_reason = self._recovery_reason(debug)
        recovery_active = recovery_reason is not None
        payload["recovery_active"] = recovery_active
        payload["recovery_reason"] = recovery_reason

        gt_xy = None
        if target_state is not None:
            gt_xy = (float(target_state.x), float(target_state.y))
            payload["gt_target_xy"] = [gt_xy[0], gt_xy[1]]

        target_pos = debug.get("target_pos") if isinstance(debug.get("target_pos"), dict) else {}
        tracker_xy_raw = target_pos.get("tracker")
        tracker_xy = self._coerce_xy(tracker_xy_raw)
        if tracker_xy is not None:
            payload["tracker_xy"] = [tracker_xy[0], tracker_xy[1]]

        if target_lost or not perception_has_target or tracker_xy is None:
            payload["status"] = "recovering" if recovery_active else ("lost" if target_lost else "no_target")
            payload["identity_state"] = payload["status"]
            # Lost frames do not accumulate mismatch time, but they also do not
            # clear it. Only a clear association back to N01 resets the watchdog.
            payload["mismatch_duration_s"] = float(self.mismatch_duration_s)
            payload["max_mismatch_duration_s"] = float(self.max_mismatch_duration_s)
            payload["switch_count"] = int(self.switch_count)
            return payload

        associated_track_id, associated_dist = self._nearest_gt_track(tracker_xy, npc_states)
        payload["associated_gt_track_id"] = associated_track_id
        payload["associated_gt_dist_m"] = associated_dist
        if gt_xy is not None:
            payload["err_to_gt_target_m"] = float(np.hypot(tracker_xy[0] - gt_xy[0], tracker_xy[1] - gt_xy[1]))

        associated_to_target = (
            associated_track_id == str(target_track_id)
            and associated_dist is not None
            and associated_dist <= self.associate_gate_m
        )
        if associated_to_target:
            self.mismatch_duration_s = 0.0
            self._was_mismatching = False
            payload["status"] = "ok"
            payload["identity_state"] = "ok"
            payload["mismatch_duration_s"] = 0.0
            payload["max_mismatch_duration_s"] = float(self.max_mismatch_duration_s)
            payload["switch_count"] = int(self.switch_count)
            return payload

        if recovery_active:
            self._was_mismatching = False
            payload["status"] = "recovering"
            payload["identity_state"] = "recovering"
            payload["mismatch_duration_s"] = float(self.mismatch_duration_s)
            payload["max_mismatch_duration_s"] = float(self.max_mismatch_duration_s)
            payload["switch_count"] = int(self.switch_count)
            return payload

        err_to_target = payload["err_to_gt_target_m"]
        mismatch = (
            eval_time_s >= self.startup_grace_sec
            and associated_track_id is not None
            and associated_track_id != str(target_track_id)
            and associated_dist is not None
            and associated_dist <= self.associate_gate_m
            and err_to_target is not None
            and err_to_target >= self.wrong_dist_m
        )

        if mismatch:
            if not self._was_mismatching:
                self.switch_count += 1
            self._was_mismatching = True
            self.mismatch_duration_s += max(float(dt), 0.0)
            self.max_mismatch_duration_s = max(self.max_mismatch_duration_s, self.mismatch_duration_s)
            self.failure_track_id = str(associated_track_id)
            payload["failure_track_id"] = self.failure_track_id
            payload["mismatch_active"] = True
            if self.mismatch_duration_s >= self.fail_sec:
                payload["status"] = "failed"
                payload["identity_state"] = "failed"
                payload["termination_triggered"] = True
                payload["termination_reason"] = "target_identity_lost"
            elif self.mismatch_duration_s >= self.confirm_sec:
                payload["status"] = "suspect"
                payload["identity_state"] = "suspect"
            else:
                payload["status"] = "warming"
                payload["identity_state"] = "warming"
        else:
            payload["status"] = "ambiguous"
            payload["identity_state"] = "ambiguous"
            self._was_mismatching = False

        payload["mismatch_duration_s"] = float(self.mismatch_duration_s)
        payload["max_mismatch_duration_s"] = float(self.max_mismatch_duration_s)
        payload["switch_count"] = int(self.switch_count)
        return payload

    @staticmethod
    def _recovery_reason(debug: dict) -> Optional[str]:
        if bool(debug.get("identity_recovery_active", False)):
            return "identity_recovery_active"
        for key in ("recovery_state", "search_detail_mode", "search_mode"):
            value = debug.get(key)
            if value is None:
                continue
            if str(value).lower() == "reacquire_transition":
                return f"{key}=reacquire_transition"
        return None

    @staticmethod
    def _coerce_xy(value) -> Optional[tuple[float, float]]:
        if value is None:
            return None
        try:
            if len(value) < 2:
                return None
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _nearest_gt_track(tracker_xy: tuple[float, float], npc_states) -> tuple[Optional[str], Optional[float]]:
        best_track = None
        best_dist = None
        tx, ty = tracker_xy
        for state in npc_states or []:
            track_id = getattr(state, "track_id", None)
            if track_id is None:
                continue
            dist = float(np.hypot(float(state.x) - tx, float(state.y) - ty))
            if best_dist is None or dist < best_dist:
                best_track = str(track_id)
                best_dist = dist
        return best_track, best_dist


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Episode manager: decoupled NPC + robot + follow policy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--load-map", action="store_true")
    p.add_argument("--map-name", type=str, default="Town10HD_Opt")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--duration-sec", type=float, default=300.0)
    p.add_argument("--max-active-sec", type=float, default=180.0, help="Maximum active simulated evaluation time before failing the episode.")
    p.add_argument("--num-walkers", type=int, default=35)
    p.add_argument(
        "--npc-flow-mode",
        choices=["ab_cd", "pair_12_34", "pair_13_24", "free_3"],
        default="ab_cd",
        help="NPC flow mode. pair_12_34=(1<->2,3<->4), pair_13_24=(1<->3,2<->4), free_3=3-point random triangle.",
    )
    p.add_argument("--npc-min-speed", type=float, default=0.7, help="Initial NPC min speed (m/s)")
    p.add_argument("--npc-max-speed", type=float, default=1.3, help="Initial NPC max speed (m/s)")
    p.add_argument(
        "--npc-rng-seed",
        type=int,
        default=None,
        help="NPC random seed. Omit for a fresh seed each run; set an integer to reproduce a crowd layout.",
    )
    p.add_argument("--npc-navigation-mode", choices=["direct", "astar"], default="direct")
    p.add_argument(
        "--npc-pair-spawn-mode",
        choices=["roi_random", "paired_endpoint"],
        default="roi_random",
        help="Spawn mode for pair_* flows. roi_random spawns anywhere in the ROI, paired_endpoint spawns near the paired start endpoint.",
    )
    p.add_argument(
        "--npc-pair-goal-policy",
        choices=["fixed_until_arrival", "retarget_on_block"],
        default="fixed_until_arrival",
        help="Goal policy for pair_* flows. fixed_until_arrival keeps the current paired endpoint until arrival; retarget_on_block flips endpoints when blocked/aged.",
    )
    p.add_argument("--npc-pair-target-band-half-width-m", type=float, default=2.5)
    p.add_argument(
        "--target-spawn-policy",
        choices=["legacy", "endpoint_center"],
        default="legacy",
        help="Target NPC spawn policy. endpoint_center keeps the target near the paired endpoint centerline.",
    )
    p.add_argument(
        "--crowd-spawn-policy",
        choices=["legacy", "roi_random"],
        default="legacy",
        help="Non-target NPC spawn policy. roi_random spreads the crowd across the full ROI.",
    )
    p.add_argument("--npc-spawn-min-sep-m", type=float, default=1.4)
    p.add_argument("--target-centerline-half-width-m", type=float, default=0.8)
    p.add_argument("--target-wall-clearance-min-m", type=float, default=1.2)
    p.add_argument("--target-endpoint-sample-radius-m", type=float, default=5.0)
    p.add_argument("--right-hand-lane-sep-m", type=float, default=0.72)
    p.add_argument("--target-lane-bias-scale", type=float, default=0.45)
    p.add_argument("--target-lane-min-offset-m", type=float, default=0.25)
    p.add_argument("--target-lane-max-offset-m", type=float, default=0.75)
    p.add_argument("--target-lateral-jitter-m", type=float, default=0.15)
    p.add_argument("--target-turnaround-forward-m", type=float, default=2.2)
    p.add_argument("--target-turnaround-merge-m", type=float, default=4.5)
    p.add_argument("--target-turnaround-side-m", type=float, default=1.6)
    p.add_argument("--target-turnaround-samples", type=int, default=20)
    p.add_argument("--target-turnaround-speed-mps", type=float, default=0.40)
    p.add_argument(
        "--crowd-turn-sublane-mode",
        choices=["off", "preserve_sublane"],
        default="off",
        help="Non-target corridor turn behavior. preserve_sublane keeps crowd lateral offsets through L-shape turns.",
    )
    p.add_argument("--crowd-turn-radius-jitter-m", type=float, default=0.0)
    p.add_argument("--crowd-turn-min-offset-keep-ratio", type=float, default=0.65)
    p.add_argument("--crowd-turn-samples", type=int, default=8)
    p.add_argument("--target-required-legs", type=int, default=2, help="Number of target endpoint-to-endpoint legs required for scenario success.")
    p.add_argument("--npc-replan-on-block", action="store_true")
    p.add_argument("--npc-block-replan-window-sec", type=float, default=2.0)
    p.add_argument("--npc-block-replan-progress-thresh", type=float, default=0.25)
    p.add_argument("--npc-block-replan-risk-thresh", type=float, default=0.5)
    p.add_argument("--npc-astar-replan-cooldown-sec", type=float, default=0.8)
    p.add_argument("--target-track-id", type=str, default="N01")
    p.add_argument("--visibility-threshold", type=int, default=400)
    p.add_argument("--grid-npz", type=str, default=os.path.join(THIS_DIR, "..", "assets", "gridmap_roi.npz"))
    p.add_argument("--flow-points-json", type=str, default=os.path.join(THIS_DIR, "..", "assets", "gridmap_roi_flow_points.json"))
    p.add_argument("--robot-spawn-x", type=float, default=-60.0)
    p.add_argument("--robot-spawn-y", type=float, default=5.0)
    p.add_argument("--robot-spawn-z", type=float, default=0.5)
    p.add_argument("--robot-spawn-yaw-deg", type=float, default=0.0)
    p.add_argument(
        "--robot-spawn-mode",
        choices=["fixed", "roi_random", "near_target", "behind_target"],
        default="near_target",
        help="Robot spawn strategy. near_target: anywhere near target. behind_target: strictly in the rear ±90° arc.",
    )
    p.add_argument("--robot-spawn-min-dist", type=float, default=2.5, help="Min robot-target spawn distance (near_target/behind_target)")
    p.add_argument("--robot-spawn-max-dist", type=float, default=8.0, help="Max robot-target spawn distance (near_target/behind_target)")
    p.add_argument("--robot-spawn-clearance-m", type=float, default=2.2, help="Minimum clearance from the robot spawn point to non-target pedestrians.")
    p.add_argument("--robot-spawn-retry-attempts", type=int, default=24, help="Number of local candidate transforms to try if the first robot spawn is blocked.")
    p.add_argument(
        "--side-follow-robot-yaw-policy",
        choices=["look_at_target", "match_target_heading"],
        default="look_at_target",
        help=(
            "Initial robot yaw for side-follow spawns. look_at_target keeps the old behavior; "
            "match_target_heading starts the robot parallel to the target route."
        ),
    )
    p.add_argument("--sensor-image-w", type=int, default=800)
    p.add_argument("--sensor-image-h", type=int, default=600)
    p.add_argument("--sensor-fov-deg", type=float, default=90.0)
    p.add_argument("--sensor-lidar-range", type=float, default=30.0)
    p.add_argument("--robot-rescue-lift-z", type=float, default=0.20)
    p.add_argument(
        "--planner",
        choices=["pid", "dwa_traj", "dwa_traj_depth_tpt", "sfm", "rda", "rda_lidar", "rda_traj", "rda_search", "rda_depth_tpt", "bso_hfc", "trackvla", "oa_vat"],
        default="pid",
        help="Follow policy: pid (default), dwa_traj, dwa_traj_depth_tpt, sfm, rda, rda_lidar, rda_traj, rda_search, rda_depth_tpt, bso_hfc, trackvla, oa_vat",
    )
    p.add_argument("--follow-position", choices=["back", "left_side", "right_side"], default="back")
    p.add_argument("--desired-distance", type=float, default=1.5)
    p.add_argument(
        "--target-lane-bias-mode",
        choices=["right_hand", "leave_follow_side_clear"],
        default="right_hand",
        help=(
            "Target lane offset strategy. right_hand keeps the old target route; "
            "leave_follow_side_clear offsets the target away from the requested side-follow point."
        ),
    )
    p.add_argument(
        "--rda-search-use-camera-visibility",
        action="store_true",
        help="Let rda_search enter search modes from camera visibility. Default keeps GT-target runs in follow mode.",
    )
    p.add_argument(
        "--rda-search-disable-target-protection",
        action="store_true",
        help="Disable the side-follow target protected zone injected for rda_search.",
    )
    p.add_argument(
        "--use-perception",
        action="store_true",
        help="Wrap the chosen --planner with a YOLO+depth+ReID perception "
             "frontend (multi-view). Target and NPC positions come from the "
             "tracker instead of GT. Compatible with every planner except "
             "rda_depth_tpt (which already has its own perception path). "
             "The --reid-* and --lost-policy flags below are only consulted "
             "when --use-perception is set.",
    )
    p.add_argument(
        "--reid-mode",
        choices=["basic", "kpr"],
        default="basic",
        help="Appearance ReID backbone (only when --use-perception): basic "
             "(ResNet, lightweight) or kpr (SOLIDER-Swin, occlusion-robust).",
    )
    p.add_argument(
        "--reid-kpr-config",
        type=str,
        default="kpr_occ_duke_test",
        help="KPR config name (only used when --reid-mode=kpr).",
    )
    p.add_argument(
        "--reid-device",
        type=str,
        default="auto",
        help="Device for the ReID backbone (auto/cuda/cpu).",
    )
    p.add_argument(
        "--lost-policy",
        choices=["gt_fallback", "brake", "last_known"],
        default="last_known",
        help="Behaviour when the perception frontend reports target lost: "
             "gt_fallback (use GT target), brake (zero command), "
             "last_known (drive toward last seen position; default). "
             "rda_search only accepts last_known.",
    )
    p.add_argument(
        "--kinematics-mode",
        choices=["ema", "kf"],
        default="kf",
        help="Per-track velocity estimator used by the depth tracker "
             "(only when --use-perception is set). "
             "'kf' (default) = constant-velocity Kalman filter (filterpy) — smoother "
             "vx/vy; recommended when downstream planners predict target trajectories "
             "(e.g. rda_traj + --use-perception). "
             "'ema' = finite-difference + EMA (lighter, use when vx/vy is not critical).",
    )
    p.add_argument(
        "--kf-pos-sigma",
        type=float,
        default=0.20,
        help="(--kinematics-mode=kf only) Measurement noise std-dev in metres. "
             "Increase to trust the depth tracker less and rely more on the "
             "CV-model prediction. Default 0.20 m.",
    )
    p.add_argument(
        "--kf-vel-sigma-q",
        type=float,
        default=0.05,
        help="(--kinematics-mode=kf only) Process noise std-dev on velocity "
             "(m/s). Reflects how quickly pedestrian velocity can change. "
             "Default 0.05 m/s.",
    )
    p.add_argument(
        "--trackvla-instruction",
        type=str,
        default="Follow the person",
        help="(planner=trackvla only) Natural-language instruction fed to the "
             "VLM each tick.",
    )
    p.add_argument(
        "--trackvla-ckpt",
        type=str,
        default=None,
        help="(planner=trackvla only) Override the default .pt checkpoint path "
             "under data/trackvla/ckpts/. Default: epoch04_step024000.pt.",
    )
    p.add_argument("--oa-vat-text-query", type=str, default="person",
                   help="(planner=oa_vat) YOLOe text prompt for first-frame and re-acq detection")
    p.add_argument("--oa-vat-kf-mode", type=str, default="confaware",
                   choices=["none", "standard", "confaware"],
                   help="(planner=oa_vat) Kalman filter mode for occlusion gap-fill")
    p.add_argument("--oa-vat-online-enhance", action="store_true", default=True,
                   help="(planner=oa_vat) EMA-update reference DINOv3 feature every N frames")
    p.add_argument("--oa-vat-no-online-enhance", dest="oa_vat_online_enhance",
                   action="store_false", help="(planner=oa_vat) disable online enhance")
    p.add_argument("--oa-vat-dinov3", type=str, default="dinov3_vitb16",
                   choices=["dinov3_vitb16", "dinov3_vits16"],
                   help="(planner=oa_vat) DINOv3 backbone")
    p.add_argument("--oa-vat-match-thresh", type=float, default=0.4,
                   help="(planner=oa_vat) DINOv3 cosine threshold for first-frame match")
    p.add_argument("--oa-vat-reacq-thresh", type=float, default=0.5,
                   help="(planner=oa_vat) DINOv3 cosine threshold for re-acquisition")
    p.add_argument("--oa-vat-yolo-conf", type=float, default=0.1,
                   help="(planner=oa_vat) YOLOe confidence threshold")
    p.add_argument("--oa-vat-target-area-ratio", type=float, default=0.12,
                   help="(planner=oa_vat) Desired bbox area / frame area for distance control")
    p.add_argument("--no-ui", action="store_true", help="Disable pygame key UI")
    p.add_argument("--debug", action="store_true", help="Enable 2D matplotlib top-down debug visualizer")
    p.add_argument("--auto-spawn-sec", type=float, default=-1.0, help="Auto trigger spawn after N seconds")
    p.add_argument("--auto-follow-sec", type=float, default=-1.0, help="Auto trigger follow after N seconds")
    p.add_argument("--output-dir", type=str, default=os.path.join(THIS_DIR, "..", "runs"))
    p.add_argument("--draw-debug", action="store_true")
    p.add_argument("--draw-laser", action="store_true", help="Draw a laser line from robot to target while following")
    p.add_argument("--async-mode", action="store_true", help="Run CARLA world in asynchronous mode")
    p.add_argument(
        "--async-after-spawn",
        action="store_true",
        help="When async-mode is enabled, keep sync mode until initial spawn finishes",
    )
    add_evaluation_args(p)
    return p.parse_args()


def make_robot_spawn_tf_fixed(args: argparse.Namespace) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=args.robot_spawn_x, y=args.robot_spawn_y, z=args.robot_spawn_z),
        carla.Rotation(yaw=args.robot_spawn_yaw_deg),
    )


def load_corridor_route_segments(flow_points_json: str, flow_mode: str) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    try:
        with open(flow_points_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[EP] warning: failed to load route segments from {flow_points_json}: {exc}")
        return []
    points = payload.get("points_world")
    if not isinstance(points, list) or len(points) < 4:
        return []
    xy = []
    for item in points[:4]:
        try:
            xy.append((float(item["x"]), float(item["y"])))
        except (TypeError, ValueError, KeyError):
            return []
    if flow_mode == "pair_13_24":
        pairs = [(0, 2), (1, 3)]
    else:
        pairs = [(0, 1), (2, 3)]
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i, j in pairs:
        ax, ay = xy[i]
        bx, by = xy[j]
        if math.hypot(bx - ax, by - ay) > 1e-6:
            segments.append(((ax, ay), (bx, by)))
    return segments


_FOLLOW_OFFSET_RAD = {
    "back":       math.pi,
    "front":      0.0,
    # CARLA target-body sides: visual-left is yaw - 90 deg, visual-right is yaw + 90 deg.
    "left_side":  -math.pi / 2.0,
    "right_side": math.pi / 2.0,
}


def follow_position_spawn_xy(target_x, target_y, target_yaw_rad,
                              follow_position, desired_distance,
                              roi_map, search_max_radius=1.5):
    offset = _FOLLOW_OFFSET_RAD.get(follow_position, math.pi)
    base_angle = target_yaw_rad + offset
    ideal_x = float(target_x + math.cos(base_angle) * desired_distance)
    ideal_y = float(target_y + math.sin(base_angle) * desired_distance)
    if roi_map.is_world_walkable(ideal_x, ideal_y):
        return ideal_x, ideal_y, True
    for radius in np.arange(0.2, search_max_radius + 1e-3, 0.2):
        for ang in np.linspace(0.0, 2 * math.pi, 12, endpoint=False):
            xx = float(ideal_x + math.cos(ang) * radius)
            yy = float(ideal_y + math.sin(ang) * radius)
            if roi_map.is_world_walkable(xx, yy):
                return xx, yy, True
    return ideal_x, ideal_y, False


def resolve_robot_spawn_yaw_deg(
    *,
    follow_position: str,
    yaw_policy: str,
    target_x: float,
    target_y: float,
    robot_x: float,
    robot_y: float,
    target_yaw_rad: float,
) -> float:
    # Side-follow starts parallel to the target route; back-follow keeps the
    # previous "look at target" initialization for camera visibility.
    if follow_position in ("left_side", "right_side") and yaw_policy == "match_target_heading":
        return float(np.degrees(target_yaw_rad))
    return float(np.degrees(np.arctan2(float(target_y) - float(robot_y), float(target_x) - float(robot_x))))


def resolve_robot_spawn_tf(args: argparse.Namespace, npc: NpcRuntime, target_track_id: str) -> carla.Transform:
    if args.robot_spawn_mode == "fixed":
        return make_robot_spawn_tf_fixed(args)

    if args.robot_spawn_mode in ("near_target", "behind_target"):
        t = npc.get_target_by_track_id(target_track_id)
        if t is not None:
            vx, vy = float(t.vx), float(t.vy)
            target_yaw_rad = (float(np.arctan2(vy, vx))
                              if abs(vx) + abs(vy) > 1e-4
                              else float(np.deg2rad(t.yaw_deg)))
            z = float(max(args.robot_spawn_z, t.z + 0.25))
            x, y, _ok = follow_position_spawn_xy(
                float(t.x), float(t.y), target_yaw_rad,
                str(args.follow_position), float(args.desired_distance),
                npc.roi_map,
            )
            robot_yaw = resolve_robot_spawn_yaw_deg(
                follow_position=str(args.follow_position),
                yaw_policy=str(args.side_follow_robot_yaw_policy),
                target_x=float(t.x),
                target_y=float(t.y),
                robot_x=x,
                robot_y=y,
                target_yaw_rad=target_yaw_rad,
            )
            return carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=robot_yaw),
            )

    # roi_random fallback
    loc = npc.roi_map.sample_world_location(npc.rng, z=args.robot_spawn_z, jitter=True)
    return carla.Transform(
        carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z)),
        carla.Rotation(yaw=args.robot_spawn_yaw_deg),
    )


def _robot_spawn_clear_of_npcs(
    tf: carla.Transform,
    npc_states: List[NpcState],
    target_track_id: str,
    min_clearance_m: float,
) -> bool:
    clearance = max(0.0, float(min_clearance_m))
    if clearance <= 1e-6:
        return True
    x = float(tf.location.x)
    y = float(tf.location.y)
    for st in npc_states:
        if str(st.track_id) == str(target_track_id):
            continue
        if float(np.hypot(float(st.x) - x, float(st.y) - y)) < clearance:
            return False
    return True


def _robot_spawn_candidate_transforms(
    base_tf: carla.Transform,
    args: argparse.Namespace,
    npc: NpcRuntime,
    target_track_id: str,
) -> List[carla.Transform]:
    max_attempts = max(1, int(args.robot_spawn_retry_attempts))
    target = npc.get_target_by_track_id(target_track_id)
    if target is None:
        return [base_tf]

    target_yaw_rad = yaw_rad_from_velocity(float(target.vx), float(target.vy), float(target.yaw_deg))
    forward = np.array([math.cos(target_yaw_rad), math.sin(target_yaw_rad)], dtype=np.float32)
    lateral = np.array([-math.sin(target_yaw_rad), math.cos(target_yaw_rad)], dtype=np.float32)
    base_xy = np.array([float(base_tf.location.x), float(base_tf.location.y)], dtype=np.float32)
    target_xy = np.array([float(target.x), float(target.y)], dtype=np.float32)

    offsets: List[np.ndarray] = [np.array([0.0, 0.0], dtype=np.float32)]
    for dist in (0.35, 0.70, 1.05, 1.40):
        offsets.append(forward * dist)
        offsets.append(-forward * dist)
        offsets.append(lateral * dist)
        offsets.append(-lateral * dist)
    for radius in (0.45, 0.90, 1.35):
        for ang in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            offsets.append(np.array([math.cos(ang) * radius, math.sin(ang) * radius], dtype=np.float32))

    out: List[carla.Transform] = []
    min_target_dist = max(0.8, float(args.robot_spawn_min_dist) * 0.75)
    max_target_dist = max(float(args.robot_spawn_max_dist), float(args.desired_distance) + 1.0)
    for off in offsets:
        xy = base_xy + off
        if not npc.roi_map.is_world_walkable(float(xy[0]), float(xy[1])):
            continue
        target_dist = float(np.linalg.norm(xy - target_xy))
        if target_dist < min_target_dist or target_dist > max_target_dist:
            continue
        yaw = resolve_robot_spawn_yaw_deg(
            follow_position=str(args.follow_position),
            yaw_policy=str(args.side_follow_robot_yaw_policy),
            target_x=float(target.x),
            target_y=float(target.y),
            robot_x=float(xy[0]),
            robot_y=float(xy[1]),
            target_yaw_rad=target_yaw_rad,
        )
        out.append(
            carla.Transform(
                carla.Location(x=float(xy[0]), y=float(xy[1]), z=float(base_tf.location.z)),
                carla.Rotation(yaw=yaw),
            )
        )
        if len(out) >= max_attempts:
            break
    return out or [base_tf]


def spawn_robot_with_retries(
    robot: ScoutRobotRuntime,
    base_tf: carla.Transform,
    args: argparse.Namespace,
    npc: NpcRuntime,
    target_track_id: str,
) -> carla.Transform:
    npc_states = npc.get_states()
    candidates = _robot_spawn_candidate_transforms(base_tf, args, npc, target_track_id)
    last_error: Optional[BaseException] = None
    skipped_for_clearance = 0

    for idx, tf in enumerate(candidates, start=1):
        if not _robot_spawn_clear_of_npcs(tf, npc_states, target_track_id, float(args.robot_spawn_clearance_m)):
            skipped_for_clearance += 1
            continue
        try:
            robot.spawn(tf)
            if idx > 1 or skipped_for_clearance > 0:
                print(
                    f"[EP] robot_spawn retry_success attempt={idx}/{len(candidates)} "
                    f"skipped_clearance={skipped_for_clearance} "
                    f"at=({tf.location.x:.2f},{tf.location.y:.2f},{tf.location.z:.2f})"
                )
            return tf
        except RuntimeError as exc:
            last_error = exc
            print(
                f"[EP][WARN] robot_spawn attempt={idx}/{len(candidates)} failed "
                f"at=({tf.location.x:.2f},{tf.location.y:.2f},{tf.location.z:.2f}): {exc}"
            )

    raise RuntimeError(
        "Failed to spawn robot scout after "
        f"{len(candidates)} local candidates; skipped_clearance={skipped_for_clearance}; "
        f"last_error={last_error}"
    )


def yaw_rad_from_velocity(vx: float, vy: float, fallback_deg: float) -> float:
    if abs(vx) + abs(vy) > 1e-4:
        return float(np.arctan2(vy, vx))
    return float(np.deg2rad(fallback_deg))


def draw_status(world: carla.World, text: str) -> None:
    world.debug.draw_string(
        carla.Location(x=0.0, y=0.0, z=3.0),
        text,
        draw_shadow=False,
        color=carla.Color(255, 255, 255),
        life_time=0.1,
    )


def step_world(world: carla.World, synchronous_mode: bool) -> None:
    if synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()


def apply_world_mode(world: carla.World, base_settings: carla.WorldSettings, sync_mode: bool, dt: float) -> None:
    s = world.get_settings()
    s.no_rendering_mode = base_settings.no_rendering_mode
    s.substepping = base_settings.substepping
    s.max_substep_delta_time = base_settings.max_substep_delta_time
    s.max_substeps = base_settings.max_substeps
    s.synchronous_mode = bool(sync_mode)
    s.fixed_delta_seconds = float(dt) if sync_mode else None
    world.apply_settings(s)


def _apply_manual_target_control(
    target_actor: Optional[carla.Actor],
    key_state,
    cam_yaw: float = 0.0,
) -> None:
    """Apply WASD walker control using camera yaw as the reference direction."""
    if target_actor is None or key_state is None:
        return
    move_forward = float(key_state[pygame.K_w]) - float(key_state[pygame.K_s])
    move_right   = float(key_state[pygame.K_d]) - float(key_state[pygame.K_a])
    sprint = bool(key_state[pygame.K_LSHIFT] or key_state[pygame.K_RSHIFT])
    speed  = MANUAL_WALKER_SPEED_MPS * (MANUAL_SPRINT_MULT if sprint else 1.0)
    if abs(move_forward) + abs(move_right) < 1e-6:
        target_actor.apply_control(
            carla.WalkerControl(direction=carla.Vector3D(1.0, 0.0, 0.0), speed=0.0, jump=False)
        )
        return
    yaw_rad = math.radians(cam_yaw)
    # CARLA/UE left-handed: forward=(cos,sin), right=(-sin,cos) at yaw angle
    dx = move_forward * math.cos(yaw_rad) - move_right * math.sin(yaw_rad)
    dy = move_forward * math.sin(yaw_rad) + move_right * math.cos(yaw_rad)
    n  = math.hypot(dx, dy)
    if n < 1e-6:
        return
    target_actor.apply_control(
        carla.WalkerControl(
            direction=carla.Vector3D(dx / n, dy / n, 0.0),
            speed=float(speed),
            jump=False,
        )
    )


def _apply_manual_robot_control(robot: ScoutRobotRuntime, key_state) -> None:
    """Apply WASD differential-drive control to the robot."""
    if key_state is None:
        return
    forward = float(key_state[pygame.K_w]) - float(key_state[pygame.K_s])
    turn    = float(key_state[pygame.K_d]) - float(key_state[pygame.K_a])
    sprint  = bool(key_state[pygame.K_LSHIFT] or key_state[pygame.K_RSHIFT])
    scale   = MANUAL_SPRINT_MULT if sprint else 1.0
    robot.apply_velocity_command(
        MANUAL_ROBOT_V_MPS * scale * forward,
        MANUAL_ROBOT_W_RADPS * scale * turn,
    )


def _update_manual_spectator(
    spectator: carla.Actor,
    actor: carla.Actor,
    cam_yaw: float,
    cam_pitch: float,
) -> None:
    """Position the spectator in 3rd-person behind `actor`."""
    loc       = actor.get_location()
    yaw_rad   = math.radians(cam_yaw)
    pitch_rad = math.radians(cam_pitch)
    ox = -MANUAL_CAM_DIST * math.cos(yaw_rad) * math.cos(pitch_rad)
    oy = -MANUAL_CAM_DIST * math.sin(yaw_rad) * math.cos(pitch_rad)
    oz =  MANUAL_CAM_HEIGHT - MANUAL_CAM_DIST * math.sin(pitch_rad)
    spectator.set_transform(carla.Transform(
        carla.Location(x=loc.x + ox, y=loc.y + oy, z=loc.z + max(oz, 0.5)),
        carla.Rotation(pitch=cam_pitch, yaw=cam_yaw, roll=0.0),
    ))


def _to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3][:, :, ::-1]


def _depth_to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    d = (
        arr[:, :, 2].astype(np.float32)
        + arr[:, :, 1].astype(np.float32) * 256.0
        + arr[:, :, 0].astype(np.float32) * 65536.0
    ) / (256.0**3 - 1.0) * 1000.0
    gray = np.clip(np.log1p(d) / np.log1p(100.0) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _depth_to_meters(image) -> Optional[np.ndarray]:
    """Decode CARLA depth RGBA buffer into HxW float32 metres."""
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return (
        arr[:, :, 2].astype(np.float32)
        + arr[:, :, 1].astype(np.float32) * 256.0
        + arr[:, :, 0].astype(np.float32) * 65536.0
    ) / (256.0**3 - 1.0) * 1000.0


def _instance_to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    b = arr[:, :, 0].astype(np.uint32)
    g = arr[:, :, 1].astype(np.uint32)
    r = arr[:, :, 2].astype(np.uint32)
    inst_id = (r << 16) | (g << 8) | b
    out = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    # deterministic pseudo-color mapping by id
    out[:, :, 0] = ((inst_id * 37) % 255).astype(np.uint8)
    out[:, :, 1] = ((inst_id * 67) % 255).astype(np.uint8)
    out[:, :, 2] = ((inst_id * 97) % 255).astype(np.uint8)
    out[inst_id == 0] = 0
    return out


def _lidar_to_points(lidar_data) -> Optional[np.ndarray]:
    """Decode CARLA LiDAR raw buffer into an (N, 4) float32 array [x, y, z, intensity]
    in the LiDAR sensor frame. Returns None when no scan is available."""
    if lidar_data is None:
        return None
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    return pts.copy()


def _draw_bboxes_on(img: Optional[np.ndarray], bboxes, stale: int) -> Optional[np.ndarray]:
    """Overlay perception-tracker bboxes onto a view's RGB image (returns a copy).

    ``bboxes`` is a list of dicts with keys {track_id, x1, y1, x2, y2, conf,
    is_target}. Green for target, yellow for others; dim when ``stale > 0``.
    Safe to call with None / empty list — returns the input unchanged.
    """
    if img is None or not bboxes:
        return img
    img = img.copy()
    for b in bboxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        tid = int(b["track_id"])
        is_tgt = bool(b.get("is_target", False))
        if is_tgt:
            color = (0, 220, 0) if stale == 0 else (0, 140, 0)
        else:
            color = (255, 215, 0) if stale == 0 else (160, 130, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"ID {tid}{' (TGT)' if is_tgt else ''}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1), (x1 + tw + 8, y1 + th + 6), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _lidar_to_bev(lidar_data, w: int = 400, h: int = 260, rng: float = 25.0) -> Optional[np.ndarray]:
    if lidar_data is None:
        return None
    img = np.zeros((h, w, 3), dtype=np.uint8)
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    m = (zs > -2.0) & (zs < 3.0) & (xs**2 + ys**2 < rng**2)
    xs, ys, zs = xs[m], ys[m], zs[m]
    sc = min(w, h) / (2.0 * rng)
    cx, cy = w // 2, h // 2
    px = (cx - ys * sc).astype(int)
    py = (cy - xs * sc).astype(int)
    v = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    zn = np.clip((zs[v] + 2.0) / 5.0, 0.0, 1.0)
    img[py[v], px[v]] = np.stack(
        [
            (zn * 255).astype(np.uint8),
            ((1.0 - zn) * 180).astype(np.uint8),
            np.full(v.sum(), 40, dtype=np.uint8),
        ],
        axis=1,
    )
    cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)
    return img


def main() -> None:
    args = build_parser()
    npc_rng_seed = (
        int(args.npc_rng_seed)
        if args.npc_rng_seed is not None
        else int(time.time_ns() % 2_147_483_647)
    )
    args.npc_rng_seed = npc_rng_seed
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_stamp)
    os.makedirs(out_dir, exist_ok=True)
    gt_path = os.path.join(out_dir, "npc_gt_stream.jsonl")
    vis_path = os.path.join(out_dir, "visibility_stream.jsonl")
    calib_path = os.path.join(out_dir, "robot_sensor_calibration.json")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    if args.load_map:
        print(f"[EP] loading map: {args.map_name}")
        world = client.load_world(args.map_name)
        time.sleep(1.0)
    else:
        world = client.get_world()
    orig_settings = world.get_settings()
    requested_async = bool(args.async_mode)
    use_sync_mode = (not requested_async) or bool(args.async_after_spawn)
    apply_world_mode(world, orig_settings, sync_mode=use_sync_mode, dt=args.dt)
    print(f"[EP] world_mode={'SYNC' if use_sync_mode else 'ASYNC'} dt={args.dt}")

    npc = NpcRuntime(
        world=world,
        grid_npz=args.grid_npz,
        flow_points_json=args.flow_points_json,
        flow_mode=args.npc_flow_mode,
        rng_seed=npc_rng_seed,
        spawn_z=args.robot_spawn_z,
    )
    print(f"[EP] npc_rng_seed={npc_rng_seed}")
    robot = ScoutRobotRuntime(world=world, dt=args.dt)
    robot.rescue_lift_z = float(args.robot_rescue_lift_z)
    policy_kwargs = {
        "desired_distance": float(args.desired_distance),
        "follow_position": str(args.follow_position),
    }
    if args.use_perception:
        perception_kwargs = dict(
            reid_mode=args.reid_mode,
            reid_kpr_config=args.reid_kpr_config,
            reid_device=args.reid_device,
            lost_policy=args.lost_policy,
            kinematics_mode=args.kinematics_mode,
            kf_pos_sigma=args.kf_pos_sigma,
            kf_vel_sigma_q=args.kf_vel_sigma_q,
        )
        perception_route_segments = load_corridor_route_segments(
            args.flow_points_json,
            args.npc_flow_mode,
        )
    else:
        perception_kwargs = None
        perception_route_segments = []

    if args.planner == "dwa_traj":
        if perception_kwargs is not None:
            from adapters.dwa_traj_perception_adapter import DwaTrajPerceptionFollowerPolicy  # noqa: PLC0415
            dwa_perception_kwargs = dict(perception_kwargs)
            dwa_perception_kwargs["route_segments"] = perception_route_segments
            policy = DwaTrajPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **dwa_perception_kwargs)
        else:
            from adapters.dwa_traj_adapter import DwaTrajFollowerPolicy  # noqa: PLC0415
            policy = DwaTrajFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "dwa_traj_depth_tpt":
        # No perception wrapper — this adapter already runs its own perception.
        from adapters.dwa_traj_depth_tpt_adapter import DwaTrajDepthTptFollowerPolicy  # noqa: PLC0415
        policy = DwaTrajDepthTptFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "sfm":
        if perception_kwargs is not None:
            from adapters.sfm_perception_adapter import SfmPerceptionFollowerPolicy  # noqa: PLC0415
            sfm_perception_kwargs = dict(perception_kwargs)
            sfm_perception_kwargs["route_segments"] = perception_route_segments
            policy = SfmPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **sfm_perception_kwargs)
        else:
            from adapters.sfm_adapter import SfmFollowerPolicy  # noqa: PLC0415
            policy = SfmFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "rda":
        if perception_kwargs is not None:
            from adapters.rda_perception_adapter import RdaPerceptionFollowerPolicy  # noqa: PLC0415
            rda_perception_kwargs = dict(perception_kwargs)
            rda_perception_kwargs["route_segments"] = perception_route_segments
            policy = RdaPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **rda_perception_kwargs)
        else:
            from adapters.rda_adapter import RdaFollowerPolicy  # noqa: PLC0415
            policy = RdaFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "rda_lidar":
        if perception_kwargs is not None:
            from adapters.rda_lidar_perception_adapter import RdaLidarPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaLidarPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from adapters.rda_lidar_adapter import RdaLidarFollowerPolicy  # noqa: PLC0415
            policy = RdaLidarFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "rda_traj":
        if perception_kwargs is not None:
            from adapters.rda_traj_perception_adapter import RdaTrajPerceptionFollowerPolicy  # noqa: PLC0415
            rda_traj_perception_kwargs = dict(perception_kwargs)
            rda_traj_perception_kwargs["route_segments"] = perception_route_segments
            policy = RdaTrajPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **rda_traj_perception_kwargs)
        else:
            from adapters.rda_traj_adapter import RdaTrajFollowerPolicy  # noqa: PLC0415
            policy = RdaTrajFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "rda_search":
        # Search-specific CLI flags are forwarded to both the GT and the
        # perception-wrapped variants (they end up on the inner search policy
        # either way — the wrapper pipes extra kwargs through to its inner).
        search_kwargs = dict(
            gt_target_always_visible=not bool(args.rda_search_use_camera_visibility),
            protect_target_for_side_follow=not bool(args.rda_search_disable_target_protection),
        )
        if perception_kwargs is not None:
            from adapters.rda_search_perception_adapter import RdaSearchPerceptionFollowerPolicy  # noqa: PLC0415
            rda_search_perception_kwargs = dict(perception_kwargs)
            rda_search_perception_kwargs["route_segments"] = perception_route_segments
            policy = RdaSearchPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **rda_search_perception_kwargs, **search_kwargs)
        else:
            from adapters.rda_search_adapter import RdaSearchFollowerPolicy  # noqa: PLC0415
            policy = RdaSearchFollowerPolicy(dt=args.dt, **policy_kwargs, **search_kwargs)
    elif args.planner == "rda_depth_tpt":
        # No perception wrapper — this adapter already runs its own perception.
        from adapters.rda_depth_tpt_adapter import RdaDepthTptFollowerPolicy  # noqa: PLC0415
        policy = RdaDepthTptFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "bso_hfc":
        if perception_kwargs is not None:
            from adapters.bso_hfc_perception_adapter import BsoHfcPerceptionFollowerPolicy  # noqa: PLC0415
            bso_perception_kwargs = dict(perception_kwargs)
            bso_perception_kwargs["route_segments"] = perception_route_segments
            policy = BsoHfcPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **bso_perception_kwargs)
        else:
            from adapters.bso_hfc_adapter import BsoHfcFollowerPolicy  # noqa: PLC0415
            policy = BsoHfcFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "trackvla":
        # End-to-end VLA — does not take desired_distance/follow_position and
        # is incompatible with the perception frontend (it has its own VLM).
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner trackvla "
                "(trackvla is itself an end-to-end vision-language-action policy)."
            )
        from adapters.trackvla_adapter import TrackVlaFollowerPolicy  # noqa: PLC0415
        policy = TrackVlaFollowerPolicy(
            dt=args.dt,
            instruction=args.trackvla_instruction,
            ckpt_path=args.trackvla_ckpt,
            render_overlay=bool(args.debug),
        )
    elif args.planner == "oa_vat":
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner oa_vat "
                "(oa_vat carries its own YOLOe + ORTrack + DINOv3 perception)."
            )
        from adapters.oa_vat_adapter import OaVatFollowerPolicy  # noqa: PLC0415
        policy = OaVatFollowerPolicy(
            dt=args.dt,
            text_query=args.oa_vat_text_query,
            kf_mode=args.oa_vat_kf_mode,
            online_enhance=args.oa_vat_online_enhance,
            dinov3_model=args.oa_vat_dinov3,
            match_thresh=args.oa_vat_match_thresh,
            reacq_match_thresh=args.oa_vat_reacq_thresh,
            yolo_conf_thresh=args.oa_vat_yolo_conf,
            target_area_ratio=args.oa_vat_target_area_ratio,
        )
    else:  # pid (default)
        if perception_kwargs is not None:
            from adapters.pid_perception_adapter import PidPerceptionFollowerPolicy  # noqa: PLC0415
            policy = PidPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from adapters.pid_adapter import PIDFollowerPolicy  # noqa: PLC0415
            policy = PIDFollowerPolicy(dt=args.dt, **policy_kwargs)
    vis_checker = InstanceVisibilityChecker(threshold=args.visibility_threshold)
    eval_enabled = bool(args.enable_evaluation)
    eval_started = False
    eval_logger: Optional[EvaluationLogger] = None
    collision_monitor = None
    identity_monitor = TargetIdentityMonitor(
        enabled=bool(eval_enabled and args.use_perception),
        associate_gate_m=float(args.target_identity_associate_gate_m),
        wrong_dist_m=float(args.target_identity_wrong_dist_m),
        startup_grace_sec=float(args.target_identity_startup_grace_sec),
        confirm_sec=float(args.target_identity_confirm_sec),
        fail_sec=float(args.target_identity_fail_sec),
    )
    termination_reason = None
    if eval_enabled:
        print("[EVAL] armed; logging starts on first follow.")

    # 2-D debug visualizer (separate process, only when --debug is set)
    dbg_vis = None
    if args.debug:
        from debug_vis.debug_visualizer import DebugVisualizer
        dbg_vis = DebugVisualizer(half_size=30.0)

    spawned = False
    following = False
    paused = False
    paused_at = 0.0
    active_target_track_id = args.target_track_id
    running = True
    tick = 0
    active_steps = 0
    episode_success: Optional[bool] = None
    desired_num_walkers = int(args.num_walkers)
    desired_min_speed = float(args.npc_min_speed)
    desired_max_speed = float(args.npc_max_speed)

    ui_enabled = (not args.no_ui) and HAS_PYGAME
    if ui_enabled:
        pygame.init()
        panel_w = 400
        panel_h = 260
        hud_h = 170
        win_w = panel_w * 3
        win_h = hud_h + panel_h * 2
        pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Episode Manager | G spawn | F follow | T target | R reset | Q quit")
        ui_font = pygame.font.SysFont("monospace", 18)
        latest_panels = {
            "rgb_left": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "rgb": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "rgb_right": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "depth": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "instance": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "lidar": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
        }
    elif not args.no_ui:
        print("[WARN] pygame not available, key UI disabled.")

    def on_signal(_sig, _frm):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)

    gt_f = open(gt_path, "w", encoding="utf-8")
    vis_f = open(vis_path, "w", encoding="utf-8")

    print("[EP] ready. Press G to spawn actors, then F to start following.")
    print(f"[EP] output_dir={out_dir}")
    print(f"[EP] npc_flow_mode={args.npc_flow_mode} nav_mode={args.npc_navigation_mode}")

    def do_spawn() -> None:
        nonlocal spawned, use_sync_mode
        if spawned:
            return
        # Cleanup stale robots from previous runs to avoid visual confusion.
        for a in world.get_actors().filter("vehicle.*"):
            try:
                is_scout = "vehicle.scout" in a.type_id
                is_known_role = a.attributes.get("role_name", "") in {"followbench_robot", "sfm_follow_robot", "scout_follow"}
                if is_scout or is_known_role:
                    a.destroy()
            except Exception:
                pass
        npc.params["min_speed"] = float(desired_min_speed)
        npc.params["max_speed"] = float(desired_max_speed)
        npc.params["navigation_mode"] = str(args.npc_navigation_mode)
        npc.params["pair_spawn_mode"] = str(args.npc_pair_spawn_mode)
        npc.params["target_spawn_policy"] = str(args.target_spawn_policy)
        npc.params["crowd_spawn_policy"] = str(args.crowd_spawn_policy)
        npc.params["npc_spawn_min_sep_m"] = float(args.npc_spawn_min_sep_m)
        npc.params["target_centerline_half_width_m"] = float(args.target_centerline_half_width_m)
        npc.params["target_wall_clearance_min_m"] = float(args.target_wall_clearance_min_m)
        npc.params["target_endpoint_sample_radius_m"] = float(args.target_endpoint_sample_radius_m)
        npc.params["right_hand_lane_sep_m"] = float(args.right_hand_lane_sep_m)
        npc.params["target_lane_bias_scale"] = float(args.target_lane_bias_scale)
        npc.params["target_lane_min_offset_m"] = float(args.target_lane_min_offset_m)
        npc.params["target_lane_max_offset_m"] = float(args.target_lane_max_offset_m)
        npc.params["target_lateral_jitter_m"] = float(args.target_lateral_jitter_m)
        npc.params["target_turnaround_forward_m"] = float(args.target_turnaround_forward_m)
        npc.params["target_turnaround_merge_m"] = float(args.target_turnaround_merge_m)
        npc.params["target_turnaround_side_m"] = float(args.target_turnaround_side_m)
        npc.params["target_turnaround_samples"] = int(args.target_turnaround_samples)
        npc.params["target_turnaround_speed_mps"] = float(args.target_turnaround_speed_mps)
        npc.params["crowd_turn_sublane_mode"] = str(args.crowd_turn_sublane_mode)
        npc.params["crowd_turn_radius_jitter_m"] = float(args.crowd_turn_radius_jitter_m)
        npc.params["crowd_turn_min_offset_keep_ratio"] = float(args.crowd_turn_min_offset_keep_ratio)
        npc.params["crowd_turn_samples"] = int(args.crowd_turn_samples)
        npc.params["follow_position"] = str(args.follow_position)
        npc.params["desired_distance"] = float(args.desired_distance)
        npc.params["target_lane_bias_mode"] = str(args.target_lane_bias_mode)
        npc.params["robot_spawn_clearance_m"] = float(args.robot_spawn_clearance_m)
        npc.params["pair_goal_policy"] = str(args.npc_pair_goal_policy)
        npc.params["pair_target_band_half_width_m"] = float(args.npc_pair_target_band_half_width_m)
        npc.params["target_required_legs"] = int(args.target_required_legs)
        npc.params["replan_on_block"] = bool(args.npc_replan_on_block)
        npc.params["block_replan_window_sec"] = float(args.npc_block_replan_window_sec)
        npc.params["block_replan_progress_thresh"] = float(args.npc_block_replan_progress_thresh)
        npc.params["block_replan_risk_thresh"] = float(args.npc_block_replan_risk_thresh)
        npc.params["astar_replan_cooldown_sec"] = float(args.npc_astar_replan_cooldown_sec)
        npc.params["sfm_dt"] = float(args.dt)
        npc.spawn(desired_num_walkers, target_track_id=active_target_track_id)
        if len(npc.agents) == 0:
            raise RuntimeError("No NPC spawned, cannot place robot near target.")
        # Materialize NPC spawn transforms so their positions are valid when
        # resolve_robot_spawn_tf reads them (CARLA needs a tick to apply set_transform).
        step_world(world, use_sync_mode)
        tgt_init = npc.get_target_by_track_id(active_target_track_id)
        tgt_actor = npc.get_actor_by_track_id(active_target_track_id)
        if tgt_actor is None or tgt_init is None:
            raise RuntimeError(
                f"Target {active_target_track_id} missing after NPC spawn; "
                "target fallback is disabled so evaluation keeps a fixed identity."
            )
        active_id = active_target_track_id

        robot_tf = resolve_robot_spawn_tf(args, npc=npc, target_track_id=active_target_track_id)
        print(
            f"[EP] robot_spawn mode={args.robot_spawn_mode} "
            f"yaw_policy={args.side_follow_robot_yaw_policy} "
            f"yaw={robot_tf.rotation.yaw:.1f} "
            f"at=({robot_tf.location.x:.2f},{robot_tf.location.y:.2f},{robot_tf.location.z:.2f})"
        )
        robot_tf = spawn_robot_with_retries(robot, robot_tf, args, npc, active_target_track_id)
        # Materialize actor transforms before reading coordinates.
        step_world(world, use_sync_mode)
        # Hard correction: ensure spawn is within configured max distance to target.
        if tgt_init is not None and robot.actor is not None and tgt_actor is not None:
            rloc = robot.actor.get_location()
            tloc = tgt_actor.get_location()
            # Some CARLA builds may still return a transient zero pose right after spawn.
            if abs(tloc.x) < 1e-4 and abs(tloc.y) < 1e-4:
                step_world(world, use_sync_mode)
                tloc = tgt_actor.get_location()
            if abs(rloc.x) < 1e-4 and abs(rloc.y) < 1e-4:
                step_world(world, use_sync_mode)
                rloc = robot.actor.get_location()
            dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
            print(
                f"[EP][DEBUG] target={active_id} actor_id={tgt_actor.id} "
                f"planned_robot_xy=({robot_tf.location.x:.2f},{robot_tf.location.y:.2f}) "
                f"target_xy=({tloc.x:.2f},{tloc.y:.2f}) "
                f"robot_xy=({rloc.x:.2f},{rloc.y:.2f}) dist={dist0:.2f}m"
            )
            if dist0 > (float(args.desired_distance) + 1.5):
                target_yaw_rad = float(np.deg2rad(float(tgt_init.yaw_deg)))
                new_x, new_y, _ok = follow_position_spawn_xy(
                    float(tloc.x), float(tloc.y), target_yaw_rad,
                    str(args.follow_position), float(args.desired_distance),
                    npc.roi_map,
                )
                yaw = resolve_robot_spawn_yaw_deg(
                    follow_position=str(args.follow_position),
                    yaw_policy=str(args.side_follow_robot_yaw_policy),
                    target_x=float(tloc.x),
                    target_y=float(tloc.y),
                    robot_x=new_x,
                    robot_y=new_y,
                    target_yaw_rad=target_yaw_rad,
                )
                fix_tf = carla.Transform(
                    carla.Location(x=new_x, y=new_y, z=robot_tf.location.z),
                    carla.Rotation(yaw=yaw),
                )
                robot.force_place(fix_tf)
                step_world(world, use_sync_mode)
                rloc = robot.actor.get_location()
                dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
                print(
                    f"[EP][DEBUG] corrected robot_xy=({rloc.x:.2f},{rloc.y:.2f}) "
                    f"target_xy=({tloc.x:.2f},{tloc.y:.2f}) dist={dist0:.2f}m"
                )
            # If still at origin, force place again with planned transform.
            if abs(rloc.x) < 1e-4 and abs(rloc.y) < 1e-4:
                robot.force_place(robot_tf)
                step_world(world, use_sync_mode)
                rloc = robot.actor.get_location()
                dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
                print(
                    f"[EP][DEBUG] force_place retry robot_xy=({rloc.x:.2f},{rloc.y:.2f}) "
                    f"target_xy=({tloc.x:.2f},{tloc.y:.2f}) dist={dist0:.2f}m"
                )
            print(f"[EP] robot_target_init_dist={dist0:.2f}m (target={active_id})")
            if args.draw_debug:
                world.debug.draw_string(
                    carla.Location(x=tloc.x, y=tloc.y, z=tloc.z + 2.8),
                    "TARGET_INIT",
                    draw_shadow=False,
                    color=carla.Color(255, 255, 0),
                    life_time=5.0,
                )
                world.debug.draw_string(
                    carla.Location(x=rloc.x, y=rloc.y, z=rloc.z + 2.8),
                    "ROBOT_SPAWN",
                    draw_shadow=False,
                    color=carla.Color(0, 255, 0),
                    life_time=5.0,
                )
        robot.spawn_sensors(
            image_w=args.sensor_image_w,
            image_h=args.sensor_image_h,
            cam_fov=args.sensor_fov_deg,
            lidar_range=args.sensor_lidar_range,
        )
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(robot.get_calibration(), f, indent=2)
        npc.set_motion_active(False)
        spawned = True
        print(
            f"[EP] spawned all actors (idle) | walkers={desired_num_walkers} "
            f"speed=[{desired_min_speed:.2f},{desired_max_speed:.2f}]"
        )
        if dbg_vis is not None:
            rs = robot.get_state()
            if rs is not None:
                dbg_vis.start(rs.x, rs.y, args.grid_npz)
                print(f"[EP][Debug] visualizer started at ({rs.x:.1f}, {rs.y:.1f})")
        try:
            world.get_spectator().set_transform(carla.Transform(
                carla.Location(x=98.13, y=33.35, z=10.69),
                carla.Rotation(pitch=-25.74, yaw=142.86, roll=0.0),
            ))
        except Exception:
            pass
        if requested_async and args.async_after_spawn and use_sync_mode:
            apply_world_mode(world, orig_settings, sync_mode=False, dt=args.dt)
            use_sync_mode = False
            print("[EP] world_mode switched to ASYNC after initial spawn.")

    manual_mode = "none"   # "none" | "target" | "robot"
    _cam_yaw    = 0.0      # 3rd-person camera yaw   (degrees, mouse-driven)
    _cam_pitch  = -20.0    # 3rd-person camera pitch  (degrees)

    def start_evaluation_if_needed() -> None:
        nonlocal eval_started, eval_logger, collision_monitor, active_steps, episode_success, termination_reason
        if not eval_enabled or eval_started or not spawned or not following:
            return
        if robot.actor is None:
            print("[EVAL][WARN] cannot start evaluation before robot actor is ready.")
            return
        target_actor = npc.get_actor_by_track_id(active_target_track_id)
        if target_actor is None:
            print(f"[EVAL][WARN] cannot start evaluation: target {active_target_track_id} missing.")
            return
        eval_logger = EvaluationLogger.from_args(args, scenario_type="corridor")
        print(f"[EVAL] started run_dir={eval_logger.run_dir}")
        eval_logger.write_meta(args, robot_actor=robot.actor, target_actor=target_actor)
        collision_monitor = HumanCollisionMonitor(
            world,
            robot.actor,
            enabled=not bool(args.disable_collision_sensor),
        )
        eval_started = True
        active_steps = 0
        identity_monitor.reset()
        episode_success = None
        termination_reason = None

    def do_reset() -> None:
        nonlocal spawned, following, tick, active_steps, episode_success, termination_reason, manual_mode, collision_monitor, eval_started, eval_logger
        if spawned:
            npc.destroy()
            robot.destroy()
        if collision_monitor is not None:
            collision_monitor.destroy()
            collision_monitor = None
        if eval_logger is not None:
            eval_logger.finalize("manual_reset", episode_success=False)
            eval_logger = None
        eval_started = False
        identity_monitor.reset()
        if dbg_vis is not None:
            dbg_vis.stop()
        spawned = False
        following = False
        tick = 0
        active_steps = 0
        episode_success = None
        termination_reason = None
        manual_mode = "none"
        if ui_enabled:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        vis_checker.reset()
        print("[EP] reset done. Press G to spawn with current params.")

    def do_toggle_manual_target() -> None:
        nonlocal manual_mode, _cam_yaw, _cam_pitch
        if not spawned:
            return
        if manual_mode == "target":
            manual_mode = "none"
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            print("[EP] manual target OFF — algorithm resumes")
        else:
            manual_mode = "target"
            ta = npc.get_actor_by_track_id(active_target_track_id)
            if ta is not None:
                _cam_yaw = float(ta.get_transform().rotation.yaw)
            _cam_pitch = -20.0
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            pygame.mouse.get_rel()   # flush accumulated delta
            print(f"[EP] manual target ON ({active_target_track_id}) | WASD move  Mouse look  Shift sprint  H release")

    def do_toggle_manual_robot() -> None:
        nonlocal manual_mode, _cam_yaw, _cam_pitch
        if not spawned:
            return
        if manual_mode == "robot":
            manual_mode = "none"
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            print("[EP] manual robot OFF — algorithm resumes")
        else:
            manual_mode = "robot"
            if robot.actor is not None:
                _cam_yaw = float(robot.actor.get_transform().rotation.yaw)
            _cam_pitch = -20.0
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            pygame.mouse.get_rel()
            print("[EP] manual robot ON | WASD move  Mouse look  Shift sprint  J release")

    def do_toggle_follow() -> None:
        nonlocal following
        if not spawned:
            return
        following = not following
        npc.set_motion_active(following)
        policy.reset()
        vis_checker.reset()
        if following:
            start_evaluation_if_needed()
        print(f"[EP] following={'ON' if following else 'OFF'} target={active_target_track_id}")

    start_t = time.time()
    auto_spawn_done = False
    auto_follow_done = False
    # Cached after spawn_sensors() so we don't keep dict-walking the calibration every tick.
    rgb_intrinsics_cached: Optional[dict] = None
    rgb_extr_cached: Optional[dict] = None
    lidar_extr_cached: Optional[dict] = None
    rgb_intrinsics_left_cached: Optional[dict] = None
    rgb_extr_left_cached: Optional[dict] = None
    rgb_intrinsics_right_cached: Optional[dict] = None
    rgb_extr_right_cached: Optional[dict] = None
    # Last-good decoded sensor frames. Used as a fallback for the very first
    # ticks after spawn_sensors() before the asynchronous sensor listeners have
    # delivered their first payload, so we never feed None to a planner that
    # expects rgb_image / lidar_points to be present.
    last_rgb_np: Optional[np.ndarray] = None
    last_lidar_np: Optional[np.ndarray] = None
    last_depth_np: Optional[np.ndarray] = None
    last_rgb_left_np: Optional[np.ndarray] = None
    last_rgb_right_np: Optional[np.ndarray] = None
    last_depth_left_np: Optional[np.ndarray] = None
    last_depth_right_np: Optional[np.ndarray] = None
    try:
        while running:
            now_t = time.time()
            if now_t - start_t >= args.duration_sec:
                if eval_started:
                    termination_reason = "duration_timeout"
                    episode_success = False
                break

            elapsed = now_t - start_t
            if args.auto_spawn_sec >= 0.0 and (not auto_spawn_done) and elapsed >= args.auto_spawn_sec:
                do_spawn()
                auto_spawn_done = True
            if args.auto_follow_sec >= 0.0 and (not auto_follow_done) and elapsed >= args.auto_follow_sec:
                if not spawned:
                    do_spawn()
                    auto_spawn_done = True
                if not following:
                    do_toggle_follow()
                auto_follow_done = True

            if ui_enabled:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        running = False
                    elif e.type == pygame.MOUSEMOTION and manual_mode != "none":
                        _cam_yaw   = (_cam_yaw + e.rel[0] * MANUAL_MOUSE_SENS) % 360.0
                        _cam_pitch = float(np.clip(
                            _cam_pitch - e.rel[1] * MANUAL_MOUSE_SENS * 0.4,
                            -60.0, 10.0))
                    elif e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_q:
                            running = False
                        elif e.key == pygame.K_ESCAPE and manual_mode != "none":
                            # ESC releases manual control without quitting
                            manual_mode = "none"
                            pygame.mouse.set_visible(True)
                            pygame.event.set_grab(False)
                            print("[EP] manual control released")
                        elif e.key == pygame.K_h and spawned:
                            do_toggle_manual_target()
                        elif e.key == pygame.K_j and spawned:
                            do_toggle_manual_robot()
                        elif e.key == pygame.K_g and not spawned:
                            do_spawn()
                        elif e.key == pygame.K_r:
                            do_reset()
                        elif e.key == pygame.K_m and spawned:
                            # toggle npc motion
                            npc.set_motion_active(not following)
                        elif e.key == pygame.K_f and spawned:
                            do_toggle_follow()
                        elif e.key == pygame.K_t and spawned:
                            states = npc.get_states()
                            if states:
                                ids = [s.track_id for s in states]
                                if active_target_track_id not in ids:
                                    active_target_track_id = ids[0]
                                else:
                                    idx = ids.index(active_target_track_id)
                                    active_target_track_id = ids[(idx + 1) % len(ids)]
                                print(f"[EP] switched target -> {active_target_track_id}")
                        # Parameter tuning (pre-spawn preferred; reset to reapply)
                        elif e.key == pygame.K_UP:
                            desired_num_walkers = min(120, desired_num_walkers + 2)
                            print(f"[EP] desired walkers -> {desired_num_walkers}")
                        elif e.key == pygame.K_DOWN:
                            desired_num_walkers = max(2, desired_num_walkers - 2)
                            print(f"[EP] desired walkers -> {desired_num_walkers}")
                        elif e.key == pygame.K_LEFTBRACKET:  # [
                            desired_min_speed = max(0.2, desired_min_speed - 0.1)
                            desired_max_speed = max(desired_max_speed, desired_min_speed + 0.1)
                            print(f"[EP] desired min_speed -> {desired_min_speed:.2f}")
                        elif e.key == pygame.K_RIGHTBRACKET:  # ]
                            desired_min_speed = min(desired_max_speed - 0.1, desired_min_speed + 0.1)
                            print(f"[EP] desired min_speed -> {desired_min_speed:.2f}")
                        elif e.key == pygame.K_MINUS:
                            desired_max_speed = max(desired_min_speed + 0.1, desired_max_speed - 0.1)
                            print(f"[EP] desired max_speed -> {desired_max_speed:.2f}")
                        elif e.key == pygame.K_EQUALS:
                            desired_max_speed = min(3.0, desired_max_speed + 0.1)
                            print(f"[EP] desired max_speed -> {desired_max_speed:.2f}")
                        elif e.key == pygame.K_SPACE:
                            paused = not paused
                            if paused:
                                paused_at = time.time()
                                print("[EP] PAUSED — press SPACE to resume")
                                if spawned:
                                    npc.set_motion_active(False)
                                    robot.hold_still()
                            else:
                                start_t += time.time() - paused_at
                                print("[EP] RESUMED")
                                if spawned and following:
                                    npc.set_motion_active(True)

                # Draw idle HUD before actors are spawned.
                if not spawned:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        surf.fill((18, 22, 28))
                        line1 = (
                            f"State: spawned={spawned} follow={following} target={active_target_track_id} "
                            f"| desired walkers={desired_num_walkers} speed=[{desired_min_speed:.2f},{desired_max_speed:.2f}]"
                        )
                        line2 = "Keys: G spawn  F follow  T switch-target  H manual-NPC  J manual-robot  R reset  SPACE pause  Q quit"
                        line3 = "Tune: Up/Down walkers | [ ] min_speed | - / = max_speed (reset then G to reapply)"
                        line4 = "Sensor grid appears after spawn."
                        for idx, text in enumerate([line1, line2, line3, line4]):
                            img = ui_font.render(text, True, (230, 235, 240))
                            surf.blit(img, (16, 20 + idx * 36))
                        pygame.display.flip()

            if paused:
                # Keep applying zero-speed controls every tick so walkers truly stop.
                # Without this, CARLA physics continues applying the last non-zero
                # WalkerControl, causing walkers to drift and invalidate their routes.
                if spawned:
                    npc.step(
                        debug_draw_ids=True,
                        label_life=max(args.dt * 2.0, 0.12),
                        robot_actor=robot.actor if robot.actor is not None else None,
                        target_track_id=active_target_track_id,
                    )
                    robot.hold_still()
                if ui_enabled:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                        overlay.fill((0, 0, 0, 140))
                        surf.blit(overlay, (0, 0))
                        pause_text = ui_font.render("PAUSED — press SPACE to resume", True, (255, 220, 0))
                        px = (surf.get_width() - pause_text.get_width()) // 2
                        py = (surf.get_height() - pause_text.get_height()) // 2
                        surf.blit(pause_text, (px, py))
                        pygame.display.flip()
                step_world(world, use_sync_mode)
                continue

            if spawned:
                npc.step(
                    debug_draw_ids=True,
                    label_life=max(args.dt * 2.0, 0.12),
                    robot_actor=robot.actor if robot.actor is not None else None,
                    target_track_id=active_target_track_id,
                )
                npc_states = npc.get_states()
                target_state = next((s for s in npc_states if s.track_id == active_target_track_id), None)
                target_actor = npc.get_actor_by_track_id(active_target_track_id)
                robot_state = robot.get_state()
                v_cmd = None
                w_cmd = None
                planner_cost_ms = None
                policy_debug_info = {}

                # ── Manual control (overrides algorithm output this tick) ──────
                if manual_mode == "target" and ui_enabled:
                    key_state = pygame.key.get_pressed()
                    _apply_manual_target_control(target_actor, key_state, cam_yaw=_cam_yaw)
                    if target_actor is not None:
                        _update_manual_spectator(
                            world.get_spectator(), target_actor, _cam_yaw, _cam_pitch)
                elif manual_mode == "robot" and ui_enabled:
                    key_state = pygame.key.get_pressed()
                    _apply_manual_robot_control(robot, key_state)
                    if robot.actor is not None:
                        _update_manual_spectator(
                            world.get_spectator(), robot.actor, _cam_yaw, _cam_pitch)

                visible = False
                pix_count = 0
                inst_img = robot.get_sensor_data("instance")
                inst_img_left = robot.get_sensor_data("instance_left")
                inst_img_right = robot.get_sensor_data("instance_right")
                rgb_img = robot.get_sensor_data("rgb")
                depth_img = robot.get_sensor_data("depth")
                lidar_img = robot.get_sensor_data("lidar")
                rgb_img_left = robot.get_sensor_data("rgb_left")
                rgb_img_right = robot.get_sensor_data("rgb_right")
                depth_img_left = robot.get_sensor_data("depth_left")
                depth_img_right = robot.get_sensor_data("depth_right")
                if target_actor is not None and "instance" in robot.sensors:
                    visible, pix_count = vis_checker.evaluate_multi(
                        views=[
                            (inst_img, robot.sensors.get("instance")),
                            (inst_img_left, robot.sensors.get("instance_left")),
                            (inst_img_right, robot.sensors.get("instance_right")),
                        ],
                        target_actor=target_actor,
                        image_w=args.sensor_image_w,
                        image_h=args.sensor_image_h,
                        fov_deg=args.sensor_fov_deg,
                    )
                if ui_enabled:
                    r = _to_rgb(rgb_img)
                    d = _depth_to_rgb(depth_img)
                    ins = _instance_to_rgb(inst_img)
                    lbev = _lidar_to_bev(lidar_img, w=400, h=260, rng=25.0)
                    # Fetch multi-view perception bboxes (may be empty dict).
                    _dbg_fn = getattr(policy, "get_debug_info", None)
                    _dbg_info = _dbg_fn() if callable(_dbg_fn) else {}
                    bboxes_by_view = _dbg_info.get("track_bboxes_by_view") or {}
                    _stale = int(_dbg_info.get("track_bboxes_age", 0))
                    # Planners that produce their own front-view overlay (e.g.
                    # trackvla draws the predicted trajectory + instruction on
                    # the 384x384 input frame) take priority over the raw RGB.
                    _rendered_front = _dbg_info.get("rendered_front")
                    if _rendered_front is not None and isinstance(_rendered_front, np.ndarray):
                        latest_panels["rgb"] = cv2.resize(_rendered_front, (400, 260))
                    elif r is not None:
                        # Draw bboxes on the original-resolution image so cv2.resize
                        # down-samples both the pixels and the rectangles consistently.
                        r = _draw_bboxes_on(r, bboxes_by_view.get("front"), _stale)
                        latest_panels["rgb"] = cv2.resize(r, (400, 260))
                    if d is not None:
                        latest_panels["depth"] = cv2.resize(d, (400, 260))
                    if ins is not None:
                        latest_panels["instance"] = cv2.resize(ins, (400, 260))
                    if lbev is not None:
                        latest_panels["lidar"] = lbev
                    rl = _to_rgb(rgb_img_left)
                    rl = _draw_bboxes_on(rl, bboxes_by_view.get("left"), _stale)
                    if rl is not None:
                        latest_panels["rgb_left"] = cv2.resize(rl, (400, 260))
                    rr = _to_rgb(rgb_img_right)
                    rr = _draw_bboxes_on(rr, bboxes_by_view.get("right"), _stale)
                    if rr is not None:
                        latest_panels["rgb_right"] = cv2.resize(rr, (400, 260))

                if target_state is not None and following:
                    target_velocity_yaw_rad = yaw_rad_from_velocity(
                        target_state.vx,
                        target_state.vy,
                        fallback_deg=target_state.yaw_deg,
                    )
                    target_actor_yaw_rad = float(np.deg2rad(target_state.yaw_deg))
                    if rgb_intrinsics_cached is None:
                        calib = robot.get_calibration()
                        rgb_calib = calib.get("rgb", {})
                        lidar_calib = calib.get("lidar", {})
                        rgb_left_calib = calib.get("rgb_left", {})
                        rgb_right_calib = calib.get("rgb_right", {})
                        rgb_intrinsics_cached = rgb_calib.get("intrinsics")
                        rgb_extr_cached = rgb_calib.get("extrinsics_robot_to_sensor")
                        lidar_extr_cached = lidar_calib.get("extrinsics_robot_to_sensor")
                        rgb_intrinsics_left_cached = rgb_left_calib.get("intrinsics")
                        rgb_extr_left_cached = rgb_left_calib.get("extrinsics_robot_to_sensor")
                        rgb_intrinsics_right_cached = rgb_right_calib.get("intrinsics")
                        rgb_extr_right_cached = rgb_right_calib.get("extrinsics_robot_to_sensor")
                    rgb_np = _to_rgb(rgb_img)
                    if rgb_np is not None:
                        last_rgb_np = rgb_np
                    lidar_np = _lidar_to_points(lidar_img)
                    if lidar_np is not None:
                        last_lidar_np = lidar_np
                    depth_np = _depth_to_meters(depth_img)
                    if depth_np is not None:
                        last_depth_np = depth_np
                    rgb_left_np = _to_rgb(rgb_img_left)
                    if rgb_left_np is not None:
                        last_rgb_left_np = rgb_left_np
                    rgb_right_np = _to_rgb(rgb_img_right)
                    if rgb_right_np is not None:
                        last_rgb_right_np = rgb_right_np
                    depth_left_np = _depth_to_meters(depth_img_left)
                    if depth_left_np is not None:
                        last_depth_left_np = depth_left_np
                    depth_right_np = _depth_to_meters(depth_img_right)
                    if depth_right_np is not None:
                        last_depth_right_np = depth_right_np
                    obs = FollowObservation(
                        tick=tick,
                        dt=args.dt,
                        robot=robot_state,
                        target=target_state,
                        npcs=npc_states,
                        target_visible=visible,
                        target_pixel_count=pix_count,
                        extras={
                            "target_yaw_rad": target_actor_yaw_rad,
                            "target_actor_yaw_rad": target_actor_yaw_rad,
                            "target_velocity_yaw_rad": target_velocity_yaw_rad,
                            "target_yaw_source": "actor",
                        },
                        rgb_image=last_rgb_np,
                        depth_image=last_depth_np,
                        lidar_points=last_lidar_np,
                        rgb_intrinsics=rgb_intrinsics_cached,
                        rgb_extrinsics_robot_to_sensor=rgb_extr_cached,
                        lidar_extrinsics_robot_to_sensor=lidar_extr_cached,
                        rgb_image_left=last_rgb_left_np,
                        rgb_image_right=last_rgb_right_np,
                        depth_image_left=last_depth_left_np,
                        depth_image_right=last_depth_right_np,
                        rgb_intrinsics_left=rgb_intrinsics_left_cached,
                        rgb_intrinsics_right=rgb_intrinsics_right_cached,
                        rgb_extrinsics_left_robot_to_sensor=rgb_extr_left_cached,
                        rgb_extrinsics_right_robot_to_sensor=rgb_extr_right_cached,
                    )
                    _t_policy = time.perf_counter()
                    action = policy.act(obs)
                    planner_cost_ms = (time.perf_counter() - _t_policy) * 1000.0
                    _dbg_fn_eval = getattr(policy, "get_debug_info", None)
                    policy_debug_info = _dbg_fn_eval() if callable(_dbg_fn_eval) else {}
                    # Robot-side safety layer: slow down / brake near pedestrians.
                    nearest_npc_dist = float("inf")
                    for s in npc_states:
                        d = float(np.hypot(robot_state.x - s.x, robot_state.y - s.y))
                        if d < nearest_npc_dist:
                            nearest_npc_dist = d

                    v_cmd = float(action.v_mps)
                    w_cmd = float(action.w_radps)
                    # if nearest_npc_dist < 2.8:
                    #     # Linear speed attenuation as pedestrians get close.
                    #     scale = float(np.clip((nearest_npc_dist - 1.0) / (2.8 - 1.0), 0.0, 1.0))
                    #     v_cmd *= scale
                    # if nearest_npc_dist < 1.0:
                    #     # Emergency stop zone.
                    #     v_cmd = 0.0
                    robot.apply_velocity_command(v_cmd, w_cmd)
                    # if args.draw_laser:
                    #     world.debug.draw_line(
                    #         carla.Location(
                    #             x=robot_state.x,
                    #             y=robot_state.y,
                    #             z=robot_state.z + 1.2,
                    #         ),
                    #         carla.Location(
                    #             x=target_state.x,
                    #             y=target_state.y,
                    #             z=target_state.z + 1.4,
                    #         ),
                    #         thickness=0.02,
                    #         color=carla.Color(50, 10, 10),
                    #         life_time=max(args.dt * 2.0, 0.12),
                    #     )
                else:
                    robot.hold_still()

                target_task_done = bool(following and npc.is_target_route_task_done(active_target_track_id))
                current_active_steps = active_steps + (1 if following else 0)
                max_active_sec = float(args.max_active_sec)
                active_timeout = bool(
                    following
                    and max_active_sec > 0.0
                    and current_active_steps * float(args.dt) >= max_active_sec
                )
                eval_clearance = None
                eval_collision = False
                eval_collision_source = None
                target_identity_status = None
                eval_termination_triggered = target_task_done or active_timeout
                if target_task_done:
                    eval_step_termination_reason = "target_round_trip_done"
                elif active_timeout:
                    eval_step_termination_reason = "max_active_time"
                else:
                    eval_step_termination_reason = None
                if eval_started and following and eval_logger is not None:
                    eval_time_s = active_steps * float(args.dt)
                    eval_clearance = eval_logger.compute_clearance(
                        robot_state,
                        target_state,
                        npc_states,
                        active_target_track_id,
                    )
                    actor_to_track = actor_track_lookup(npc_states)
                    sensor_events = (
                        enrich_sensor_collision_events(
                            collision_monitor.drain(),
                            tick=tick,
                            time_s=eval_time_s,
                            episode_active=True,
                            actor_to_track=actor_to_track,
                            target_track_id=active_target_track_id,
                        )
                        if collision_monitor is not None
                        else []
                    )
                    geometry_event = (
                        make_geometry_collision_event(
                            tick,
                            eval_time_s,
                            eval_clearance,
                            active_target_track_id,
                            margin=float(args.human_collision_margin),
                        )
                    )
                    eval_collision_events = list(sensor_events)
                    if geometry_event is not None:
                        eval_collision_events.append(geometry_event)
                    if eval_collision_events:
                        eval_logger.write_collision_events(eval_collision_events)
                        eval_collision = True
                        eval_collision_source = eval_collision_events[0].get("source")
                        if bool(args.terminate_on_human_collision):
                            termination_reason = "human_collision"
                            episode_success = False
                            eval_termination_triggered = True
                            eval_step_termination_reason = "human_collision"
                    target_identity_status = identity_monitor.update(
                        dt=float(args.dt),
                        eval_time_s=eval_time_s,
                        target_track_id=active_target_track_id,
                        target_state=target_state,
                        npc_states=npc_states,
                        policy_debug_info=policy_debug_info,
                    )
                    identity_loss_triggered = bool(target_identity_status.get("termination_triggered"))
                    if identity_loss_triggered and not bool(args.terminate_on_target_identity_loss):
                        target_identity_status["termination_triggered"] = False
                        target_identity_status["termination_reason"] = None
                    if identity_loss_triggered and bool(args.terminate_on_target_identity_loss):
                        termination_reason = str(target_identity_status.get("termination_reason"))
                        episode_success = False
                        eval_termination_triggered = True
                        eval_step_termination_reason = termination_reason
                    eval_logger.write_step(
                        tick=tick,
                        time_s=eval_time_s,
                        episode_active=True,
                        paused=paused,
                        target_track_id=active_target_track_id,
                        robot_state=robot_state,
                        target_state=target_state,
                        npc_states=npc_states,
                        visible=visible,
                        pixel_count=pix_count,
                        visibility_threshold=args.visibility_threshold,
                        command=None if v_cmd is None else {"v_mps": v_cmd, "w_radps": w_cmd},
                        planner_cost_ms=planner_cost_ms,
                        debug_info=policy_debug_info,
                        collision=eval_collision,
                        collision_source=eval_collision_source,
                        termination_triggered=eval_termination_triggered,
                        termination_reason=eval_step_termination_reason,
                        clearance=eval_clearance,
                        target_identity=target_identity_status,
                    )
                    if eval_termination_triggered:
                        if eval_step_termination_reason == "target_identity_lost":
                            print("[EVAL] terminating episode: target_identity_lost")
                        elif target_task_done:
                            termination_reason = "target_round_trip_done"
                            episode_success = True
                            print("[EVAL] terminating episode: target_round_trip_done")
                        elif active_timeout:
                            termination_reason = "max_active_time"
                            episode_success = False
                            print("[EVAL] terminating episode: max_active_time")
                        else:
                            print("[EVAL] terminating episode: human_collision")
                        npc.set_motion_active(False)
                        robot.hold_still()
                        running = False

                if eval_logger is None and (target_task_done or active_timeout):
                    if target_task_done:
                        termination_reason = "target_round_trip_done"
                        episode_success = True
                    else:
                        termination_reason = "max_active_time"
                        episode_success = False
                    npc.set_motion_active(False)
                    robot.hold_still()
                    running = False

                if following:
                    active_steps = current_active_steps

                gt_f.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "time_s": now_t - start_t,
                            "target_track_id": active_target_track_id,
                            "robot": {"x": robot_state.x, "y": robot_state.y, "z": robot_state.z, "yaw_rad": robot_state.yaw_rad, "speed": robot_state.speed},
                            "npcs": [s.__dict__ for s in npc_states],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                vis_f.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "time_s": now_t - start_t,
                            "target_track_id": active_target_track_id,
                            "target_visible": bool(visible),
                            "target_pixel_count": int(pix_count),
                            "threshold": int(args.visibility_threshold),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # if args.draw_debug and target_state is not None:
                #     world.debug.draw_string(
                #         carla.Location(
                #             x=target_state.x,
                #             y=target_state.y,
                #             z=target_state.z + 2.6,
                #         ),
                #         f"TGT {active_target_track_id} vis={int(visible)} pix={pix_count}",
                #         draw_shadow=False,
                #         color=carla.Color(255, 255, 0),
                #         life_time=max(args.dt * 2.0, 0.12),
                #     )

                # Debug visualizer update (non-blocking, drops frames if busy)
                if dbg_vis is not None and robot_state is not None:
                    _dbg = getattr(policy, "get_debug_info", None)
                    _dbg_info = _dbg() if callable(_dbg) else {}
                    dbg_vis.update(
                        tick=tick,
                        robot_state=robot_state,
                        npc_states=npc_states,
                        active_target_id=active_target_track_id,
                        obstacles=_dbg_info.get("obstacles"),
                        traj_points=_dbg_info.get("traj_points"),
                        goal_point=_dbg_info.get("goal_point"),
                        search_samples=_dbg_info.get("search_samples"),
                        search_goal=_dbg_info.get("search_goal"),
                        predicted_target_traj=_dbg_info.get("predicted_target_traj"),
                        map_occupied_cells=_dbg_info.get("map_occupied_cells"),
                        map_observed_free_cells=_dbg_info.get("map_observed_free_cells"),
                        map_outline=_dbg_info.get("map_outline"),
                        map_occupancy_rgba=_dbg_info.get("map_occupancy_rgba"),
                        map_esdf_rgba=_dbg_info.get("map_esdf_rgba"),
                        map_hybrid_rgba=_dbg_info.get("map_hybrid_rgba"),
                        map_debug_extent=_dbg_info.get("map_debug_extent"),
                        map_debug_mode=_dbg_info.get("map_debug_mode"),
                        lidar_range_max=_dbg_info.get("lidar_range_max"),
                    )

                draw_status(
                    world,
                    f"spawned={spawned} follow={following} target={active_target_track_id} pix={pix_count}",
                )
                if ui_enabled:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        surf.fill((18, 22, 28))
                        line1 = (
                            f"State: spawned={spawned} follow={following} target={active_target_track_id} "
                            f"| walkers={desired_num_walkers} speed=[{desired_min_speed:.2f},{desired_max_speed:.2f}]"
                        )
                        line2 = "Keys: G spawn  F follow  T switch-target  H manual-NPC  J manual-robot  R reset  SPACE pause  Q quit"
                        line3 = "Tune: Up/Down walkers | [ ] min_speed | - / = max_speed (reset then G)"
                        line4 = (
                            f"Visibility(instance): pix={pix_count} thr={args.visibility_threshold} "
                            f"=> visible={int(visible)}"
                        )
                        for idx, text in enumerate([line1, line2, line3, line4]):
                            img = ui_font.render(text, True, (230, 235, 240))
                            surf.blit(img, (16, 16 + idx * 34))

                        labels = [
                            ("RGB Left",  0, 0, "rgb_left"),
                            ("RGB Front", 1, 0, "rgb"),
                            ("RGB Right", 2, 0, "rgb_right"),
                            ("Depth",     0, 1, "depth"),
                            ("Instance",  1, 1, "instance"),
                            ("LiDAR BEV", 2, 1, "lidar"),
                        ]
                        base_y = 170
                        for title, cx, cy, key in labels:
                            x = cx * 400
                            y = base_y + cy * 260
                            arr = latest_panels[key]
                            surf.blit(pygame.surfarray.make_surface(arr.transpose(1, 0, 2)), (x, y))
                            pygame.draw.rect(surf, (60, 60, 60), pygame.Rect(x, y, 400, 260), 1)
                            t = ui_font.render(title, True, (255, 220, 0))
                            surf.blit(t, (x + 6, y + 6))
                        pygame.display.flip()

            step_world(world, use_sync_mode)
            tick += 1
    finally:
        if collision_monitor is not None:
            collision_monitor.destroy()
        gt_f.close()
        vis_f.close()
        npc.destroy()
        robot.destroy()
        world.apply_settings(orig_settings)
        if ui_enabled:
            pygame.quit()
        if dbg_vis is not None:
            dbg_vis.stop()
        if eval_logger is not None:
            result = eval_logger.finalize(termination_reason, episode_success=episode_success)
            if result is not None:
                print(f"[EVAL] result: {os.path.join(eval_logger.run_dir, 'eval_result.json')}")
                print(
                    f"[EVAL] obstacle_avoidance_success={result.get('obstacle_avoidance_success')} "
                    f"search_success={result.get('search_success')} "
                    f"TVR={result.get('target_visibility_ratio')}"
                )
        print("[EP] cleanup done.")
        print(f"[EP] GT stream: {gt_path}")
        print(f"[EP] visibility stream: {vis_path}")
        print(f"[EP] calibration: {calib_path}")


if __name__ == "__main__":
    main()
