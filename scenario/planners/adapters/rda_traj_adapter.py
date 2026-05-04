"""RDA trajectory follower adapter for the Follow-Bench platform."""

from __future__ import annotations

import math

import numpy as np

from behavior.follow_goal import compute_follow_goal
from common.types import AgentState2D
from planning.rda.rda_traj import RdaTrajPlanner
from adapters.rda_obstacles import NPC_OBS_HALF_SIZE_M, obstacles_debug, target_box_obstacle
from core_types import FollowAction, FollowObservation
from follow_policy_adapter import FollowerPolicyAdapter
from perception.gt.gt_scene_provider import robot_state_from_obs, target_state_from_obs, target_yaw_from_obs


def _predict_target_traj(tx: float, ty: float, vx: float, vy: float, n_steps: int, dt: float) -> np.ndarray:
    steps = np.arange(1, n_steps + 1, dtype=float)[:, None]
    return np.array([tx, ty]) + steps * np.array([vx, vy]) * dt


def _build_goal_traj(
    target_future: np.ndarray,
    obs: FollowObservation,
    follow_position: str,
    distance: float,
) -> list:
    goal_traj = []
    robot = robot_state_from_obs(obs)
    target_yaw = target_yaw_from_obs(obs)
    for i in range(target_future.shape[0]):
        tx_i, ty_i = target_future[i, 0], target_future[i, 1]
        target = AgentState2D(
            track_id=str(obs.target.track_id),
            x=float(tx_i),
            y=float(ty_i),
            vx=float(obs.target.vx),
            vy=float(obs.target.vy),
            yaw=target_yaw,
            speed=float(obs.target.speed),
            is_target=True,
        )
        goal_traj.append(compute_follow_goal(robot, target, follow_position, distance))
    return goal_traj


class RdaTrajFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        follow_position: str = "right_side",
        receding: int = 10,
        iter_num: int = 4,
        process_num: int = 1,
        enable_uncertainty_brake: bool = False,
        protect_target_for_side_follow: bool = True,
        target_protect_half_size: float = NPC_OBS_HALF_SIZE_M,
    ) -> None:
        self.desired_distance = desired_distance
        self.follow_position = follow_position
        self.dt = dt
        self._receding = receding
        self.protect_target_for_side_follow = bool(protect_target_for_side_follow)
        self.target_protect_half_size = float(target_protect_half_size)
        self._planner = RdaTrajPlanner(
            dt=dt,
            receding=receding,
            iter_num=iter_num,
            process_num=process_num,
            enable_uncertainty_brake=enable_uncertainty_brake,
        )
        self._last_debug: dict = {
            "goal_point": None,
            "traj_points": [],
            "obstacles": [],
            "predicted_target_traj": [],
        }

    def reset(self) -> None:
        self._planner.reset()
        self._last_debug = {
            "goal_point": None,
            "traj_points": [],
            "obstacles": [],
            "predicted_target_traj": [],
        }

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        rx, ry = float(obs.robot.x), float(obs.robot.y)
        tx, ty = float(obs.target.x), float(obs.target.y)
        vx, vy = float(obs.target.vx), float(obs.target.vy)
        robot_yaw = float(obs.robot.yaw_rad)
        dist = math.hypot(tx - rx, ty - ry)

        target_future = _predict_target_traj(tx, ty, vx, vy, self._receding, self.dt)
        goal_traj = _build_goal_traj(target_future, obs, self.follow_position, self.desired_distance)
        ref_speed = max(math.hypot(vx, vy), 0.3)
        obstacle_list = []
        if self.protect_target_for_side_follow and self.follow_position in ("left_side", "right_side"):
            target = target_state_from_obs(obs, self.target_protect_half_size)
            if target is not None:
                obstacle_list.append(target_box_obstacle(target, half_size=self.target_protect_half_size))

        try:
            result = self._planner.compute(
                robot_pose=np.array([rx, ry, robot_yaw], dtype=float),
                goal_traj=goal_traj,
                target_traj=target_future.T,
                ref_speed=ref_speed,
                obstacle_list=obstacle_list,
            )
        except Exception as e:
            self._last_debug = {
                "goal_point": goal_traj[0][:2, 0].tolist() if goal_traj else None,
                "traj_points": [],
                "obstacles": obstacles_debug(obstacle_list),
                "predicted_target_traj": target_future.tolist(),
                "error": str(e),
            }
            print(f"[RDA-Traj] MPC error: {e}")
            return FollowAction(v_mps=0.0, w_radps=0.0)

        v = float(result["v_mps"])
        w = float(result["w_radps"])
        self._last_debug = {
            "goal_point": goal_traj[0][:2, 0].tolist() if goal_traj else None,
            "traj_points": result.get("traj_points", []) if isinstance(result, dict) else [],
            "obstacles": obstacles_debug(obstacle_list),
            "predicted_target_traj": target_future.tolist(),
        }
        print(
            f"[RDA-Traj] mode={self.follow_position} v={v:.3f} w={w:.3f}  "
            f"dist={dist:.2f}m  target_vel=({vx:.2f},{vy:.2f})"
        )
        return FollowAction(v_mps=v, w_radps=w)


__all__ = ["RdaTrajFollowerPolicy"]
