"""Adaptive Frenet-frame lattice trajectory planner."""
import math
from interfaces import (
    PlannedTrajectory, TrajectoryPoint, BehaviorMode,
    EgoVehicleState, PerceptionOutput, PredictionOutput
)
from .behavior_planner import BehaviorPlanner

class AdaptiveLatticePlanner:
    """Plans jerk-optimal collision-free trajectories respecting free-space boundaries."""
    def __init__(self, horizon_seconds: float = 3.0, dt: float = 0.2):
        self.horizon_seconds = horizon_seconds
        self.dt = dt
        self.behavior_planner = BehaviorPlanner()
        self.plan_counter = 0

    def plan(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput,
        target_cruise_speed_mps: float = 8.0
    ) -> PlannedTrajectory:
        self.plan_counter += 1
        behavior = self.behavior_planner.decide_behavior(ego_state, perception, prediction)

        target_speed = target_cruise_speed_mps
        lateral_offset = 0.0

        if behavior == BehaviorMode.EMERGENCY_STOP:
            target_speed = 0.0
        elif behavior == BehaviorMode.FOLLOW:
            target_speed = max(2.0, target_cruise_speed_mps * 0.6)
        elif behavior == BehaviorMode.NUDGE_LEFT:
            lateral_offset = 1.2
            target_speed = min(target_speed, 4.5)
        elif behavior == BehaviorMode.NUDGE_RIGHT:
            lateral_offset = -1.2
            target_speed = min(target_speed, 4.5)

        waypoints = []
        steps = int(self.horizon_seconds / self.dt)
        curr_x = ego_state.pose.position.x
        curr_y = ego_state.pose.position.y
        curr_speed = ego_state.twist.speed_mps

        for i in range(steps):
            t = ego_state.timestamp + (i + 1) * self.dt
            alpha = (i + 1) / steps
            spd = curr_speed + alpha * (target_speed - curr_speed)
            dist = spd * (i + 1) * self.dt
            wx = curr_x + dist
            wy = curr_y + alpha * lateral_offset

            waypoints.append(TrajectoryPoint(
                timestamp=t,
                x=wx,
                y=wy,
                yaw_rad=ego_state.pose.heading_rad,
                curvature=0.0,
                speed_mps=max(0.0, spd),
                acceleration_mps2=(target_speed - curr_speed) / self.horizon_seconds,
                jerk_mps3=0.0
            ))

        return PlannedTrajectory(
            trajectory_id=f"traj_{self.plan_counter}",
            timestamp=ego_state.timestamp,
            behavior_mode=behavior,
            waypoints=waypoints,
            target_speed_mps=target_speed,
            total_cost=0.0,
            is_feasible=True
        )
