"""Rigid Lane-Following / Pure Pursuit Baseline Planner.

Represents conventional rule-based AV planning that assumes fixed lane centerlines
and lacks multi-candidate adaptive sampling and unstructured uncertainty cost optimization.
"""
import math
from typing import List
from interfaces import (
    PlannedTrajectory, TrajectoryPoint, BehaviorMode,
    EgoVehicleState, PerceptionOutput, PredictionOutput
)


class BaselinePlanner:
    """Conventional baseline planner following a rigid centerline with simple headway keeping.

    Unlike the Adaptive Lattice Planner, it does not sample lateral nudges, does not
    evaluate boundary traversability or multi-modal agent covariance, and relies solely
    on crude longitudinal braking when obstacles block the nominal path.
    """

    def __init__(self, horizon_seconds: float = 3.0, dt: float = 0.2):
        self.horizon_seconds = horizon_seconds
        self.dt = dt
        self.plan_counter = 0

    def plan(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput,
        target_cruise_speed_mps: float = 8.0
    ) -> PlannedTrajectory:
        """Generate a rigid centerline trajectory without adaptive lateral swerving."""
        self.plan_counter += 1
        num_steps = int(self.horizon_seconds / self.dt)

        current_x = ego_state.pose.position.x
        current_y = ego_state.pose.position.y
        current_speed = ego_state.twist.speed_mps
        current_yaw = ego_state.pose.heading_rad

        # Baseline obstacle check: check for any obstacle directly ahead within a fixed lateral corridor (+- 1.2m)
        min_lead_distance = float('inf')
        for obs in perception.obstacles:
            dx = obs.bbox.center.x - current_x
            dy = obs.bbox.center.y - current_y
            # Longitudinal forward distance and lateral offset
            lon_dist = dx * math.cos(current_yaw) + dy * math.sin(current_yaw)
            lat_dist = abs(-dx * math.sin(current_yaw) + dy * math.cos(current_yaw))

            if lon_dist > 0.5 and lat_dist < 1.3:
                if lon_dist < min_lead_distance:
                    min_lead_distance = lon_dist

        # Rigid longitudinal speed rule
        if min_lead_distance < 6.0:
            target_speed = 0.0  # Harsh stop
            behavior = BehaviorMode.EMERGENCY_STOP
        elif min_lead_distance < 15.0:
            target_speed = max(0.0, current_speed * 0.5)  # Abrupt deceleration
            behavior = BehaviorMode.FOLLOW
        else:
            target_speed = target_cruise_speed_mps
            behavior = BehaviorMode.CRUISE

        # Generate straight/rigid line points directly along ego yaw with zero lateral adaptation
        points: List[TrajectoryPoint] = []
        x = current_x
        y = current_y
        speed = current_speed

        for i in range(num_steps):
            t = ego_state.timestamp + (i + 1) * self.dt
            speed = speed + 0.3 * (target_speed - speed)
            x += speed * math.cos(current_yaw) * self.dt
            y += speed * math.sin(current_yaw) * self.dt

            points.append(TrajectoryPoint(
                timestamp=t,
                x=x,
                y=y,
                yaw_rad=current_yaw,
                curvature=0.0,
                speed_mps=max(0.0, speed),
                acceleration_mps2=0.0,
                jerk_mps3=0.0
            ))

        return PlannedTrajectory(
            trajectory_id=f"baseline_traj_{self.plan_counter}",
            timestamp=ego_state.timestamp,
            behavior_mode=behavior,
            waypoints=points,
            target_speed_mps=target_speed,
            total_cost=50.0,
            is_feasible=True
        )
