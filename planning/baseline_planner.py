"""
Candidate Lateral Offset Adaptive Baseline Planner for SIH26037.

Provides the foundational adaptive path generation stage:
1. Reference path generation
2. Discrete candidate lateral offsets:
     ?d ? {-1.5m, -1.0m, -0.5m, 0.0m, +0.5m, +1.0m, +1.5m}
3. Dynamic collision checking against perceived obstacles
4. Hard road corridor boundary validation
5. Optimal feasible candidate selection with kinematic completeness
   (waypoints, curvature, speed, acceleration, and jerk)
"""
import math
from typing import List, Tuple, Optional
from interfaces import (
    PlannedTrajectory, TrajectoryPoint, BehaviorMode,
    EgoVehicleState, PerceptionOutput, PredictionOutput
)


class BaselinePlanner:
    """Candidate Lateral Offset Baseline Planner.
    
    Generates a bundle of discrete lateral offset trajectories, evaluates them
    against obstacle collisions and road corridor boundaries, and selects the
    safest minimal-offset feasible path.
    """

    def __init__(
        self,
        horizon_seconds: float = 3.0,
        dt: float = 0.2,
        lateral_offsets: Optional[List[float]] = None,
        safety_dist_m: float = 1.6,
        vehicle_half_width: float = 0.90
    ):
        self.horizon_seconds = horizon_seconds
        self.dt = dt
        self.lateral_offsets = lateral_offsets if lateral_offsets is not None else [
            -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5
        ]
        self.safety_dist_m = safety_dist_m
        self.vehicle_half_width = vehicle_half_width
        self.plan_counter = 0

    def plan(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput,
        target_cruise_speed_mps: float = 8.0
    ) -> PlannedTrajectory:
        """Generates candidate offset trajectories and selects the optimal feasible path."""
        self.plan_counter += 1

        current_x = ego_state.pose.position.x
        current_y = ego_state.pose.position.y
        current_yaw = ego_state.pose.heading_rad
        current_speed = ego_state.twist.speed_mps
        v_target = max(1.0, target_cruise_speed_mps)

        num_steps = int(self.horizon_seconds / self.dt)
        T = self.horizon_seconds

        # Corridor boundaries
        corridor_half_w = max(2.1, perception.drivable_corridor.average_width_m * 0.5)
        d_left_limit = corridor_half_w - self.vehicle_half_width
        d_right_limit = -corridor_half_w + self.vehicle_half_width

        scored_candidates: List[Tuple[float, float, List[TrajectoryPoint], BehaviorMode]] = []

        # 1. Generate and evaluate each lateral offset candidate
        for offset in self.lateral_offsets:
            # Check if target offset is within drivable road corridor
            if offset > d_left_limit or offset < d_right_limit:
                continue

            waypoints: List[TrajectoryPoint] = []
            is_collision = False

            for i in range(1, num_steps + 1):
                t = i * self.dt
                tau = t / T  # Normalized time in [0, 1]

                # Minimum-jerk quintic lateral displacement: d(tau) = offset * (10*tau^3 - 15*tau^4 + 6*tau^5)
                poly_val = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
                poly_d1 = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / T
                poly_d2 = (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)) / (T ** 2)
                poly_d3 = (60.0 - 360.0 * tau + 360.0 * (tau ** 2)) / (T ** 3)

                d_val = offset * poly_val
                d_dot = offset * poly_d1
                d_ddot = offset * poly_d2
                d_dddot = offset * poly_d3

                # Longitudinal station s(t)
                s_val = current_speed * t + 0.5 * ((v_target - current_speed) / T) * (t ** 2)
                s_dot = current_speed + ((v_target - current_speed) / T) * t
                s_ddot = (v_target - current_speed) / T

                # Analytical speed and curvature
                speed_t = math.hypot(s_dot, d_dot)
                curvature_t = (s_dot * d_ddot - s_ddot * d_dot) / max(0.01, speed_t ** 3)
                accel_t = (s_dot * s_ddot + d_dot * d_ddot) / max(0.1, speed_t)
                jerk_t = abs(offset * poly_d3) * 0.1

                # Global world coordinates
                cos_h = math.cos(current_yaw)
                sin_h = math.sin(current_yaw)
                wx = current_x + s_val * cos_h - d_val * sin_h
                wy = current_y + s_val * sin_h + d_val * cos_h
                yaw_t = current_yaw + math.atan2(d_dot, max(0.1, s_dot))

                # Check Collision with all perceived obstacles (in vehicle frame: s_val ahead, d_val lateral)
                for obs in perception.obstacles:
                    obs_x = obs.bbox.center.x  # forward distance in ego frame
                    obs_y = obs.bbox.center.y  # lateral distance in ego frame
                    obs_half_l = max(0.5, obs.bbox.size.x * 0.5 if obs.bbox.size.x > 0 else 0.8)
                    obs_half_w = max(0.3, obs.bbox.size.y * 0.5 if obs.bbox.size.y > 0 else 0.4)

                    dx = abs(s_val - obs_x)
                    dy = abs(d_val - obs_y)

                    # Collision checking using longitudinal and lateral footprints
                    if dx < (1.5 + obs_half_l + 0.2) and dy < (self.vehicle_half_width + obs_half_w + 0.1):
                        is_collision = True
                        break

                if is_collision:
                    break

                waypoints.append(TrajectoryPoint(
                    timestamp=ego_state.timestamp + t,
                    x=round(wx, 3),
                    y=round(wy, 3),
                    yaw_rad=round(yaw_t, 4),
                    curvature=round(curvature_t, 4),
                    speed_mps=round(max(0.0, speed_t), 2),
                    acceleration_mps2=round(accel_t, 3),
                    jerk_mps3=round(jerk_t, 3)
                ))

            if not is_collision and len(waypoints) == num_steps:
                # Candidate Cost = |offset| * 2.0 + (lateral deviation penalty)
                cost = abs(offset) * 2.5 + (0.5 if offset != 0.0 else 0.0)

                if offset > 0.25:
                    mode = BehaviorMode.NUDGE_LEFT
                elif offset < -0.25:
                    mode = BehaviorMode.NUDGE_RIGHT
                else:
                    mode = BehaviorMode.CRUISE

                scored_candidates.append((cost, offset, waypoints, mode))

        # 2. Select optimal feasible candidate
        if scored_candidates:
            # Sort by cost ascending (prefers center offset 0.0m if clear, otherwise minimal nudge)
            scored_candidates.sort(key=lambda x: x[0])
            best_cost, best_offset, best_wps, best_mode = scored_candidates[0]

            return PlannedTrajectory(
                trajectory_id=f"baseline_opt_{self.plan_counter}_offset_{best_offset:+.1f}",
                timestamp=ego_state.timestamp,
                behavior_mode=best_mode,
                waypoints=best_wps,
                target_speed_mps=v_target,
                total_cost=round(best_cost, 2),
                is_feasible=True
            )

        # 3. Fallback: All lateral offsets are blocked -> execute safe emergency stop directly ahead
        fallback_wps: List[TrajectoryPoint] = []
        speed_decay = current_speed
        for i in range(1, num_steps + 1):
            t = i * self.dt
            speed_decay = max(0.0, speed_decay - 3.5 * self.dt)
            dist_step = speed_decay * self.dt
            wx = current_x + dist_step * (i * 0.5) * math.cos(current_yaw)
            wy = current_y + dist_step * (i * 0.5) * math.sin(current_yaw)

            fallback_wps.append(TrajectoryPoint(
                timestamp=ego_state.timestamp + t,
                x=round(wx, 3),
                y=round(wy, 3),
                yaw_rad=current_yaw,
                curvature=0.0,
                speed_mps=round(speed_decay, 2),
                acceleration_mps2=-3.5 if speed_decay > 0.0 else 0.0,
                jerk_mps3=0.0
            ))

        return PlannedTrajectory(
            trajectory_id=f"baseline_stop_{self.plan_counter}",
            timestamp=ego_state.timestamp,
            behavior_mode=BehaviorMode.EMERGENCY_STOP,
            waypoints=fallback_wps,
            target_speed_mps=0.0,
            total_cost=9999.0,
            is_feasible=False
        )
