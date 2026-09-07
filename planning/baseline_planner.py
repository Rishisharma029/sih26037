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
        self.last_candidates_summary: List[dict] = []

    def plan(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput,
        target_cruise_speed_mps: float = 8.0
    ) -> PlannedTrajectory:
        """Generates candidate offset trajectories and selects the optimal feasible path."""
        self.plan_counter += 1
        self.last_candidates_summary = []

        current_x = ego_state.pose.position.x
        current_y = ego_state.pose.position.y
        current_yaw = ego_state.pose.heading_rad
        current_speed = ego_state.twist.speed_mps
        v_target = max(1.0, target_cruise_speed_mps)

        num_steps = int(self.horizon_seconds / self.dt)
        T = self.horizon_seconds

        # Helper to interpolate corridor boundaries at relative station s_rel
        corridor_bps = perception.drivable_corridor.boundary_points

        def get_local_corridor_limits(s_rel: float) -> Tuple[float, float]:
            if not corridor_bps:
                hw = max(2.1, perception.drivable_corridor.average_width_m * 0.5)
                return hw, -hw
            # Find nearest or bracket
            for idx in range(len(corridor_bps) - 1):
                bp0 = corridor_bps[idx]
                bp1 = corridor_bps[idx + 1]
                if bp0.s <= s_rel <= bp1.s:
                    alpha = (s_rel - bp0.s) / max(0.01, bp1.s - bp0.s)
                    dl = bp0.d_left + alpha * (bp1.d_left - bp0.d_left)
                    dr = bp0.d_right + alpha * (bp1.d_right - bp0.d_right)
                    return dl, dr
            if s_rel < corridor_bps[0].s:
                return corridor_bps[0].d_left, corridor_bps[0].d_right
            return corridor_bps[-1].d_left, corridor_bps[-1].d_right

        scored_candidates: List[Tuple[float, float, List[TrajectoryPoint], BehaviorMode]] = []
        raw_candidates_info: List[dict] = []

        # 1. Generate and evaluate each lateral offset candidate across the irregular corridor
        for offset in self.lateral_offsets:
            waypoints: List[TrajectoryPoint] = []
            is_collision = False
            collision_obs_id = ""
            collision_pt = None
            anomaly_cost = 0.0

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

                # Check dynamic corridor limits at this longitudinal station s_val
                dl_curr, dr_curr = get_local_corridor_limits(s_val)
                if (d_val + self.vehicle_half_width) > (dl_curr - 0.10) or (d_val - self.vehicle_half_width) < (dr_curr + 0.10):
                    is_collision = True
                    collision_obs_id = "ROAD_EDGE_DITCH"
                    # Global world coordinates for collision marker
                    cos_h = math.cos(current_yaw)
                    sin_h = math.sin(current_yaw)
                    collision_pt = {
                        "x": round(current_x + s_val * cos_h - d_val * sin_h, 3),
                        "y": round(current_y + s_val * sin_h + d_val * cos_h, 3)
                    }
                    break

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

                # Check Collision with perceived dynamic/static obstacles
                for obs in perception.obstacles:
                    obs_x = obs.bbox.center.x  # forward distance in ego frame
                    obs_y = obs.bbox.center.y  # lateral distance in ego frame
                    obs_half_l = max(0.5, obs.bbox.size.x * 0.5 if obs.bbox.size.x > 0 else 0.8)
                    obs_half_w = max(0.3, obs.bbox.size.y * 0.5 if obs.bbox.size.y > 0 else 0.4)

                    dx = abs(s_val - obs_x)
                    dy = abs(d_val - obs_y)

                    if dx < (1.5 + obs_half_l + 0.2) and dy < (self.vehicle_half_width + obs_half_w + 0.1):
                        is_collision = True
                        collision_obs_id = obs.id
                        collision_pt = {"x": round(wx, 3), "y": round(wy, 3)}
                        break

                if is_collision:
                    break

                # Check Collision / Penalty with Road Anomalies (Potholes, Gravel heaps)
                for anom in perception.anomalies:
                    # Anomaly distance from vehicle position (wx, wy)
                    adx = wx - anom.position.x
                    ady = wy - anom.position.y
                    adist = math.hypot(adx, ady)
                    if adist < (anom.radius_m + self.vehicle_half_width * 0.7):
                        if abs(anom.depth_or_height_m) > 0.12 or anom.anomaly_type == "GRAVEL":
                            # Severe crater / blocked patch -> reject candidate
                            is_collision = True
                            collision_obs_id = f"{anom.anomaly_type}_{anom.id}"
                            collision_pt = {"x": round(wx, 3), "y": round(wy, 3)}
                            break
                        else:
                            # Minor pothole -> penalize cost
                            anomaly_cost += 2.5

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
                # Candidate Cost = |offset| * 2.5 + (deviation penalty) + anomaly_cost
                cost = abs(offset) * 2.5 + (0.5 if offset != 0.0 else 0.0) + anomaly_cost

                if offset > 0.25:
                    mode = BehaviorMode.NUDGE_LEFT
                elif offset < -0.25:
                    mode = BehaviorMode.NUDGE_RIGHT
                else:
                    mode = BehaviorMode.CRUISE

                scored_candidates.append((cost, offset, waypoints, mode))
                raw_candidates_info.append({
                    "offset": offset,
                    "is_feasible": True,
                    "is_selected": False,
                    "cost": round(cost, 2),
                    "mode": mode.value,
                    "rejection_reason": "CLEAR",
                    "waypoints": [{"x": wp.x, "y": wp.y} for wp in waypoints]
                })
            else:
                raw_candidates_info.append({
                    "offset": offset,
                    "is_feasible": False,
                    "is_selected": False,
                    "cost": 9999.0,
                    "mode": "COLLISION",
                    "rejection_reason": f"COLLISION ({collision_obs_id})" if collision_obs_id else "INCOMPLETE",
                    "collision_point": collision_pt,
                    "collision_obstacle_id": collision_obs_id,
                    "waypoints": [{"x": wp.x, "y": wp.y} for wp in waypoints]
                })

        # 2. Select optimal feasible candidate
        if scored_candidates:
            # Sort by cost ascending (prefers center offset 0.0m if clear, otherwise minimal nudge)
            scored_candidates.sort(key=lambda x: x[0])
            best_cost, best_offset, best_wps, best_mode = scored_candidates[0]

            for cand in raw_candidates_info:
                if cand["offset"] == best_offset and cand["is_feasible"]:
                    cand["is_selected"] = True

            self.last_candidates_summary = raw_candidates_info

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
        self.last_candidates_summary = raw_candidates_info
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
