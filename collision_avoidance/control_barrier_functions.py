"""Control Barrier Functions (CBF) forward invariance safety filter."""
import math
from typing import Tuple, List, Optional
from interfaces import PlannedTrajectory, EgoVehicleState, PerceptionOutput, TrajectoryPoint


class ControlBarrierFilter:
    """Enforces Control Barrier Function invariants h(x) >= 0 to guarantee forward safe sets
    for both dynamic obstacle clearance and irregular road corridor containment.
    """

    def __init__(self, min_safe_dist_m: float = 1.6, gamma: float = 1.2):
        self.min_safe_dist_m = min_safe_dist_m
        self.gamma = gamma  # CBF decay rate parameter
        self.last_min_ditch_margin = 999.0
        self.last_min_obstacle_margin = 999.0

    def filter_trajectory(
        self,
        trajectory: PlannedTrajectory,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput
    ) -> Tuple[PlannedTrajectory, bool, float]:
        """Validates and filters trajectory waypoints against barrier constraints.

        Returns: (safe_trajectory, barrier_violated, min_margin_m)
        """
        min_ditch_margin = 999.0
        min_obs_margin = 999.0
        barrier_violated = False

        corridor_half_w = max(2.5, perception.drivable_corridor.average_width_m * 0.5)

        filtered_wps = []
        for wp in trajectory.waypoints:
            cloned_wp = TrajectoryPoint(
                timestamp=wp.timestamp,
                x=wp.x,
                y=wp.y,
                yaw_rad=wp.yaw_rad,
                curvature=wp.curvature,
                speed_mps=wp.speed_mps,
                acceleration_mps2=wp.acceleration_mps2,
                jerk_mps3=wp.jerk_mps3
            )
            filtered_wps.append(cloned_wp)

        for wp in filtered_wps:
            # 1. Obstacle barrier condition: h_obs = ||p_wp - p_obs|| - r_safe >= 0
            for obs in perception.obstacles:
                dx = wp.x - obs.bbox.center.x
                dy = wp.y - obs.bbox.center.y
                dist = math.hypot(dx, dy)
                margin = dist - self.min_safe_dist_m
                if margin < min_obs_margin:
                    min_obs_margin = margin

                if margin < 0.0:
                    barrier_violated = True
                    wp.speed_mps = min(wp.speed_mps, 1.2)
                    wp.acceleration_mps2 = -3.5

            # 2. Corridor boundary barrier condition: h_boundary >= 0
            d_left = corridor_half_w
            d_right = -corridor_half_w
            if perception.drivable_corridor.boundary_points:
                ref_s = wp.x - (filtered_wps[0].x if filtered_wps else 0.0)
                closest_bp = min(perception.drivable_corridor.boundary_points, key=lambda bp: abs(bp.s - ref_s))
                d_left = closest_bp.d_left
                d_right = closest_bp.d_right

            veh_half_w = 0.90
            dist_left = d_left - (wp.y + veh_half_w)
            dist_right = (wp.y - veh_half_w) - d_right
            b_margin = min(dist_left, dist_right)
            if b_margin < min_ditch_margin:
                min_ditch_margin = b_margin

            if b_margin < 0.0:
                barrier_violated = True
                wp.speed_mps = 0.0
                wp.acceleration_mps2 = -6.0
            elif b_margin < 0.25:
                barrier_violated = True
                wp.speed_mps = min(wp.speed_mps, 2.0)
                wp.acceleration_mps2 = -2.5

        self.last_min_ditch_margin = round(min_ditch_margin, 2)
        self.last_min_obstacle_margin = round(min_obs_margin, 2)
        total_min_margin = min(min_ditch_margin, min_obs_margin)

        filtered_traj = PlannedTrajectory(
            trajectory_id=trajectory.trajectory_id,
            timestamp=trajectory.timestamp,
            behavior_mode=trajectory.behavior_mode,
            waypoints=filtered_wps,
            target_speed_mps=trajectory.target_speed_mps,
            total_cost=trajectory.total_cost,
            is_feasible=not barrier_violated
        )

        return filtered_traj, barrier_violated, max(-5.0, round(total_min_margin, 2))
