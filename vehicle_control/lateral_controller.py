"""Stanley lateral tracking controller with front-axle cross-track error compensation."""
import math
from typing import Optional
from interfaces import SafeTrajectory, EgoVehicleState


class StanleyLateralController:
    """Calculates front steering angle delta from trajectory waypoints using the Stanley method."""

    def __init__(
        self,
        k_gain: float = 1.4,
        k_soft: float = 1.0,
        max_steer_rad: float = 0.785, # 45 degrees
        max_steer_rate_rad_s: float = 0.60,
        wheelbase_m: float = 2.7
    ):
        self.k_gain = k_gain
        self.k_soft = k_soft
        self.max_steer_rad = max_steer_rad
        self.max_steer_rate_rad_s = max_steer_rate_rad_s
        self.wheelbase_m = wheelbase_m
        self.prev_steer_rad = 0.0

    def compute_steering(
        self,
        ego_state: EgoVehicleState,
        trajectory: SafeTrajectory,
        dt: float = 0.05
    ) -> float:
        """Computes rate-limited front steering angle to track the safe trajectory."""
        if not trajectory.waypoints:
            return 0.0

        ego_x = ego_state.pose.position.x
        ego_y = ego_state.pose.position.y
        ego_yaw = ego_state.pose.heading_rad
        speed = max(0.2, ego_state.twist.speed_mps)

        # 1. Project to front axle
        front_x = ego_x + self.wheelbase_m * math.cos(ego_yaw)
        front_y = ego_y + self.wheelbase_m * math.sin(ego_yaw)

        # 2. Find closest waypoint ahead on trajectory
        best_idx = 0
        min_dist = float("inf")
        for idx, wp in enumerate(trajectory.waypoints):
            d = math.hypot(wp.x - front_x, wp.y - front_y)
            if d < min_dist:
                min_dist = d
                best_idx = idx

        target_wp = trajectory.waypoints[min(best_idx + 1, len(trajectory.waypoints) - 1)]

        # 3. Path heading
        if best_idx < len(trajectory.waypoints) - 1:
            next_wp = trajectory.waypoints[best_idx + 1]
            path_yaw = math.atan2(next_wp.y - target_wp.y, next_wp.x - target_wp.x)
            if math.hypot(next_wp.x - target_wp.x, next_wp.y - target_wp.y) < 0.01:
                path_yaw = target_wp.yaw_rad
        else:
            path_yaw = target_wp.yaw_rad

        # 4. Heading error normalized to [-pi, pi]
        heading_error = path_yaw - ego_yaw
        heading_error = (heading_error + math.pi) % (2.0 * math.pi) - math.pi

        # 5. Cross-track error to front axle
        # Vector from target waypoint to front axle
        dx = front_x - target_wp.x
        dy = front_y - target_wp.y
        # Cross-track error (positive if front axle is to the right of path)
        crosstrack_error = -math.sin(path_yaw) * dx + math.cos(path_yaw) * dy

        # 6. Stanley control law
        delta_raw = heading_error + math.atan2(self.k_gain * (-crosstrack_error), speed + self.k_soft)

        # 7. Steering rate limiter
        max_delta_change = self.max_steer_rate_rad_s * dt
        delta_change = delta_raw - self.prev_steer_rad
        clamped_change = max(-max_delta_change, min(max_delta_change, delta_change))
        delta_cmd = self.prev_steer_rad + clamped_change

        # 8. Maximum steering angle clamp
        delta_final = max(-self.max_steer_rad, min(self.max_steer_rad, delta_cmd))
        self.prev_steer_rad = delta_final

        return round(delta_final, 4)
