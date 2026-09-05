"""Stanley lateral tracking controller with front-axle cross-track error compensation."""
import math
import numpy as np
from interfaces import SafeTrajectory, EgoVehicleState

class StanleyLateralController:
    """Calculates front steering angle delta from trajectory waypoints."""
    def __init__(self, k_gain: float = 1.2, max_steer_rad: float = 0.785):
        self.k_gain = k_gain
        self.max_steer_rad = max_steer_rad

    def compute_steering(self, ego_state: EgoVehicleState, trajectory: SafeTrajectory) -> float:
        if not trajectory.waypoints:
            return 0.0

        target_wp = trajectory.waypoints[min(1, len(trajectory.waypoints)-1)]
        dx = target_wp.x - ego_state.pose.position.x
        dy = target_wp.y - ego_state.pose.position.y

        path_yaw = math.atan2(dy, dx)
        yaw_error = path_yaw - ego_state.pose.heading_rad
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        # Cross-track error
        crosstrack_error = dy
        speed = max(0.5, ego_state.twist.speed_mps)

        delta = yaw_error + math.atan2(self.k_gain * crosstrack_error, speed)
        return float(np.clip(delta, -self.max_steer_rad, self.max_steer_rad))
