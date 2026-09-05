"""Longitudinal PID and feedforward acceleration controller."""
from interfaces import SafeTrajectory, EgoVehicleState


class LongitudinalPIDController:
    """Generates throttle and brake demand percentages from target speed and feedforward acceleration."""

    def __init__(
        self,
        kp: float = 35.0,
        ki: float = 1.2,
        kd: float = 3.0,
        max_integral: float = 8.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.integral_error = 0.0
        self.prev_error = 0.0

    def compute_throttle_brake(
        self,
        ego_state: EgoVehicleState,
        trajectory: SafeTrajectory,
        dt: float = 0.05
    ) -> tuple[float, float]:
        """Computes throttle (0-100%) and brake (0-100%) percentages."""
        if trajectory.is_emergency_stop:
            self.integral_error = 0.0
            return 0.0, 100.0

        if not trajectory.waypoints:
            return 0.0, 35.0

        # Lookahead target point for smoother and faster responsiveness
        lookahead_idx = min(2, len(trajectory.waypoints) - 1)
        target_wp = trajectory.waypoints[lookahead_idx]
        target_speed = target_wp.speed_mps
        target_accel = target_wp.acceleration_mps2
        curr_speed = ego_state.twist.speed_mps

        error = target_speed - curr_speed

        # Anti-windup integration
        self.integral_error += error * dt
        self.integral_error = max(-self.max_integral, min(self.max_integral, self.integral_error))

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # Feedback control output
        fb_output = self.kp * error + self.ki * self.integral_error + self.kd * derivative

        # Feedforward acceleration contribution (scaled)
        ff_output = target_accel * 18.0

        total_demand = fb_output + ff_output

        if total_demand > 0.5:
            throttle = min(100.0, max(0.0, total_demand * 1.5))
            brake = 0.0
        elif total_demand < -0.5:
            throttle = 0.0
            brake = min(100.0, max(0.0, -total_demand * 1.5))
        else:
            throttle = 0.0
            brake = 0.0

        return round(throttle, 1), round(brake, 1)
