"""Longitudinal PID speed and acceleration tracker."""
from interfaces import SafeTrajectory, EgoVehicleState

class LongitudinalPIDController:
    """Generates throttle and brake percentages from target speed error."""
    def __init__(self, kp: float = 20.0, ki: float = 0.5, kd: float = 2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0.0
        self.prev_error = 0.0

    def compute_throttle_brake(
        self,
        ego_state: EgoVehicleState,
        trajectory: SafeTrajectory,
        dt: float = 0.05
    ) -> tuple[float, float]:
        if trajectory.is_emergency_stop:
            return 0.0, 100.0

        if not trajectory.waypoints:
            return 0.0, 30.0

        target_speed = trajectory.waypoints[0].speed_mps
        error = target_speed - ego_state.twist.speed_mps

        self.integral_error += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral_error + self.kd * derivative

        if output > 0:
            throttle = min(100.0, output)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(100.0, -output)

        return throttle, brake
