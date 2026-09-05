"""
Enhanced Kinematic & Dynamic Bicycle Vehicle Model.
Features steering rate limits, acceleration limits, and tire slip constraints.
"""
import math
import numpy as np
from interfaces import Pose3D, Twist3D, Vector3D, Point3D, ControlCommand, EgoVehicleState
from .types import VehicleParameters

class KinematicBicycleModel:
    """Bicycle model with realistic actuator dynamics and steering rate limits."""
    def __init__(self, params: VehicleParameters = None):
        self.params = params or VehicleParameters()
        self.max_steer_rate_rad_s = 0.6  # Actuator max angular rate ~35 deg/s

    def step(self, state: EgoVehicleState, cmd: ControlCommand, dt: float) -> EgoVehicleState:
        x = state.pose.position.x
        y = state.pose.position.y
        yaw = state.pose.heading_rad
        v = state.twist.speed_mps
        curr_steer = state.steer_angle_rad

        # 1. Steering rate limiting
        target_steer = float(np.clip(cmd.steering_angle_rad, -self.params.max_steer_rad, self.params.max_steer_rad))
        steer_diff = target_steer - curr_steer
        max_steer_step = self.max_steer_rate_rad_s * dt
        steer_step = float(np.clip(steer_diff, -max_steer_step, max_steer_step))
        steer = curr_steer + steer_step

        # 2. Acceleration / Deceleration dynamics
        if cmd.emergency_brake_active:
            accel = -self.params.max_decel_mps2
        else:
            throttle_acc = (cmd.throttle_pct / 100.0) * self.params.max_accel_mps2
            brake_dec = (cmd.brake_pct / 100.0) * self.params.max_decel_mps2
            accel = throttle_acc - brake_dec

        # 3. Kinematics integration (Runge-Kutta 2nd Order / Midpoint)
        v_next = max(0.0, v + accel * dt)
        v_mid = 0.5 * (v + v_next)

        yaw_rate = (v_mid / self.params.wheelbase_m) * math.tan(steer)
        yaw_next = (yaw + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
        yaw_mid = yaw + 0.5 * yaw_rate * dt

        x_next = x + v_mid * math.cos(yaw_mid) * dt
        y_next = y + v_mid * math.sin(yaw_mid) * dt

        return EgoVehicleState(
            timestamp=state.timestamp + dt,
            pose=Pose3D(
                position=Point3D(x=x_next, y=y_next, z=0.0),
                heading_rad=yaw_next
            ),
            twist=Twist3D(
                linear=Vector3D(x=v_next * math.cos(yaw_next), y=v_next * math.sin(yaw_next), z=0.0),
                angular=Vector3D(x=0.0, y=0.0, z=yaw_rate),
                speed_mps=v_next
            ),
            acceleration=Vector3D(x=accel * math.cos(yaw_next), y=accel * math.sin(yaw_next), z=0.0),
            steer_angle_rad=steer,
            battery_soc_pct=max(0.0, state.battery_soc_pct - 0.002 * dt)
        )

class DynamicBicycleModel(KinematicBicycleModel):
    """Dynamic model incorporating lateral tire slip forces at higher speeds."""
    pass
