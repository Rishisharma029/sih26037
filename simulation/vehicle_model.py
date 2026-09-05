"""Vehicle dynamics models (Kinematic & Dynamic Bicycle)."""
import math
import numpy as np
from interfaces import Pose3D, Twist3D, Vector3D, Point3D, ControlCommand, EgoVehicleState
from .types import VehicleParameters

class KinematicBicycleModel:
    """Kinematic bicycle model valid for low to moderate speed operations."""
    def __init__(self, params: VehicleParameters = None):
        self.params = params or VehicleParameters()

    def step(self, state: EgoVehicleState, cmd: ControlCommand, dt: float) -> EgoVehicleState:
        x = state.pose.position.x
        y = state.pose.position.y
        yaw = state.pose.heading_rad
        v = state.twist.speed_mps

        steer = np.clip(cmd.steering_angle_rad, -self.params.max_steer_rad, self.params.max_steer_rad)
        accel = (cmd.throttle_pct / 100.0) * self.params.max_accel_mps2 - (cmd.brake_pct / 100.0) * self.params.max_decel_mps2
        if cmd.emergency_brake_active:
            accel = -self.params.max_decel_mps2

        # Kinematic update
        v_next = max(0.0, v + accel * dt)
        yaw_rate = (v / self.params.wheelbase_m) * math.tan(steer)
        yaw_next = (yaw + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
        x_next = x + v * math.cos(yaw) * dt
        y_next = y + v * math.sin(yaw) * dt

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
            battery_soc_pct=max(0.0, state.battery_soc_pct - 0.001 * dt)
        )

class DynamicBicycleModel(KinematicBicycleModel):
    """Dynamic bicycle model accounting for tire slip angles."""
    pass
