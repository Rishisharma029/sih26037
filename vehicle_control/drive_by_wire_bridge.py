"""Drive-by-Wire gateway translating commands to CAN frames and ROS2 format."""
from interfaces import SafeTrajectory, EgoVehicleState, ControlCommand, GearMode
from .lateral_controller import StanleyLateralController
from .longitudinal_controller import LongitudinalPIDController

class DriveByWireBridge:
    """Executes control loop and serializes commands."""
    def __init__(self):
        self.lateral = StanleyLateralController()
        self.longitudinal = LongitudinalPIDController()

    def generate_command(self, ego_state: EgoVehicleState, trajectory: SafeTrajectory, dt: float = 0.05) -> ControlCommand:
        steer = self.lateral.compute_steering(ego_state, trajectory)
        throttle, brake = self.longitudinal.compute_throttle_brake(ego_state, trajectory, dt)

        return ControlCommand(
            timestamp=ego_state.timestamp,
            steering_angle_rad=steer,
            throttle_pct=throttle,
            brake_pct=brake,
            gear=GearMode.DRIVE,
            emergency_brake_active=trajectory.is_emergency_stop
        )

    def to_can_frames(self, cmd: ControlCommand) -> dict:
        """Translates ControlCommand to mock CAN arbitration IDs."""
        return {
            "CAN_ID_STEERING_0x101": {"angle_rad": cmd.steering_angle_rad},
            "CAN_ID_THROTTLE_0x102": {"throttle_pct": cmd.throttle_pct},
            "CAN_ID_BRAKE_0x103": {"brake_pct": cmd.brake_pct, "e_stop": cmd.emergency_brake_active},
            "CAN_ID_GEAR_0x104": {"gear": cmd.gear.value}
        }
