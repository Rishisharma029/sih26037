"""Unit tests for lateral and longitudinal control."""
from vehicle_control.lateral_controller import StanleyLateralController
from vehicle_control.longitudinal_controller import LongitudinalPIDController
from vehicle_control.drive_by_wire_bridge import DriveByWireBridge
from interfaces import (
    EgoVehicleState, Pose3D, Point3D, Twist3D, Vector3D,
    SafeTrajectory, TrajectoryPoint, SafetyAction
)

def test_stanley_steering():
    stanley = StanleyLateralController()
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=5.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )
    # Target path slightly to the left (y = 1.0)
    safe_traj = SafeTrajectory(
        timestamp=0.0,
        source_trajectory_id="s1",
        waypoints=[TrajectoryPoint(timestamp=0.1, x=5.0, y=1.0, speed_mps=5.0)],
        safety_action=SafetyAction.NONE,
        is_emergency_stop=False,
        barrier_margin_m=5.0,
        min_ttc_seconds=10.0
    )
    steer = stanley.compute_steering(ego, safe_traj)
    # Positive steer to turn left towards y=1.0
    assert steer > 0.0

def test_longitudinal_throttle_and_can():
    bridge = DriveByWireBridge()
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0)),
        twist=Twist3D(speed_mps=2.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )
    safe_traj = SafeTrajectory(
        timestamp=0.0,
        source_trajectory_id="s1",
        waypoints=[TrajectoryPoint(timestamp=0.1, x=5.0, y=0.0, speed_mps=6.0)],
        safety_action=SafetyAction.NONE,
        is_emergency_stop=False,
        barrier_margin_m=5.0,
        min_ttc_seconds=10.0
    )
    cmd = bridge.generate_command(ego, safe_traj)
    assert cmd.throttle_pct > 0.0
    can = bridge.to_can_frames(cmd)
    assert "CAN_ID_STEERING_0x101" in can
