"""Strict type & interface contract validation test suite."""
import pytest
from interfaces import (
    Point3D, Vector3D, Pose3D, Twist3D, ObstacleClass, MotionIntent, BehaviorMode,
    SafetyAction, TrackedObstacle, FreeSpaceCorridor, CorridorBoundaryPoint,
    PerceptionOutput, PredictedAgent, PredictedTrajectory, PredictionOutput,
    PlannedTrajectory, TrajectoryPoint, SafeTrajectory, ControlCommand,
    EgoVehicleState, RawSensorFrame, VehicleTelemetry
)

def test_primitive_types():
    pt = Point3D(x=10.0, y=-2.5, z=0.0)
    assert pt.x == 10.0
    assert pt.y == -2.5

    v = Vector3D(x=1.0, y=2.0, z=3.0)
    assert v.z == 3.0

def test_perception_output_contract():
    corridor = FreeSpaceCorridor(
        timestamp=1.0,
        boundary_points=[CorridorBoundaryPoint(s=0.0, d_left=3.0, d_right=-3.0)],
        average_width_m=6.0
    )
    p_out = PerceptionOutput(
        timestamp=1.0,
        frame_id=1,
        obstacles=[],
        drivable_corridor=corridor
    )
    json_str = p_out.model_dump_json()
    reloaded = PerceptionOutput.model_validate_json(json_str)
    assert reloaded.frame_id == 1
    assert reloaded.drivable_corridor.average_width_m == 6.0

def test_prediction_output_contract():
    pred = PredictionOutput(
        timestamp=1.0,
        horizon_seconds=3.0,
        agents=[],
        high_risk_agent_ids=[]
    )
    assert pred.horizon_seconds == 3.0

def test_planning_output_contract():
    traj = PlannedTrajectory(
        trajectory_id="traj_001",
        timestamp=1.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[TrajectoryPoint(timestamp=1.0, x=0.0, y=0.0, speed_mps=5.0)],
        target_speed_mps=5.0
    )
    assert traj.behavior_mode == BehaviorMode.CRUISE
    assert len(traj.waypoints) == 1

def test_control_command_limits():
    cmd = ControlCommand(
        timestamp=1.0,
        steering_angle_rad=0.35,
        throttle_pct=50.0,
        brake_pct=0.0
    )
    assert cmd.steering_angle_rad == 0.35
    with pytest.raises(Exception):
        # Exceeds max steering angle 45 deg (0.785 rad)
        ControlCommand(timestamp=1.0, steering_angle_rad=1.5)
