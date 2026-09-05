"""Comprehensive Unit & Closed-Loop Integration Tests for Phase 8 Vehicle Control."""
import math
import pytest

from interfaces import (
    EgoVehicleState, Pose3D, Point3D, Twist3D, Vector3D,
    SafeTrajectory, TrajectoryPoint, SafetyAction
)
from vehicle_control.lateral_controller import StanleyLateralController
from vehicle_control.longitudinal_controller import LongitudinalPIDController
from vehicle_control.drive_by_wire_bridge import DriveByWireBridge
from simulation.closed_loop_pipeline import ClosedLoopAutonomyPipeline
from simulation.environment import SimulationEnvironment, RoadGeometry, RoadCorridorProfile
from simulation.actors import SimulationActor
from interfaces import ObstacleClass


def test_stanley_steering_correction():
    """Verify Stanley controller steers towards left target waypoint."""
    stanley = StanleyLateralController(k_gain=1.4)
    ego = EgoVehicleState(
        timestamp=10.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=5.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )
    # Target path nudging left (y = 1.2m)
    safe_traj = SafeTrajectory(
        timestamp=10.0,
        source_trajectory_id="s1",
        waypoints=[
            TrajectoryPoint(timestamp=10.2, x=2.0, y=0.5, speed_mps=5.0),
            TrajectoryPoint(timestamp=10.4, x=5.0, y=1.2, speed_mps=5.0)
        ],
        safety_action=SafetyAction.CORRIDOR_NUDGE,
        is_emergency_stop=False,
        barrier_margin_m=4.0,
        min_ttc_seconds=10.0
    )
    steer = stanley.compute_steering(ego, safe_traj, dt=0.05)
    # Must produce positive steering angle to steer towards y=1.2
    assert steer > 0.0
    assert steer <= 0.785


def test_longitudinal_throttle_and_can():
    """Verify DriveByWireBridge converts speed deficit to throttle demand and CAN arbitration IDs."""
    bridge = DriveByWireBridge()
    ego = EgoVehicleState(
        timestamp=10.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0)),
        twist=Twist3D(speed_mps=2.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )
    safe_traj = SafeTrajectory(
        timestamp=10.0,
        source_trajectory_id="s1",
        waypoints=[TrajectoryPoint(timestamp=10.1, x=5.0, y=0.0, speed_mps=6.0, acceleration_mps2=1.0)],
        safety_action=SafetyAction.NONE,
        is_emergency_stop=False,
        barrier_margin_m=5.0,
        min_ttc_seconds=10.0
    )
    cmd = bridge.generate_command(ego, safe_traj, dt=0.05)
    assert cmd.throttle_pct > 20.0
    assert cmd.brake_pct == 0.0

    can = bridge.to_can_frames(cmd)
    assert "CAN_ID_STEERING_0x101" in can
    assert "CAN_ID_THROTTLE_0x102" in can
    assert "CAN_ID_BRAKE_0x103" in can
    assert "CAN_ID_GEAR_0x104" in can


def test_closed_loop_single_step():
    """Verify single-step execution of the complete Sense-Plan-Act closed loop."""
    pipeline = ClosedLoopAutonomyPipeline(target_cruise_speed_mps=5.0, dt=0.05)
    new_state, safe_plan, cmd, telemetry = pipeline.run_step()

    assert new_state.timestamp > 0.0
    assert safe_plan.safety_action in [SafetyAction.NONE, SafetyAction.CORRIDOR_NUDGE, SafetyAction.ADAPTIVE_CRUISE_SLOWDOWN]
    assert cmd.gear.value == "DRIVE"
    assert telemetry["step"] == 1
    assert telemetry["ego"]["speed_kph"] >= 0.0


def test_closed_loop_multi_step_traversal():
    """Verify vehicle executes 40 closed-loop cycles traversing Indian village corridor without collision."""
    geom = RoadGeometry(length_m=100.0, base_width_m=6.5, profile=RoadCorridorProfile.UNMARKED_VILLAGE)
    env = SimulationEnvironment(geometry=geom, dt=0.05)

    # Add oncoming tractor/truck in opposing lane
    tractor = SimulationActor(
        id="tractor_1",
        obstacle_class=ObstacleClass.TRUCK,
        x=50.0,
        y=1.8, # Left side of opposing lane
        speed_mps=-3.0,
        yaw_rad=math.pi,
        length_m=3.8,
        width_m=2.0
    )
    env.add_actor(tractor)

    pipeline = ClosedLoopAutonomyPipeline(environment=env, target_cruise_speed_mps=5.0, dt=0.05)
    env.ego_state.twist.speed_mps = 2.5

    # Run 50 closed-loop steps (2.5 seconds of autonomous driving)
    for _ in range(50):
        pipeline.run_step()

    metrics = pipeline.metrics
    assert metrics.step_count == 50
    assert metrics.total_distance_m > 4.0
    assert metrics.collisions == 0
    assert metrics.min_corridor_margin_m > 0.30
    assert metrics.rms_crosstrack_error_m < 0.65
