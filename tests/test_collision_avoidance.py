"""Comprehensive Unit & Stress Tests for Phase 7 Dedicated Safety Supervisory Layer."""
import math
import pytest

from interfaces import (
    TrackedObstacle, ObstacleClass, BoundingBox3D, Point3D, Vector3D,
    EgoVehicleState, Pose3D, Twist3D, PlannedTrajectory, TrajectoryPoint,
    BehaviorMode, PerceptionOutput, FreeSpaceCorridor, PredictionOutput,
    PredictedAgent, PredictedTrajectory, PredictedTrajectoryPoint,
    MotionIntent, SafetyAction
)
from collision_avoidance.ttc_calculator import TTCCalculator
from collision_avoidance.control_barrier_functions import ControlBarrierFilter
from collision_avoidance.safety_supervisor import SafetySupervisoryLayer


def _create_mock_ego(speed: float = 8.0) -> EgoVehicleState:
    return EgoVehicleState(
        timestamp=10.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=speed),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )


def test_ttc_calculation():
    """Verify 2D relative speed TTC computation."""
    calc = TTCCalculator(critical_threshold_s=0.85)
    ego = _create_mock_ego(speed=10.0)
    obs = TrackedObstacle(
        id="lead_car",
        obstacle_class=ObstacleClass.CAR,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=10.0, y=0.0, z=0.5), size=Vector3D(x=4.0, y=1.8, z=1.5)),
        velocity=Vector3D(x=5.0, y=0.0, z=0.0),
        distance_m=10.0,
        is_static=False
    )
    risk = calc.compute_ttc(ego, [obs])
    # Relative dx=10m, relative vx=5m/s -> TTC=2.0s
    assert abs(risk.min_ttc_seconds - 2.0) < 0.1


def test_aeb_trigger():
    """Verify immediate Autonomous Emergency Braking (AEB) when TTC < threshold."""
    supervisor = SafetySupervisoryLayer(aeb_ttc_threshold_s=1.0)
    ego = _create_mock_ego(speed=10.0)

    # Obstacle 4m ahead stationary -> TTC = 0.4s (Critical)
    obs = TrackedObstacle(
        id="darting_ped",
        obstacle_class=ObstacleClass.PEDESTRIAN,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=4.0, y=0.0, z=0.8), size=Vector3D(x=0.5, y=0.5, z=1.7)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=4.0,
        is_static=True
    )
    planned = PlannedTrajectory(
        trajectory_id="p1",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[TrajectoryPoint(timestamp=10.1, x=1.0, y=0.0, speed_mps=10.0)],
        target_speed_mps=10.0
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[])
    safe = supervisor.supervise(planned, ego, perc, pred)

    assert safe.is_emergency_stop is True
    assert safe.safety_action == SafetyAction.EMERGENCY_BRAKE
    assert safe.waypoints[0].speed_mps == 0.0
    assert safe.waypoints[0].acceleration_mps2 <= -6.0


def test_cbf_barrier_override():
    """Verify Control Barrier Function (CBF) overrides speed when margin breaches minimum threshold."""
    supervisor = SafetySupervisoryLayer(min_barrier_dist_m=2.0)
    ego = _create_mock_ego(speed=5.0)

    # Obstacle located close to waypoint (dx=1.0m, dy=0.5m -> dist ~ 1.1m < 2.0m)
    obs = TrackedObstacle(
        id="road_stone",
        obstacle_class=ObstacleClass.STATIC_DEBRIS,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=5.0, y=0.5, z=0.3), size=Vector3D(x=1.0, y=1.0, z=0.5)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=5.0,
        is_static=True
    )

    planned = PlannedTrajectory(
        trajectory_id="p_nudge",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.2, x=2.5, y=0.0, speed_mps=6.0),
            TrajectoryPoint(timestamp=10.4, x=5.0, y=0.2, speed_mps=6.0), # Too close to obstacle at (5.0, 0.5)
        ],
        target_speed_mps=6.0
    )

    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[])

    safe = supervisor.supervise(planned, ego, perc, pred)

    assert safe.safety_action == SafetyAction.CONTROL_BARRIER_OVERRIDE
    # Speed at hazardous waypoint is clipped to safe crawl
    assert safe.waypoints[1].speed_mps <= 1.5


def test_unexpected_obstacle_incursion_replan():
    """Verify sudden cutting-in vehicle triggers EMERGENCY_REPLAN."""
    supervisor = SafetySupervisoryLayer()
    ego = _create_mock_ego(speed=7.0)

    # Cutting in auto-rickshaw encroaching on planned path
    rick_incursion = PredictedAgent(
        id="rick_swerving",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        primary_intent=MotionIntent.CUTTING_IN,
        is_high_risk=True,
        trajectories=[
            PredictedTrajectory(
                probability=0.8,
                mode_name="cut_in",
                waypoints=[
                    PredictedTrajectoryPoint(
                        timestamp=10.2,
                        position=Point3D(x=3.0, y=0.2, z=0.5), # Encroaching into center path at x=3
                        velocity=Vector3D(x=4.0, y=-1.5, z=0.0),
                        yaw_rad=0.0
                    ),
                    PredictedTrajectoryPoint(
                        timestamp=10.4,
                        position=Point3D(x=5.5, y=0.0, z=0.5),
                        velocity=Vector3D(x=4.0, y=-1.5, z=0.0),
                        yaw_rad=0.0
                    )
                ]
            )
        ]
    )

    planned = PlannedTrajectory(
        trajectory_id="p_cruise",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.2, x=3.0, y=0.0, speed_mps=7.0),
            TrajectoryPoint(timestamp=10.4, x=5.5, y=0.0, speed_mps=7.0),
        ],
        target_speed_mps=7.0
    )

    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[rick_incursion])

    safe = supervisor.supervise(planned, ego, perc, pred)

    assert safe.safety_action == SafetyAction.EMERGENCY_REPLAN
    assert safe.replan_recommended is True
    assert "UNEXPECTED_OBSTACLE_INCURSION" in safe.safety_status_reason


def test_nominal_safe_passthrough():
    """Verify clean pass-through when planned trajectory has clear envelope."""
    supervisor = SafetySupervisoryLayer()
    ego = _create_mock_ego(speed=6.0)

    planned = PlannedTrajectory(
        trajectory_id="p_clear",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.2, x=2.0, y=0.0, speed_mps=6.0),
            TrajectoryPoint(timestamp=10.4, x=4.0, y=0.0, speed_mps=6.0),
        ],
        target_speed_mps=6.0
    )

    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.5)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[])

    safe = supervisor.supervise(planned, ego, perc, pred)

    assert safe.safety_action == SafetyAction.NONE
    assert safe.is_emergency_stop is False
    assert safe.replan_recommended is False
    assert safe.safety_status_reason == "TRAJECTORY_VERIFIED_SAFE"
