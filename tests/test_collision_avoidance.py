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
    ego = _create_mock_ego(speed=4.0)

    # Obstacle located 15m ahead at y=0.5m (TTC = 15/4 = 3.75s, within caution range)
    obs = TrackedObstacle(
        id="road_stone",
        obstacle_class=ObstacleClass.STATIC_DEBRIS,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=15.0, y=0.5, z=0.3), size=Vector3D(x=1.0, y=1.0, z=0.5)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=15.0,
        is_static=True
    )

    planned = PlannedTrajectory(
        trajectory_id="p_nudge",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.2, x=5.0, y=0.0, speed_mps=6.0),
            TrajectoryPoint(timestamp=10.4, x=15.0, y=0.2, speed_mps=6.0), # Too close to obstacle at (15.0, 0.5)
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


def test_ttc_tier_normal_greater_than_4s():
    """Verify TTC > 4.0s maintains normal cruise without speed overrides."""
    supervisor = SafetySupervisoryLayer(aeb_ttc_threshold_s=1.0, replan_ttc_threshold_s=2.0, slowdown_ttc_threshold_s=4.0)
    ego = _create_mock_ego(speed=6.0)
    # Lead obstacle 30m ahead moving at 1m/s -> closing speed 5m/s -> TTC = 6.0s (> 4.0s)
    obs = TrackedObstacle(
        id="distant_car",
        obstacle_class=ObstacleClass.CAR,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=30.0, y=0.0, z=0.5), size=Vector3D(x=4.0, y=1.8, z=1.5)),
        velocity=Vector3D(x=-5.0, y=0.0, z=0.0),
        distance_m=30.0,
        is_static=False
    )
    planned = PlannedTrajectory(
        trajectory_id="p_normal",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[TrajectoryPoint(timestamp=10.2, x=2.0, y=0.0, speed_mps=6.0)],
        target_speed_mps=6.0
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    safe = supervisor.supervise(planned, ego, perc, PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[]))
    assert safe.safety_action == SafetyAction.NONE
    assert safe.is_emergency_stop is False
    assert safe.replan_recommended is False
    assert safe.waypoints[0].speed_mps == 6.0


def test_ttc_tier_slowdown_caution_2_to_4s():
    """Verify 2.0s <= TTC <= 4.0s triggers ADAPTIVE_CRUISE_SLOWDOWN and trims waypoint speed."""
    supervisor = SafetySupervisoryLayer(aeb_ttc_threshold_s=1.0, replan_ttc_threshold_s=2.0, slowdown_ttc_threshold_s=4.0)
    ego = _create_mock_ego(speed=8.0)
    # Obstacle 15m ahead closing at 5.0 m/s -> TTC = 3.0s (in [2.0s, 4.0s])
    obs = TrackedObstacle(
        id="slowing_auto",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=15.0, y=0.0, z=0.5), size=Vector3D(x=2.5, y=1.3, z=1.6)),
        velocity=Vector3D(x=-5.0, y=0.0, z=0.0),
        distance_m=15.0,
        is_static=False
    )
    planned = PlannedTrajectory(
        trajectory_id="p_fast",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[TrajectoryPoint(timestamp=10.2, x=2.0, y=0.0, speed_mps=8.0)],
        target_speed_mps=8.0
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    safe = supervisor.supervise(planned, ego, perc, PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[]))
    assert safe.safety_action == SafetyAction.ADAPTIVE_CRUISE_SLOWDOWN
    assert safe.is_emergency_stop is False
    assert safe.replan_recommended is False
    # Speed should be clamped down from 8.0 m/s
    assert safe.waypoints[0].speed_mps < 8.0
    assert safe.waypoints[0].speed_mps <= 3.0 * 1.35 + 0.1


def test_ttc_tier_emergency_replan_1_to_2s():
    """Verify 1.0s <= TTC < 2.0s triggers EMERGENCY_REPLAN with defensive deceleration."""
    supervisor = SafetySupervisoryLayer(aeb_ttc_threshold_s=1.0, replan_ttc_threshold_s=2.0, slowdown_ttc_threshold_s=4.0)
    ego = _create_mock_ego(speed=8.0)
    # Obstacle 9m ahead closing at 6.0 m/s -> TTC = 1.5s (in [1.0s, 2.0s))
    obs = TrackedObstacle(
        id="cutting_motorcycle",
        obstacle_class=ObstacleClass.MOTORCYCLE,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=9.0, y=0.0, z=0.5), size=Vector3D(x=2.0, y=0.8, z=1.2)),
        velocity=Vector3D(x=-6.0, y=0.0, z=0.0),
        distance_m=9.0,
        is_static=False
    )
    planned = PlannedTrajectory(
        trajectory_id="p_high_speed",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[TrajectoryPoint(timestamp=10.2, x=2.0, y=0.0, speed_mps=8.0)],
        target_speed_mps=8.0
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    safe = supervisor.supervise(planned, ego, perc, PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[]))
    assert safe.safety_action == SafetyAction.EMERGENCY_REPLAN
    assert safe.replan_recommended is True
    assert safe.is_emergency_stop is False
    assert safe.waypoints[0].speed_mps <= 4.0
    assert safe.waypoints[0].acceleration_mps2 <= -2.0


def test_ttc_tier_emergency_brake_less_than_1s():
    """Verify TTC < 1.0s triggers EMERGENCY_BRAKE with zero speed and max braking."""
    supervisor = SafetySupervisoryLayer(aeb_ttc_threshold_s=1.0, replan_ttc_threshold_s=2.0, slowdown_ttc_threshold_s=4.0)
    ego = _create_mock_ego(speed=8.0)
    # Obstacle 4m ahead closing at 6.0 m/s -> TTC = 0.67s (< 1.0s)
    obs = TrackedObstacle(
        id="stopped_truck",
        obstacle_class=ObstacleClass.TRUCK,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=4.0, y=0.0, z=1.0), size=Vector3D(x=6.0, y=2.2, z=2.5)),
        velocity=Vector3D(x=-6.0, y=0.0, z=0.0),
        distance_m=4.0,
        is_static=False
    )
    planned = PlannedTrajectory(
        trajectory_id="p_unsafe",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[TrajectoryPoint(timestamp=10.2, x=2.0, y=0.0, speed_mps=8.0)],
        target_speed_mps=8.0
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    safe = supervisor.supervise(planned, ego, perc, PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[]))
    assert safe.safety_action == SafetyAction.EMERGENCY_BRAKE
    assert safe.is_emergency_stop is True
    assert safe.waypoints[0].speed_mps == 0.0
    assert safe.waypoints[0].acceleration_mps2 <= -6.0


def test_safety_overrides_unsafe_high_speed_plan():
    """Verify that even if planner proposes a reckless 15 m/s trajectory through an obstacle, safety overrides it."""
    supervisor = SafetySupervisoryLayer()
    ego = _create_mock_ego(speed=12.0)
    # Static obstacle right in front at 5m
    obs = TrackedObstacle(
        id="debris_hazard",
        obstacle_class=ObstacleClass.STATIC_DEBRIS,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=5.0, y=0.0, z=0.3), size=Vector3D(x=1.2, y=1.2, z=0.5)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=5.0,
        is_static=True
    )
    reckless_plan = PlannedTrajectory(
        trajectory_id="p_reckless",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.1, x=1.5, y=0.0, speed_mps=15.0),
            TrajectoryPoint(timestamp=10.2, x=3.0, y=0.0, speed_mps=15.0),
            TrajectoryPoint(timestamp=10.3, x=5.0, y=0.0, speed_mps=15.0), # Crash waypoint
        ],
        target_speed_mps=15.0
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    safe = supervisor.supervise(reckless_plan, ego, perc, PredictionOutput(timestamp=10.0, horizon_seconds=2.0, agents=[]))
    
    # Safety MUST override reckless plan
    assert safe.safety_action == SafetyAction.EMERGENCY_BRAKE
    assert safe.is_emergency_stop is True
    assert all(wp.speed_mps == 0.0 for wp in safe.waypoints)

