"""Comprehensive Unit Tests for Hard Collision Avoidance Safety Supervisor."""
import math
import pytest

from interfaces import (
    EgoVehicleState, Pose3D, Point3D, Twist3D, Vector3D,
    PlannedTrajectory, TrajectoryPoint, BehaviorMode,
    PerceptionOutput, FreeSpaceCorridor, CorridorBoundaryPoint,
    PredictionOutput, PredictedAgent, PredictedTrajectory, PredictedTrajectoryPoint,
    ObstacleClass, MotionIntent, SafetyAction, TrackedObstacle, BoundingBox3D
)
from collision_avoidance.safety_supervisor import SafetySupervisoryLayer


def _create_mock_ego(x: float = 0.0, y: float = 0.0, speed: float = 6.0, yaw: float = 0.0) -> EgoVehicleState:
    return EgoVehicleState(
        timestamp=10.0,
        pose=Pose3D(position=Point3D(x=x, y=y, z=0.0), heading_rad=yaw),
        twist=Twist3D(speed_mps=speed),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )


def test_supervisor_accepts_nominal_safe_trajectory():
    """Verify that a safe trajectory on a clear road passes the safety supervisor with full clearance."""
    supervisor = SafetySupervisoryLayer()
    ego = _create_mock_ego(speed=6.0)

    planned = PlannedTrajectory(
        trajectory_id="plan_nominal",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.0 + i * 0.2, x=i * 1.2, y=0.0, speed_mps=6.0)
            for i in range(1, 15)
        ],
        target_speed_mps=6.0,
        total_cost=5.0,
        is_feasible=True
    )

    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.5, agents=[])

    safe = supervisor.supervise(planned, ego, perc, pred)

    assert safe.is_rejected is False
    assert safe.rejection_reason is None
    assert safe.supervisor_gate_status == "PASSED_SAFE"
    assert safe.safety_action == SafetyAction.NONE
    assert safe.is_emergency_stop is False
    assert safe.replan_recommended is False
    assert len(safe.waypoints) == 14
    assert safe.waypoints[0].speed_mps == 6.0


def test_supervisor_hard_rejection_on_ditch_breach():
    """Verify that a planned trajectory breaching road boundary is REJECTED and replaced with emergency stop."""
    supervisor = SafetySupervisoryLayer()
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)

    # Road corridor boundary is only +/- 1.8m wide
    corridor_pts = [CorridorBoundaryPoint(s=float(s), d_left=1.8, d_right=-1.8) for s in range(0, 30, 2)]
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=corridor_pts, average_width_m=3.6)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.5, agents=[])

    # Planner dangerously swerves into left ditch (y = 2.4m > 1.8m - 0.9m = 0.9m max safe)
    planned_unsafe = PlannedTrajectory(
        trajectory_id="plan_ditch_breach",
        timestamp=10.0,
        behavior_mode=BehaviorMode.NUDGE_LEFT,
        waypoints=[
            TrajectoryPoint(timestamp=10.0 + i * 0.2, x=i * 1.2, y=0.5 + i * 0.25, speed_mps=6.0)
            for i in range(1, 15)
        ],
        target_speed_mps=6.0,
        total_cost=10.0,
        is_feasible=True
    )

    safe = supervisor.supervise(planned_unsafe, ego, perc, pred)

    assert safe.is_rejected is True
    assert safe.rejection_reason == "REJECTED_DITCH_BREACH"
    assert safe.supervisor_gate_status == "REJECTED_AEB"
    assert safe.safety_action == SafetyAction.EMERGENCY_BRAKE
    assert safe.is_emergency_stop is True
    # Fallback waypoints must decelerate to 0
    assert safe.waypoints[-1].speed_mps == 0.0
    assert safe.waypoints[-1].acceleration_mps2 == -6.5


def test_supervisor_hard_rejection_on_critical_ttc():
    """Verify that imminent obstacle collision (TTC < 1.0s) triggers hard rejection and AEB."""
    supervisor = SafetySupervisoryLayer(aeb_ttc_threshold_s=1.0)
    ego = _create_mock_ego(x=0.0, y=0.0, speed=8.0)

    # Obstacle directly ahead at x=3.5m, stationary (TTC = 3.5 / 8.0 = 0.43s < 1.0s)
    obs_tractor = TrackedObstacle(
        id="oncoming_tractor",
        obstacle_class=ObstacleClass.TRUCK,
        confidence=0.99,
        bbox=BoundingBox3D(center=Point3D(x=3.5, y=0.0, z=0.5), size=Vector3D(x=3.0, y=1.8, z=2.0)),
        velocity=Vector3D(x=-6.0, y=0.0, z=0.0),
        distance_m=3.5,
        is_static=False
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs_tractor],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.5, agents=[])

    planned = PlannedTrajectory(
        trajectory_id="plan_straight",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.0 + i * 0.2, x=i * 1.5, y=0.0, speed_mps=8.0)
            for i in range(1, 15)
        ],
        target_speed_mps=8.0,
        total_cost=0.0,
        is_feasible=True
    )

    safe = supervisor.supervise(planned, ego, perc, pred)

    assert safe.is_rejected is True
    assert safe.rejection_reason == "REJECTED_CRITICAL_TTC"
    assert safe.supervisor_gate_status == "REJECTED_AEB"
    assert safe.safety_action == SafetyAction.EMERGENCY_BRAKE
    assert safe.is_emergency_stop is True
    assert safe.min_ttc_seconds < 1.0


def test_supervisor_emergency_replan_on_rapid_cut_in():
    """Verify that rapid cut-in prediction triggers emergency replan and defensive crawl."""
    supervisor = SafetySupervisoryLayer(replan_ttc_threshold_s=2.0)
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)

    # Motorcycle cutting into ego swath at x=6m, y=0.2m
    pred_moto = PredictedAgent(
        id="aggressive_biker",
        obstacle_class=ObstacleClass.MOTORCYCLE,
        primary_intent=MotionIntent.CUTTING_IN,
        is_high_risk=True,
        corridor_invasion_prob=0.85,
        trajectories=[
            PredictedTrajectory(
                probability=0.75,
                mode_name="cut_in",
                waypoints=[
                    PredictedTrajectoryPoint(
                        timestamp=10.0 + step * 0.2,
                        position=Point3D(x=6.0 + step * 0.5, y=0.2, z=0.5),
                        velocity=Vector3D(x=2.5, y=-0.5, z=0.0),
                        yaw_rad=0.0,
                        sigma_x=0.2,
                        sigma_y=0.2
                    )
                    for step in range(1, 15)
                ]
            )
        ]
    )

    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.5, agents=[pred_moto])

    planned = PlannedTrajectory(
        trajectory_id="plan_straight",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.0 + i * 0.2, x=i * 1.2, y=0.0, speed_mps=6.0)
            for i in range(1, 15)
        ],
        target_speed_mps=6.0,
        total_cost=0.0,
        is_feasible=True
    )

    safe = supervisor.supervise(planned, ego, perc, pred)

    assert safe.is_rejected is True
    assert safe.rejection_reason == "REJECTED_UNEXPECTED_INCURSION"
    assert safe.supervisor_gate_status == "REJECTED_REPLAN"
    assert safe.replan_recommended is True
    assert safe.safety_action == SafetyAction.EMERGENCY_REPLAN
    # Waypoints must be defensively slowed
    assert safe.waypoints[-1].speed_mps <= 3.0


def test_supervisor_caution_slowdown():
    """Verify that 2.0s <= TTC <= 4.0s triggers proportional speed modulation without emergency stop."""
    supervisor = SafetySupervisoryLayer(slowdown_ttc_threshold_s=4.0, replan_ttc_threshold_s=2.0)
    ego = _create_mock_ego(x=0.0, y=0.0, speed=7.0)

    # Obstacle ahead at x=21.0m (TTC = 21.0 / 7.0 = 3.0s in caution band)
    obs_slow = TrackedObstacle(
        id="slow_rickshaw",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=21.0, y=0.0, z=0.5), size=Vector3D(x=2.5, y=1.4, z=1.6)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=21.0,
        is_static=True
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs_slow],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.5, agents=[])

    # Trajectory planned up to 12m ahead (safe distance from 21m obstacle)
    planned = PlannedTrajectory(
        trajectory_id="plan_straight",
        timestamp=10.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=10.0 + i * 0.2, x=i * 1.0, y=0.0, speed_mps=7.0)
            for i in range(1, 12)
        ],
        target_speed_mps=7.0,
        total_cost=0.0,
        is_feasible=True
    )

    safe = supervisor.supervise(planned, ego, perc, pred)

    assert safe.is_rejected is False
    assert safe.supervisor_gate_status == "PASSED_WITH_SLOWDOWN"
    assert safe.safety_action == SafetyAction.ADAPTIVE_CRUISE_SLOWDOWN
    assert safe.is_emergency_stop is False
    # Speed should be modulated down
    assert safe.waypoints[-1].speed_mps < 7.0
