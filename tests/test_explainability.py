"""Unit tests for Phase 12 Explainability Flight Recorder."""
import pytest
from interfaces import (
    EgoVehicleState, Point3D, Vector3D, Twist3D, Pose3D,
    PerceptionOutput, PredictionOutput, FreeSpaceCorridor,
    PlannedTrajectory, SafeTrajectory, SafetyAction, BehaviorMode,
    TrackedObstacle, BoundingBox3D, ObstacleClass, TrajectoryPoint
)
from explainability.flight_recorder import FlightRecorder
from explainability.audit_exporter import AuditLogExporter
from explainability.types import RiskLevel, DecisionEvent


def test_flight_recorder_event_generation():
    recorder = FlightRecorder(scenario_name="01_village_road")

    ego = EgoVehicleState(
        timestamp=1.2,
        pose=Pose3D(position=Point3D(x=10.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(linear=Vector3D(x=5.0, y=0.0, z=0.0), speed_mps=5.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        steer_angle_rad=0.0,
        battery_soc_pct=95.0
    )

    motorcycle = TrackedObstacle(
        id="moto_1",
        obstacle_class=ObstacleClass.MOTORCYCLE,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=18.0, y=1.0, z=0.5), size=Vector3D(x=2.0, y=0.8, z=1.2)),
        velocity=Vector3D(x=-3.0, y=0.0, z=0.0),
        distance_m=8.0,
        is_static=False
    )

    perception = PerceptionOutput(
        timestamp=1.2,
        frame_id=24,
        obstacles=[motorcycle],
        drivable_corridor=FreeSpaceCorridor(timestamp=1.2, average_width_m=4.5),
        anomalies=[]
    )
    prediction = PredictionOutput(timestamp=1.2, horizon_seconds=3.0, agents=[])

    planned = PlannedTrajectory(
        trajectory_id="Path #4",
        timestamp=1.2,
        behavior_mode=BehaviorMode.NUDGE_LEFT,
        waypoints=[TrajectoryPoint(timestamp=1.4, x=11.0, y=0.4, yaw_rad=0.05, curvature=0.02, speed_mps=4.5)],
        target_speed_mps=4.5,
        total_cost=12.5,
        is_feasible=True
    )

    safe = SafeTrajectory(
        timestamp=1.2,
        source_trajectory_id="Path #4",
        waypoints=planned.waypoints,
        safety_action=SafetyAction.CORRIDOR_NUDGE,
        barrier_margin_m=2.1,
        min_ttc_seconds=2.31
    )

    event = recorder.record_decision(ego, perception, prediction, planned, safe, candidate_scores=[1, 2, 3, 4, 5, 6, 7])

    assert event.event_id == 1
    assert "Motorcycle" in event.hazard.hazard_type
    assert event.hazard.ttc_seconds == pytest.approx(2.31)
    assert event.total_candidates_evaluated == 7
    assert event.selected_candidate_id == "Path #4"
    assert "safety margin" in event.rationale.primary_reason.lower()

    # Verify event card format
    card = event.format_event_card()
    assert "EVENT #1" in card
    assert "Hazard:" in card
    assert "Reason:" in card


def test_audit_log_exporter_markdown():
    recorder = FlightRecorder()
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(linear=Vector3D(x=0.0, y=0.0, z=0.0), speed_mps=0.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        steer_angle_rad=0.0,
        battery_soc_pct=95.0
    )
    perception = PerceptionOutput(
        timestamp=0.0,
        frame_id=1,
        obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=0.0, average_width_m=4.5),
        anomalies=[]
    )
    prediction = PredictionOutput(timestamp=0.0, horizon_seconds=3.0, agents=[])
    planned = PlannedTrajectory(
        trajectory_id="Path #1",
        timestamp=0.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[],
        target_speed_mps=6.0,
        total_cost=5.0,
        is_feasible=True
    )
    safe = SafeTrajectory(
        timestamp=0.0,
        source_trajectory_id="Path #1",
        waypoints=[],
        safety_action=SafetyAction.NONE,
        barrier_margin_m=5.0,
        min_ttc_seconds=10.0
    )

    ev = recorder.record_decision(ego, perception, prediction, planned, safe)
    md = AuditLogExporter.export_markdown([ev], title="Test Audit")
    assert "Autonomous Decision Audit Log" in md
    assert "EVENT #1" in md
