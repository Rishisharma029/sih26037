"""Unit tests for Road Surface Traversability, Potholes, and Indian Road Anomalies."""
import pytest
import math
from interfaces import (
    TraversabilityClass, RoadAnomaly, Point3D, Pose3D, Twist3D, Vector3D,
    EgoVehicleState, PerceptionOutput, PredictionOutput, FreeSpaceCorridor,
    CorridorBoundaryPoint, PlannedTrajectory, TrajectoryPoint
)
from simulation.environment import VillageRoadGeometry
from planning.cost_evaluator import MultiObjectiveCostEvaluator
from planning.frenet_lattice import FrenetLatticeGenerator, CandidateTrajectory
from collision_avoidance.safety_supervisor import SafetySupervisoryLayer


def test_traversability_class_taxonomy_and_models():
    """Verify the complete 7-class road surface condition taxonomy."""
    assert TraversabilityClass.SAFE == "SAFE"
    assert TraversabilityClass.DEGRADED == "DEGRADED"
    assert TraversabilityClass.POTHOLE == "POTHOLE"
    assert TraversabilityClass.GRAVEL == "GRAVEL"
    assert TraversabilityClass.WATERLOGGED == "WATERLOGGED"
    assert TraversabilityClass.SPEED_BUMP == "SPEED_BUMP"
    assert TraversabilityClass.BLOCKED == "BLOCKED"

    pothole = RoadAnomaly(
        id="crater_1",
        anomaly_type="POTHOLE",
        position=Point3D(x=25.0, y=0.0, z=-0.15),
        radius_m=0.75,
        depth_or_height_m=-0.15,
        traversability_class=TraversabilityClass.POTHOLE,
        severity=0.9,
        is_passable=False,
        max_safe_speed_mps=0.0,
        traversability_score=0.05,
        description="Severe rim-damaging pothole"
    )
    assert not pothole.is_passable
    assert pothole.max_safe_speed_mps == 0.0
    assert pothole.traversability_score == 0.05


def test_spatial_surface_condition_query():
    """Verify VillageRoadGeometry spatial surface condition lookup."""
    geom = VillageRoadGeometry(length_m=200.0)
    anomalies = geom.get_default_anomalies()

    # 1. Pothole location at s=24.0, d=0.45
    t_cls, score, anom = geom.get_surface_condition_at(24.0, 0.45, anomalies)
    assert t_cls == TraversabilityClass.POTHOLE
    assert score == 0.05
    assert anom is not None and anom.id == "pothole_km_24"

    # 2. Gravel location at s=38.0, d=-0.40
    t_cls, score, anom = geom.get_surface_condition_at(38.0, -0.40, anomalies)
    assert t_cls == TraversabilityClass.GRAVEL
    assert score == 0.50

    # 3. Off-road ditch breach
    t_cls, score, _ = geom.get_surface_condition_at(10.0, 5.0, anomalies)
    assert t_cls == TraversabilityClass.BLOCKED
    assert score == 0.0

    # 4. Safe nominal road
    t_cls, score, _ = geom.get_surface_condition_at(10.0, 0.0, anomalies)
    assert t_cls == TraversabilityClass.SAFE
    assert score == 1.0


def test_cost_evaluator_hard_rejection_on_deep_pothole():
    """Verify planner cost evaluator rejects candidate trajectories cutting through deep potholes at speed."""
    evaluator = MultiObjectiveCostEvaluator()

    # Create straight candidate trajectory through (x=10..30, y=0.0) at v=5.0 m/s
    waypoints = [
        TrajectoryPoint(timestamp=i*0.2, x=x, y=0.0, yaw_rad=0.0, curvature=0.0, speed_mps=5.0)
        for i, x in enumerate([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    ]
    candidate = CandidateTrajectory(
        candidate_id="P_center",
        target_d=0.0,
        target_v=5.0,
        horizon_t=2.5,
        waypoints=waypoints,
        label="Center Cruise"
    )

    pothole = RoadAnomaly(
        id="crater_pothole",
        anomaly_type="POTHOLE",
        position=Point3D(x=15.0, y=0.0, z=-0.14),
        radius_m=0.75,
        depth_or_height_m=-0.14,
        traversability_class=TraversabilityClass.POTHOLE,
        severity=0.9,
        is_passable=False,
        max_safe_speed_mps=0.0,
        traversability_score=0.05,
        description="Deep crater pothole (depth 14cm)"
    )

    perception = PerceptionOutput(
        timestamp=0.0,
        frame_id=1,
        obstacles=[],
        drivable_corridor=FreeSpaceCorridor(
            timestamp=0.0,
            boundary_points=[CorridorBoundaryPoint(s=s, d_left=2.2, d_right=-2.2, curvature=0.0) for s in range(0, 35, 5)],
            average_width_m=4.4,
            is_blocked=False
        ),
        anomalies=[pothole]
    )

    prediction = PredictionOutput(
        timestamp=0.0,
        agents=[]
    )

    score = evaluator.evaluate(candidate, perception, prediction, target_cruise_speed_mps=5.0)

    # Must be marked non-feasible with UNSAFE_TRAVERSABILITY tag and explainable string
    assert not score.is_feasible
    assert score.status_tag == "UNSAFE_TRAVERSABILITY"
    assert "Technically open, but unsafe to drive through" in score.explanation
    assert "Deep crater pothole" in score.explanation


def test_cost_evaluator_avoidance_path_selection():
    """Verify that an avoidance trajectory around a waterlogged pool is preferred over center path."""
    evaluator = MultiObjectiveCostEvaluator()

    # Center candidate (crosses pool at y=0.0)
    wps_center = [
        TrajectoryPoint(timestamp=i*0.2, x=x, y=0.0, yaw_rad=0.0, curvature=0.0, speed_mps=4.5)
        for i, x in enumerate([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    ]
    cand_center = CandidateTrajectory(
        candidate_id="P_center",
        target_d=0.0,
        target_v=4.5,
        horizon_t=2.5,
        waypoints=wps_center,
        label="Center"
    )

    # Nudge left candidate (avoids pool at y=1.65m with wide corridor)
    wps_left = [
        TrajectoryPoint(timestamp=i*0.2, x=x, y=min(1.65, x * 0.15), yaw_rad=0.05, curvature=0.02, speed_mps=4.5)
        for i, x in enumerate([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    ]
    cand_left = CandidateTrajectory(
        candidate_id="P_left",
        target_d=1.65,
        target_v=4.5,
        horizon_t=2.5,
        waypoints=wps_left,
        label="Nudge Left"
    )

    waterlog = RoadAnomaly(
        id="submerged_pool",
        anomaly_type="WATER_LOGGING",
        position=Point3D(x=15.0, y=0.0, z=-0.16),
        radius_m=1.0,
        depth_or_height_m=-0.16,
        traversability_class=TraversabilityClass.WATERLOGGED,
        severity=0.9,
        is_passable=False,
        max_safe_speed_mps=0.5,
        traversability_score=0.20,
        description="Flooded muddy depression"
    )

    perception = PerceptionOutput(
        timestamp=0.0,
        frame_id=1,
        obstacles=[],
        drivable_corridor=FreeSpaceCorridor(
            timestamp=0.0,
            boundary_points=[CorridorBoundaryPoint(s=s, d_left=3.2, d_right=-3.2, curvature=0.0) for s in range(0, 30, 5)],
            average_width_m=6.4,
            is_blocked=False
        ),
        anomalies=[waterlog]
    )

    prediction = PredictionOutput(timestamp=0.0, agents=[])

    score_center = evaluator.evaluate(cand_center, perception, prediction, target_cruise_speed_mps=4.5)
    score_left = evaluator.evaluate(cand_left, perception, prediction, target_cruise_speed_mps=4.5)

    assert not score_center.is_feasible
    assert score_center.status_tag == "UNSAFE_TRAVERSABILITY"

    assert score_left.is_feasible
    assert score_left.total_cost < 60.0


def test_safety_supervisor_blocks_unsafe_surface_trajectory():
    """Verify SafetySupervisoryLayer triggers emergency replan if ego trajectory breaches deep pothole at speed."""
    from interfaces import BehaviorMode
    supervisor = SafetySupervisoryLayer()

    ego_state = EgoVehicleState(
        timestamp=1.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=4.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )

    planned = PlannedTrajectory(
        timestamp=1.0,
        trajectory_id="plan_straight",
        behavior_mode=BehaviorMode.CRUISE,
        target_speed_mps=4.0,
        waypoints=[
            TrajectoryPoint(timestamp=1.0 + i*0.2, x=x, y=0.0, yaw_rad=0.0, curvature=0.0, speed_mps=4.0)
            for i, x in enumerate([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        ]
    )

    pothole = RoadAnomaly(
        id="severe_pothole_6m",
        anomaly_type="POTHOLE",
        position=Point3D(x=6.0, y=0.0, z=-0.18),
        radius_m=0.7,
        depth_or_height_m=-0.18,
        traversability_class=TraversabilityClass.POTHOLE,
        severity=0.95,
        is_passable=False,
        max_safe_speed_mps=0.0,
        traversability_score=0.05,
        description="Deep chassis-cracking crater"
    )

    perception = PerceptionOutput(
        timestamp=1.0,
        frame_id=2,
        obstacles=[],
        drivable_corridor=FreeSpaceCorridor(
            timestamp=1.0,
            boundary_points=[CorridorBoundaryPoint(s=s, d_left=2.2, d_right=-2.2, curvature=0.0) for s in range(0, 15, 3)],
            average_width_m=4.4,
            is_blocked=False
        ),
        anomalies=[pothole]
    )

    prediction = PredictionOutput(timestamp=1.0, agents=[])

    safe_traj = supervisor.supervise(planned, ego_state, perception, prediction)

    assert safe_traj.is_rejected
    assert safe_traj.rejection_reason == "REJECTED_UNSAFE_SURFACE"
    assert safe_traj.supervisor_gate_status == "REJECTED_REPLAN"
    assert safe_traj.replan_recommended

    assert safe_traj.rejection_reason == "REJECTED_UNSAFE_SURFACE"
    assert safe_traj.supervisor_gate_status == "REJECTED_REPLAN"
    assert safe_traj.replan_recommended
