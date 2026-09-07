"""Comprehensive Unit Tests for Real Road Corridor & Hard Safety Invariant."""
import math
import pytest
from interfaces import (
    Point3D, Vector3D, Pose3D, Twist3D, PlannedTrajectory,
    TrajectoryPoint, BehaviorMode, PerceptionOutput, FreeSpaceCorridor,
    PredictionOutput, SafetyAction, EgoVehicleState
)
from simulation.environment import VillageRoadGeometry, RoadCorridorProfile
from perception.boundary_detector import FreeSpaceBoundaryDetector
from planning.cost_evaluator import TrajectoryCostEvaluator
from planning.frenet_lattice import CandidateTrajectory
from collision_avoidance.control_barrier_functions import ControlBarrierFilter
from collision_avoidance.safety_supervisor import SafetySupervisoryLayer


def test_corridor_boundary_detection_with_geometry():
    """Verify that FreeSpaceBoundaryDetector generates accurate irregular boundary points."""
    geom = VillageRoadGeometry(length_m=250.0, base_width_m=4.3)
    detector = FreeSpaceBoundaryDetector()

    corridor = detector.detect_corridor(
        timestamp=1.0,
        lookahead_m=30.0,
        step_m=5.0,
        current_s=50.0,
        geometry=geom
    )

    assert len(corridor.boundary_points) == 7 # 0, 5, 10, 15, 20, 25, 30m
    assert abs(corridor.average_width_m - 4.3) < 0.5
    for bp in corridor.boundary_points:
        assert bp.d_left > 1.8
        assert bp.d_right < -1.8
        assert bp.d_left > bp.d_right


def test_frenet_cartesian_projection_invariance():
    """Verify Cartesian <-> Frenet projection consistency across straight and curved road."""
    geom = VillageRoadGeometry(length_m=250.0)

    # 1. Straight section (s=45m, d=+0.5m left)
    x, y, yaw = geom.frenet_to_cartesian(s=45.0, d=0.5)
    assert abs(x - 45.0) < 1e-3
    assert abs(y - 0.5) < 1e-3
    assert abs(yaw - 0.0) < 1e-3

    s_rec, d_rec = geom.cartesian_to_frenet(x, y)
    assert abs(s_rec - 45.0) < 1e-2
    assert abs(d_rec - 0.5) < 1e-2

    # 2. Curved section (s=135m, d=-0.3m right)
    xc, yc, yawc = geom.frenet_to_cartesian(s=135.0, d=-0.3)
    s_curv, d_curv = geom.cartesian_to_frenet(xc, yc, s_guess=135.0)
    assert abs(s_curv - 135.0) < 0.2
    assert abs(d_curv - (-0.3)) < 0.1


def test_ditch_margin_calculation():
    """Verify ditch margin calculation for inside and outside positions."""
    geom = VillageRoadGeometry(length_m=250.0, base_width_m=4.4)
    # At s=0, road is ~4.4m wide (d_left ~ +2.2, d_right ~ -2.2)
    # Vehicle half width = 0.9m
    # Vehicle centered at d=0.0 -> margin should be ~ 2.2 - 0.9 = 1.3m
    margin_center = geom.get_ditch_margin(s=0.0, d=0.0, vehicle_half_width=0.9)
    assert abs(margin_center - 1.3) < 0.2
    assert margin_center > 1.0 # Deep inside drivable road

    # Vehicle dangerously close to left edge (d = +1.2m)
    # margin = 2.2 - (1.2 + 0.9) = 0.1m
    margin_edge = geom.get_ditch_margin(s=0.0, d=1.2, vehicle_half_width=0.9)
    assert margin_edge < 0.35
    assert margin_edge > 0.0

    # Vehicle in ditch (d = +2.0m) -> margin = 2.2 - (2.0 + 0.9) = -0.7m < 0
    margin_ditch = geom.get_ditch_margin(s=0.0, d=2.0, vehicle_half_width=0.9)
    assert margin_ditch < 0.0 # Negative margin indicates ditch breach!


def test_hard_corridor_safety_invariant_rejection():
    """Verify TrajectoryCostEvaluator rejects candidate trajectories that breach road boundary."""
    evaluator = TrajectoryCostEvaluator()

    detector = FreeSpaceBoundaryDetector(default_width_m=4.4)
    corridor = detector.detect_corridor(timestamp=1.0, lookahead_m=30.0, step_m=5.0)
    perception = PerceptionOutput(timestamp=1.0, frame_id=1, obstacles=[], drivable_corridor=corridor)
    prediction = PredictionOutput(timestamp=1.0, horizon_seconds=2.0, agents=[])

    # 1. Valid Candidate within road (y=0.2m)
    cand_safe = CandidateTrajectory(
        candidate_id="cand_safe",
        target_d=0.2,
        target_v=6.0,
        horizon_t=2.0,
        waypoints=[
            TrajectoryPoint(timestamp=1.1, x=2.0, y=0.0, speed_mps=6.0),
            TrajectoryPoint(timestamp=1.2, x=5.0, y=0.2, speed_mps=6.0),
            TrajectoryPoint(timestamp=1.4, x=10.0, y=0.2, speed_mps=6.0),
        ]
    )
    score_safe = evaluator.evaluate(cand_safe, perception, prediction, 6.0)
    assert score_safe.is_feasible is True
    assert score_safe.total_cost < 9000.0

    # 2. Invalid Candidate swerving into ditch (y=2.5m -> exceeds d_left ~ 2.2m)
    cand_ditch = CandidateTrajectory(
        candidate_id="cand_ditch",
        target_d=2.5,
        target_v=6.0,
        horizon_t=2.0,
        waypoints=[
            TrajectoryPoint(timestamp=1.1, x=2.0, y=1.0, speed_mps=6.0),
            TrajectoryPoint(timestamp=1.2, x=5.0, y=2.0, speed_mps=6.0),
            TrajectoryPoint(timestamp=1.4, x=10.0, y=2.5, speed_mps=6.0), # Outside corridor!
        ]
    )
    score_ditch = evaluator.evaluate(cand_ditch, perception, prediction, 6.0)
    assert score_ditch.is_feasible is False
    assert score_ditch.total_cost >= 99999.0
    assert "traversability" in score_ditch.cost_breakdown


def test_control_barrier_ditch_override():
    """Verify ControlBarrierFilter halts or overrides speed on corridor breach."""
    cbf = ControlBarrierFilter()
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=6.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )

    detector = FreeSpaceBoundaryDetector(default_width_m=4.4)
    corridor = detector.detect_corridor(timestamp=0.0, lookahead_m=30.0, step_m=5.0)
    perception = PerceptionOutput(timestamp=0.0, frame_id=1, obstacles=[], drivable_corridor=corridor)

    # Trajectory that breaches ditch verge at y=2.5m
    traj_breach = PlannedTrajectory(
        trajectory_id="p_ditch",
        timestamp=0.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[
            TrajectoryPoint(timestamp=0.2, x=3.0, y=1.5, speed_mps=6.0),
            TrajectoryPoint(timestamp=0.4, x=6.0, y=2.6, speed_mps=6.0), # Critical breach
        ],
        target_speed_mps=6.0
    )

    filtered_traj, violated, margin = cbf.filter_trajectory(traj_breach, ego, perception)
    assert violated is True
    # Speed at critical breach waypoint must be commanded to 0.0 with emergency braking
    assert filtered_traj.waypoints[1].speed_mps == 0.0
    assert filtered_traj.waypoints[1].acceleration_mps2 <= -5.0
