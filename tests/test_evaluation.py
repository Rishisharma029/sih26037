"""Comprehensive unit and integration tests for Phase 10 Evaluation System."""
import pytest
import math
from interfaces import (
    EgoVehicleState, Point3D, Vector3D, Twist3D, Pose3D,
    PerceptionOutput, PredictionOutput, FreeSpaceCorridor
)
from evaluation.metrics import (
    compute_path_smoothness, compute_velocity_std,
    FullEvaluationResult, SafetyClearanceMetrics,
    KinematicSmoothnessMetrics, ComputationalMetrics, ScenarioMissionMetrics
)
from evaluation.evaluator import BenchmarkHarness
from evaluation.reporter import ScorecardReporter
from planning.baseline_planner import BaselinePlanner


def _create_mock_states(num_steps: int = 20, dt: float = 0.05) -> list:
    states = []
    for i in range(num_steps):
        t = i * dt
        x = t * 5.0
        y = 0.1 * math.sin(x * 0.2)
        states.append(EgoVehicleState(
            timestamp=t,
            pose=Pose3D(position=Point3D(x=x, y=y, z=0.0), heading_rad=0.02),
            twist=Twist3D(linear=Vector3D(x=5.0 + 0.1 * math.sin(t), y=0.0, z=0.0), speed_mps=5.0 + 0.1 * math.sin(t)),
            acceleration=Vector3D(x=0.1, y=0.05, z=0.0),
            steer_angle_rad=0.01,
            battery_soc_pct=95.0
        ))
    return states


def test_path_smoothness_and_speed_stability_math():
    straight_line = [(float(i), 0.0) for i in range(20)]
    assert compute_path_smoothness(straight_line) == pytest.approx(0.0, abs=1e-5)

    curved_line = [(float(i), math.sin(i * 0.2)) for i in range(20)]
    assert compute_path_smoothness(curved_line) > 0.0

    constant_speeds = [5.0] * 10
    assert compute_velocity_std(constant_speeds) == pytest.approx(0.0, abs=1e-5)

    varying_speeds = [4.0, 5.0, 6.0, 5.0, 4.0]
    assert compute_velocity_std(varying_speeds) > 0.0


def test_evaluator_harness_output_completeness():
    harness = BenchmarkHarness()
    states = _create_mock_states()
    clearances = [2.5, 2.3, 2.0, 1.8, 2.2]
    ttcs = [5.0, 4.5, 4.0, 3.5, 4.0]
    latencies = [3.2, 4.1, 2.9, 3.5]

    res = harness.evaluate(
        scenario_name="test_scenario",
        difficulty="MEDIUM",
        planner_type="Adaptive Planner",
        states=states,
        min_clearance_records=clearances,
        ttc_records=ttcs,
        latencies_ms=latencies,
        aeb_triggers=0,
        emergency_replans=0
    )

    assert isinstance(res, FullEvaluationResult)
    assert res.safety.passed_safety is True
    assert res.safety.min_clearance_m == pytest.approx(1.8)
    assert res.safety.min_ttc_s == pytest.approx(3.5)
    assert res.computational.mean_latency_ms > 0.0
    assert res.kinematics.passed_comfort is True
    assert res.mission.composite_score > 70.0


def test_baseline_planner_execution():
    planner = BaselinePlanner(horizon_seconds=2.0, dt=0.2)
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(linear=Vector3D(x=6.0, y=0.0, z=0.0), speed_mps=6.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        steer_angle_rad=0.0,
        battery_soc_pct=95.0
    )
    dummy_perception = PerceptionOutput(
        timestamp=0.0,
        frame_id=1,
        obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=0.0, average_width_m=4.5),
        anomalies=[]
    )
    dummy_prediction = PredictionOutput(timestamp=0.0, horizon_seconds=3.0, agents=[])

    traj = planner.plan(ego, dummy_perception, dummy_prediction, target_cruise_speed_mps=6.0)
    assert traj.is_feasible is True
    assert len(traj.waypoints) == 10
    assert traj.waypoints[-1].x > 0.0


def test_comparative_scorecard_markdown_generation():
    harness = BenchmarkHarness()
    states = _create_mock_states()
    res_base = harness.evaluate("01_village_road", "MEDIUM", "Baseline Planner", states, [1.0], [2.0], [1.0])
    res_adapt = harness.evaluate("01_village_road", "MEDIUM", "Adaptive Planner", states, [2.2], [4.5], [3.5])

    md = ScorecardReporter.generate_comparative_markdown([res_base], [res_adapt])
    assert "Head-to-Head Evaluation Report" in md
    assert "Baseline Planner" in md
    assert "Adaptive Planner" in md
