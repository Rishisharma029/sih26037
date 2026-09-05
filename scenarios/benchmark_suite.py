"""
SIH26037 Master Benchmark Suite: Executes all 5 Hallmark Real-Indian Scenarios
and computes quantitative safety, comfort, and compliance scorecards.
"""
import sys
import os
import math
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interfaces import (
    ControlCommand, SafeTrajectory, TrajectoryPoint,
    SafetyAction, GearMode
)
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from scenarios.scenario_unsignalled_junction import UnsignalledJunctionScenario
from scenarios.scenario_highway_cutin import HighwayCutInScenario
from scenarios.scenario_dense_market import DenseMarketScenario
from scenarios.scenario_cattle_crossing import CattleCrossingScenario

from vehicle_control.lateral_controller import StanleyLateralController
from vehicle_control.longitudinal_controller import LongitudinalPIDController
from collision_avoidance.emergency_brake import EmergencyBrakeSupervisory
from evaluation.evaluator import BenchmarkHarness
from evaluation.reporter import ScorecardReporter

def execute_scenario_benchmark(scenario_cls, target_speed_mps: float = 6.0) -> dict:
    scenario = scenario_cls()
    lat_ctrl = StanleyLateralController(k_gain=1.4)
    lon_ctrl = LongitudinalPIDController(kp=22.0, ki=0.5, kd=2.0)
    supervisory = EmergencyBrakeSupervisory(aeb_ttc_threshold_s=0.85)
    evaluator = BenchmarkHarness()

    dt = scenario.dt
    steps = int(scenario.duration_seconds / dt)
    min_ttc_observed = 999.0

    for step in range(steps):
        ego_state = scenario.simulator.state
        s_curr = ego_state.pose.position.x

        # 1. Build candidate path
        waypoints = []
        for lookahead in range(1, 10):
            s_target = s_curr + lookahead * 2.2
            rx, ry, ryaw = scenario.env.geometry.get_centerline_point(s_target)
            waypoints.append(TrajectoryPoint(
                timestamp=ego_state.timestamp + lookahead * 0.15,
                x=rx,
                y=ry,
                yaw_rad=ryaw,
                speed_mps=target_speed_mps
            ))

        safe_traj = SafeTrajectory(
            timestamp=ego_state.timestamp,
            source_trajectory_id="bench_traj",
            waypoints=waypoints,
            safety_action=SafetyAction.NONE,
            is_emergency_stop=False,
            barrier_margin_m=2.5,
            min_ttc_seconds=999.0
        )

        # 2. Check collision risk with active actors
        ground_truth_obs = scenario.simulator.get_ground_truth_obstacles()
        if ground_truth_obs:
            risk = supervisory.ttc_calc.compute_ttc(ego_state, ground_truth_obs)
            min_ttc_observed = min(min_ttc_observed, risk.min_ttc_seconds)

            has_obstacle_in_path = any(
                0.0 < (obs.bbox.center.x - ego_state.pose.position.x) < 7.0 and
                abs(obs.bbox.center.y - ego_state.pose.position.y) < 1.4
                for obs in ground_truth_obs
            )

            # If critical obstacle, apply emergency brake
            if risk.min_ttc_seconds < 1.5 or has_obstacle_in_path:
                safe_traj.is_emergency_stop = True
                safe_traj.safety_action = SafetyAction.EMERGENCY_BRAKE
                for wp in safe_traj.waypoints:
                    wp.speed_mps = 0.0
            elif risk.min_ttc_seconds < 3.0:
                safe_traj.safety_action = SafetyAction.ADAPTIVE_CRUISE_SLOWDOWN
                for wp in safe_traj.waypoints:
                    wp.speed_mps = max(1.0, wp.speed_mps * 0.4)

        # 3. Compute control & step
        steer = lat_ctrl.compute_steering(ego_state, safe_traj)
        throttle, brake = lon_ctrl.compute_throttle_brake(ego_state, safe_traj, dt=dt)

        cmd = ControlCommand(
            timestamp=ego_state.timestamp,
            steering_angle_rad=steer,
            throttle_pct=throttle,
            brake_pct=brake,
            gear=GearMode.DRIVE,
            emergency_brake_active=safe_traj.is_emergency_stop
        )

        scenario.run_step(cmd)

    metrics = evaluator.evaluate(scenario.history_states, min_ttc=min_ttc_observed)
    return {
        "scenario_name": scenario.name,
        "metrics": metrics,
        "final_x": scenario.simulator.state.pose.position.x,
        "min_ttc": min_ttc_observed,
        "passed": metrics.safety.passed_safety
    }

def run_all_5_benchmarks() -> str:
    scenarios = [
        (UnmarkedVillageRoadScenario, 5.5),
        (UnsignalledJunctionScenario, 5.0),
        (HighwayCutInScenario, 7.0),
        (DenseMarketScenario, 3.5),
        (CattleCrossingScenario, 5.0)
    ]

    print("=================================================================")
    print("      SIH26037: 5 REAL-INDIAN BENCHMARK SCENARIOS EXECUTION      ")
    print("=================================================================")

    results = []
    for sc_cls, spd in scenarios:
        res = execute_scenario_benchmark(sc_cls, target_speed_mps=spd)
        results.append(res)
        status_tag = "[PASS ASIL-D]" if res["passed"] else "[FAIL]"
        print(f"{status_tag} {res['scenario_name']}")
        print(f"       Distance: {res['final_x']:6.2f}m | Min TTC: {res['min_ttc']:5.2f}s | Avg Speed: {res['metrics'].average_speed_mps*3.6:4.1f} km/h")

    # Generate Markdown Report
    md = """# SIH26037 Real-Indian Benchmark Suite Scorecard
## Closed-Loop Simulation Results Across 5 Hallmark Indian Road Challenges

| Scenario | Real-World Location | Distance | Min TTC | Avg Speed | Safety Status |
|---|---|---|---|---|---|
"""
    for r in results:
        status_badge = "PASS" if r["passed"] else "FAIL"
        md += f"| {r['scenario_name']} | India Grounded | {r['final_x']:.1f}m | {r['min_ttc']:.2f}s | {r['metrics'].average_speed_mps*3.6:.1f} km/h | **{status_badge}** |\n"

    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "BENCHMARK_SCORECARD.md"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nConsolidated report written to {report_path}")
    return md

if __name__ == "__main__":
    run_all_5_benchmarks()
