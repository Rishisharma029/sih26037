"""Advanced Automated Benchmark Evaluator Harness."""
import math
import time
from typing import List, Optional
from interfaces import EgoVehicleState, SafeTrajectory, PlannedTrajectory
from .metrics import (
    FullEvaluationResult, ScenarioMissionMetrics,
    SafetyClearanceMetrics, KinematicSmoothnessMetrics,
    ComputationalMetrics, compute_path_smoothness, compute_velocity_std
)


class BenchmarkHarness:
    """Evaluates closed-loop autonomous driving runs against full metric specifications."""

    def evaluate(
        self,
        scenario_name: str,
        difficulty: str,
        planner_type: str,
        states: List[EgoVehicleState],
        min_clearance_records: List[float],
        ttc_records: List[float],
        latencies_ms: List[float],
        aeb_triggers: int = 0,
        emergency_replans: int = 0,
        target_distance_m: float = 50.0
    ) -> FullEvaluationResult:
        """Compute full comprehensive metric set from episode telemetry."""
        total_steps = len(states)
        if total_steps == 0:
            raise ValueError("Telemetry states list cannot be empty for evaluation.")

        # 1. Mission Metrics
        init_state = states[0]
        final_state = states[-1]
        completion_time_s = final_state.timestamp - init_state.timestamp
        total_distance = math.hypot(
            final_state.pose.position.x - init_state.pose.position.x,
            final_state.pose.position.y - init_state.pose.position.y
        )
        completion_rate = min(100.0, (total_distance / max(1.0, target_distance_m)) * 100.0)
        speeds = [s.twist.speed_mps for s in states]
        avg_speed = sum(speeds) / len(speeds)

        # 2. Safety & Clearance Metrics
        valid_clearances = [c for c in min_clearance_records if c < 100.0]
        min_clearance = min(valid_clearances) if valid_clearances else 10.0
        mean_clearance = sum(valid_clearances) / max(1, len(valid_clearances)) if valid_clearances else 10.0

        valid_ttcs = [t for t in ttc_records if t < 100.0]
        min_ttc = min(valid_ttcs) if valid_ttcs else 10.0
        mean_ttc = sum(valid_ttcs) / max(1, len(valid_ttcs)) if valid_ttcs else 10.0

        # Collisions: minimum clearance < 0.2m
        collisions = sum(1 for c in valid_clearances if c < 0.20)
        collision_rate = (collisions / max(1, len(valid_clearances))) * 100.0

        # Near-collisions: clearance < 0.8m or TTC < 1.8s
        near_collisions = sum(1 for i, c in enumerate(min_clearance_records) if c < 0.80 or (i < len(ttc_records) and ttc_records[i] < 1.8))
        near_collision_rate = (near_collisions / max(1, len(min_clearance_records))) * 100.0

        boundary_violations = sum(1 for s in states if abs(s.pose.position.y) > 2.6)  # Outside standard corridor
        passed_safety = (collisions == 0) and (min_ttc >= 1.0) and (boundary_violations == 0)

        safety_metrics = SafetyClearanceMetrics(
            min_clearance_m=min_clearance,
            mean_clearance_m=mean_clearance,
            min_ttc_s=min_ttc,
            mean_ttc_s=mean_ttc,
            collision_count=collisions,
            collision_rate_pct=collision_rate,
            near_collision_count=near_collisions,
            near_collision_rate_pct=near_collision_rate,
            boundary_violations=boundary_violations,
            passed_safety=passed_safety
        )

        # 3. Kinematic Smoothness Metrics
        traj_points = [(s.pose.position.x, s.pose.position.y) for s in states]
        smoothness = compute_path_smoothness(traj_points)
        speed_std = compute_velocity_std(speeds)

        max_lat_accel = 0.0
        max_decel = 0.0
        lateral_errors = [abs(s.pose.position.y) for s in states]
        rms_cte = math.sqrt(sum(e ** 2 for e in lateral_errors) / len(lateral_errors))

        jerks = []
        for i in range(1, len(states)):
            dt = max(1e-3, states[i].timestamp - states[i-1].timestamp)
            da_lat = (states[i].acceleration.y - states[i-1].acceleration.y) / dt
            jerks.append(abs(da_lat))
            if abs(states[i].acceleration.y) > max_lat_accel:
                max_lat_accel = abs(states[i].acceleration.y)
            if states[i].acceleration.x < -max_decel:
                max_decel = abs(states[i].acceleration.x)

        mean_jerk = sum(jerks) / max(1, len(jerks)) if jerks else 0.0
        max_jerk = max(jerks) if jerks else 0.0
        passed_comfort = (max_decel <= 5.5) and (max_lat_accel <= 3.0) and (max_jerk <= 6.0)

        kinematics_metrics = KinematicSmoothnessMetrics(
            path_length_m=total_distance,
            path_smoothness_rad_m=smoothness,
            mean_lateral_jerk_mps3=mean_jerk,
            max_lateral_jerk_mps3=max_jerk,
            max_lat_accel_mps2=max_lat_accel,
            max_lon_decel_mps2=max_decel,
            lateral_deviation_rms_m=rms_cte,
            speed_stability_mps=speed_std,
            passed_comfort=passed_comfort
        )

        # 4. Computational Metrics
        latencies = latencies_ms if latencies_ms else [10.0]
        mean_lat = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))
        p95_lat = sorted_lat[p95_idx]
        max_lat = max(latencies)
        frequency = 1000.0 / max(1e-2, mean_lat)

        comp_metrics = ComputationalMetrics(
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            max_latency_ms=max_lat,
            planning_frequency_hz=min(50.0, frequency),
            aeb_intervention_count=aeb_triggers,
            emergency_replan_count=emergency_replans
        )

        # 5. Composite Score Calculation (0 - 100)
        safety_pts = 40.0 if passed_safety else (20.0 if collisions == 0 else 0.0)
        clearance_pts = min(20.0, min_clearance * 10.0)
        comfort_pts = 20.0 if passed_comfort else max(0.0, 20.0 - mean_jerk * 2.0)
        comp_pts = min(10.0, (100.0 / max(1.0, mean_lat)))
        progress_pts = (completion_rate / 100.0) * 10.0
        composite_score = max(0.0, min(100.0, safety_pts + clearance_pts + comfort_pts + comp_pts + progress_pts))

        mission_metrics = ScenarioMissionMetrics(
            scenario_name=scenario_name,
            difficulty=difficulty,
            planner_type=planner_type,
            completion_rate_pct=completion_rate,
            completion_time_s=completion_time_s,
            average_speed_mps=avg_speed,
            total_distance_traveled_m=total_distance,
            composite_score=composite_score
        )

        return FullEvaluationResult(
            mission=mission_metrics,
            safety=safety_metrics,
            kinematics=kinematics_metrics,
            computational=comp_metrics
        )
