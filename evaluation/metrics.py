"""Quantitative safety, comfort, kinematics, computational, and mission metrics."""
import math
from dataclasses import dataclass
from typing import List
from interfaces import EgoVehicleState, SafeTrajectory, PlannedTrajectory


@dataclass
class SafetyClearanceMetrics:
    """Safety and distance margin metrics."""
    min_clearance_m: float
    mean_clearance_m: float
    min_ttc_s: float
    mean_ttc_s: float
    collision_count: int
    collision_rate_pct: float
    near_collision_count: int  # Clearance < 1.0m or TTC < 2.0s
    near_collision_rate_pct: float
    boundary_violations: int
    passed_safety: bool


@dataclass
class KinematicSmoothnessMetrics:
    """Path geometry, smoothness, and ride quality metrics."""
    path_length_m: float
    path_smoothness_rad_m: float  # Integral of squared curvature along arc length: ∫ κ² ds
    mean_lateral_jerk_mps3: float
    max_lateral_jerk_mps3: float
    max_lat_accel_mps2: float
    max_lon_decel_mps2: float
    lateral_deviation_rms_m: float  # RMS Cross-Track Error (CTE) relative to road centerline
    speed_stability_mps: float  # Velocity standard deviation (lower = smoother cruise)
    passed_comfort: bool


@dataclass
class ComputationalMetrics:
    """Real-time performance and computational complexity metrics."""
    mean_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    planning_frequency_hz: float
    aeb_intervention_count: int
    emergency_replan_count: int


@dataclass
class ScenarioMissionMetrics:
    """Mission success and progress metrics."""
    scenario_name: str
    difficulty: str
    planner_type: str
    completion_rate_pct: float
    completion_time_s: float
    average_speed_mps: float
    total_distance_traveled_m: float
    composite_score: float  # 0 to 100 overall score


@dataclass
class FullEvaluationResult:
    """Unified evaluation container."""
    mission: ScenarioMissionMetrics
    safety: SafetyClearanceMetrics
    kinematics: KinematicSmoothnessMetrics
    computational: ComputationalMetrics


def compute_path_smoothness(trajectory_points: List[tuple]) -> float:
    """Compute the path smoothness integral: ∫ κ² ds.

    Lower value indicates a smoother path with fewer abrupt steering changes.
    """
    if len(trajectory_points) < 3:
        return 0.0

    total_integral = 0.0
    for i in range(1, len(trajectory_points) - 1):
        p_prev = trajectory_points[i - 1]
        p_curr = trajectory_points[i]
        p_next = trajectory_points[i + 1]

        # Segment lengths
        ds1 = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        ds2 = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        ds = (ds1 + ds2) / 2.0

        if ds < 1e-4:
            continue

        # Curvature κ = |dθ / ds|
        theta1 = math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])
        theta2 = math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])
        dtheta = math.atan2(math.sin(theta2 - theta1), math.cos(theta2 - theta1))
        kappa = abs(dtheta) / ds

        total_integral += (kappa ** 2) * ds

    return total_integral


def compute_velocity_std(speeds: List[float]) -> float:
    """Compute standard deviation of speed to measure velocity stability."""
    if len(speeds) < 2:
        return 0.0
    mean_speed = sum(speeds) / len(speeds)
    variance = sum((s - mean_speed) ** 2 for s in speeds) / (len(speeds) - 1)
    return math.sqrt(variance)

# Backward-compatibility aliases
SafetyMetrics = SafetyClearanceMetrics
ComfortMetrics = KinematicSmoothnessMetrics
EvaluatorMetrics = FullEvaluationResult
