"""Quantitative safety, comfort, and mission efficiency metrics."""
from dataclasses import dataclass
from typing import List
from interfaces import EgoVehicleState, SafeTrajectory

@dataclass
class SafetyMetrics:
    min_ttc_seconds: float
    total_collisions: int
    boundary_violations: int
    passed_safety: bool

@dataclass
class ComfortMetrics:
    max_accel_mps2: float
    max_decel_mps2: float
    max_lateral_jerk_mps3: float
    passed_comfort: bool

@dataclass
class EvaluatorMetrics:
    safety: SafetyMetrics
    comfort: ComfortMetrics
    completion_time_s: float
    average_speed_mps: float
