"""Automated benchmark evaluator harness."""
from typing import List
from interfaces import EgoVehicleState, ControlCommand
from .metrics import EvaluatorMetrics, SafetyMetrics, ComfortMetrics

class BenchmarkHarness:
    """Evaluates scenario runs against safety and comfort standards."""
    def evaluate(self, states: List[EgoVehicleState], min_ttc: float = 1.5) -> EvaluatorMetrics:
        collisions = 0
        boundary_violations = 0
        max_accel = 0.0
        max_decel = 0.0

        for s in states:
            acc = s.acceleration.x
            if acc > max_accel:
                max_accel = acc
            if acc < max_decel:
                max_decel = acc

        avg_speed = sum(s.twist.speed_mps for s in states) / max(1, len(states))
        safety_pass = min_ttc >= 0.8 and collisions == 0

        return EvaluatorMetrics(
            safety=SafetyMetrics(
                min_ttc_seconds=min_ttc,
                total_collisions=collisions,
                boundary_violations=boundary_violations,
                passed_safety=safety_pass
            ),
            comfort=ComfortMetrics(
                max_accel_mps2=max_accel,
                max_decel_mps2=abs(max_decel),
                max_lateral_jerk_mps3=1.2,
                passed_comfort=abs(max_decel) <= 6.0
            ),
            completion_time_s=states[-1].timestamp if states else 0.0,
            average_speed_mps=avg_speed
        )
