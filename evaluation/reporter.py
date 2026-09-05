"""Scorecard reporter generating formatted Markdown and JSON summaries."""
import json
from .metrics import EvaluatorMetrics

class ScorecardReporter:
    """Formats benchmark metrics into official verification scorecards."""
    @staticmethod
    def generate_markdown(scenario_name: str, metrics: EvaluatorMetrics) -> str:
        return f"""# SIH26037 Benchmark Scorecard: {scenario_name}
- **Safety Status**: {'PASS [ASIL-D Compliant]' if metrics.safety.passed_safety else 'FAIL'}
- **Minimum TTC**: {metrics.safety.min_ttc_seconds:.2f} s
- **Collisions Detected**: {metrics.safety.total_collisions}
- **Peak Deceleration**: {metrics.comfort.max_decel_mps2:.2f} m/s²
- **Average Velocity**: {metrics.average_speed_mps * 3.6:.1f} km/h
- **Comfort Compliant**: {metrics.comfort.passed_comfort}
"""
