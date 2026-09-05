"""Audit Log Exporters generating persistent JSON and Markdown flight logs."""
import json
from typing import List
from .types import DecisionEvent


class AuditLogExporter:
    """Exports flight recorder events into Markdown decision logs and JSONL traces."""

    @staticmethod
    def export_markdown(events: List[DecisionEvent], title: str = "Decision Audit Log") -> str:
        """Generate a complete Markdown audit flight record document."""
        md = f"""# SIH26037 — Autonomous Decision Audit Log
## {title}

This audit record captures real-time explainable decisions made by the autonomous driving stack.

---

"""
        for event in events:
            md += f"""### EVENT #{event.event_id} (t = {event.timestamp:.2f}s | Speed = {event.ego_speed_kph:.1f} km/h)

```
Hazard:
{event.hazard.hazard_type}

Risk:
{event.risk_level.value}

TTC:
{event.hazard.ttc_seconds:.2f} s (Distance: {event.hazard.relative_distance_m:.1f} m, Side: {event.hazard.corridor_side})

Current path:
{event.path_status}

Candidates Evaluated:
{event.total_candidates_evaluated}

Selected:
{event.selected_candidate_id} [{event.selected_behavior.value}]

Safety Action:
{event.safety_action.value}

Reason:
{event.rationale.primary_reason}
```

- **Safety Margin**: {event.rationale.safety_margin_m:.2f} m
- **Curvature Rating**: {event.rationale.curvature_rating} | **Lateral Deviation**: {event.rationale.lateral_deviation_rating}

---
"""
        return md

    @staticmethod
    def export_jsonl(events: List[DecisionEvent], filepath: str):
        """Export events as sequential JSONL lines."""
        with open(filepath, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(ev.model_dump_json() + "\n")
