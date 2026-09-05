"""Typed data schemas for Explainability & Decision Audit Flight Recorder."""
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from interfaces import BehaviorMode, SafetyAction, ObstacleClass


class RiskLevel(str, Enum):
    """Dynamic risk assessment tier."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HazardContext(BaseModel):
    """Contextual description of the primary road hazard."""
    model_config = ConfigDict(extra="forbid")
    hazard_id: Optional[str] = None
    hazard_type: str = Field(..., description="e.g. 'Motorcycle entering ego trajectory', 'Stray cattle on road'")
    obstacle_class: Optional[ObstacleClass] = None
    relative_distance_m: float
    relative_speed_mps: float
    ttc_seconds: float
    corridor_side: str = Field("FRONT", description="'LEFT', 'RIGHT', 'FRONT', 'OPPOSING'")


class CandidateSummary(BaseModel):
    """Summary of an individual candidate trajectory evaluated by the planner."""
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    target_lateral_offset_m: float
    target_speed_mps: float
    safety_cost: float
    clearance_cost: float
    progress_cost: float
    comfort_cost: float
    uncertainty_cost: float
    total_cost: float
    is_feasible: bool
    rejection_reason: Optional[str] = None


class DecisionRationale(BaseModel):
    """Human-readable explanation and decomposition of the chosen action."""
    model_config = ConfigDict(extra="forbid")
    primary_reason: str = Field(..., description="e.g. 'Highest safety margin + acceptable curvature + low deviation'")
    safety_margin_m: float
    curvature_rating: str = Field("ACCEPTABLE", description="'GENTLE', 'ACCEPTABLE', 'SHARP'")
    lateral_deviation_rating: str = Field("LOW", description="'MINIMAL', 'LOW', 'MODERATE', 'HIGH'")
    tradeoff_notes: Optional[str] = None


class DecisionEvent(BaseModel):
    """Comprehensive, explainable decision audit record for a single planning instant."""
    model_config = ConfigDict(extra="forbid")
    event_id: int = Field(..., description="Sequential event number (e.g. 42 for EVENT #42)")
    timestamp: float
    scenario_name: str = "Indian Unstructured Road"
    ego_speed_kph: float
    ego_position_x: float
    ego_position_y: float
    hazard: HazardContext
    risk_level: RiskLevel
    path_status: str = Field("SAFE", description="'SAFE', 'UNSAFE_NUDGE_REQUIRED', 'BLOCKED', 'CRITICAL'")
    total_candidates_evaluated: int
    selected_candidate_id: str
    selected_behavior: BehaviorMode
    safety_action: SafetyAction
    rationale: DecisionRationale
    safety_override_active: bool = False
    notes: Optional[str] = None

    def format_event_card(self) -> str:
        """Render a formatted human-readable ASCII / Markdown event card."""
        return (
            f"EVENT #{self.event_id}\n\n"
            f"Hazard:\n"
            f"{self.hazard.hazard_type}\n\n"
            f"Risk:\n"
            f"{self.risk_level.value}\n\n"
            f"TTC:\n"
            f"{self.hazard.ttc_seconds:.2f} s (Dist: {self.hazard.relative_distance_m:.1f}m)\n\n"
            f"Current path:\n"
            f"{self.path_status}\n\n"
            f"Candidates:\n"
            f"{self.total_candidates_evaluated}\n\n"
            f"Selected:\n"
            f"{self.selected_candidate_id} ({self.selected_behavior.value})\n\n"
            f"Reason:\n"
            f"{self.rationale.primary_reason}"
        )
