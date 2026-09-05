"""Schemas for logging and replaying Indian driving datasets (IDD style)."""
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class TelemetrySample(BaseModel):
    timestamp: float
    x: float
    y: float
    speed_mps: float
    steer_rad: float
    min_ttc: float

class IndianDrivingEpisodeSchema(BaseModel):
    episode_id: str
    scenario_name: str
    total_duration_s: float
    samples: List[TelemetrySample] = Field(default_factory=list)
