"""Difficulty levels and configuration for Indian benchmark scenarios."""
from enum import Enum


class DifficultyLevel(str, Enum):
    """Standardized benchmark difficulty tiers."""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    EXTREME = "EXTREME"
