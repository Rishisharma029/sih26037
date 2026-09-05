"""Dataset log loader and replay parser."""
import json
from .schemas import IndianDrivingEpisodeSchema

class DatasetLogLoader:
    """Replays sensor and telemetry logs from file."""
    @staticmethod
    def load_from_json(json_str: str) -> IndianDrivingEpisodeSchema:
        return IndianDrivingEpisodeSchema.model_validate_json(json_str)
