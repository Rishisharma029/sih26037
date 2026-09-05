"""
SIH26037 Dynamic World Model.
Maintains persistent tracks of all perceived road objects with:
- Object ID (Persistent tracking UUID)
- Type (ObstacleClass)
- Position (3D cartesian coordinates)
- Velocity (3D velocity vector)
- Acceleration (3D acceleration vector)
- Heading (Yaw angle in radians)
- Size (Length, Width, Height)
- Confidence (Bayesian fused confidence)
- Tracking History (Time-stamped coordinate history)
- Risk Score (Calculated threat level)
"""
from typing import List, Dict, Optional, Tuple
import math
from pydantic import BaseModel, Field, ConfigDict
from interfaces import ObstacleClass, Point3D, Vector3D, BoundingBox3D, TrackedObstacle

class TrackHistoryPoint(BaseModel):
    timestamp: float
    position: Point3D
    velocity: Vector3D
    heading_rad: float

class WorldModelTrack(BaseModel):
    """Rich dynamic track representing a physical road actor in the environment."""
    model_config = ConfigDict(extra="forbid")
    object_id: str = Field(..., description="Unique persistent tracking ID")
    obstacle_type: ObstacleClass = Field(..., description="Perceived object classification")
    position: Point3D = Field(..., description="Estimated spatial centroid in vehicle coordinates")
    velocity: Vector3D = Field(default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0))
    acceleration: Vector3D = Field(default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0))
    heading_rad: float = Field(0.0, description="Estimated heading yaw in radians")
    size: Vector3D = Field(..., description="Dimensions (Length, Width, Height) in meters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Fused detection confidence")
    tracking_history: List[TrackHistoryPoint] = Field(default_factory=list)
    risk_score: float = Field(0.0, ge=0.0, le=1.0, description="Evaluated collision risk score [0, 1]")
    age_steps: int = 1
    missed_steps: int = 0
    is_static: bool = False

    def to_tracked_obstacle(self) -> TrackedObstacle:
        """Converts internal world track to standardized TrackedObstacle contract."""
        dist = math.hypot(self.position.x, self.position.y)
        return TrackedObstacle(
            id=self.object_id,
            obstacle_class=self.obstacle_type,
            confidence=self.confidence,
            bbox=BoundingBox3D(
                center=self.position,
                size=self.size,
                yaw_rad=self.heading_rad
            ),
            velocity=self.velocity,
            acceleration=self.acceleration,
            distance_m=dist,
            is_static=self.is_static
        )

class WorldModel(BaseModel):
    """Complete dynamic state representation of the operational environment."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    tracks: Dict[str, WorldModelTrack] = Field(default_factory=dict)
    active_object_count: int = 0
    high_risk_objects: List[str] = Field(default_factory=list)
