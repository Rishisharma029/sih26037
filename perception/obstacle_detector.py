"""Mixed-traffic obstacle detector for Indian roads."""
from typing import List
from interfaces import TrackedObstacle, ObstacleClass, BoundingBox3D, Point3D, Vector3D

class MixedTrafficObstacleDetector:
    """Detects heterogeneous road users (Auto-rickshaw, cattle, bikes, pedestrians)."""
    def detect_from_mock(self, mock_obstacles: List[dict]) -> List[TrackedObstacle]:
        results = []
        for o in mock_obstacles:
            results.append(TrackedObstacle(
                id=o["id"],
                obstacle_class=ObstacleClass(o.get("class", "CAR")),
                confidence=o.get("confidence", 0.95),
                bbox=BoundingBox3D(
                    center=Point3D(x=o["x"], y=o["y"], z=o.get("z", 0.8)),
                    size=Vector3D(x=o.get("length", 2.5), y=o.get("width", 1.4), z=o.get("height", 1.6)),
                    yaw_rad=o.get("yaw", 0.0)
                ),
                velocity=Vector3D(x=o.get("vx", 0.0), y=o.get("vy", 0.0), z=0.0),
                distance_m=(o["x"]**2 + o["y"]**2)**0.5,
                is_static=o.get("is_static", False)
            ))
        return results
