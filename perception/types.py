"""Perception type helpers."""
from interfaces import TrackedObstacle, FreeSpaceCorridor, PerceptionOutput, ObstacleClass, Point3D, Vector3D

class CameraDetection:
    def __init__(self, class_name: ObstacleClass, bbox_2d: tuple, confidence: float, estimated_depth: float):
        self.class_name = class_name
        self.bbox_2d = bbox_2d # (xmin, ymin, xmax, ymax)
        self.confidence = confidence
        self.estimated_depth = estimated_depth

class LidarCluster:
    def __init__(self, centroid: Point3D, size: Vector3D, num_points: int):
        self.centroid = centroid
        self.size = size
        self.num_points = num_points

class RadarTarget:
    def __init__(self, target_id: str, range_m: float, azimuth_rad: float, doppler_speed_mps: float, rcs_db: float):
        self.target_id = target_id
        self.range_m = range_m
        self.azimuth_rad = azimuth_rad
        self.doppler_speed_mps = doppler_speed_mps
        self.rcs_db = rcs_db

__all__ = [
    "TrackedObstacle", "FreeSpaceCorridor", "PerceptionOutput",
    "ObstacleClass", "CameraDetection", "LidarCluster", "RadarTarget"
]

