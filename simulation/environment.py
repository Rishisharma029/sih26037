"""
Unmarked Road Environment with irregular edges, lateral drop-offs, potholes, and obstacles.
"""
import math
import numpy as np
from typing import List, Tuple, Optional
from interfaces import Point3D, Vector3D, ObstacleClass, RoadAnomaly
from .actors import SimulationActor

class VillageRoadGeometry:
    """Models a continuous spline-based unmarked road without painted lane lines."""
    def __init__(self, length_m: float = 250.0):
        self.length_m = length_m

    def get_centerline_point(self, s: float) -> tuple[float, float, float]:
        """Returns (x, y, heading_rad) at longitudinal station s."""
        if s <= 100.0:
            return s, 0.0, 0.0
        elif s <= 180.0:
            # Gentle curved section
            ds = s - 100.0
            curv = 0.015
            yaw = curv * ds
            x = 100.0 + (math.sin(yaw) / curv)
            y = (1.0 - math.cos(yaw)) / curv
            return x, y, yaw
        else:
            # Straight exit section
            ds = s - 180.0
            yaw = 0.015 * 80.0 # ~1.2 rad
            base_x = 100.0 + (math.sin(yaw) / 0.015)
            base_y = (1.0 - math.cos(yaw)) / 0.015
            x = base_x + ds * math.cos(yaw)
            y = base_y + ds * math.sin(yaw)
            return x, y, yaw

    def get_corridor_widths(self, s: float) -> tuple[float, float]:
        """Returns irregular left and right boundary widths (d_left, d_right) at s."""
        # Unstructured variation between 3.8m and 4.6m total width
        base_half_w = 2.15
        left_noise = 0.15 * math.sin(s * 0.08) + 0.08 * math.cos(s * 0.2)
        right_noise = 0.18 * math.cos(s * 0.07) - 0.05 * math.sin(s * 0.25)
        d_left = base_half_w + left_noise
        d_right = -(base_half_w + right_noise)
        return d_left, d_right

class RoadEnvironment:
    """Complete simulation environment with road geometry, actors, and anomalies."""
    def __init__(self, length_m: float = 250.0):
        self.geometry = VillageRoadGeometry(length_m=length_m)
        self.actors: List[SimulationActor] = []
        self.anomalies: List[RoadAnomaly] = []

    def add_actor(self, actor: SimulationActor):
        self.actors.append(actor)

    def add_anomaly(self, anomaly: RoadAnomaly):
        self.anomalies.append(anomaly)

    def step_actors(self, dt: float):
        for a in self.actors:
            a.step(dt)

    def get_active_actors(self) -> List[SimulationActor]:
        return self.actors
