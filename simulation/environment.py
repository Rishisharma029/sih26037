"""Unmarked Road Environment with irregular edges, lateral drop-offs, potholes, and obstacles."""
from enum import Enum
import math
import numpy as np
from typing import List, Tuple, Optional
from interfaces import Point3D, Vector3D, Pose3D, Twist3D, ObstacleClass, RoadAnomaly, EgoVehicleState, ControlCommand
from .actors import SimulationActor
from .vehicle_model import KinematicBicycleModel


class RoadCorridorProfile(str, Enum):
    """Road profile classifications for Indian traffic settings."""
    UNMARKED_VILLAGE = "UNMARKED_VILLAGE"
    URBAN_INTERSECTION = "URBAN_INTERSECTION"
    HIGHWAY_MERGE = "HIGHWAY_MERGE"
    DENSE_MARKET = "DENSE_MARKET"
    CATTLE_CROSSING = "CATTLE_CROSSING"


class VillageRoadGeometry:
    """Models a continuous spline-based unmarked road without painted lane lines."""

    def __init__(self, length_m: float = 250.0, base_width_m: float = 4.3, profile: RoadCorridorProfile = RoadCorridorProfile.UNMARKED_VILLAGE):
        self.length_m = length_m
        self.base_width_m = base_width_m
        self.profile = profile

    def get_centerline_point(self, s: float) -> Tuple[float, float, float]:
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

    def get_corridor_widths(self, s: float) -> Tuple[float, float]:
        """Returns irregular left and right boundary widths (d_left, d_right) at s."""
        base_half_w = self.base_width_m * 0.5
        left_noise = 0.15 * math.sin(s * 0.08) + 0.08 * math.cos(s * 0.2)
        right_noise = 0.18 * math.cos(s * 0.07) - 0.05 * math.sin(s * 0.25)
        d_left = base_half_w + left_noise
        d_right = -(base_half_w + right_noise)
        return d_left, d_right


# Aliases
RoadGeometry = VillageRoadGeometry


class RoadEnvironment:
    """Complete simulation environment with road geometry, vehicle physics, actors, and anomalies."""

    def __init__(self, geometry: Optional[VillageRoadGeometry] = None, length_m: float = 250.0, dt: float = 0.05):
        self.dt = dt
        self.geometry = geometry if geometry is not None else VillageRoadGeometry(length_m=length_m)
        self.actors: List[SimulationActor] = []
        self.anomalies: List[RoadAnomaly] = []
        self.vehicle_model = KinematicBicycleModel()
        self.ego_state = EgoVehicleState(
            timestamp=0.0,
            pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
            twist=Twist3D(speed_mps=0.0),
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
        )

    def add_actor(self, actor: SimulationActor):
        self.actors.append(actor)

    def add_anomaly(self, anomaly: RoadAnomaly):
        self.anomalies.append(anomaly)

    def step(self, cmd: ControlCommand) -> Tuple[EgoVehicleState, dict]:
        """Step ego vehicle physics and dynamic traffic actors."""
        self.ego_state = self.vehicle_model.step(self.ego_state, cmd, dt=self.dt)
        for a in self.actors:
            a.step(self.dt)
        return self.ego_state, {}

    def step_actors(self, dt: float):
        for a in self.actors:
            a.step(dt)

    def get_active_actors(self) -> List[SimulationActor]:
        return self.actors


# Aliases
SimulationEnvironment = RoadEnvironment
