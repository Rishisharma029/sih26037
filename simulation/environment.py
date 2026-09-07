"""Unmarked Road Environment with irregular edges, lateral drop-offs, potholes, and obstacles."""
from enum import Enum
import math
import numpy as np
from typing import List, Tuple, Optional
from interfaces import (
    Point3D, Vector3D, Pose3D, Twist3D, ObstacleClass,
    RoadAnomaly, TraversabilityClass, EgoVehicleState, ControlCommand
)
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
            yaw = 0.015 * 80.0  # ~1.2 rad
            base_x = 100.0 + (math.sin(yaw) / 0.015)
            base_y = (1.0 - math.cos(yaw)) / 0.015
            x = base_x + ds * math.cos(yaw)
            y = base_y + ds * math.sin(yaw)
            return x, y, yaw

    def get_corridor_widths(self, s: float) -> Tuple[float, float]:
        """Returns irregular, non-rectangular left and right boundary widths (d_left, d_right) at s."""
        base_half_w = self.base_width_m * 0.5

        # Macro width variation features
        choke1 = -0.45 * math.exp(-((s - 38.0) / 7.0) ** 2)
        bay1 = +0.65 * math.exp(-((s - 82.0) / 9.0) ** 2)
        choke2 = -0.40 * math.exp(-((s - 125.0) / 8.0) ** 2)
        bay2 = +0.55 * math.exp(-((s - 175.0) / 10.0) ** 2)

        macro_w_left = choke1 + bay1 + choke2 + bay2
        macro_w_right = choke1 * 0.9 + bay1 * 0.85 + choke2 * 0.95 + bay2 * 0.9

        # High-frequency edge roughness
        left_noise = 0.14 * math.sin(s * 0.08) + 0.06 * math.cos(s * 0.22)
        right_noise = 0.15 * math.cos(s * 0.07) - 0.05 * math.sin(s * 0.25)

        d_left = base_half_w + macro_w_left + left_noise
        d_right = -(base_half_w + macro_w_right + right_noise)
        return d_left, d_right

    def get_default_anomalies(self) -> List[RoadAnomaly]:
        """Returns standard unstructured Indian road anomalies (potholes, waterlogged pools, gravel, speed bumps)."""
        anomalies = []
        p1_x, p1_y, _ = self.frenet_to_cartesian(24.0, 0.45)
        anomalies.append(RoadAnomaly(
            id="pothole_km_24",
            anomaly_type="POTHOLE",
            position=Point3D(x=round(p1_x, 2), y=round(p1_y, 2), z=-0.12),
            radius_m=0.65,
            depth_or_height_m=-0.12,
            traversability_class=TraversabilityClass.POTHOLE,
            severity=0.85,
            is_passable=False,
            max_safe_speed_mps=0.0,
            traversability_score=0.05,
            description="Deep rim-damaging pothole crater (depth -12cm)"
        ))
        g1_x, g1_y, _ = self.frenet_to_cartesian(38.0, -0.40)
        anomalies.append(RoadAnomaly(
            id="gravel_km_38",
            anomaly_type="GRAVEL",
            position=Point3D(x=round(g1_x, 2), y=round(g1_y, 2), z=-0.02),
            radius_m=1.8,
            depth_or_height_m=-0.02,
            traversability_class=TraversabilityClass.GRAVEL,
            severity=0.45,
            is_passable=True,
            max_safe_speed_mps=2.5,
            traversability_score=0.50,
            description="Loose gravel stone aggregates"
        ))
        w1_x, w1_y, _ = self.frenet_to_cartesian(54.0, 0.35)
        anomalies.append(RoadAnomaly(
            id="waterlog_km_54",
            anomaly_type="WATER_LOGGING",
            position=Point3D(x=round(w1_x, 2), y=round(w1_y, 2), z=-0.15),
            radius_m=1.6,
            depth_or_height_m=-0.15,
            traversability_class=TraversabilityClass.WATERLOGGED,
            severity=0.90,
            is_passable=False,
            max_safe_speed_mps=0.8,
            traversability_score=0.20,
            description="Submerged murky flood pool hiding subsurface depth"
        ))
        sb1_x, sb1_y, _ = self.frenet_to_cartesian(72.0, 0.0)
        anomalies.append(RoadAnomaly(
            id="speed_bump_km_72",
            anomaly_type="SPEED_BUMP",
            position=Point3D(x=round(sb1_x, 2), y=round(sb1_y, 2), z=0.11),
            radius_m=1.5,
            depth_or_height_m=0.11,
            traversability_class=TraversabilityClass.SPEED_BUMP,
            severity=0.60,
            is_passable=True,
            max_safe_speed_mps=1.5,
            traversability_score=0.40,
            description="Unmarked steep concrete speed hump"
        ))
        return anomalies

    def get_surface_condition_at(
        self,
        s: float,
        d: float,
        anomalies: Optional[List[RoadAnomaly]] = None
    ) -> Tuple[TraversabilityClass, float, Optional[RoadAnomaly]]:
        """Evaluates road surface condition and traversability at Frenet station (s, d)."""
        x, y, _ = self.frenet_to_cartesian(s, d)
        d_left, d_right = self.get_corridor_widths(s)

        # Off-road ditch breach
        if d > d_left or d < d_right:
            return TraversabilityClass.BLOCKED, 0.0, None

        if anomalies:
            for anom in anomalies:
                dx = x - anom.position.x
                dy = y - anom.position.y
                if math.hypot(dx, dy) <= anom.radius_m:
                    return anom.traversability_class, anom.traversability_score, anom

        # Degraded rough tarmac zones
        if 85.0 <= s <= 98.0:
            return TraversabilityClass.DEGRADED, 0.75, None

        return TraversabilityClass.SAFE, 1.0, None

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        cx, cy, cyaw = self.get_centerline_point(s)
        x = cx - d * math.sin(cyaw)
        y = cy + d * math.cos(cyaw)
        return x, y, cyaw

    def cartesian_to_frenet(self, x: float, y: float, s_guess: Optional[float] = None) -> Tuple[float, float]:
        search_start = max(0.0, s_guess - 20.0) if s_guess is not None else max(0.0, x - 30.0)
        search_end = min(self.length_m, search_start + 60.0)
        
        best_s = search_start
        min_dist_sq = float("inf")
        
        s = search_start
        while s <= search_end:
            cx, cy, _ = self.get_centerline_point(s)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_s = s
            s += 1.0

        s_fine_start = max(0.0, best_s - 1.5)
        s_fine_end = min(self.length_m, best_s + 1.5)
        s = s_fine_start
        while s <= s_fine_end:
            cx, cy, _ = self.get_centerline_point(s)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_s = s
            s += 0.05

        cx, cy, cyaw = self.get_centerline_point(best_s)
        dx = x - cx
        dy = y - cy
        d = -dx * math.sin(cyaw) + dy * math.cos(cyaw)
        return best_s, d

    def get_ditch_margin(self, s: float, d: float, vehicle_half_width: float = 0.90) -> float:
        d_left, d_right = self.get_corridor_widths(s)
        margin_left = d_left - (d + vehicle_half_width)
        margin_right = (d - vehicle_half_width) - d_right
        return min(margin_left, margin_right)


RoadGeometry = VillageRoadGeometry


class RoadEnvironment:
    """Complete simulation environment with road geometry, vehicle physics, actors, and anomalies."""

    def __init__(self, geometry: Optional[VillageRoadGeometry] = None, length_m: float = 250.0, dt: float = 0.05):
        self.dt = dt
        self.geometry = geometry if geometry is not None else VillageRoadGeometry(length_m=length_m)
        self.actors: List[SimulationActor] = []
        self.anomalies: List[RoadAnomaly] = list(self.geometry.get_default_anomalies())
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
            a.step(self.dt, self.ego_state)
        return self.ego_state, {}

    def step_actors(self, dt: float, ego_state: Optional[EgoVehicleState] = None):
        for a in self.actors:
            a.step(dt, ego_state or self.ego_state)

    def get_active_actors(self) -> List[SimulationActor]:
        return self.actors


SimulationEnvironment = RoadEnvironment
