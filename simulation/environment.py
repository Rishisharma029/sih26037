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
        """Returns irregular, non-rectangular left and right boundary widths (d_left, d_right) at s.
        
        Models realistic unstructured Indian rural roads:
        - Narrow choke points (culverts/trees/erosion) where width drops to ~3.4m
        - Widened passing bays/junctions where width expands to ~5.6m
        - High-frequency eroded shoulder roughness
        """
        base_half_w = self.base_width_m * 0.5

        # Macro width variation features
        # 1. Culvert choke point near s=38m (narrows)
        choke1 = -0.45 * math.exp(-((s - 38.0) / 7.0) ** 2)
        # 2. Passing bay near s=82m (widens)
        bay1 = +0.65 * math.exp(-((s - 82.0) / 9.0) ** 2)
        # 3. Embankment constriction near s=125m (narrows)
        choke2 = -0.40 * math.exp(-((s - 125.0) / 8.0) ** 2)
        # 4. Market opening near s=175m (widens)
        bay2 = +0.55 * math.exp(-((s - 175.0) / 10.0) ** 2)

        macro_w_left = choke1 + bay1 + choke2 + bay2
        macro_w_right = choke1 * 0.9 + bay1 * 0.85 + choke2 * 0.95 + bay2 * 0.9

        # High-frequency edge roughness (broken bitumen edge & shoulder erosion)
        left_noise = 0.14 * math.sin(s * 0.08) + 0.06 * math.cos(s * 0.22)
        right_noise = 0.15 * math.cos(s * 0.07) - 0.05 * math.sin(s * 0.25)

        d_left = base_half_w + macro_w_left + left_noise
        d_right = -(base_half_w + macro_w_right + right_noise)
        return d_left, d_right

    def get_default_anomalies(self) -> List[RoadAnomaly]:
        """Returns standard unstructured Indian road anomalies (potholes, speed bumps, gravel heaps)."""
        anomalies = []
        # Pothole 1 (s=24m, slight left of center)
        p1_x, p1_y, _ = self.frenet_to_cartesian(24.0, 0.5)
        anomalies.append(RoadAnomaly(
            id="pothole_km_24",
            anomaly_type="POTHOLE",
            position=Point3D(x=round(p1_x, 2), y=round(p1_y, 2), z=-0.09),
            radius_m=0.55,
            depth_or_height_m=-0.09
        ))
        # Speed Bump 1 (s=52m, unmarked rural speed hump)
        sb1_x, sb1_y, _ = self.frenet_to_cartesian(52.0, 0.0)
        anomalies.append(RoadAnomaly(
            id="speed_bump_km_52",
            anomaly_type="SPEED_BUMP",
            position=Point3D(x=round(sb1_x, 2), y=round(sb1_y, 2), z=0.10),
            radius_m=1.6,
            depth_or_height_m=0.10
        ))
        # Pothole 2 (s=76m, right side of carriageway)
        p2_x, p2_y, _ = self.frenet_to_cartesian(76.0, -0.65)
        anomalies.append(RoadAnomaly(
            id="pothole_km_76",
            anomaly_type="POTHOLE",
            position=Point3D(x=round(p2_x, 2), y=round(p2_y, 2), z=-0.11),
            radius_m=0.65,
            depth_or_height_m=-0.11
        ))
        # Blocked gravel patch on shoulder (s=92m)
        gp1_x, gp1_y, _ = self.frenet_to_cartesian(92.0, 1.9)
        anomalies.append(RoadAnomaly(
            id="gravel_heap_km_92",
            anomaly_type="GRAVEL",
            position=Point3D(x=round(gp1_x, 2), y=round(gp1_y, 2), z=0.15),
            radius_m=1.2,
            depth_or_height_m=0.15
        ))
        # Severe Center Crater (s=122m, deep pothole in road center)
        p3_x, p3_y, _ = self.frenet_to_cartesian(122.0, 0.1)
        anomalies.append(RoadAnomaly(
            id="crater_pothole_km_122",
            anomaly_type="POTHOLE",
            position=Point3D(x=round(p3_x, 2), y=round(p3_y, 2), z=-0.16),
            radius_m=0.85,
            depth_or_height_m=-0.16
        ))
        return anomalies

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        """Converts Frenet coordinates (s, d) to global Cartesian (x, y, yaw_rad).
        +d is to the left of the centerline, -d is to the right.
        """
        cx, cy, cyaw = self.get_centerline_point(s)
        x = cx - d * math.sin(cyaw)
        y = cy + d * math.cos(cyaw)
        return x, y, cyaw

    def cartesian_to_frenet(self, x: float, y: float, s_guess: Optional[float] = None) -> Tuple[float, float]:
        """Projects global Cartesian (x, y) onto road centerline to compute Frenet (s, d).
        
        Returns:
            (s, d):
                s: Arc-length station along centerline (meters)
                d: Lateral offset from centerline (meters, positive left, negative right)
        """
        if x <= 100.0 and s_guess is None:
            return x, y

        # Robust local search around guess or station sweep
        search_start = max(0.0, s_guess - 20.0) if s_guess is not None else max(0.0, x - 30.0)
        search_end = min(self.length_m, search_start + 60.0)
        
        best_s = search_start
        min_dist_sq = float("inf")
        
        # Coarse step
        s = search_start
        while s <= search_end:
            cx, cy, _ = self.get_centerline_point(s)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_s = s
            s += 1.0

        # Fine refinement around best_s
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
        # Lateral offset: d = -dx * sin(cyaw) + dy * cos(cyaw)
        d = -dx * math.sin(cyaw) + dy * math.cos(cyaw)
        return best_s, d

    def get_ditch_margin(self, s: float, d: float, vehicle_half_width: float = 0.90) -> float:
        """Computes true physical clearance margin (meters) between the vehicle boundary
        and the left/right irregular road edges/ditches.
        
        A value >= 0 means the vehicle is completely on the drivable road surface.
        A value < 0 means the vehicle envelope has breached the road edge.
        """
        d_left, d_right = self.get_corridor_widths(s)
        margin_left = d_left - (d + vehicle_half_width)
        margin_right = (d - vehicle_half_width) - d_right
        return min(margin_left, margin_right)


# Aliases
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
            a.step(self.dt)
        return self.ego_state, {}

    def step_actors(self, dt: float):
        for a in self.actors:
            a.step(dt)

    def get_active_actors(self) -> List[SimulationActor]:
        return self.actors


# Aliases
SimulationEnvironment = RoadEnvironment
