"""Scenario 01: Unmarked Indian Village Road (Khed Shivapur, Maharashtra)."""
import math
from interfaces import ObstacleClass, RoadAnomaly, Point3D
from simulation.actors import SimulationActor
from simulation.environment import RoadEnvironment
from .scenario_base import BaseScenario
from .difficulty import DifficultyLevel


class UnmarkedVillageRoadScenario(BaseScenario):
    """Hallmark Scenario 1: Unmarked Village Road with irregular edges, lateral drop-offs,

    potholes, parked vehicles, and oncoming tractors across 4 difficulty tiers.
    """

    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM, duration_seconds: float = 12.0, dt: float = 0.05):
        super().__init__(name="01_village_road", difficulty=difficulty, duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        if self.difficulty == DifficultyLevel.EASY:
            # Distant stationary auto on wide shoulder
            auto = SimulationActor(
                id="parked_auto_rickshaw",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=45.0, y=2.2, speed_mps=0.0, yaw_rad=0.0,
                length_m=2.6, width_m=1.3, is_static=True
            )
            self.env.add_actor(auto)

        elif self.difficulty == DifficultyLevel.MEDIUM:
            # Parked auto + roadside boulder
            auto = SimulationActor(
                id="parked_auto_rickshaw",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=25.0, y=1.5, speed_mps=0.0, yaw_rad=0.0,
                length_m=2.6, width_m=1.3, is_static=True
            )
            boulder = SimulationActor(
                id="roadside_boulder",
                obstacle_class=ObstacleClass.STATIC_DEBRIS,
                x=40.0, y=-1.8, speed_mps=0.0, yaw_rad=0.0,
                length_m=1.2, width_m=1.2, is_static=True
            )
            self.env.add_actor(auto)
            self.env.add_actor(boulder)

        elif self.difficulty == DifficultyLevel.HARD:
            # Oncoming tractor in opposing lane + crossing pedestrian + pothole
            tractor = SimulationActor(
                id="oncoming_tractor",
                obstacle_class=ObstacleClass.TRUCK,
                x=55.0, y=1.6, speed_mps=-3.5, yaw_rad=math.pi,
                length_m=4.2, width_m=2.0
            )
            ped = SimulationActor(
                id="crossing_pedestrian",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=30.0, y=-2.2, speed_mps=1.0, yaw_rad=math.pi / 2.0,
                length_m=0.5, width_m=0.5
            )
            auto = SimulationActor(
                id="parked_auto_rickshaw",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=18.0, y=1.6, speed_mps=0.0, yaw_rad=0.0,
                length_m=2.6, width_m=1.3, is_static=True
            )
            pothole = RoadAnomaly(
                id="pothole_1",
                anomaly_type="POTHOLE",
                position=Point3D(x=22.0, y=-0.4, z=-0.15),
                radius_m=0.6,
                depth_or_height_m=-0.12
            )
            self.env.add_actor(tractor)
            self.env.add_actor(ped)
            self.env.add_actor(auto)
            self.env.add_anomaly(pothole)

        elif self.difficulty == DifficultyLevel.EXTREME:
            # Wide oncoming tractor + sudden crossing pedestrian + deep pothole in narrow corridor
            tractor = SimulationActor(
                id="oncoming_tractor_wide",
                obstacle_class=ObstacleClass.TRUCK,
                x=40.0, y=1.4, speed_mps=-4.5, yaw_rad=math.pi,
                length_m=4.5, width_m=2.2
            )
            ped = SimulationActor(
                id="darting_villager",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=22.0, y=-2.0, speed_mps=1.8, yaw_rad=math.pi / 2.0,
                length_m=0.5, width_m=0.5
            )
            auto = SimulationActor(
                id="parked_auto_blocking",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=14.0, y=1.2, speed_mps=0.0, yaw_rad=0.0,
                length_m=2.6, width_m=1.3, is_static=True
            )
            pothole = RoadAnomaly(
                id="deep_pothole",
                anomaly_type="POTHOLE",
                position=Point3D(x=18.0, y=-0.2, z=-0.20),
                radius_m=0.8,
                depth_or_height_m=-0.18
            )
            self.env.add_actor(tractor)
            self.env.add_actor(ped)
            self.env.add_actor(auto)
            self.env.add_anomaly(pothole)


def build_unmarked_village_environment() -> RoadEnvironment:
    """Helper for testing and quick baseline instantiation."""
    sc = UnmarkedVillageRoadScenario(difficulty=DifficultyLevel.HARD)
    boulder = SimulationActor(
        id="roadside_boulder",
        obstacle_class=ObstacleClass.STATIC_DEBRIS,
        x=40.0, y=-1.8, speed_mps=0.0, yaw_rad=0.0,
        length_m=1.2, width_m=1.2, is_static=True
    )
    sc.env.add_actor(boulder)
    return sc.env
