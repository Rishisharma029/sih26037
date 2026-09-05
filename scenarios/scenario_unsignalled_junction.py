"""Scenario 02: Uncontrolled / Unsignalled Urban Intersection (Silk Board, Bengaluru)."""
import math
from interfaces import ObstacleClass, RoadAnomaly, Point3D
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario
from .difficulty import DifficultyLevel


class UnsignalledJunctionScenario(BaseScenario):
    """Hallmark Scenario 2: Unsignalled Intersection with non-lane-respecting two-wheelers,

    cross-turning auto-rickshaws, and multi-directional flows across 4 difficulty tiers.
    """

    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM, duration_seconds: float = 12.0, dt: float = 0.05):
        super().__init__(name="02_uncontrolled_intersection", difficulty=difficulty, duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        if self.difficulty == DifficultyLevel.EASY:
            # Single slow cross-traffic motorcycle
            moto = SimulationActor(
                id="cross_moto",
                obstacle_class=ObstacleClass.MOTORCYCLE,
                x=35.0, y=-6.0, speed_mps=2.0, yaw_rad=math.pi / 2.0,
                length_m=2.0, width_m=0.8
            )
            self.env.add_actor(moto)

        elif self.difficulty == DifficultyLevel.MEDIUM:
            # Auto-rickshaw turning across intersection + pedestrian
            rick = SimulationActor(
                id="turning_rick",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=25.0, y=-4.0, speed_mps=3.0, yaw_rad=math.pi / 3.0,
                length_m=2.6, width_m=1.3
            )
            ped = SimulationActor(
                id="ped_crossing",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=30.0, y=3.0, speed_mps=0.8, yaw_rad=-math.pi / 2.0,
                length_m=0.5, width_m=0.5
            )
            self.env.add_actor(rick)
            self.env.add_actor(ped)

        elif self.difficulty == DifficultyLevel.HARD:
            # Aggressive cutting motorcycle + right-turning bus + crossing pedestrian
            moto = SimulationActor(
                id="swerving_bike",
                obstacle_class=ObstacleClass.MOTORCYCLE,
                x=18.0, y=-3.5, speed_mps=5.0, yaw_rad=math.pi / 2.5,
                length_m=2.0, width_m=0.8
            )
            bus = SimulationActor(
                id="cross_bus",
                obstacle_class=ObstacleClass.BUS,
                x=32.0, y=5.0, speed_mps=3.5, yaw_rad=-math.pi / 2.0,
                length_m=9.0, width_m=2.5
            )
            ped = SimulationActor(
                id="hesitating_ped",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=14.0, y=-2.5, speed_mps=1.0, yaw_rad=math.pi / 2.0,
                length_m=0.5, width_m=0.5
            )
            self.env.add_actor(moto)
            self.env.add_actor(bus)
            self.env.add_actor(ped)

        elif self.difficulty == DifficultyLevel.EXTREME:
            # High-speed crossing SUV + simultaneous multi-angle motorcycle swarm
            suv = SimulationActor(
                id="fast_crossing_suv",
                obstacle_class=ObstacleClass.CAR,
                x=16.0, y=-8.0, speed_mps=6.5, yaw_rad=math.pi / 2.0,
                length_m=4.8, width_m=1.9
            )
            bike_1 = SimulationActor(
                id="blind_corner_bike",
                obstacle_class=ObstacleClass.MOTORCYCLE,
                x=12.0, y=3.5, speed_mps=5.5, yaw_rad=-math.pi / 2.2,
                length_m=2.0, width_m=0.8
            )
            bike_2 = SimulationActor(
                id="rear_filtering_bike",
                obstacle_class=ObstacleClass.MOTORCYCLE,
                x=8.0, y=-1.5, speed_mps=6.0, yaw_rad=0.1,
                length_m=2.0, width_m=0.8
            )
            self.env.add_actor(suv)
            self.env.add_actor(bike_1)
            self.env.add_actor(bike_2)
