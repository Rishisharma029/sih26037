"""Scenario 05: Sudden Cattle Crossing and Curve Obstruction (Varanasi Highway Bend, UP)."""
import math
from interfaces import ObstacleClass
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario
from .difficulty import DifficultyLevel


class CattleCrossingScenario(BaseScenario):
    """Hallmark Scenario 5: Sudden Cattle Crossing with livestock hesitation, road freezing,

    and oncoming traffic obstruction across 4 difficulty tiers.
    """

    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM, duration_seconds: float = 12.0, dt: float = 0.05):
        super().__init__(name="05_cattle_crossing", difficulty=difficulty, duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        if self.difficulty == DifficultyLevel.EASY:
            # Animal far away on road shoulder
            cow = SimulationActor(
                id="distant_cow",
                obstacle_class=ObstacleClass.CATTLE_ANIMAL,
                x=45.0, y=2.5, speed_mps=0.0, yaw_rad=0.0,
                length_m=2.2, width_m=1.0, is_static=True
            )
            self.env.add_actor(cow)

        elif self.difficulty == DifficultyLevel.MEDIUM:
            # Animal slowly enters road from shoulder
            cow = SimulationActor(
                id="slow_crossing_cow",
                obstacle_class=ObstacleClass.CATTLE_ANIMAL,
                x=25.0, y=2.2, speed_mps=0.6, yaw_rad=-math.pi / 2.0,
                length_m=2.2, width_m=1.0
            )
            self.env.add_actor(cow)

        elif self.difficulty == DifficultyLevel.HARD:
            # Sudden crossing followed by freeze in road center
            cow = SimulationActor(
                id="sudden_crossing_cow",
                obstacle_class=ObstacleClass.CATTLE_ANIMAL,
                x=15.0, y=1.8, speed_mps=1.2, yaw_rad=-math.pi / 2.0,
                length_m=2.2, width_m=1.0
            )
            calf = SimulationActor(
                id="trailing_calf",
                obstacle_class=ObstacleClass.CATTLE_ANIMAL,
                x=18.0, y=2.8, speed_mps=0.8, yaw_rad=-math.pi / 2.0,
                length_m=1.4, width_m=0.7
            )
            self.env.add_actor(cow)
            self.env.add_actor(calf)

        elif self.difficulty == DifficultyLevel.EXTREME:
            # Cattle crossing + abrupt freeze in lane center + fast oncoming truck in opposing lane
            cow_blocking = SimulationActor(
                id="frozen_cow_center",
                obstacle_class=ObstacleClass.CATTLE_ANIMAL,
                x=12.0, y=0.2, speed_mps=0.2, yaw_rad=-math.pi / 2.0,
                length_m=2.2, width_m=1.0
            )
            oncoming_truck = SimulationActor(
                id="oncoming_highway_truck",
                obstacle_class=ObstacleClass.TRUCK,
                x=40.0, y=1.8, speed_mps=-6.0, yaw_rad=math.pi,
                length_m=8.5, width_m=2.5
            )
            self.env.add_actor(cow_blocking)
            self.env.add_actor(oncoming_truck)
