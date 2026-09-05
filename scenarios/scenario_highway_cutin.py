"""Scenario 03: Highway Merge with Slow Vehicles (NH-48 Jaipur, Rajasthan)."""
import math
from interfaces import ObstacleClass
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario
from .difficulty import DifficultyLevel


class HighwayCutInScenario(BaseScenario):
    """Hallmark Scenario 3: Highway Merge with speed differentials, slow commercial traffic,

    and aggressive lateral cut-ins across 4 difficulty tiers.
    """

    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM, duration_seconds: float = 12.0, dt: float = 0.05):
        super().__init__(name="03_highway_merge", difficulty=difficulty, duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        if self.difficulty == DifficultyLevel.EASY:
            # Distant slow truck merging with 45m buffer
            truck = SimulationActor(
                id="slow_truck_merge",
                obstacle_class=ObstacleClass.TRUCK,
                x=45.0, y=2.5, speed_mps=5.0, yaw_rad=-0.08,
                length_m=7.5, width_m=2.4
            )
            self.env.add_actor(truck)

        elif self.difficulty == DifficultyLevel.MEDIUM:
            # Auto-rickshaw merging at 3.5 m/s at x=28m
            rick = SimulationActor(
                id="merging_rickshaw",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=28.0, y=2.2, speed_mps=4.0, yaw_rad=-0.15,
                length_m=2.6, width_m=1.3
            )
            car_lead = SimulationActor(
                id="lead_highway_car",
                obstacle_class=ObstacleClass.CAR,
                x=55.0, y=0.0, speed_mps=8.0, yaw_rad=0.0,
                length_m=4.5, width_m=1.8
            )
            self.env.add_actor(rick)
            self.env.add_actor(car_lead)

        elif self.difficulty == DifficultyLevel.HARD:
            # Aggressive cut-in auto-rickshaw + fast overtaking car on adjacent lane
            cut_in_auto = SimulationActor(
                id="aggressive_cut_in_rick",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=18.0, y=2.0, speed_mps=5.0, yaw_rad=-0.35, # Steep cut-in angle
                length_m=2.6, width_m=1.3
            )
            fast_car = SimulationActor(
                id="fast_passing_car",
                obstacle_class=ObstacleClass.CAR,
                x=35.0, y=-2.0, speed_mps=12.0, yaw_rad=0.0,
                length_m=4.5, width_m=1.8
            )
            self.env.add_actor(cut_in_auto)
            self.env.add_actor(fast_car)

        elif self.difficulty == DifficultyLevel.EXTREME:
            # Ultra-short range slow tractor merge + fast motorcycle overtaking on left shoulder
            slow_tractor = SimulationActor(
                id="sudden_merge_tractor",
                obstacle_class=ObstacleClass.TRUCK,
                x=14.0, y=1.8, speed_mps=2.2, yaw_rad=-0.40,
                length_m=4.0, width_m=2.0
            )
            fast_bike = SimulationActor(
                id="shoulder_overtake_bike",
                obstacle_class=ObstacleClass.MOTORCYCLE,
                x=6.0, y=-1.8, speed_mps=11.0, yaw_rad=0.05,
                length_m=2.0, width_m=0.8
            )
            self.env.add_actor(slow_tractor)
            self.env.add_actor(fast_bike)
