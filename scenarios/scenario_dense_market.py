"""Scenario 04: Dense Market Road with Extreme Pedestrian Density (Chandni Chowk, Old Delhi)."""
import math
from interfaces import ObstacleClass
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario
from .difficulty import DifficultyLevel


class DenseMarketScenario(BaseScenario):
    """Hallmark Scenario 4: Dense Bazaar Street with pushcarts, bidirectional pedestrian flows,

    parked vehicles, and minimal lateral clearance across 4 difficulty tiers.
    """

    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM, duration_seconds: float = 12.0, dt: float = 0.05):
        super().__init__(name="04_dense_market", difficulty=difficulty, duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        if self.difficulty == DifficultyLevel.EASY:
            # Static pushcart and pedestrian on wide side
            cart = SimulationActor(
                id="market_cart",
                obstacle_class=ObstacleClass.PUSHCART,
                x=30.0, y=1.5, speed_mps=0.0, yaw_rad=0.0,
                length_m=1.8, width_m=1.0, is_static=True
            )
            ped = SimulationActor(
                id="shopper_1",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=35.0, y=-1.6, speed_mps=0.6, yaw_rad=math.pi / 4.0,
                length_m=0.5, width_m=0.5
            )
            self.env.add_actor(cart)
            self.env.add_actor(ped)

        elif self.difficulty == DifficultyLevel.MEDIUM:
            # Two walking pedestrians + parked auto + roadside pushcart
            cart = SimulationActor(
                id="fruit_cart",
                obstacle_class=ObstacleClass.PUSHCART,
                x=22.0, y=1.4, speed_mps=0.0, yaw_rad=0.0,
                length_m=1.8, width_m=1.0, is_static=True
            )
            ped_1 = SimulationActor(
                id="crossing_shopper",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=18.0, y=-1.8, speed_mps=0.8, yaw_rad=math.pi / 2.0,
                length_m=0.5, width_m=0.5
            )
            auto_parked = SimulationActor(
                id="loading_auto",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=35.0, y=-1.5, speed_mps=0.0, yaw_rad=0.0,
                length_m=2.6, width_m=1.3, is_static=True
            )
            self.env.add_actor(cart)
            self.env.add_actor(ped_1)
            self.env.add_actor(auto_parked)

        elif self.difficulty == DifficultyLevel.HARD:
            # Narrow corridor, reversing pedestrian, moving pushcart, nudging auto
            cart = SimulationActor(
                id="moving_cart",
                obstacle_class=ObstacleClass.PUSHCART,
                x=25.0, y=1.2, speed_mps=0.8, yaw_rad=math.pi, # Moving against traffic
                length_m=1.8, width_m=1.0
            )
            hesitating_ped = SimulationActor(
                id="hesitating_pedestrian",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=16.0, y=0.4, speed_mps=0.5, yaw_rad=-math.pi / 2.0,
                length_m=0.5, width_m=0.5
            )
            nudging_auto = SimulationActor(
                id="nudging_market_auto",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=32.0, y=-1.2, speed_mps=2.0, yaw_rad=0.2,
                length_m=2.6, width_m=1.3
            )
            self.env.add_actor(cart)
            self.env.add_actor(hesitating_ped)
            self.env.add_actor(nudging_auto)

        elif self.difficulty == DifficultyLevel.EXTREME:
            # Dense swarm: 3 pedestrians walking abreast, oncoming pushcart, darting child
            ped_child = SimulationActor(
                id="darting_child",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=10.0, y=-1.8, speed_mps=2.2, yaw_rad=math.pi / 2.0,
                length_m=0.4, width_m=0.4
            )
            ped_group_1 = SimulationActor(
                id="ped_group_left",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=18.0, y=0.8, speed_mps=0.7, yaw_rad=-math.pi / 4.0,
                length_m=0.5, width_m=0.5
            )
            ped_group_2 = SimulationActor(
                id="ped_group_right",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=18.0, y=-0.6, speed_mps=0.7, yaw_rad=math.pi / 4.0,
                length_m=0.5, width_m=0.5
            )
            pushcart = SimulationActor(
                id="heavy_pushcart_blocking",
                obstacle_class=ObstacleClass.PUSHCART,
                x=26.0, y=0.0, speed_mps=-0.5, yaw_rad=math.pi,
                length_m=2.0, width_m=1.2
            )
            self.env.add_actor(ped_child)
            self.env.add_actor(ped_group_1)
            self.env.add_actor(ped_group_2)
            self.env.add_actor(pushcart)
