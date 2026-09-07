"""Unmarked Village Road Scenario for SIH26037 Evaluation."""
import math
from typing import List, Optional
from interfaces import ObstacleClass, RoadAnomaly, Point3D
from simulation.environment import RoadEnvironment, SimulationActor, VillageRoadGeometry
from simulation.actors import (
    TractorActor, PedestrianActor, MotorcycleActor,
    AutoRickshawActor, CattleActor
)
from simulation.simulator import ClosedLoopSimulator
from .scenario_base import BaseScenario
from .difficulty import DifficultyLevel


class UnmarkedVillageRoadScenario(BaseScenario):
    """Authentic Indian Rural Road Scenario with non-rectangular road geometry,
    potholes, speed bumps, irregular shoulders, and behaviorally intelligent dynamic hazards.
    """

    def __init__(
        self,
        difficulty: DifficultyLevel = DifficultyLevel.HARD,
        duration_seconds: float = 18.0,
        dt: float = 0.05
    ):
        super().__init__(
            name="UnmarkedVillageRoadScenario",
            difficulty=difficulty,
            duration_seconds=duration_seconds,
            dt=dt
        )

    def setup_environment(self):
        """Populate hazards and road surface features corresponding to difficulty."""
        if self.difficulty == DifficultyLevel.EASY:
            auto = AutoRickshawActor(
                id="parked_auto_shoulder",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=25.0, y=1.8, speed_mps=0.0, yaw_rad=0.0,
                length_m=2.6, width_m=1.3, is_static=False
            )
            self.env.add_actor(auto)

        elif self.difficulty == DifficultyLevel.MEDIUM:
            boulder = SimulationActor(
                id="fallen_boulder",
                obstacle_class=ObstacleClass.STATIC_DEBRIS,
                x=30.0, y=-0.8, speed_mps=0.0, yaw_rad=0.0,
                length_m=1.1, width_m=1.1, is_static=True
            )
            auto = AutoRickshawActor(
                id="slow_auto_ahead",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=45.0, y=1.5, speed_mps=0.0, yaw_rad=0.0,
                length_m=2.6, width_m=1.3, is_static=False
            )
            self.env.add_actor(boulder)
            self.env.add_actor(auto)

        elif self.difficulty in [DifficultyLevel.HARD, DifficultyLevel.EXTREME]:
            tractor_speed = 3.5 if self.difficulty == DifficultyLevel.HARD else 4.8
            tractor = TractorActor(
                id="oncoming_tractor",
                obstacle_class=ObstacleClass.TRUCK,
                x=38.0, y=0.7, speed_mps=tractor_speed,
                base_lateral_y=0.7,
                drift_amplitude_m=0.55 if self.difficulty == DifficultyLevel.HARD else 0.75
            )
            ped = PedestrianActor(
                id="crossing_pedestrian",
                obstacle_class=ObstacleClass.PEDESTRIAN,
                x=22.0, y=-2.0, speed_mps=1.4,
                crossing_target_y=2.0
            )
            auto = AutoRickshawActor(
                id="parked_auto_rickshaw",
                obstacle_class=ObstacleClass.AUTO_RICKSHAW,
                x=14.0, y=1.5, speed_mps=0.0, yaw_rad=0.0,
                target_lane_y=0.6
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

    def spawn_oncoming_tractor(self, dist_ahead: float = 35.0, y: float = 0.8, speed_mps: float = 3.5):
        """Dynamically inject an oncoming agricultural tractor with center-drift behavior."""
        ego_x = self.simulator.state.pose.position.x
        tractor = TractorActor(
            id=f"oncoming_tractor_{len(self.env.actors)+1}",
            obstacle_class=ObstacleClass.TRUCK,
            x=ego_x + dist_ahead,
            y=y,
            speed_mps=abs(speed_mps),
            base_lateral_y=y,
            drift_amplitude_m=0.6
        )
        self.env.add_actor(tractor)
        return tractor

    def spawn_oncoming_motorcycle(self, dist_ahead: float = 30.0, y: float = 1.4, speed_mps: float = 5.0):
        """Dynamically inject an oncoming fast motorcycle executing a dynamic corridor cut-in."""
        ego_x = self.simulator.state.pose.position.x
        bike = MotorcycleActor(
            id=f"oncoming_motorcycle_{len(self.env.actors)+1}",
            obstacle_class=ObstacleClass.MOTORCYCLE,
            x=ego_x + dist_ahead,
            y=y,
            speed_mps=abs(speed_mps),
            cut_in_target_y=-0.2
        )
        self.env.add_actor(bike)
        return bike

    def spawn_crossing_pedestrian(self, dist_ahead: float = 20.0, start_y: float = -2.2, speed_mps: float = 1.4):
        """Dynamically inject a villager with hesitation and darting behavior."""
        ego_x = self.simulator.state.pose.position.x
        ped = PedestrianActor(
            id=f"crossing_pedestrian_{len(self.env.actors)+1}",
            obstacle_class=ObstacleClass.PEDESTRIAN,
            x=ego_x + dist_ahead,
            y=start_y,
            speed_mps=abs(speed_mps),
            crossing_target_y=2.2
        )
        self.env.add_actor(ped)
        return ped

    def spawn_cattle(self, dist_ahead: float = 24.0, y: float = -1.2, speed_mps: float = 0.6):
        """Dynamically inject cattle with wandering and center-corridor freeze behavior."""
        ego_x = self.simulator.state.pose.position.x
        cow = CattleActor(
            id=f"wandering_cattle_{len(self.env.actors)+1}",
            obstacle_class=ObstacleClass.CATTLE_ANIMAL,
            x=ego_x + dist_ahead,
            y=y,
            speed_mps=abs(speed_mps),
            crossing_dir=1.0 if y < 0 else -1.0
        )
        self.env.add_actor(cow)
        return cow

    def spawn_parked_auto(self, dist_ahead: float = 18.0, y: float = 1.6):
        """Dynamically inject a roadside parked auto rickshaw that pulls out into the lane."""
        ego_x = self.simulator.state.pose.position.x
        auto = AutoRickshawActor(
            id=f"parked_auto_{len(self.env.actors)+1}",
            obstacle_class=ObstacleClass.AUTO_RICKSHAW,
            x=ego_x + dist_ahead,
            y=y,
            speed_mps=0.0,
            target_lane_y=0.6
        )
        self.env.add_actor(auto)
        return auto


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
