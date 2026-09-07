"""Abstract base class for benchmark scenarios."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from interfaces import (
    EgoVehicleState, ControlCommand, SafeTrajectory,
    ObstacleClass, RoadAnomaly, Point3D, TraversabilityClass
)
from simulation.simulator import ClosedLoopSimulator
from simulation.environment import RoadEnvironment
from simulation.actors import (
    SimulationActor, TractorActor, PedestrianActor,
    MotorcycleActor, AutoRickshawActor, CattleActor
)
from .difficulty import DifficultyLevel


class BaseScenario(ABC):
    """Base harness executing a closed-loop scenario episode across difficulty levels."""

    def __init__(
        self,
        name: str,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        duration_seconds: float = 10.0,
        dt: float = 0.05
    ):
        self.name = name
        self.difficulty = difficulty
        self.duration_seconds = duration_seconds
        self.dt = dt
        self.env = RoadEnvironment(dt=dt)
        self.setup_environment()
        self.simulator = ClosedLoopSimulator(env=self.env, dt=self.dt)
        self.history_states: List[EgoVehicleState] = []
        self.history_commands: List[ControlCommand] = []

    @abstractmethod
    def setup_environment(self):
        """Populate obstacles, road properties, and anomalies based on difficulty."""
        pass

    def run_step(self, command: ControlCommand):
        state, raw_sensor = self.simulator.step(command)
        self.history_states.append(state)
        self.history_commands.append(command)
        return state, raw_sensor

    def spawn_oncoming_tractor(self, dist_ahead: float = 35.0, y: float = 0.8, speed_mps: float = 3.5):
        """Dynamically inject an oncoming agricultural tractor with center-drift behavior."""
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
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
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
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
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
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
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
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
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
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

    def spawn_boulder(self, dist_ahead: float = 26.0, y: float = -0.9, size_m: float = 1.2):
        """Dynamically inject a fallen roadside boulder."""
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
        boulder = SimulationActor(
            id=f"boulder_{len(self.env.actors)+1}",
            obstacle_class=ObstacleClass.STATIC_DEBRIS,
            x=ego_x + dist_ahead,
            y=y,
            speed_mps=0.0,
            yaw_rad=0.0,
            length_m=size_m,
            width_m=size_m,
            is_static=True
        )
        self.env.add_actor(boulder)
        return boulder

    def spawn_pothole(self, dist_ahead: float = 25.0, y: float = 0.0, depth_m: float = -0.15, radius_m: float = 0.75):
        """Dynamically inject a severe structural pothole."""
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
        pothole = RoadAnomaly(
            id=f"pothole_{len(self.env.anomalies)+1}",
            anomaly_type="POTHOLE",
            position=Point3D(x=round(ego_x + dist_ahead, 2), y=round(y, 2), z=round(depth_m, 2)),
            radius_m=radius_m,
            depth_or_height_m=depth_m,
            traversability_class=TraversabilityClass.POTHOLE,
            severity=0.90,
            is_passable=False,
            max_safe_speed_mps=0.0,
            traversability_score=0.05,
            description=f"Deep crater pothole ({int(depth_m*100)}cm depth)"
        )
        self.env.add_anomaly(pothole)
        return pothole

    def spawn_waterlogged_area(self, dist_ahead: float = 28.0, y: float = 0.2, depth_m: float = -0.16, radius_m: float = 1.6):
        """Dynamically inject a flooded waterlogged hazard zone."""
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
        waterlog = RoadAnomaly(
            id=f"waterlog_{len(self.env.anomalies)+1}",
            anomaly_type="WATER_LOGGING",
            position=Point3D(x=round(ego_x + dist_ahead, 2), y=round(y, 2), z=round(depth_m, 2)),
            radius_m=radius_m,
            depth_or_height_m=depth_m,
            traversability_class=TraversabilityClass.WATERLOGGED,
            severity=0.88,
            is_passable=False,
            max_safe_speed_mps=0.8,
            traversability_score=0.20,
            description="Submerged murky flood pool"
        )
        self.env.add_anomaly(waterlog)
        return waterlog

    def spawn_gravel_patch(self, dist_ahead: float = 22.0, y: float = -0.3, radius_m: float = 1.8):
        """Dynamically inject a loose gravel aggregate patch."""
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
        gravel = RoadAnomaly(
            id=f"gravel_{len(self.env.anomalies)+1}",
            anomaly_type="GRAVEL",
            position=Point3D(x=round(ego_x + dist_ahead, 2), y=round(y, 2), z=-0.02),
            radius_m=radius_m,
            depth_or_height_m=-0.02,
            traversability_class=TraversabilityClass.GRAVEL,
            severity=0.45,
            is_passable=True,
            max_safe_speed_mps=2.5,
            traversability_score=0.50,
            description="Loose unpaved stone gravel"
        )
        self.env.add_anomaly(gravel)
        return gravel

    def spawn_speed_bump(self, dist_ahead: float = 30.0, y: float = 0.0, height_m: float = 0.12, radius_m: float = 1.5):
        """Dynamically inject an unmarked concrete speed hump."""
        ego_x = self.simulator.state.pose.position.x if hasattr(self, "simulator") and self.simulator else 0.0
        bump = RoadAnomaly(
            id=f"speed_bump_{len(self.env.anomalies)+1}",
            anomaly_type="SPEED_BUMP",
            position=Point3D(x=round(ego_x + dist_ahead, 2), y=round(y, 2), z=round(height_m, 2)),
            radius_m=radius_m,
            depth_or_height_m=height_m,
            traversability_class=TraversabilityClass.SPEED_BUMP,
            severity=0.60,
            is_passable=True,
            max_safe_speed_mps=1.5,
            traversability_score=0.40,
            description="Unmarked steep speed hump"
        )
        self.env.add_anomaly(bump)
        return bump

