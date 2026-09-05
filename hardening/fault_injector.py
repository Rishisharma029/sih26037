"""Adversarial Fault Injection Engine testing 10 critical edge failure modes."""
import random
import math
from typing import List, Optional
from interfaces import (
    EgoVehicleState, PerceptionOutput, PredictionOutput,
    TrackedObstacle, BoundingBox3D, Point3D, Vector3D,
    ObstacleClass, FreeSpaceCorridor, CorridorBoundaryPoint
)


class AdversarialFaultInjector:
    """Deliberately perturbs sensor data, perception frames, and compute pipelines

    to stress-test system resilience and trigger fail-operational fallbacks.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)

    # 1. Sensor Noise (Gaussian LiDAR/Radar jitter)
    def inject_sensor_noise(self, perception: PerceptionOutput, sigma_pos: float = 0.3, sigma_vel: float = 0.5) -> PerceptionOutput:
        noisy_obstacles = []
        for obs in perception.obstacles:
            nx = obs.bbox.center.x + self.rng.gauss(0.0, sigma_pos)
            ny = obs.bbox.center.y + self.rng.gauss(0.0, sigma_pos)
            nvx = obs.velocity.x + self.rng.gauss(0.0, sigma_vel)
            noisy_obs = obs.model_copy(deep=True)
            noisy_obs.bbox.center.x = nx
            noisy_obs.bbox.center.y = ny
            noisy_obs.velocity.x = nvx
            noisy_obstacles.append(noisy_obs)
        return perception.model_copy(update={"obstacles": noisy_obstacles}, deep=True)

    # 2. Detection Failure / False Negatives (Missed Obstacle Detection)
    def inject_detection_dropout(self, perception: PerceptionOutput, drop_rate: float = 0.4) -> PerceptionOutput:
        survived = [obs for obs in perception.obstacles if self.rng.random() > drop_rate]
        return perception.model_copy(update={"obstacles": survived}, deep=True)

    # 3. Occlusion (Blind Spots behind large vehicles)
    def inject_occlusion(self, perception: PerceptionOutput, ego_x: float = 0.0) -> PerceptionOutput:
        # Find large trucks or buses
        occluders = [obs for obs in perception.obstacles if obs.obstacle_class in [ObstacleClass.TRUCK, ObstacleClass.BUS]]
        if not occluders:
            return perception
        visible = []
        for obs in perception.obstacles:
            is_hidden = False
            for occ in occluders:
                if occ.id != obs.id and occ.bbox.center.x < obs.bbox.center.x and abs(occ.bbox.center.y - obs.bbox.center.y) < 1.5:
                    is_hidden = True
                    break
            if not is_hidden:
                visible.append(obs)
        return perception.model_copy(update={"obstacles": visible}, deep=True)

    # 4. Sudden Incursion / Pop-Up Obstacle (Sub-5m dynamic appearance)
    def inject_sudden_incursion(self, perception: PerceptionOutput, ego_x: float, ego_y: float) -> PerceptionOutput:
        popup = TrackedObstacle(
            id="sudden_popup_motorcycle",
            obstacle_class=ObstacleClass.MOTORCYCLE,
            confidence=0.98,
            bbox=BoundingBox3D(
                center=Point3D(x=ego_x + 3.8, y=ego_y + 0.5, z=0.5),
                size=Vector3D(x=2.0, y=0.8, z=1.2),
                yaw_rad=math.pi
            ),
            velocity=Vector3D(x=-4.0, y=0.0, z=0.0),
            distance_m=3.8,
            is_static=False
        )
        obs_list = list(perception.obstacles) + [popup]
        return perception.model_copy(update={"obstacles": obs_list}, deep=True)

    # 5. False Positive Ghost Detections
    def inject_ghost_detections(self, perception: PerceptionOutput, ego_x: float, ego_y: float, count: int = 3) -> PerceptionOutput:
        ghosts = []
        for i in range(count):
            ghosts.append(TrackedObstacle(
                id=f"ghost_radar_reflection_{i}",
                obstacle_class=ObstacleClass.STATIC_DEBRIS,
                confidence=0.45,
                bbox=BoundingBox3D(
                    center=Point3D(x=ego_x + 5.0 + i * 4.0, y=ego_y + (i - 1) * 1.2, z=0.2),
                    size=Vector3D(x=0.5, y=0.5, z=0.2),
                    yaw_rad=0.0
                ),
                velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                distance_m=5.0 + i * 4.0,
                is_static=True
            ))
        return perception.model_copy(update={"obstacles": list(perception.obstacles) + ghosts}, deep=True)

    # 6. Multiple Swarming Obstacles (10+ simultaneous conflicting agents)
    def inject_swarming_crowd(self, perception: PerceptionOutput, ego_x: float, ego_y: float) -> PerceptionOutput:
        swarm = []
        for i in range(12):
            swarm.append(TrackedObstacle(
                id=f"swarm_actor_{i}",
                obstacle_class=ObstacleClass.PEDESTRIAN if i % 2 == 0 else ObstacleClass.BICYCLE,
                confidence=0.90,
                bbox=BoundingBox3D(
                    center=Point3D(x=ego_x + 6.0 + (i * 2.0), y=ego_y + ((i % 5) - 2) * 0.8, z=0.5),
                    size=Vector3D(x=0.6, y=0.6, z=1.6),
                    yaw_rad=0.0
                ),
                velocity=Vector3D(x=self.rng.uniform(-1.0, 1.0), y=self.rng.uniform(-0.5, 0.5), z=0.0),
                distance_m=6.0 + (i * 2.0),
                is_static=False
            ))
        return perception.model_copy(update={"obstacles": swarm}, deep=True)

    # 7. Low Visibility / Adverse Weather (Simulates dense fog / monsoon rain)
    def inject_low_visibility(self, perception: PerceptionOutput) -> PerceptionOutput:
        degraded = []
        for obs in perception.obstacles:
            d_obs = obs.model_copy(deep=True)
            d_obs.confidence = max(0.2, obs.confidence * 0.4) # Severely degraded confidence
            degraded.append(d_obs)
        health = {"camera": False, "lidar": True, "radar": True, "gnss": True}
        return perception.model_copy(update={"obstacles": degraded, "sensor_health": health}, deep=True)
