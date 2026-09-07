"""
Dynamic and Static actor definitions for simulation.
Models behaviorally intelligent non-ego Indian road users:
  - TractorActor: Occasionally encroaches toward centerline to avoid shoulder erosion
  - PedestrianActor: Multi-modal states (waiting, crossing, hesitating stop, darting)
  - MotorcycleActor: Dynamic lateral cut-in and swerving maneuvers
  - AutoRickshawActor: Roadside pull-out and lane merge acceleration
  - CattleActor: Slow crossing, corridor freeze, and erratic turnback
"""
import math
import random
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from interfaces import ObstacleClass, Point3D, Vector3D, TrackedObstacle, BoundingBox3D, EgoVehicleState


class ActorBehaviorState(str, Enum):
    """Behavioral mode of a simulated road actor."""
    NOMINAL = "NOMINAL"
    CENTER_DRIFT = "CENTER_DRIFT"
    WAITING_ROADSIDE = "WAITING_ROADSIDE"
    CROSSING_NORMAL = "CROSSING_NORMAL"
    HESITATION_STOP = "HESITATION_STOP"
    SUDDEN_DART = "SUDDEN_DART"
    CUTTING_IN = "CUTTING_IN"
    PARKED_ROADSIDE = "PARKED_ROADSIDE"
    PULLING_OUT = "PULLING_OUT"
    ROAD_FREEZE = "ROAD_FREEZE"
    ERRATIC_TURNBACK = "ERRATIC_TURNBACK"


@dataclass
class SimulationActor:
    """Base simulated actor in the scene with kinematic motion."""
    id: str
    obstacle_class: ObstacleClass
    x: float
    y: float
    z: float = 0.0
    yaw_rad: float = 0.0
    speed_mps: float = 0.0
    length_m: float = 2.5
    width_m: float = 1.4
    height_m: float = 1.6
    is_static: bool = False
    trajectory_waypoints: List[Tuple[float, float]] = field(default_factory=list)
    current_wp_idx: int = 0
    behavior_state: ActorBehaviorState = ActorBehaviorState.NOMINAL
    behavior_description: str = "Nominal trajectory"

    def step(self, dt: float, ego_state: Optional[EgoVehicleState] = None):
        """Update actor position based on velocity and waypoint sequence."""
        if self.is_static:
            return

        if self.trajectory_waypoints and self.current_wp_idx < len(self.trajectory_waypoints):
            target_x, target_y = self.trajectory_waypoints[self.current_wp_idx]
            dx = target_x - self.x
            dy = target_y - self.y
            dist = math.hypot(dx, dy)

            if dist < 0.8:
                self.current_wp_idx += 1
                if self.current_wp_idx >= len(self.trajectory_waypoints):
                    self.speed_mps = 0.0
                    return
                target_x, target_y = self.trajectory_waypoints[self.current_wp_idx]
                dx = target_x - self.x
                dy = target_y - self.y
                dist = math.hypot(dx, dy)

            self.yaw_rad = math.atan2(dy, dx)
            step_dist = min(dist, self.speed_mps * dt)
            self.x += step_dist * math.cos(self.yaw_rad)
            self.y += step_dist * math.sin(self.yaw_rad)
        else:
            # Constant velocity motion along yaw
            self.x += self.speed_mps * math.cos(self.yaw_rad) * dt
            self.y += self.speed_mps * math.sin(self.yaw_rad) * dt

    def to_tracked_obstacle(self, ego_x: float, ego_y: float) -> TrackedObstacle:
        """Convert simulation actor to Perception TrackedObstacle."""
        rel_x = (self.x - ego_x)
        rel_y = (self.y - ego_y)
        dist = math.hypot(rel_x, rel_y)
        vx = self.speed_mps * math.cos(self.yaw_rad)
        vy = self.speed_mps * math.sin(self.yaw_rad)

        return TrackedObstacle(
            id=self.id,
            obstacle_class=self.obstacle_class,
            confidence=0.98,
            bbox=BoundingBox3D(
                center=Point3D(x=self.x, y=self.y, z=self.z + self.height_m / 2.0),
                size=Vector3D(x=self.length_m, y=self.width_m, z=self.height_m),
                yaw_rad=self.yaw_rad
            ),
            velocity=Vector3D(x=vx, y=vy, z=0.0),
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
            distance_m=dist,
            is_static=self.is_static
        )


@dataclass
class TractorActor(SimulationActor):
    """Heavy rural tractor that periodically drifts toward centerline to avoid potholes and eroded shoulders."""
    base_lateral_y: float = 0.7
    drift_period_s: float = 6.0
    drift_amplitude_m: float = 0.55
    elapsed_time_s: float = 0.0

    def __post_init__(self):
        self.obstacle_class = ObstacleClass.TRUCK
        self.length_m = 4.5
        self.width_m = 2.1
        self.height_m = 2.4
        self.yaw_rad = math.pi  # Oncoming traffic heading
        self.base_lateral_y = self.y

    def step(self, dt: float, ego_state: Optional[EgoVehicleState] = None):
        self.elapsed_time_s += dt
        
        # Center drift oscillation: sine wave encroaching toward y=0.0
        drift_offset = self.drift_amplitude_m * math.sin((2.0 * math.pi / self.drift_period_s) * self.elapsed_time_s)
        target_y = self.base_lateral_y - abs(drift_offset)
        
        # Move oncoming along -X
        self.x -= self.speed_mps * dt
        self.y += (target_y - self.y) * min(1.0, 2.5 * dt)
        
        if abs(self.y) < 0.35:
            self.behavior_state = ActorBehaviorState.CENTER_DRIFT
            self.behavior_description = "Encroaching centerline to bypass shoulder edge"
        else:
            self.behavior_state = ActorBehaviorState.NOMINAL
            self.behavior_description = "Cruising along oncoming corridor"


@dataclass
class PedestrianActor(SimulationActor):
    """Pedestrian with multi-modal behavior: waiting, crossing, hesitating stop, and darting."""
    crossing_target_y: float = 2.2
    hesitation_prob: float = 0.35
    dart_prob: float = 0.25
    decision_made: bool = False
    hesitation_timer_s: float = 0.0

    def __post_init__(self):
        self.obstacle_class = ObstacleClass.PEDESTRIAN
        self.length_m = 0.6
        self.width_m = 0.6
        self.height_m = 1.7
        self.behavior_state = ActorBehaviorState.WAITING_ROADSIDE
        self.behavior_description = "Waiting at roadside"

    def step(self, dt: float, ego_state: Optional[EgoVehicleState] = None):
        dist_to_ego = 999.0
        if ego_state is not None:
            dist_to_ego = math.hypot(self.x - ego_state.pose.position.x, self.y - ego_state.pose.position.y)

        # Interactive Decision Point when ego is within 22m
        if not self.decision_made and dist_to_ego < 22.0:
            self.decision_made = True
            rand_val = random.random()
            if rand_val < self.hesitation_prob:
                self.behavior_state = ActorBehaviorState.HESITATION_STOP
                self.speed_mps = 0.0
                self.behavior_description = "Hesitating and halting at road edge"
            elif rand_val < (self.hesitation_prob + self.dart_prob):
                self.behavior_state = ActorBehaviorState.SUDDEN_DART
                self.speed_mps = 2.4
                self.behavior_description = "Suddenly darting across roadway"
            else:
                self.behavior_state = ActorBehaviorState.CROSSING_NORMAL
                self.speed_mps = 1.3
                self.behavior_description = "Crossing roadway normally"

        if self.behavior_state == ActorBehaviorState.HESITATION_STOP:
            self.hesitation_timer_s += dt
            if self.hesitation_timer_s > 1.8:
                # Resume crossing after hesitation
                self.behavior_state = ActorBehaviorState.CROSSING_NORMAL
                self.speed_mps = 1.1
                self.behavior_description = "Resuming crossing after hesitation"

        if self.behavior_state in [ActorBehaviorState.CROSSING_NORMAL, ActorBehaviorState.SUDDEN_DART]:
            direction = 1.0 if self.crossing_target_y > self.y else -1.0
            self.yaw_rad = math.pi / 2.0 if direction > 0 else -math.pi / 2.0
            self.y += direction * self.speed_mps * dt
            if abs(self.y - self.crossing_target_y) < 0.2:
                self.speed_mps = 0.0
                self.behavior_state = ActorBehaviorState.NOMINAL
                self.behavior_description = "Reached opposite road verge"


@dataclass
class MotorcycleActor(SimulationActor):
    """Two-wheeler executing dynamic swerving and rapid corridor cut-in maneuvers."""
    cut_in_triggered: bool = False
    cut_in_target_y: float = -0.3
    cut_in_rate: float = 1.8
    elapsed_time_s: float = 0.0

    def __post_init__(self):
        self.obstacle_class = ObstacleClass.MOTORCYCLE
        self.length_m = 2.0
        self.width_m = 0.8
        self.height_m = 1.4
        self.yaw_rad = math.pi  # Oncoming

    def step(self, dt: float, ego_state: Optional[EgoVehicleState] = None):
        self.elapsed_time_s += dt
        dist_to_ego = 999.0
        if ego_state is not None:
            dist_to_ego = self.x - ego_state.pose.position.x

        # Trigger sharp cut-in when approaching 25m ahead
        if not self.cut_in_triggered and dist_to_ego < 26.0:
            self.cut_in_triggered = True
            self.behavior_state = ActorBehaviorState.CUTTING_IN
            self.behavior_description = "Executing aggressive cut-in into ego lane"

        if self.behavior_state == ActorBehaviorState.CUTTING_IN:
            dy = self.cut_in_target_y - self.y
            self.y += math.copysign(min(abs(dy), self.cut_in_rate * dt), dy)
            self.yaw_rad = math.pi + math.atan2(dy * 1.5, -self.speed_mps)
            if abs(self.y - self.cut_in_target_y) < 0.1:
                self.yaw_rad = math.pi
                self.behavior_state = ActorBehaviorState.NOMINAL
                self.behavior_description = "Cut-in complete, stabilized in lane"

        self.x -= self.speed_mps * dt


@dataclass
class AutoRickshawActor(SimulationActor):
    """Three-wheeler parked on the road shoulder that pulls out into the active lane."""
    pull_out_triggered: bool = False
    target_lane_y: float = 0.6
    target_cruise_v: float = 4.5
    pull_out_dist_trigger_m: float = 24.0

    def __post_init__(self):
        self.obstacle_class = ObstacleClass.AUTO_RICKSHAW
        self.length_m = 2.6
        self.width_m = 1.4
        self.height_m = 1.7
        self.speed_mps = 0.0
        self.is_static = False
        self.behavior_state = ActorBehaviorState.PARKED_ROADSIDE
        self.behavior_description = "Parked on road shoulder"

    def step(self, dt: float, ego_state: Optional[EgoVehicleState] = None):
        dist_to_ego = 999.0
        if ego_state is not None:
            dist_to_ego = self.x - ego_state.pose.position.x

        # Pull-out trigger when ego approaches
        if not self.pull_out_triggered and 0.0 < dist_to_ego < self.pull_out_dist_trigger_m:
            self.pull_out_triggered = True
            self.behavior_state = ActorBehaviorState.PULLING_OUT
            self.behavior_description = "Pulling out from roadside into travel corridor"

        if self.behavior_state == ActorBehaviorState.PULLING_OUT:
            # Accelerate from 0 to cruising speed
            self.speed_mps = min(self.target_cruise_v, self.speed_mps + 2.2 * dt)
            dy = self.target_lane_y - self.y
            self.y += math.copysign(min(abs(dy), 1.2 * dt), dy)
            self.yaw_rad = math.atan2(dy * 1.8, max(0.5, self.speed_mps))
            self.x += self.speed_mps * dt

            if abs(self.y - self.target_lane_y) < 0.1:
                self.yaw_rad = 0.0
                self.behavior_state = ActorBehaviorState.NOMINAL
                self.behavior_description = "Cruising forward in lane after pull-out"
        elif self.behavior_state == ActorBehaviorState.NOMINAL:
            self.x += self.speed_mps * dt


@dataclass
class CattleActor(SimulationActor):
    """Cattle with rural behavior: slow crossing, freezing in the center corridor, and startled turnback."""
    crossing_dir: float = 1.0  # +1 (left-to-right) or -1 (right-to-left)
    freeze_prob: float = 0.45
    freeze_timer_s: float = 0.0
    has_frozen: bool = False

    def __post_init__(self):
        self.obstacle_class = ObstacleClass.CATTLE_ANIMAL
        self.length_m = 2.2
        self.width_m = 1.0
        self.height_m = 1.5
        self.speed_mps = 0.65
        self.yaw_rad = math.pi / 2.0 if self.crossing_dir > 0 else -math.pi / 2.0
        self.behavior_state = ActorBehaviorState.CROSSING_NORMAL
        self.behavior_description = "Slowly wandering across road"

    def step(self, dt: float, ego_state: Optional[EgoVehicleState] = None):
        dist_to_ego = 999.0
        if ego_state is not None:
            dist_to_ego = math.hypot(self.x - ego_state.pose.position.x, self.y - ego_state.pose.position.y)

        # Mid-road freeze when reaching center corridor (|y| < 0.6) and vehicle approaches
        if not self.has_frozen and abs(self.y) < 0.6 and dist_to_ego < 20.0:
            self.has_frozen = True
            if random.random() < self.freeze_prob:
                self.behavior_state = ActorBehaviorState.ROAD_FREEZE
                self.speed_mps = 0.0
                self.behavior_description = "Stationary freeze directly in center corridor"

        if self.behavior_state == ActorBehaviorState.ROAD_FREEZE:
            self.freeze_timer_s += dt
            if self.freeze_timer_s > 3.0:
                # Resume slow walk
                self.behavior_state = ActorBehaviorState.CROSSING_NORMAL
                self.speed_mps = 0.55
                self.behavior_description = "Resuming slow crossing"

        if self.behavior_state == ActorBehaviorState.CROSSING_NORMAL:
            self.y += self.crossing_dir * self.speed_mps * dt
