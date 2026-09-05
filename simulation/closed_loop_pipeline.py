"""End-to-End Closed-Loop Autonomy Pipeline integrating Perception, Prediction, Planning, Safety, and Control."""
import math
from typing import Dict, Any, List, Optional, Tuple

from interfaces import (
    EgoVehicleState, ControlCommand, PlannedTrajectory, SafeTrajectory,
    PerceptionOutput, PredictionOutput, FreeSpaceCorridor, TrackedObstacle,
    BoundingBox3D, Point3D, Vector3D, GearMode, SafetyAction, BehaviorMode
)
from simulation.environment import SimulationEnvironment, RoadGeometry, RoadCorridorProfile
from simulation.vehicle_model import KinematicBicycleModel
from perception.boundary_detector import FreeSpaceBoundaryDetector
from perception.sensor_fusion import PerceptionFusion
from perception.world_model import WorldModel
from prediction.trajectory_predictor import TrajectoryPredictor
from planning.local_planner import AdaptiveLatticePlanner
from collision_avoidance.safety_supervisor import SafetySupervisoryLayer
from vehicle_control.drive_by_wire_bridge import DriveByWireBridge


class ClosedLoopMetrics:
    """Tracks end-to-end performance and safety telemetry over simulation trajectory."""

    def __init__(self):
        self.step_count = 0
        self.total_distance_m = 0.0
        self.crosstrack_errors: List[float] = []
        self.heading_errors: List[float] = []
        self.speed_errors: List[float] = []
        self.min_corridor_margin_m = 999.0
        self.min_obstacle_clearance_m = 999.0
        self.safety_interventions = 0
        self.collisions = 0

    @property
    def rms_crosstrack_error_m(self) -> float:
        if not self.crosstrack_errors:
            return 0.0
        sq_sum = sum(e * e for e in self.crosstrack_errors)
        return round(math.sqrt(sq_sum / len(self.crosstrack_errors)), 3)

    @property
    def rms_heading_error_deg(self) -> float:
        if not self.heading_errors:
            return 0.0
        sq_sum = sum(e * e for e in self.heading_errors)
        return round(math.degrees(math.sqrt(sq_sum / len(self.heading_errors))), 2)

    @property
    def mean_speed_tracking_error_mps(self) -> float:
        if not self.speed_errors:
            return 0.0
        return round(sum(abs(e) for e in self.speed_errors) / len(self.speed_errors), 3)


class ClosedLoopAutonomyPipeline:
    """Master orchestrator executing the full Sense-Plan-Act closed loop in real-time."""

    def __init__(
        self,
        environment: Optional[SimulationEnvironment] = None,
        target_cruise_speed_mps: float = 6.0,
        dt: float = 0.05
    ):
        self.dt = dt
        self.target_cruise_speed_mps = target_cruise_speed_mps

        # Simulation & Dynamics
        if environment is None:
            geom = RoadGeometry(length_m=120.0, base_width_m=6.5, profile=RoadCorridorProfile.UNMARKED_VILLAGE)
            self.env = SimulationEnvironment(geometry=geom, dt=dt)
        else:
            self.env = environment

        # Subsystems
        self.boundary_detector = FreeSpaceBoundaryDetector()
        self.fusion = PerceptionFusion()
        self.world_model = WorldModel(timestamp=0.0)
        self.predictor = TrajectoryPredictor(horizon_seconds=3.0, dt=0.2, mode="ensemble")
        self.planner = AdaptiveLatticePlanner(horizon_seconds=3.0, dt=0.2)
        self.safety_layer = SafetySupervisoryLayer(aeb_ttc_threshold_s=0.85)
        self.dbw_bridge = DriveByWireBridge()

        # State
        self.metrics = ClosedLoopMetrics()
        self.current_plan: Optional[PlannedTrajectory] = None
        self.current_safe_plan: Optional[SafeTrajectory] = None
        self.latest_command: Optional[ControlCommand] = None
        self.plan_replan_interval_steps = 4 # Re-plan at 5 Hz (every 4 steps of 0.05s = 0.2s)

    def run_step(self) -> Tuple[EgoVehicleState, SafeTrajectory, ControlCommand, Dict[str, Any]]:
        """Executes a single closed-loop autonomy step."""
        ego_state = self.env.ego_state
        step_idx = self.metrics.step_count

        # 1. SENSE & PERCEPTION: Extract detected obstacles & drivable corridor
        obstacles: List[TrackedObstacle] = []
        for actor in self.env.actors:
            dx = actor.x - ego_state.pose.position.x
            dy = actor.y - ego_state.pose.position.y
            dist = math.hypot(dx, dy)
            vx = actor.speed_mps * math.cos(actor.yaw_rad)
            vy = actor.speed_mps * math.sin(actor.yaw_rad)

            # Check sensor field-of-view (within 45m range)
            if dist <= 45.0:
                obs = TrackedObstacle(
                    id=actor.id,
                    obstacle_class=actor.obstacle_class,
                    confidence=0.95,
                    bbox=BoundingBox3D(
                        center=Point3D(x=actor.x, y=actor.y, z=0.5),
                        size=Vector3D(x=actor.length_m, y=actor.width_m, z=1.5),
                        yaw_rad=actor.yaw_rad
                    ),
                    velocity=Vector3D(x=vx, y=vy, z=0.0),
                    distance_m=dist,
                    is_static=actor.is_static
                )
                obstacles.append(obs)

        d_left, d_right = self.env.geometry.get_corridor_widths(ego_state.pose.position.x)
        perception = PerceptionOutput(
            timestamp=ego_state.timestamp,
            frame_id=step_idx,
            obstacles=obstacles,
            drivable_corridor=FreeSpaceCorridor(
                timestamp=ego_state.timestamp,
                boundary_points=[],
                average_width_m=float(d_left - d_right)
            )
        )

        # 2. PREDICTION: Multi-modal trajectory and intention forecasting
        prediction = self.predictor.predict(perception, ego_speed=ego_state.twist.speed_mps)

        # 3. PLANNING: Re-plan every 0.2s or on first step
        if self.current_plan is None or (step_idx % self.plan_replan_interval_steps == 0):
            self.current_plan = self.planner.plan(
                ego_state=ego_state,
                perception=perception,
                prediction=prediction,
                target_cruise_speed_mps=self.target_cruise_speed_mps
            )

        # 4. SAFETY SUPERVISION: Independent invariant check & collision arbitration
        self.current_safe_plan = self.safety_layer.supervise(
            planned=self.current_plan,
            ego_state=ego_state,
            perception=perception,
            prediction=prediction
        )

        # Trigger emergency replan if requested by safety supervisor
        if self.current_safe_plan.replan_recommended:
            self.metrics.safety_interventions += 1
            self.current_plan = self.planner.plan(
                ego_state=ego_state,
                perception=perception,
                prediction=prediction,
                target_cruise_speed_mps=max(2.0, self.target_cruise_speed_mps * 0.5)
            )
            self.current_safe_plan = self.safety_layer.supervise(
                planned=self.current_plan,
                ego_state=ego_state,
                perception=perception,
                prediction=prediction
            )

        # 5. VEHICLE CONTROL: Drive-by-wire steering & longitudinal command
        self.latest_command = self.dbw_bridge.generate_command(
            ego_state=ego_state,
            trajectory=self.current_safe_plan,
            dt=self.dt
        )

        # 6. ACTUATION & DYNAMICS: Step physical simulation
        new_ego_state, raw_sensor_dict = self.env.step(self.latest_command)

        # 7. METRICS & TELEMETRY
        self._update_metrics(new_ego_state, d_left, d_right, obstacles)

        telemetry_frame = {
            "timestamp": new_ego_state.timestamp,
            "step": self.metrics.step_count,
            "ego": {
                "x": new_ego_state.pose.position.x,
                "y": new_ego_state.pose.position.y,
                "speed_kph": new_ego_state.twist.speed_mps * 3.6,
                "heading_deg": math.degrees(new_ego_state.pose.heading_rad)
            },
            "safety_action": self.current_safe_plan.safety_action.value,
            "min_ttc_s": self.current_safe_plan.min_ttc_seconds,
            "corridor_margin_m": self.metrics.min_corridor_margin_m,
            "rms_cte_m": self.metrics.rms_crosstrack_error_m
        }

        return new_ego_state, self.current_safe_plan, self.latest_command, telemetry_frame

    def _update_metrics(
        self,
        state: EgoVehicleState,
        d_left: float,
        d_right: float,
        obstacles: List[TrackedObstacle]
    ):
        self.metrics.step_count += 1
        speed = state.twist.speed_mps
        self.metrics.total_distance_m += speed * self.dt

        # Cross-track error against centerline (y=0 for straight sections or geometry)
        cx, cy, cyaw = self.env.geometry.get_centerline_point(state.pose.position.x)
        cte = state.pose.position.y - cy
        self.metrics.crosstrack_errors.append(cte)

        head_err = state.pose.heading_rad - cyaw
        self.metrics.heading_errors.append((head_err + math.pi) % (2.0 * math.pi) - math.pi)

        # Speed tracking error
        target_v = self.current_safe_plan.waypoints[0].speed_mps if self.current_safe_plan and self.current_safe_plan.waypoints else self.target_cruise_speed_mps
        self.metrics.speed_errors.append(target_v - speed)

        # Corridor margins
        m_left = d_left - state.pose.position.y
        m_right = state.pose.position.y - d_right
        margin = min(m_left, m_right)
        self.metrics.min_corridor_margin_m = min(self.metrics.min_corridor_margin_m, margin)

        # Obstacle clearance
        for obs in obstacles:
            if obs.distance_m < self.metrics.min_obstacle_clearance_m:
                self.metrics.min_obstacle_clearance_m = obs.distance_m
            if obs.distance_m < 0.6:
                self.metrics.collisions += 1
