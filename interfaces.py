"""
SIH26037 Core Interfaces and Typed Contracts
Defines strictly typed Pydantic V2 schemas for every subsystem in the autonomous driving stack.
All units follow the SI standard: meters (m), seconds (s), radians (rad), meters per second (m/s),
and meters per second squared (m/s^2).
Coordinate Frame: ISO 8855 / SAE J670 Vehicle Coordinate System (X-Forward, Y-Left, Z-Up).
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# 1. Coordinate & Kinematic Primitives
# ---------------------------------------------------------------------------

class Vector2D(BaseModel):
    """2D vector in cartesian coordinates."""
    model_config = ConfigDict(extra="forbid")
    x: float = Field(..., description="Forward / Easting component (meters)")
    y: float = Field(..., description="Lateral / Northing component (meters)")


class Vector3D(BaseModel):
    """3D vector in cartesian coordinates."""
    model_config = ConfigDict(extra="forbid")
    x: float = Field(..., description="Forward component (meters)")
    y: float = Field(..., description="Lateral component (meters)")
    z: float = Field(0.0, description="Vertical component (meters)")


class Point3D(BaseModel):
    """3D position coordinate."""
    model_config = ConfigDict(extra="forbid")
    x: float = Field(..., description="X coordinate (meters)")
    y: float = Field(..., description="Y coordinate (meters)")
    z: float = Field(0.0, description="Z coordinate (meters)")


class Quaternion(BaseModel):
    """Orientation quaternion."""
    model_config = ConfigDict(extra="forbid")
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


class Pose3D(BaseModel):
    """6-DOF Pose representing position and orientation."""
    model_config = ConfigDict(extra="forbid")
    position: Point3D = Field(default_factory=lambda: Point3D(x=0.0, y=0.0, z=0.0))
    orientation: Quaternion = Field(default_factory=Quaternion)
    heading_rad: float = Field(0.0, description="Yaw angle around Z-axis in radians (-pi to pi)")


class Twist3D(BaseModel):
    """Linear and angular velocity."""
    model_config = ConfigDict(extra="forbid")
    linear: Vector3D = Field(default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0))
    angular: Vector3D = Field(default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0))
    speed_mps: float = Field(0.0, description="Absolute scalar speed in m/s")


# ---------------------------------------------------------------------------
# 2. Enumerations for Unstructured Indian Road Domains
# ---------------------------------------------------------------------------

class ObstacleClass(str, Enum):
    """Classes of objects specifically encountered on Indian roads."""
    CAR = "CAR"
    BUS = "BUS"
    TRUCK = "TRUCK"
    AUTO_RICKSHAW = "AUTO_RICKSHAW"
    MOTORCYCLE = "MOTORCYCLE"
    BICYCLE = "BICYCLE"
    PEDESTRIAN = "PEDESTRIAN"
    CATTLE_ANIMAL = "CATTLE_ANIMAL"
    PUSHCART = "PUSHCART"
    POTHOLE = "POTHOLE"
    SPEED_BUMP = "SPEED_BUMP"
    STATIC_DEBRIS = "STATIC_DEBRIS"
    UNKNOWN = "UNKNOWN"


class MotionIntent(str, Enum):
    """Predicted intention of surrounding road actors."""
    CRUISING = "CRUISING"
    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    STOPPING = "STOPPING"
    TURNING_LEFT = "TURNING_LEFT"
    TURNING_RIGHT = "TURNING_RIGHT"
    CUTTING_IN = "CUTTING_IN"
    CROSSING_PATH = "CROSSING_PATH"
    ERRATIC_SWERVE = "ERRATIC_SWERVE"
    STATIONARY = "STATIONARY"
    UNKNOWN = "UNKNOWN"


class BehaviorMode(str, Enum):
    """Operational mode decided by the Behavioral State Machine."""
    CRUISE = "CRUISE"
    FOLLOW = "FOLLOW"
    NUDGE_LEFT = "NUDGE_LEFT"
    NUDGE_RIGHT = "NUDGE_RIGHT"
    YIELD = "YIELD"
    OVERTAKE = "OVERTAKE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SAFE_STOP = "SAFE_STOP"


class SafetyAction(str, Enum):
    """Intervention triggered by the Collision Avoidance supervisory layer."""
    NONE = "NONE"
    ADAPTIVE_CRUISE_SLOWDOWN = "ADAPTIVE_CRUISE_SLOWDOWN"
    CORRIDOR_NUDGE = "CORRIDOR_NUDGE"
    CONTROL_BARRIER_OVERRIDE = "CONTROL_BARRIER_OVERRIDE"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


class GearMode(str, Enum):
    """Drive-by-wire transmission gear."""
    PARK = "PARK"
    REVERSE = "REVERSE"
    NEUTRAL = "NEUTRAL"
    DRIVE = "DRIVE"


# ---------------------------------------------------------------------------
# 3. Perception Interfaces
# ---------------------------------------------------------------------------

class BoundingBox3D(BaseModel):
    """3D oriented bounding box of an obstacle."""
    model_config = ConfigDict(extra="forbid")
    center: Point3D
    size: Vector3D = Field(..., description="Length (x), Width (y), Height (z) in meters")
    yaw_rad: float = Field(0.0, description="Heading orientation in radians")


class TrackedObstacle(BaseModel):
    """Fused object detection with tracking state and dynamic estimates."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., description="Unique persistent tracking ID")
    obstacle_class: ObstacleClass
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox3D
    velocity: Vector3D = Field(default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0))
    acceleration: Vector3D = Field(default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0))
    distance_m: float = Field(..., ge=0.0, description="Radial distance to ego vehicle")
    is_static: bool = Field(False, description="True if stationary, False if moving")


class CorridorBoundaryPoint(BaseModel):
    """Point defining a free-space corridor boundary."""
    model_config = ConfigDict(extra="forbid")
    s: float = Field(..., description="Longitudinal distance along reference path (meters)")
    d_left: float = Field(..., description="Lateral distance to left boundary/edge (meters)")
    d_right: float = Field(..., description="Lateral distance to right boundary/edge (meters)")
    curvature: float = Field(0.0, description="Local road curvature (1/m)")


class FreeSpaceCorridor(BaseModel):
    """Drivable corridor boundary without relying on painted lane markers."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    boundary_points: List[CorridorBoundaryPoint] = Field(default_factory=list)
    average_width_m: float = Field(..., ge=1.0, description="Average drivable road width")
    is_blocked: bool = Field(False, description="True if road is completely blocked")


class RoadAnomaly(BaseModel):
    """Potholes, unmarked speed humps, or broken road surfaces."""
    model_config = ConfigDict(extra="forbid")
    id: str
    anomaly_type: str = Field(..., description="'POTHOLE', 'SPEED_BUMP', 'GRAVEL', 'WATER_LOGGING'")
    position: Point3D
    radius_m: float = Field(..., ge=0.1)
    depth_or_height_m: float = Field(0.0, description="Depth if negative, height if positive")


class PerceptionOutput(BaseModel):
    """Standardized output frame from the Perception Subsystem."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float = Field(..., description="UNIX epoch or simulation time in seconds")
    frame_id: int = Field(..., ge=0)
    obstacles: List[TrackedObstacle] = Field(default_factory=list)
    drivable_corridor: FreeSpaceCorridor
    anomalies: List[RoadAnomaly] = Field(default_factory=list)
    sensor_health: Dict[str, bool] = Field(
        default_factory=lambda: {"camera": True, "lidar": True, "radar": True, "gnss": True}
    )


# ---------------------------------------------------------------------------
# 4. Prediction Interfaces
# ---------------------------------------------------------------------------

class PredictedTrajectoryPoint(BaseModel):
    """Waypoint in an agent's forecasted trajectory."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    position: Point3D
    velocity: Vector3D
    yaw_rad: float
    sigma_x: float = Field(0.0, description="1-sigma longitudinal spatial uncertainty (m)")
    sigma_y: float = Field(0.0, description="1-sigma lateral spatial uncertainty (m)")


class PredictedTrajectory(BaseModel):
    """A single predicted future path option with probability."""
    model_config = ConfigDict(extra="forbid")
    probability: float = Field(..., ge=0.0, le=1.0)
    mode_name: str = Field("continuation", description="Modal identifier (e.g. continue, nudge, cut_in, stall)")
    collision_risk: float = Field(0.0, ge=0.0, le=1.0, description="Estimated collision risk score [0, 1]")
    waypoints: List[PredictedTrajectoryPoint] = Field(default_factory=list)


class PredictedAgent(BaseModel):
    """Full prediction package for an individual road actor."""
    model_config = ConfigDict(extra="forbid")
    id: str
    obstacle_class: ObstacleClass
    primary_intent: MotionIntent
    trajectories: List[PredictedTrajectory] = Field(default_factory=list)
    is_high_risk: bool = Field(False, description="Flagged if potential conflict path exists")


class PredictionOutput(BaseModel):
    """Output from the Prediction Subsystem."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    horizon_seconds: float = Field(3.0, description="Prediction time horizon")
    agents: List[PredictedAgent] = Field(default_factory=list)
    high_risk_agent_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 5. Planning Interfaces
# ---------------------------------------------------------------------------

class TrajectoryPoint(BaseModel):
    """Waypoint in the planned trajectory for the ego vehicle."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    x: float = Field(..., description="Ego forward position (m)")
    y: float = Field(..., description="Ego lateral position (m)")
    yaw_rad: float = Field(0.0, description="Heading angle (rad)")
    curvature: float = Field(0.0, description="Path curvature kappa (1/m)")
    speed_mps: float = Field(..., description="Target linear velocity (m/s)")
    acceleration_mps2: float = Field(0.0, description="Target linear acceleration (m/s^2)")
    jerk_mps3: float = Field(0.0, description="Derivative of acceleration (m/s^3)")


class PlannedTrajectory(BaseModel):
    """Full trajectory plan generated by local planner."""
    model_config = ConfigDict(extra="forbid")
    trajectory_id: str
    timestamp: float
    behavior_mode: BehaviorMode
    waypoints: List[TrajectoryPoint] = Field(default_factory=list)
    target_speed_mps: float = Field(..., ge=0.0)
    total_cost: float = Field(0.0, description="Cost evaluation score of the selected trajectory")
    is_feasible: bool = Field(True, description="True if satisfies dynamic and geometric limits")


# ---------------------------------------------------------------------------
# 6. Collision Avoidance Interfaces
# ---------------------------------------------------------------------------

class TTCResult(BaseModel):
    """Time-to-Collision metrics with a specific obstacle."""
    model_config = ConfigDict(extra="forbid")
    obstacle_id: str
    ttc_seconds: float = Field(..., description="Estimated time until impact in seconds (inf if no conflict)")
    distance_at_cpa_m: float = Field(..., description="Distance at Closest Point of Approach in meters")
    is_critical: bool = Field(False, description="True if TTC < safety threshold")


class CollisionRisk(BaseModel):
    """Overall collision risk assessment."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    min_ttc_seconds: float = Field(..., description="Minimum TTC across all obstacles")
    closest_obstacle_id: Optional[str] = None
    risk_level: float = Field(..., ge=0.0, le=1.0, description="0 (safe) to 1 (imminent crash)")
    ttc_evaluations: List[TTCResult] = Field(default_factory=list)


class SafeTrajectory(BaseModel):
    """Validated or safety-overridden trajectory passed to Vehicle Control."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    source_trajectory_id: str
    waypoints: List[TrajectoryPoint] = Field(default_factory=list)
    safety_action: SafetyAction
    is_emergency_stop: bool = Field(False)
    barrier_margin_m: float = Field(..., description="Minimum distance margin to nearest constraint")
    min_ttc_seconds: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# 7. Vehicle Control Interfaces
# ---------------------------------------------------------------------------

class ControlCommand(BaseModel):
    """Actuation command transmitted to drive-by-wire system."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    steering_angle_rad: float = Field(..., ge=-0.785, le=0.785, description="Steering angle [-45 deg, +45 deg]")
    throttle_pct: float = Field(0.0, ge=0.0, le=100.0, description="Throttle demand 0 to 100%")
    brake_pct: float = Field(0.0, ge=0.0, le=100.0, description="Brake demand 0 to 100%")
    gear: GearMode = Field(GearMode.DRIVE)
    emergency_brake_active: bool = Field(False)


class ActuatorFeedback(BaseModel):
    """Feedback from physical or simulated steering, motor, and brake actuators."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    actual_steering_angle_rad: float
    actual_throttle_pct: float
    actual_brake_pct: float
    wheel_speeds_rad_per_s: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0],
        description="[FL, FR, RL, RR]"
    )


# ---------------------------------------------------------------------------
# 8. Ego Vehicle State & Raw Sensors
# ---------------------------------------------------------------------------

class EgoVehicleState(BaseModel):
    """Complete ground-truth or localized state of the ego vehicle."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    pose: Pose3D
    twist: Twist3D
    acceleration: Vector3D
    steer_angle_rad: float = 0.0
    battery_soc_pct: float = Field(95.0, ge=0.0, le=100.0)


class RawSensorFrame(BaseModel):
    """Raw sensor readings prior to perception fusion."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    frame_id: int
    lidar_points_count: int = 0
    camera_detections_count: int = 0
    radar_targets_count: int = 0
    gnss_fix: bool = True
    imu_angular_velocity: Vector3D = Field(default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0))
    imu_linear_acceleration: Vector3D = Field(default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0))


# ---------------------------------------------------------------------------
# 9. Telemetry, Goal, and SOS Contracts (CampusOS Compatible)
# ---------------------------------------------------------------------------

class VehicleTelemetry(BaseModel):
    """CampusOS / ROS2 compatible live telemetry payload."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    vehicle_id: str = "SIH26037-AV-01"
    pose: Pose3D
    twist: Twist3D
    battery_soc: float
    behavior_mode: BehaviorMode
    safety_action: SafetyAction
    current_speed_kph: float
    steering_angle_deg: float
    min_ttc_seconds: float
    is_e_stop_active: bool


class MissionGoal(BaseModel):
    """Target destination waypoint from operator or routing engine."""
    model_config = ConfigDict(extra="forbid")
    goal_id: str
    target_pose: Pose3D
    tolerance_radius_m: float = 1.0


class EmergencyStopCommand(BaseModel):
    """Emergency stop signal from CampusOS SOS or onboard watchdog."""
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    source: str = Field(..., description="'CAMPUS_OS_SOS', 'ONBOARD_WATCHDOG', 'MANUAL_BUTTON'")
    hard_stop: bool = True
