"""Intent classification for non-lane-respecting actors in Indian traffic environments."""
import math
from typing import List, Optional
from interfaces import TrackedObstacle, MotionIntent, ObstacleClass


class IntentClassifier:
    """Infers dynamic actor intentions (e.g. cut-in, swerve, crossing, stopping)
    tailored for Indian traffic dynamics such as two-wheelers weaving, autorickshaws
    nudging across corridors, pedestrians hesitating, and livestock wandering.
    """

    def classify_intent(self, obstacle: TrackedObstacle) -> MotionIntent:
        """Classify the primary intent of an obstacle using temporal history and kinematics."""
        if obstacle.is_static:
            return MotionIntent.STATIONARY

        # 1. Prioritize multi-frame temporal tracker intent if confident
        if getattr(obstacle, "inferred_intent", None) not in (None, MotionIntent.UNKNOWN):
            return obstacle.inferred_intent

        vx = obstacle.velocity.x
        vy = obstacle.velocity.y
        speed = math.sqrt(vx * vx + vy * vy)

        if speed < 0.15:
            return MotionIntent.STATIONARY

        # Acceleration cues
        ax = obstacle.acceleration.x
        ay = obstacle.acceleration.y

        # Heading angle in vehicle coordinate frame (0 rad = forward along X)
        heading = obstacle.bbox.yaw_rad
        lat_heading = abs(math.sin(heading))

        # Pedestrians and Cattle crossing detection
        if obstacle.obstacle_class in [ObstacleClass.PEDESTRIAN, ObstacleClass.CATTLE_ANIMAL]:
            # Perpendicular or high lateral motion indicates active road crossing
            if abs(vy) > 0.35 or lat_heading > 0.6:
                return MotionIntent.CROSSING_PATH
            if speed < 0.4:
                return MotionIntent.DECELERATING

        # Deceleration / Braking
        if ax < -1.5:
            return MotionIntent.DECELERATING
        if ax > 1.2:
            return MotionIntent.ACCELERATING

        # Aggressive lateral cutting in (towards ego center line |y| -> 0)
        y_pos = obstacle.bbox.center.y
        is_moving_towards_center = (y_pos > 0.5 and vy < -0.3) or (y_pos < -0.5 and vy > 0.3)
        if is_moving_towards_center or abs(vy) > 0.75:
            return MotionIntent.CUTTING_IN

        # Erratic swerving / weaving (common with motorcycles and autorickshaws)
        if abs(ay) > 1.2 or (abs(vy) > 0.4 and abs(vx) > 1.0):
            return MotionIntent.ERRATIC_SWERVE

        # Turning maneuvers
        if vy > 0.5 and vx > 0.5:
            return MotionIntent.TURNING_LEFT
        if vy < -0.5 and vx > 0.5:
            return MotionIntent.TURNING_RIGHT

        return MotionIntent.CRUISING
