"""Multi-modal rule-based and kinematic trajectory predictor for Indian road actors."""
import math
from typing import List, Tuple
from interfaces import (
    TrackedObstacle, ObstacleClass, MotionIntent,
    PredictedTrajectory, PredictedTrajectoryPoint, Point3D, Vector3D
)


class KinematicPredictor:
    """Generates multi-modal kinematic trajectory branches with uncertainty covariance

    and collision risk assessment for varied Indian road users (motorcycles, autos,
    pedestrians, cattle, and commercial vehicles).
    """

    def __init__(self, horizon_seconds: float = 3.0, dt: float = 0.5):
        self.horizon_seconds = horizon_seconds
        self.dt = dt

    def predict_modes(
        self,
        obstacle: TrackedObstacle,
        intent: MotionIntent,
        current_time: float,
        ego_speed: float = 6.0
    ) -> List[PredictedTrajectory]:
        """Generate K discrete trajectory options with mode probabilities and uncertainty."""
        if obstacle.is_static:
            return [self._create_static_trajectory(obstacle, current_time)]

        # Class-specific multi-modal generators
        if obstacle.obstacle_class == ObstacleClass.MOTORCYCLE:
            return self._predict_motorcycle_modes(obstacle, intent, current_time, ego_speed)
        elif obstacle.obstacle_class == ObstacleClass.AUTO_RICKSHAW:
            return self._predict_autorickshaw_modes(obstacle, intent, current_time, ego_speed)
        elif obstacle.obstacle_class == ObstacleClass.PEDESTRIAN:
            return self._predict_pedestrian_modes(obstacle, intent, current_time, ego_speed)
        elif obstacle.obstacle_class == ObstacleClass.CATTLE_ANIMAL:
            return self._predict_cattle_modes(obstacle, intent, current_time, ego_speed)
        else:
            return self._predict_vehicle_modes(obstacle, intent, current_time, ego_speed)

    def _create_static_trajectory(self, obstacle: TrackedObstacle, current_time: float) -> PredictedTrajectory:
        """Stationary obstacle stays fixed at current position."""
        steps = int(self.horizon_seconds / self.dt)
        pts = []
        for step in range(1, steps + 1):
            t_future = current_time + step * self.dt
            pts.append(PredictedTrajectoryPoint(
                timestamp=t_future,
                position=Point3D(x=obstacle.bbox.center.x, y=obstacle.bbox.center.y, z=obstacle.bbox.center.z),
                velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                yaw_rad=obstacle.bbox.yaw_rad,
                sigma_x=0.05,
                sigma_y=0.05
            ))
        return PredictedTrajectory(
            probability=1.0,
            mode_name="stationary",
            collision_risk=0.0 if obstacle.distance_m > 8.0 else 0.4,
            waypoints=pts
        )

    def _predict_motorcycle_modes(
        self,
        obs: TrackedObstacle,
        intent: MotionIntent,
        t0: float,
        ego_speed: float
    ) -> List[PredictedTrajectory]:
        """Motorcycles exhibit high agility: straight cruise, obstacle weave, or cut-in."""
        steps = int(self.horizon_seconds / self.dt)
        x0, y0 = obs.bbox.center.x, obs.bbox.center.y
        vx, vy = obs.velocity.x, obs.velocity.y
        yaw = obs.bbox.yaw_rad

        # Assign mode probabilities based on intent
        if intent == MotionIntent.CUTTING_IN:
            p1, p2, p3 = 0.20, 0.15, 0.65
        elif intent == MotionIntent.ERRATIC_SWERVE:
            p1, p2, p3 = 0.25, 0.50, 0.25
        else:
            p1, p2, p3 = 0.65, 0.20, 0.15

        # Mode 1: Continuation (CTRA / Constant Velocity)
        pts_t1 = []
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            fx = x0 + vx * dt_step
            fy = y0 + vy * dt_step
            sig_x = 0.20 + 0.15 * dt_step + 0.05 * abs(vx) * dt_step
            sig_y = 0.20 + 0.20 * dt_step + 0.10 * abs(vy) * dt_step
            pts_t1.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=fx, y=fy, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=vy, z=0.0),
                yaw_rad=yaw,
                sigma_x=sig_x,
                sigma_y=sig_y
            ))

        # Mode 2: Nudge / Weave around obstacle (lateral offset +0.7m away from current path)
        pts_t2 = []
        nudge_dir = 1.0 if y0 >= 0 else -1.0
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            fx = x0 + vx * dt_step
            # Smooth lateral sigmoid nudge
            lateral_offset = nudge_dir * 0.8 * (1.0 - math.exp(-1.5 * dt_step))
            fy = y0 + vy * dt_step + lateral_offset
            pts_t2.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=fx, y=fy, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=vy + nudge_dir * 0.5, z=0.0),
                yaw_rad=yaw + nudge_dir * 0.15,
                sigma_x=0.25 + 0.25 * dt_step,
                sigma_y=0.25 + 0.35 * dt_step
            ))

        # Mode 3: Aggressive Cut-in / Lane merge (merges toward center y -> 0)
        pts_t3 = []
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            fx = x0 + vx * dt_step
            # Decays y towards center 0
            fy = y0 * math.exp(-1.8 * dt_step)
            lat_v = -1.8 * fy
            pts_t3.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=fx, y=fy, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=lat_v, z=0.0),
                yaw_rad=math.atan2(lat_v, vx) if abs(vx) > 0.1 else yaw,
                sigma_x=0.30 + 0.30 * dt_step,
                sigma_y=0.30 + 0.40 * dt_step
            ))

        t1 = PredictedTrajectory(
            probability=p1,
            mode_name="continuation",
            collision_risk=self._calc_risk(pts_t1, p1, ego_speed),
            waypoints=pts_t1
        )
        t2 = PredictedTrajectory(
            probability=p2,
            mode_name="nudge_obstacle",
            collision_risk=self._calc_risk(pts_t2, p2, ego_speed),
            waypoints=pts_t2
        )
        t3 = PredictedTrajectory(
            probability=p3,
            mode_name="cut_in_merge",
            collision_risk=self._calc_risk(pts_t3, p3, ego_speed),
            waypoints=pts_t3
        )
        return [t1, t2, t3]

    def _predict_autorickshaw_modes(
        self,
        obs: TrackedObstacle,
        intent: MotionIntent,
        t0: float,
        ego_speed: float
    ) -> List[PredictedTrajectory]:
        """Auto-rickshaws: Moderate speed, frequent nudges, passenger stops."""
        steps = int(self.horizon_seconds / self.dt)
        x0, y0 = obs.bbox.center.x, obs.bbox.center.y
        vx, vy = obs.velocity.x, obs.velocity.y
        yaw = obs.bbox.yaw_rad

        if intent == MotionIntent.DECELERATING or intent == MotionIntent.STOPPING:
            p1, p2, p3 = 0.20, 0.60, 0.20
        elif intent == MotionIntent.CUTTING_IN:
            p1, p2, p3 = 0.20, 0.15, 0.65
        else:
            p1, p2, p3 = 0.60, 0.25, 0.15

        # Mode 1: Continuation
        pts_t1 = [
            PredictedTrajectoryPoint(
                timestamp=t0 + step * self.dt,
                position=Point3D(x=x0 + vx * step * self.dt, y=y0 + vy * step * self.dt, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=vy, z=0.0),
                yaw_rad=yaw,
                sigma_x=0.18 + 0.12 * step * self.dt,
                sigma_y=0.18 + 0.15 * step * self.dt
            )
            for step in range(1, steps + 1)
        ]

        # Mode 2: Decelerate / Roadside Halting (moves towards road curb while slowing down)
        curb_y = 2.0 if y0 >= 0 else -2.0
        pts_t2 = []
        curr_vx = vx
        curr_x = x0
        curr_y = y0
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            curr_vx = max(0.0, vx * (1.0 - 0.35 * dt_step))
            curr_x += curr_vx * self.dt
            curr_y += (curb_y - curr_y) * 0.20
            pts_t2.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=curr_x, y=curr_y, z=obs.bbox.center.z),
                velocity=Vector3D(x=curr_vx, y=(curb_y - curr_y) * 0.20 / self.dt, z=0.0),
                yaw_rad=yaw,
                sigma_x=0.20 + 0.15 * dt_step,
                sigma_y=0.20 + 0.20 * dt_step
            ))

        # Mode 3: Nudge / Cut across lane
        pts_t3 = [
            PredictedTrajectoryPoint(
                timestamp=t0 + step * self.dt,
                position=Point3D(x=x0 + vx * step * self.dt, y=y0 * math.exp(-1.2 * step * self.dt), z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=-1.2 * y0 * math.exp(-1.2 * step * self.dt), z=0.0),
                yaw_rad=yaw,
                sigma_x=0.25 + 0.20 * step * self.dt,
                sigma_y=0.25 + 0.30 * step * self.dt
            )
            for step in range(1, steps + 1)
        ]

        return [
            PredictedTrajectory(probability=p1, mode_name="continuation", collision_risk=self._calc_risk(pts_t1, p1, ego_speed), waypoints=pts_t1),
            PredictedTrajectory(probability=p2, mode_name="roadside_halt", collision_risk=self._calc_risk(pts_t2, p2, ego_speed), waypoints=pts_t2),
            PredictedTrajectory(probability=p3, mode_name="nudge_cut_in", collision_risk=self._calc_risk(pts_t3, p3, ego_speed), waypoints=pts_t3),
        ]

    def _predict_pedestrian_modes(
        self,
        obs: TrackedObstacle,
        intent: MotionIntent,
        t0: float,
        ego_speed: float
    ) -> List[PredictedTrajectory]:
        """Pedestrians: Direct crossing, hesitation/stall in center, or turn-back."""
        steps = int(self.horizon_seconds / self.dt)
        x0, y0 = obs.bbox.center.x, obs.bbox.center.y
        vx, vy = obs.velocity.x, obs.velocity.y
        yaw = obs.bbox.yaw_rad

        if intent == MotionIntent.CROSSING_PATH:
            p1, p2, p3 = 0.65, 0.25, 0.10
        elif intent == MotionIntent.DECELERATING:
            p1, p2, p3 = 0.20, 0.65, 0.15
        else:
            p1, p2, p3 = 0.50, 0.30, 0.20

        # Mode 1: Constant velocity crossing
        pts_t1 = [
            PredictedTrajectoryPoint(
                timestamp=t0 + step * self.dt,
                position=Point3D(x=x0 + vx * step * self.dt, y=y0 + vy * step * self.dt, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=vy, z=0.0),
                yaw_rad=yaw,
                sigma_x=0.15 + 0.10 * step * self.dt,
                sigma_y=0.15 + 0.15 * step * self.dt
            )
            for step in range(1, steps + 1)
        ]

        # Mode 2: Hesitation / sudden stop mid-road
        pts_t2 = []
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            # Velocity drops sharply
            decay = math.exp(-2.5 * dt_step)
            fx = x0 + vx * (1.0 - decay) / 2.5
            fy = y0 + vy * (1.0 - decay) / 2.5
            pts_t2.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=fx, y=fy, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx * decay, y=vy * decay, z=0.0),
                yaw_rad=yaw,
                sigma_x=0.20 + 0.10 * dt_step,
                sigma_y=0.20 + 0.15 * dt_step
            ))

        # Mode 3: Reversal (walks back towards starting shoulder)
        pts_t3 = []
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            # Reverses lateral motion after 0.8s
            if dt_step < 0.8:
                fy = y0 + vy * dt_step
                rev_vy = vy
            else:
                fy = y0 + vy * 0.8 - vy * (dt_step - 0.8)
                rev_vy = -vy
            pts_t3.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=x0 + vx * dt_step * 0.5, y=fy, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx * 0.5, y=rev_vy, z=0.0),
                yaw_rad=yaw + math.pi if rev_vy * vy < 0 else yaw,
                sigma_x=0.25 + 0.15 * dt_step,
                sigma_y=0.25 + 0.25 * dt_step
            ))

        return [
            PredictedTrajectory(probability=p1, mode_name="constant_crossing", collision_risk=self._calc_risk(pts_t1, p1, ego_speed), waypoints=pts_t1),
            PredictedTrajectory(probability=p2, mode_name="hesitate_halt", collision_risk=self._calc_risk(pts_t2, p2, ego_speed), waypoints=pts_t2),
            PredictedTrajectory(probability=p3, mode_name="reversal_to_shoulder", collision_risk=self._calc_risk(pts_t3, p3, ego_speed), waypoints=pts_t3),
        ]

    def _predict_cattle_modes(
        self,
        obs: TrackedObstacle,
        intent: MotionIntent,
        t0: float,
        ego_speed: float
    ) -> List[PredictedTrajectory]:
        """Cattle: Slow drift, abrupt freeze/blocking road, erratic turning."""
        steps = int(self.horizon_seconds / self.dt)
        x0, y0 = obs.bbox.center.x, obs.bbox.center.y
        vx, vy = obs.velocity.x, obs.velocity.y
        yaw = obs.bbox.yaw_rad

        p1, p2, p3 = 0.40, 0.45, 0.15

        # Mode 1: Slow continuous drift
        pts_t1 = [
            PredictedTrajectoryPoint(
                timestamp=t0 + step * self.dt,
                position=Point3D(x=x0 + vx * step * self.dt, y=y0 + vy * step * self.dt, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=vy, z=0.0),
                yaw_rad=yaw,
                sigma_x=0.25 + 0.15 * step * self.dt,
                sigma_y=0.25 + 0.25 * step * self.dt
            )
            for step in range(1, steps + 1)
        ]

        # Mode 2: Abrupt freeze in middle of road (high probability for cattle)
        pts_t2 = []
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            decay = math.exp(-3.5 * dt_step)
            fx = x0 + vx * (1.0 - decay) / 3.5
            fy = y0 + vy * (1.0 - decay) / 3.5
            pts_t2.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=fx, y=fy, z=obs.bbox.center.z),
                velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                yaw_rad=yaw,
                sigma_x=0.15 + 0.08 * dt_step,
                sigma_y=0.15 + 0.10 * dt_step
            ))

        # Mode 3: Erratic 45 deg wander / direction change
        veer_angle = 0.60
        pts_t3 = []
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            sp = math.sqrt(vx * vx + vy * vy)
            new_vx = sp * math.cos(yaw + veer_angle)
            new_vy = sp * math.sin(yaw + veer_angle)
            pts_t3.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=x0 + new_vx * dt_step, y=y0 + new_vy * dt_step, z=obs.bbox.center.z),
                velocity=Vector3D(x=new_vx, y=new_vy, z=0.0),
                yaw_rad=yaw + veer_angle,
                sigma_x=0.30 + 0.25 * dt_step,
                sigma_y=0.30 + 0.35 * dt_step
            ))

        return [
            PredictedTrajectory(probability=p1, mode_name="slow_drift", collision_risk=self._calc_risk(pts_t1, p1, ego_speed), waypoints=pts_t1),
            PredictedTrajectory(probability=p2, mode_name="road_freeze", collision_risk=self._calc_risk(pts_t2, p2, ego_speed), waypoints=pts_t2),
            PredictedTrajectory(probability=p3, mode_name="erratic_veer", collision_risk=self._calc_risk(pts_t3, p3, ego_speed), waypoints=pts_t3),
        ]

    def _predict_vehicle_modes(
        self,
        obs: TrackedObstacle,
        intent: MotionIntent,
        t0: float,
        ego_speed: float
    ) -> List[PredictedTrajectory]:
        """Cars, Trucks, Buses: Forward continuation, lane nudge, or braking."""
        steps = int(self.horizon_seconds / self.dt)
        x0, y0 = obs.bbox.center.x, obs.bbox.center.y
        vx, vy = obs.velocity.x, obs.velocity.y
        yaw = obs.bbox.yaw_rad

        if intent == MotionIntent.DECELERATING:
            p1, p2, p3 = 0.25, 0.60, 0.15
        elif intent == MotionIntent.CUTTING_IN:
            p1, p2, p3 = 0.25, 0.15, 0.60
        else:
            p1, p2, p3 = 0.70, 0.20, 0.10

        pts_t1 = [
            PredictedTrajectoryPoint(
                timestamp=t0 + step * self.dt,
                position=Point3D(x=x0 + vx * step * self.dt, y=y0 + vy * step * self.dt, z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=vy, z=0.0),
                yaw_rad=yaw,
                sigma_x=0.15 + 0.10 * step * self.dt,
                sigma_y=0.15 + 0.12 * step * self.dt
            )
            for step in range(1, steps + 1)
        ]

        # Mode 2: Deceleration
        pts_t2 = []
        for step in range(1, steps + 1):
            dt_step = step * self.dt
            dec_vx = max(0.0, vx - 1.2 * dt_step)
            fx = x0 + (vx + dec_vx) * 0.5 * dt_step
            pts_t2.append(PredictedTrajectoryPoint(
                timestamp=t0 + dt_step,
                position=Point3D(x=fx, y=y0 + vy * dt_step, z=obs.bbox.center.z),
                velocity=Vector3D(x=dec_vx, y=vy, z=0.0),
                yaw_rad=yaw,
                sigma_x=0.20 + 0.15 * dt_step,
                sigma_y=0.20 + 0.15 * dt_step
            ))

        # Mode 3: Lateral Swerve / Nudge
        nudge_dir = 1.0 if y0 >= 0 else -1.0
        pts_t3 = [
            PredictedTrajectoryPoint(
                timestamp=t0 + step * self.dt,
                position=Point3D(x=x0 + vx * step * self.dt, y=y0 + nudge_dir * 0.6 * math.sin(step * self.dt), z=obs.bbox.center.z),
                velocity=Vector3D(x=vx, y=nudge_dir * 0.6 * math.cos(step * self.dt), z=0.0),
                yaw_rad=yaw,
                sigma_x=0.22 + 0.18 * step * self.dt,
                sigma_y=0.22 + 0.25 * step * self.dt
            )
            for step in range(1, steps + 1)
        ]

        return [
            PredictedTrajectory(probability=p1, mode_name="continuation", collision_risk=self._calc_risk(pts_t1, p1, ego_speed), waypoints=pts_t1),
            PredictedTrajectory(probability=p2, mode_name="deceleration", collision_risk=self._calc_risk(pts_t2, p2, ego_speed), waypoints=pts_t2),
            PredictedTrajectory(probability=p3, mode_name="nudge", collision_risk=self._calc_risk(pts_t3, p3, ego_speed), waypoints=pts_t3),
        ]

    def _calc_risk(self, waypoints: List[PredictedTrajectoryPoint], prob: float, ego_speed: float) -> float:
        """Calculate collision risk score [0, 1] against nominal ego path (forward along X)."""
        if not waypoints:
            return 0.0

        min_d = 999.0
        for pt in waypoints:
            dt_future = pt.timestamp - waypoints[0].timestamp + self.dt
            ego_x_fut = ego_speed * dt_future
            ego_y_fut = 0.0

            dx = pt.position.x - ego_x_fut
            dy = pt.position.y - ego_y_fut
            dist = math.hypot(dx, dy)
            if dist < min_d:
                min_d = dist

        # Closer distance yields higher risk, weighted by mode probability
        if min_d < 2.5:
            base_risk = 1.0 - (min_d / 2.5)
            return round(min(1.0, max(0.0, base_risk * (0.5 + 0.5 * prob))), 3)
        return 0.0
