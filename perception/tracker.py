"""
Multi-Object Tracker (MOT) & Velocity Estimator.
Maintains persistent track IDs and estimates linear velocities (vx, vy)
using a Constant-Velocity Kalman Filter on 3D Bounding Box centroids.
"""
import math
from typing import List, Dict, Any, Optional, Tuple
from interfaces import ObstacleClass, Point3D, Vector3D, BoundingBox3D, TrackedObstacle
from .idd_detector import DetectedObject2D

class TrackState:
    """Individual tracked object state with Kalman filter estimates."""
    def __init__(
        self,
        track_id: str,
        detection: DetectedObject2D,
        timestamp: float
    ):
        self.track_id = track_id
        self.obstacle_class = detection.class_name
        self.confidence = detection.confidence
        self.bbox_3d = detection.to_3d_bbox()
        self.last_timestamp = timestamp
        self.first_timestamp = timestamp
        self.age_steps = 1
        self.missed_steps = 0

        # State: [x, y, vx, vy] in vehicle body frame
        self.x = self.bbox_3d.center.x
        self.y = self.bbox_3d.center.y
        self.vx = 0.0
        self.vy = 0.0
        self.size = detection.estimated_size_m
        self.yaw_rad = detection.estimated_yaw_rad
        self.is_static = False

    def predict(self, dt: float):
        """State prediction step."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def update(self, detection: DetectedObject2D, timestamp: float):
        """Measurement update with Kalman smoothing."""
        dt = max(0.01, timestamp - self.last_timestamp)
        new_bbox = detection.to_3d_bbox()
        meas_x = new_bbox.center.x
        meas_y = new_bbox.center.y

        # Velocity estimation: numerical derivative with alpha filter
        raw_vx = (meas_x - self.x) / dt
        raw_vy = (meas_y - self.y) / dt

        alpha_pos = 0.70
        alpha_vel = 0.60

        self.x = self.x + alpha_pos * (meas_x - self.x)
        self.y = self.y + alpha_pos * (meas_y - self.y)
        self.vx = self.vx + alpha_vel * (raw_vx - self.vx)
        self.vy = self.vy + alpha_vel * (raw_vy - self.vy)

        self.size = detection.estimated_size_m
        self.yaw_rad = detection.estimated_yaw_rad
        self.confidence = max(0.85, 0.9 * self.confidence + 0.1 * detection.confidence)
        self.obstacle_class = detection.class_name
        self.last_timestamp = timestamp
        self.age_steps += 1
        self.missed_steps = 0

        # Static check
        speed = math.hypot(self.vx, self.vy)
        self.is_static = (speed < 0.3 and self.age_steps > 3)

    def to_tracked_obstacle(self) -> TrackedObstacle:
        """Exports to standard TrackedObstacle contract."""
        dist = math.hypot(self.x, self.y)
        return TrackedObstacle(
            id=self.track_id,
            obstacle_class=self.obstacle_class,
            confidence=round(self.confidence, 3),
            bbox=BoundingBox3D(
                center=Point3D(x=round(self.x, 2), y=round(self.y, 2), z=round(self.size.z * 0.5, 2)),
                size=self.size,
                yaw_rad=round(self.yaw_rad, 4)
            ),
            velocity=Vector3D(x=round(self.vx, 2), y=round(self.vy, 2), z=0.0),
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
            distance_m=round(dist, 2),
            is_static=self.is_static
        )


class MultiObjectTracker:
    """Maintains persistent tracking across detection frames."""
    def __init__(self, max_missed_steps: int = 5, match_distance_threshold_m: float = 3.5):
        self.tracks: Dict[str, TrackState] = {}
        self.max_missed_steps = max_missed_steps
        self.match_dist_thresh = match_distance_threshold_m
        self.track_counter = 0

    def update(self, detections: List[DetectedObject2D], timestamp: float, dt: float = 0.05) -> List[TrackedObstacle]:
        """Associates detections, updates Kalman state, and returns active TrackedObstacles."""
        # 1. Predict existing tracks
        for t in self.tracks.values():
            t.predict(dt)
            t.missed_steps += 1

        # 2. Match detections to tracks using spatial Euclidean distance
        unmatched_dets = []
        matched_tracks = set()

        for det in detections:
            bbox = det.to_3d_bbox()
            det_x = bbox.center.x
            det_y = bbox.center.y

            best_id = None
            min_d = float("inf")

            for t_id, track in self.tracks.items():
                if t_id in matched_tracks:
                    continue
                d = math.hypot(det_x - track.x, det_y - track.y)
                if d < self.match_dist_thresh and d < min_d:
                    min_d = d
                    best_id = t_id

            if best_id is not None:
                self.tracks[best_id].update(det, timestamp)
                matched_tracks.add(best_id)
            else:
                unmatched_dets.append(det)

        # 3. Instantiate new tracks for unmatched detections
        for det in unmatched_dets:
            self.track_counter += 1
            cls_name = det.class_name.value.lower()
            new_id = f"trk_{self.track_counter:02d}_{cls_name}"
            new_track = TrackState(track_id=new_id, detection=det, timestamp=timestamp)
            self.tracks[new_id] = new_track

        # 4. Prune stale tracks
        active_ids = [t_id for t_id, t in self.tracks.items() if t.missed_steps <= self.max_missed_steps]
        self.tracks = {t_id: self.tracks[t_id] for t_id in active_ids}

        # 5. Return active tracked obstacles sorted by proximity
        tracked_obs = [t.to_tracked_obstacle() for t in self.tracks.values() if t.x > -2.0]
        tracked_obs.sort(key=lambda o: o.distance_m)
        return tracked_obs
