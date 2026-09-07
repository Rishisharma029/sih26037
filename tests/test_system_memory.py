"""
Test Suite: System Memory, Multi-Frame Track History, Kinematic Derivatives & Intent Inference
SIH26037 — Step 10 Verification
"""

import pytest
import math
from interfaces import (
    Point3D, Vector3D, BoundingBox3D, ObstacleClass,
    MotionIntent, TrackedObstacle, TrackHistoryFrame
)
from perception.idd_detector import DetectedObject2D
from perception.tracker import MultiObjectTracker
from prediction.intent_classifier import IntentClassifier


def create_detection(x: float, y: float, obs_class: ObstacleClass = ObstacleClass.MOTORCYCLE, conf: float = 0.90, yaw: float = 0.0) -> DetectedObject2D:
    return DetectedObject2D(
        class_name=obs_class,
        confidence=conf,
        bbox_2d=(100.0, 100.0, 200.0, 200.0),
        estimated_depth_m=x,
        lateral_offset_m=y,
        estimated_size_m=Vector3D(x=2.0, y=0.8, z=1.2),
        estimated_yaw_rad=yaw
    )


class TestSystemMemoryTracker:
    """Verifies that the multi-object tracker maintains multi-frame temporal memory."""

    def test_multi_frame_history_accumulation(self):
        """Test that history buffer accumulates over consecutive frames."""
        tracker = MultiObjectTracker(max_missed_steps=5)

        # Frame 0: t=0.0, x=30.0, y=1.0
        det0 = [create_detection(x=30.0, y=1.0)]
        tracks0 = tracker.update(det0, timestamp=0.0, dt=0.05)
        assert len(tracks0) == 1
        assert tracks0[0].history_length == 1
        assert len(tracks0[0].history) == 1
        assert tracks0[0].history[0].position.x == 30.0

        # Frame 1: t=0.05, x=29.7, y=1.0 (moving towards ego at -6.0 m/s)
        det1 = [create_detection(x=29.7, y=1.0)]
        tracks1 = tracker.update(det1, timestamp=0.05, dt=0.05)
        assert len(tracks1) == 1
        assert tracks1[0].history_length == 2
        assert len(tracks1[0].history) == 2
        assert pytest.approx(tracks1[0].velocity.x, rel=0.1) == -6.0

        # Frame 2: t=0.10, x=29.4, y=1.0
        det2 = [create_detection(x=29.4, y=1.0)]
        tracks2 = tracker.update(det2, timestamp=0.10, dt=0.05)
        assert len(tracks2) == 1
        assert tracks2[0].history_length == 3
        assert len(tracks2[0].history) == 3
        assert pytest.approx(tracks2[0].speed_mps, rel=0.1) == 6.0

    def test_acceleration_numerical_derivative(self):
        """Test that linear acceleration is accurately derived from velocity changes."""
        tracker = MultiObjectTracker(max_missed_steps=5)

        # Initial steady state at 4 m/s (dx = -0.20 per frame)
        x = 30.0
        t = 0.0
        for _ in range(5):
            tracker.update([create_detection(x=x, y=0.0, obs_class=ObstacleClass.CAR)], timestamp=t, dt=0.05)
            x -= 0.20
            t += 0.05

        # Sudden deceleration: step change from 4 m/s to 1 m/s (dx = -0.05 per frame)
        x -= 0.05
        t += 0.05
        tracks_decel = tracker.update([create_detection(x=x, y=0.0, obs_class=ObstacleClass.CAR)], timestamp=t, dt=0.05)
        
        # Acceleration during transient deceleration should spike
        assert len(tracks_decel) == 1
        assert abs(tracks_decel[0].acceleration.x) > 5.0
        # In history, transient acceleration is captured
        accels = [abs(h.acceleration.x) for h in tracks_decel[0].history]
        assert max(accels) > 5.0

    def test_intent_classification_cutting_in(self):
        """Test that an actor with substantial lateral velocity toward center is classified as CUTTING_IN."""
        tracker = MultiObjectTracker()

        # Motorcycle starting on right shoulder (y=2.0) and cutting into ego lane (y -> 0.0)
        x = 25.0
        y = 2.0
        t = 0.0
        for _ in range(6):
            tracker.update([create_detection(x=x, y=y, obs_class=ObstacleClass.MOTORCYCLE)], timestamp=t, dt=0.05)
            x -= 0.25  # vx = -5.0 m/s
            y -= 0.08  # vy = -1.6 m/s (lateral cut-in)
            t += 0.05

        tracks = tracker.update([create_detection(x=x, y=y, obs_class=ObstacleClass.MOTORCYCLE)], timestamp=t, dt=0.05)
        assert len(tracks) == 1
        assert tracks[0].inferred_intent == MotionIntent.CUTTING_IN

    def test_intent_classification_crossing_pedestrian(self):
        """Test that a pedestrian crossing the road perpendicularly is classified as CROSSING_PATH."""
        tracker = MultiObjectTracker()

        # Pedestrian crossing from y = -2.5 to y = 1.0 at x = 15.0
        x = 15.0
        y = -2.5
        t = 0.0
        for _ in range(8):
            tracker.update([create_detection(x=x, y=y, obs_class=ObstacleClass.PEDESTRIAN)], timestamp=t, dt=0.05)
            y += 0.07  # vy = 1.4 m/s
            t += 0.05

        tracks = tracker.update([create_detection(x=x, y=y, obs_class=ObstacleClass.PEDESTRIAN)], timestamp=t, dt=0.05)
        assert len(tracks) == 1
        assert tracks[0].inferred_intent == MotionIntent.CROSSING_PATH

    def test_intent_classification_stationary(self):
        """Test that a stationary parked vehicle is classified as STATIONARY."""
        tracker = MultiObjectTracker()

        # Static auto rickshaw
        x = 20.0
        t = 0.0
        for _ in range(6):
            tracker.update([create_detection(x=x, y=1.0, obs_class=ObstacleClass.AUTO_RICKSHAW)], timestamp=t, dt=0.05)
            t += 0.05

        tracks = tracker.update([create_detection(x=x, y=1.0, obs_class=ObstacleClass.AUTO_RICKSHAW)], timestamp=t, dt=0.05)
        assert len(tracks) == 1
        assert tracks[0].inferred_intent == MotionIntent.STATIONARY

    def test_intent_classifier_consumes_tracker_intent(self):
        """Test that IntentClassifier respects the temporal intent inferred by MultiObjectTracker."""
        classifier = IntentClassifier()

        tracker = MultiObjectTracker()
        # Setup cutting in motorcycle
        x, y, t = 22.0, 1.8, 0.0
        for _ in range(6):
            tracker.update([create_detection(x=x, y=y, obs_class=ObstacleClass.MOTORCYCLE)], timestamp=t, dt=0.05)
            x -= 0.25
            y -= 0.08
            t += 0.05

        tracks = tracker.update([create_detection(x=x, y=y, obs_class=ObstacleClass.MOTORCYCLE)], timestamp=t, dt=0.05)
        tracked_obs = tracks[0]

        # Verify IntentClassifier prioritizes tracked_obs.inferred_intent
        primary_intent = classifier.classify_intent(tracked_obs)
        assert primary_intent == MotionIntent.CUTTING_IN
