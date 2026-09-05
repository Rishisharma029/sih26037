"""Unit tests for Phase 13 Hardening & Fault Fallbacks."""
import pytest
from interfaces import (
    EgoVehicleState, Point3D, Vector3D, Twist3D, Pose3D,
    PerceptionOutput, TrackedObstacle, BoundingBox3D, ObstacleClass, FreeSpaceCorridor
)
from hardening.fault_injector import AdversarialFaultInjector
from hardening.fallback_manager import FallbackManager
from hardening.stress_suite import SystemHardeningTestSuite


def test_fault_injection_noise_and_dropout():
    injector = AdversarialFaultInjector(random_seed=42)
    obs = TrackedObstacle(
        id="test_obs",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=10.0, y=0.0, z=0.5), size=Vector3D(x=2.5, y=1.2, z=1.5)),
        velocity=Vector3D(x=2.0, y=0.0, z=0.0),
        distance_m=10.0
    )
    p = PerceptionOutput(
        timestamp=0.0,
        frame_id=1,
        obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=0.0, average_width_m=4.5),
        anomalies=[]
    )

    noisy = injector.inject_sensor_noise(p, sigma_pos=0.5)
    assert noisy.obstacles[0].bbox.center.x != 10.0

    dropped = injector.inject_detection_dropout(p, drop_rate=1.0)
    assert len(dropped.obstacles) == 0


def test_fallback_manager_mrm_on_timeout():
    fallback = FallbackManager()
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(linear=Vector3D(x=5.0, y=0.0, z=0.0), speed_mps=5.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        steer_angle_rad=0.0,
        battery_soc_pct=95.0
    )

    mrm_plan, triggered, reason = fallback.evaluate_and_apply_fallback(
        ego_state=ego,
        perception=PerceptionOutput(timestamp=0.0, frame_id=1, obstacles=[], drivable_corridor=FreeSpaceCorridor(timestamp=0.0, average_width_m=4.5), anomalies=[]),
        safe_plan=None,
        planner_latency_ms=100.0  # Timeout
    )

    assert triggered is True
    assert "PLANNER_TIMEOUT" in reason
    assert mrm_plan.is_emergency_stop is True
    assert len(mrm_plan.waypoints) > 0


def test_hardening_stress_suite_execution():
    suite = SystemHardeningTestSuite()
    results = suite.run_all_stress_tests()
    assert len(results) == 10
    assert all(results.values()) is True
