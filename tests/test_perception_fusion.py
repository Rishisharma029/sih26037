"""
Tests for Multi-Sensor Fusion Engine (Camera + LiDAR + Radar).
"""
import pytest
from interfaces import ObstacleClass, Point3D, Vector3D, EgoVehicleState, Pose3D, Twist3D
from simulation.environment import RoadEnvironment
from simulation.actors import SimulationActor
from simulation.sensor_sim import SyntheticSensorSuite, CameraDetection, LidarCluster, RadarTarget
from perception.sensor_fusion import MultiSensorKalmanFusion

def test_multi_sensor_pipeline_fusion():
    env = RoadEnvironment(length_m=100.0)
    # Add oncoming auto-rickshaw
    env.add_actor(SimulationActor(
        id="auto_01",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        x=25.0,
        y=0.5,
        speed_mps=5.0,
        yaw_rad=3.14159,
        is_static=False
    ))

    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=6.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )

    sensor_suite = SyntheticSensorSuite(env)
    cam_dets, lidar_clusters, rad_targets = sensor_suite.capture_sensor_measurements(ego)

    assert len(cam_dets) >= 1
    assert len(lidar_clusters) >= 1
    assert len(rad_targets) >= 1

    fusion = MultiSensorKalmanFusion(road_width_m=6.0)
    world = fusion.update(
        timestamp=0.1,
        camera_dets=cam_dets,
        lidar_clusters=lidar_clusters,
        radar_targets=rad_targets,
        ego_state=ego
    )

    assert world.active_object_count >= 1
    track = list(world.tracks.values())[0]
    assert track.confidence > 0.8
    assert track.obstacle_type == ObstacleClass.AUTO_RICKSHAW
    assert track.risk_score > 0.0
