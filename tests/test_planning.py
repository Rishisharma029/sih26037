"""Unit tests for planning subsystem."""
from planning.local_planner import AdaptiveLatticePlanner
from interfaces import (
    EgoVehicleState, Pose3D, Point3D, Twist3D, Vector3D,
    PerceptionOutput, FreeSpaceCorridor, PredictionOutput, BehaviorMode
)

def test_cruise_planning():
    planner = AdaptiveLatticePlanner(horizon_seconds=2.0, dt=0.5)
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0)),
        twist=Twist3D(speed_mps=5.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )
    perc = PerceptionOutput(
        timestamp=0.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=0.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=0.0, horizon_seconds=2.0, agents=[])
    plan = planner.plan(ego, perc, pred, target_cruise_speed_mps=6.0)

    assert plan.behavior_mode == BehaviorMode.CRUISE
    assert len(plan.waypoints) == 4
    assert plan.waypoints[-1].x > 0.0
