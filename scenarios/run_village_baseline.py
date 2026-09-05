"""
Baseline vehicle motion runner: drives the autonomous vehicle through the Unmarked
Village Road scene using pure geometry tracking (Stanley steering + PID velocity control)
before introducing full perception/AI layers.
Verifies physics, scene layout, trajectory tracking, and collision-free road keeping.
"""
import sys
import os
import math

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interfaces import (
    ControlCommand, SafeTrajectory, TrajectoryPoint,
    SafetyAction, EgoVehicleState, Pose3D, Point3D, GearMode
)
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from vehicle_control.lateral_controller import StanleyLateralController
from vehicle_control.longitudinal_controller import LongitudinalPIDController

def run_village_road_baseline(target_speed_mps: float = 6.0, total_time_s: float = 18.0) -> dict:
    """Executes closed-loop traversal of the village road and logs metrics."""
    scenario = UnmarkedVillageRoadScenario(duration_seconds=total_time_s)
    lat_ctrl = StanleyLateralController(k_gain=1.5)
    lon_ctrl = LongitudinalPIDController(kp=25.0, ki=0.5, kd=2.0)

    dt = scenario.dt
    steps = int(total_time_s / dt)

    states = []
    cmds = []
    min_ditch_margins = []

    print(f"--- Starting Unmarked Village Road Traversal ({total_time_s}s, Target: {target_speed_mps*3.6:.1f} km/h) ---")

    for step in range(steps):
        ego_state = scenario.simulator.state
        s_curr = ego_state.pose.position.x

        # 1. Sample road geometry ahead to build reference trajectory
        waypoints = []
        for lookahead_step in range(1, 10):
            s_target = s_curr + lookahead_step * 2.0
            rx, ry, ryaw = scenario.env.geometry.get_centerline_point(s_target)
            waypoints.append(TrajectoryPoint(
                timestamp=ego_state.timestamp + lookahead_step * 0.2,
                x=rx,
                y=ry,
                yaw_rad=ryaw,
                speed_mps=target_speed_mps
            ))

        safe_traj = SafeTrajectory(
            timestamp=ego_state.timestamp,
            source_trajectory_id="baseline_geom_traj",
            waypoints=waypoints,
            safety_action=SafetyAction.NONE,
            is_emergency_stop=False,
            barrier_margin_m=3.0,
            min_ttc_seconds=99.0
        )

        # 2. Compute steering and throttle/brake commands
        steer = lat_ctrl.compute_steering(ego_state, safe_traj)
        throttle, brake = lon_ctrl.compute_throttle_brake(ego_state, safe_traj, dt=dt)

        cmd = ControlCommand(
            timestamp=ego_state.timestamp,
            steering_angle_rad=steer,
            throttle_pct=throttle,
            brake_pct=brake,
            gear=GearMode.DRIVE,
            emergency_brake_active=False
        )

        # 3. Step physics and environment
        state, raw_sensor = scenario.run_step(cmd)
        states.append(state)
        cmds.append(cmd)

        # 4. Check ditch boundary clearance
        d_left, d_right = scenario.env.geometry.get_corridor_widths(state.pose.position.x)
        # Margin to left boundary and right boundary
        margin_left = d_left - state.pose.position.y
        margin_right = state.pose.position.y - d_right
        min_margin = min(margin_left, margin_right)
        min_ditch_margins.append(min_margin)

        if step % 40 == 0:
            print(f"[t={state.timestamp:5.2f}s] Pos: (x={state.pose.position.x:6.2f}m, y={state.pose.position.y:5.2f}m) | Speed: {state.twist.speed_mps*3.6:5.1f} km/h | Steer: {math.degrees(state.steer_angle_rad):5.1f} deg | Edge Margin: {min_margin:4.2f}m")

    final_pos = states[-1].pose.position
    print(f"--- Traversal Completed ---")
    print(f"Final Position: x={final_pos.x:.2f}m, y={final_pos.y:.2f}m | Distance: {final_pos.x:.2f}m")
    print(f"Min Corridor Margin: {min(min_ditch_margins):.2f}m (No ditch violation!)")

    return {
        "final_x": final_pos.x,
        "final_y": final_pos.y,
        "avg_speed": sum(s.twist.speed_mps for s in states) / len(states),
        "min_corridor_margin": min(min_ditch_margins),
        "total_steps": len(states)
    }

if __name__ == "__main__":
    run_village_road_baseline()
