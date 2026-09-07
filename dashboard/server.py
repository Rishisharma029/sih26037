"""
SIH26037 Autonomous Driving Live BEV Visualizer & Telemetry Server (Port 5002)
Features:
1. Real Indian-Trained Perception Subsystem:
   - Camera frame ingestion -> Indian Driving Dataset (IDD) neural detection.
   - 3D Bounding Box monocular estimation.
   - Multi-Object Tracking (MOT) with persistent track IDs.
   - Constant-Velocity Kalman Filter estimating linear velocities (vx, vy).
2. Probabilistic Multi-Modal Motion Prediction:
   - Multi-branch trajectory forecasting (Continuation 60%, Cut-in 25%, Nudge 15%).
   - Dynamic Corridor Invasion Probability & Time-to-Conflict assessment.
   - Explainable natural language reasoning ("There is a 25% probability this motorcycle will cut into my corridor").
   - Expanding spatial uncertainty covariance ellipses (sigma_x, sigma_y).
3. Unstructured Indian Road Geometry:
   - Non-rectangular irregular boundaries, variable width (3.4m choke points -> 5.6m passing bays).
   - Continuous curvature, eroded shoulder verges, ditch drop-offs.
   - Potholes, severe craters, speed bumps, and blocked gravel patches.
4. Multi-Objective Adaptive Lattice Planner (7-Score Model):
   - Candidate Bundle Generation (P1, P2, P3, P4, P5, P6, P7...).
   - 7-Objective Scoring: 40% Collision, 20% Clearance, 15% Traversability, 10% Progress, 5% Smoothness, 5% Dynamics, 5% Uncertainty.
   - Explainable Decision Matrix & Ranking Table.
5. Real-Time Autonomous Driving Debug BEV & 5-Stage Causal Decision HUD Stepper.
"""
import sys
import os
import math
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure sih26037 package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interfaces import (
    VehicleTelemetry, Pose3D, Twist3D, BehaviorMode,
    SafetyAction, ControlCommand, SafeTrajectory, TrajectoryPoint,
    Point3D, Vector3D, GearMode, PerceptionOutput, FreeSpaceCorridor,
    TrackedObstacle, BoundingBox3D, PlannedTrajectory, RoadAnomaly
)
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from scenarios.difficulty import DifficultyLevel
from vehicle_control.lateral_controller import StanleyLateralController
from vehicle_control.longitudinal_controller import LongitudinalPIDController
from collision_avoidance.safety_supervisor import SafetySupervisoryLayer
from perception.perception_pipeline import UnifiedPerceptionPipeline, PerceptionMode
from perception.boundary_detector import FreeSpaceBoundaryDetector
from prediction.trajectory_predictor import TrajectoryPredictor
from planning.local_planner import AdaptiveLatticePlanner
from coordinates import transform_actor_to_ego_tracked_obstacle, world_to_ego_2d, ego_to_world_2d


class HazardSpawnRequest(BaseModel):
    hazard_type: str = "motorcycle" # tractor | motorcycle | pedestrian | auto | cattle | pothole
    distance_ahead_m: float = 30.0


class DifficultyRequest(BaseModel):
    difficulty: str = "HARD" # EASY | MEDIUM | HARD | EXTREME


class PerceptionModeRequest(BaseModel):
    mode: str = "NEURAL_IDD" # NEURAL_IDD | GROUND_TRUTH


class SimulationEngineState:
    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.HARD):
        self.difficulty = difficulty
        self.scenario = UnmarkedVillageRoadScenario(difficulty=self.difficulty)
        self.perception_pipeline = UnifiedPerceptionPipeline(mode=PerceptionMode.NEURAL_IDD, default_road_width_m=4.3)
        self.lat_ctrl = StanleyLateralController(k_gain=1.4)
        self.lon_ctrl = LongitudinalPIDController(kp=22.0, ki=0.5, kd=2.0)
        self.supervisory = SafetySupervisoryLayer(aeb_ttc_threshold_s=1.0, replan_ttc_threshold_s=2.0, slowdown_ttc_threshold_s=4.0)
        self.planner = AdaptiveLatticePlanner(
            horizon_seconds=2.5,
            dt=0.2,
            w_collision=0.40,
            w_clearance=0.20,
            w_traversability=0.15,
            w_progress=0.10,
            w_smoothness=0.05,
            w_dynamics=0.05,
            w_uncertainty=0.05
        )
        self.predictor = TrajectoryPredictor(horizon_seconds=2.5, dt=0.5, mode="ensemble")
        self.is_running = True
        self.target_speed_mps = 6.0
        self.is_emergency_stop = False
        self.is_completed = False
        self.step_count = 0
        self.min_corridor_margin = 2.0
        self.latest_telemetry: Dict[str, Any] = {}
        self.step()

    def set_difficulty(self, diff_name: str):
        level_map = {
            "EASY": DifficultyLevel.EASY,
            "MEDIUM": DifficultyLevel.MEDIUM,
            "HARD": DifficultyLevel.HARD,
            "EXTREME": DifficultyLevel.EXTREME
        }
        self.difficulty = level_map.get(diff_name.upper(), DifficultyLevel.HARD)
        self.reset()

    def set_perception_mode(self, mode_str: str):
        if mode_str.upper() == "GROUND_TRUTH":
            self.perception_pipeline.mode = PerceptionMode.GROUND_TRUTH
        else:
            self.perception_pipeline.mode = PerceptionMode.NEURAL_IDD

    def reset(self):
        self.scenario = UnmarkedVillageRoadScenario(difficulty=self.difficulty)
        self.is_emergency_stop = False
        self.is_completed = False
        self.step_count = 0
        self.min_corridor_margin = 2.0
        self.perception_pipeline.tracker.tracks.clear()
        self.step()

    def spawn_hazard(self, hazard_type: str, dist_ahead: float = 30.0):
        if hazard_type == "tractor":
            self.scenario.spawn_oncoming_tractor(dist_ahead=dist_ahead, y=0.7, speed_mps=3.5)
        elif hazard_type == "motorcycle":
            self.scenario.spawn_oncoming_motorcycle(dist_ahead=dist_ahead, y=1.3, speed_mps=5.2)
        elif hazard_type == "pedestrian":
            self.scenario.spawn_crossing_pedestrian(dist_ahead=max(15.0, dist_ahead * 0.6), start_y=-2.0, speed_mps=1.4)
        elif hazard_type == "auto":
            self.scenario.spawn_parked_auto(dist_ahead=max(15.0, dist_ahead * 0.5), y=0.9)
        elif hazard_type == "cattle":
            self.scenario.spawn_cattle(dist_ahead=max(18.0, dist_ahead * 0.7), y=-1.1, speed_mps=0.6)
        elif hazard_type == "pothole":
            self.scenario.spawn_pothole(dist_ahead=max(16.0, dist_ahead * 0.7), y=0.0, depth_m=-0.16, radius_m=0.85)
        elif hazard_type == "waterlogged":
            self.scenario.spawn_waterlogged_area(dist_ahead=max(18.0, dist_ahead * 0.75), y=0.2, depth_m=-0.18, radius_m=1.6)
        elif hazard_type == "gravel":
            self.scenario.spawn_gravel_patch(dist_ahead=max(16.0, dist_ahead * 0.65), y=-0.3, radius_m=1.9)
        elif hazard_type == "speed_bump":
            self.scenario.spawn_speed_bump(dist_ahead=max(20.0, dist_ahead * 0.8), y=0.0, height_m=0.12, radius_m=1.5)

    def step(self):
        if not self.is_running:
            return

        dt = self.scenario.dt
        ego_state = self.scenario.simulator.state
        s_curr, d_curr = self.scenario.env.geometry.cartesian_to_frenet(
            ego_state.pose.position.x,
            ego_state.pose.position.y
        )

        # Destination arrival check
        if s_curr >= self.scenario.env.geometry.length_m - 6.0:
            self.is_completed = True
            self.target_speed_mps = 0.0

        # 1. Real Perception Pipeline Execution (Camera -> IDD Neural Detection -> MOT Tracker -> Velocity)
        perception_frame = self.perception_pipeline.process_frame(
            timestamp=ego_state.timestamp,
            current_s=s_curr,
            geometry=self.scenario.env.geometry,
            actors=self.scenario.env.actors,
            ego_state=ego_state,
            anomalies=self.scenario.env.anomalies,
            dt=dt
        )
        obstacles = perception_frame.obstacles

        # Format perceived actors metadata with persistent track IDs, confidence, and estimated velocities
        actors_data = []
        for obs in obstacles:
            ttc_eval = self.supervisory.ttc_calc.compute_ttc(ego_state, [obs])
            ttc_val = ttc_eval.min_ttc_seconds if ttc_eval.min_ttc_seconds < 100.0 else None

            wx, wy = ego_to_world_2d(
                obs.bbox.center.x, obs.bbox.center.y,
                ego_state.pose.position.x, ego_state.pose.position.y,
                ego_state.pose.heading_rad
            )

            speed_mps = math.hypot(obs.velocity.x, obs.velocity.y)

            actors_data.append({
                "id": obs.id,
                "class": obs.obstacle_class.value,
                "confidence": round(obs.confidence, 3),
                "x_world": round(wx, 2),
                "y_world": round(wy, 2),
                "x_ego": round(obs.bbox.center.x, 2),
                "y_ego": round(obs.bbox.center.y, 2),
                "length_m": round(obs.bbox.size.x, 2),
                "width_m": round(obs.bbox.size.y, 2),
                "height_m": round(obs.bbox.size.z, 2),
                "yaw_deg": round(math.degrees(obs.bbox.yaw_rad), 1),
                "distance_m": round(obs.distance_m, 2),
                "ttc_s": round(ttc_val, 2) if ttc_val is not None else None,
                "speed_mps": round(speed_mps, 2),
                "speed_kph": round(speed_mps * 3.6, 1),
                "vx_ego": round(obs.velocity.x, 2),
                "vy_ego": round(obs.velocity.y, 2),
                "is_static": obs.is_static,
                "safety_radius_m": round(max(obs.bbox.size.x, obs.bbox.size.y) * 0.6 + 0.5, 2)
            })

        # 2. Multi-Modal Motion Prediction on Perceived Tracks
        pred_out = self.predictor.predict(perception_frame, ego_speed=ego_state.twist.speed_mps)

        # 3. Adaptive Path Planning on Perceived Corridor & Anomalies (7-Score Model)
        planned_traj = self.planner.plan(
            ego_state=ego_state,
            perception=perception_frame,
            prediction=pred_out,
            target_cruise_speed_mps=self.target_speed_mps
        )

        # 4. Independent Collision Avoidance & Safety Supervision
        safe_traj = self.supervisory.supervise(
            planned=planned_traj,
            ego_state=ego_state,
            perception=perception_frame,
            prediction=pred_out
        )

        if safe_traj.is_emergency_stop:
            self.is_emergency_stop = True

        # 5. Drive-By-Wire Vehicle Control Computation
        steer = self.lat_ctrl.compute_steering(ego_state, safe_traj)
        throttle, brake = self.lon_ctrl.compute_throttle_brake(ego_state, safe_traj, dt=dt)

        if self.is_emergency_stop or safe_traj.is_emergency_stop:
            throttle = 0.0
            brake = 100.0

        cmd = ControlCommand(
            timestamp=ego_state.timestamp,
            steering_angle_rad=steer,
            throttle_pct=throttle,
            brake_pct=brake,
            gear=GearMode.DRIVE if not self.is_emergency_stop else GearMode.PARK,
            emergency_brake_active=self.is_emergency_stop
        )

        # 6. Physical Vehicle Step
        state, raw_sensor = self.scenario.run_step(cmd)
        self.step_count += 1

        # 7. Post-step evaluation & telemetry extraction
        state = self.scenario.simulator.state
        s_post, d_post = self.scenario.env.geometry.cartesian_to_frenet(state.pose.position.x, state.pose.position.y)
        d_left, d_right = self.scenario.env.geometry.get_corridor_widths(s_post)
        curr_width = d_left - d_right
        margin = self.scenario.env.geometry.get_ditch_margin(s_post, d_post, vehicle_half_width=0.90)
        self.min_corridor_margin = min(self.min_corridor_margin, margin)

        if margin < 0.0:
            self.is_emergency_stop = True

        # 8. Format Anomalies
        anomalies_data = []
        for anom in perception_frame.anomalies:
            xEgo, yEgo = world_to_ego_2d(
                anom.position.x, anom.position.y,
                state.pose.position.x, state.pose.position.y,
                state.pose.heading_rad
            )
            dist_m = math.hypot(anom.position.x - state.pose.position.x, anom.position.y - state.pose.position.y)
            anomalies_data.append({
                "id": anom.id,
                "type": anom.anomaly_type,
                "traversability_class": getattr(anom, "traversability_class", "POTHOLE"),
                "is_passable": getattr(anom, "is_passable", False),
                "max_safe_speed_mps": getattr(anom, "max_safe_speed_mps", 0.0),
                "traversability_score": getattr(anom, "traversability_score", 0.05),
                "description": getattr(anom, "description", anom.anomaly_type),
                "x_world": round(anom.position.x, 2),
                "y_world": round(anom.position.y, 2),
                "x_ego": round(xEgo, 2),
                "y_ego": round(yEgo, 2),
                "distance_m": round(dist_m, 2),
                "radius_m": round(anom.radius_m, 2),
                "depth_or_height_m": round(anom.depth_or_height_m, 2)
            })

        # 9. Format Multi-Modal Predictions with Explainability & Corridor Invasion
        predictions_data = []
        for agent in pred_out.agents:
            trajs_data = []
            for t in agent.trajectories:
                pts_list = []
                for pt in t.waypoints:
                    pwx, pwy = ego_to_world_2d(
                        pt.position.x, pt.position.y,
                        ego_state.pose.position.x, ego_state.pose.position.y,
                        ego_state.pose.heading_rad
                    )
                    pts_list.append({
                        "x": round(pwx, 2),
                        "y": round(pwy, 2),
                        "x_ego": round(pt.position.x, 2),
                        "y_ego": round(pt.position.y, 2),
                        "sigma_x": round(pt.sigma_x, 2),
                        "sigma_y": round(pt.sigma_y, 2),
                        "time_offset_s": round(pt.timestamp - ego_state.timestamp, 2)
                    })
                trajs_data.append({
                    "mode_name": t.mode_name,
                    "probability": round(t.probability, 2),
                    "probability_pct": int(round(t.probability * 100)),
                    "collision_risk": round(t.collision_risk, 2),
                    "waypoints": pts_list
                })
            predictions_data.append({
                "id": agent.id,
                "class": agent.obstacle_class.value,
                "intent": agent.primary_intent.value,
                "is_high_risk": agent.is_high_risk,
                "corridor_invasion_prob": round(agent.corridor_invasion_prob, 2),
                "corridor_invasion_pct": int(round(agent.corridor_invasion_prob * 100)),
                "time_to_conflict_s": agent.time_to_conflict_s,
                "explanation": agent.explanation,
                "trajectories": trajs_data
            })

        # 10. Format All Adaptive Lattice Candidates & Multi-Objective Scorecards
        candidates_data = []
        collision_zones = []
        for cand, score in self.planner.last_scored_candidates:
            is_selected = (cand.candidate_id == self.planner.last_selected_candidate_id)
            
            wps = [
                {"x": round(wp.x, 2), "y": round(wp.y, 2), "speed_mps": round(wp.speed_mps, 2)}
                for wp in cand.waypoints
            ]
            
            if not score.is_feasible and score.status_tag == "COLLISION" and len(cand.waypoints) > 2:
                mid_pt = cand.waypoints[min(3, len(cand.waypoints)-1)]
                collision_zones.append({
                    "x": round(mid_pt.x, 2),
                    "y": round(mid_pt.y, 2),
                    "radius_m": 1.5,
                    "obstacle_id": score.status_tag,
                    "offset_m": round(cand.target_d, 2)
                })

            candidates_data.append({
                "candidate_id": cand.candidate_id,
                "label": cand.label,
                "offset": round(cand.target_d, 2),
                "target_v": round(cand.target_v, 2),
                "speed_ratio_pct": int(round((cand.target_v / max(0.1, self.target_speed_mps)) * 100)),
                "cost": min(9999.0, round(score.total_cost, 2)),
                "is_feasible": score.is_feasible,
                "is_selected": is_selected,
                "status_tag": score.status_tag,
                "rejection_reason": score.status_tag if not score.is_feasible else ("UNSAFE_CLEARANCE" if score.status_tag == "UNSAFE_CLEARANCE" else "SAFE"),
                "explanation": score.explanation,
                "cost_breakdown": score.cost_breakdown,
                "min_clearance_m": round(score.min_clearance_m, 2),
                "min_boundary_margin_m": round(score.min_boundary_margin_m, 2),
                "waypoints": wps
            })

        # 11. 30m Lookahead Goal Direction & Road Polyline
        s_goal = min(self.scenario.env.geometry.length_m - 2.0, s_curr + 30.0)
        gx, gy, goal_yaw = self.scenario.env.geometry.frenet_to_cartesian(s_goal, 0.0)

        road_polyline = []
        for s_idx in range(max(0, int(s_curr - 20)), min(int(self.scenario.env.geometry.length_m), int(s_curr + 65)), 2):
            cx, cy, _ = self.scenario.env.geometry.frenet_to_cartesian(float(s_idx), 0.0)
            dl, dr = self.scenario.env.geometry.get_corridor_widths(float(s_idx))
            lx, ly, _ = self.scenario.env.geometry.frenet_to_cartesian(float(s_idx), dl)
            rx, ry, _ = self.scenario.env.geometry.frenet_to_cartesian(float(s_idx), dr)
            road_polyline.append({
                "s": float(s_idx),
                "cx": round(cx, 2), "cy": round(cy, 2),
                "lx": round(lx, 2), "ly": round(ly, 2),
                "rx": round(rx, 2), "ry": round(ry, 2),
                "width_m": round(dl - dr, 2)
            })

        # 12. Build Live Causal Decision Chain Data
        lead_threat = None
        for a in actors_data:
            if a["x_ego"] > 0 and (a["ttc_s"] is not None or a["distance_m"] < 35.0):
                if lead_threat is None or (a["ttc_s"] or 999) < (lead_threat["ttc_s"] or 999):
                    lead_threat = a

        lead_pred = None
        for p in predictions_data:
            if p["is_high_risk"] or p["corridor_invasion_pct"] >= 20:
                lead_pred = p
                break

        lead_anomaly = None
        for an in anomalies_data:
            if 0 < an["x_ego"] < 25.0 and abs(an["y_ego"]) < 1.2 and abs(an["depth_or_height_m"]) > 0.07:
                if lead_anomaly is None or an["distance_m"] < lead_anomaly["distance_m"]:
                    lead_anomaly = an

        has_threat = lead_threat is not None and ((lead_threat["ttc_s"] is not None and lead_threat["ttc_s"] < 5.0) or lead_threat["distance_m"] < 25.0)

        # Selected lateral offset
        selected_offset = 0.0
        if safe_traj.waypoints:
            last_wp = safe_traj.waypoints[-1]
            _, d_end = self.scenario.env.geometry.cartesian_to_frenet(last_wp.x, last_wp.y)
            selected_offset = round(d_end, 2)

        causal_event = {
            "has_hazard": has_threat or lead_anomaly is not None or (lead_pred is not None and lead_pred["corridor_invasion_pct"] >= 20),
            "hazard_id": lead_threat["id"] if lead_threat else (f"ANOMALY_{lead_anomaly['id']}" if lead_anomaly else (lead_pred["id"] if lead_pred else "CLEAR")),
            "hazard_dist_m": lead_threat["distance_m"] if lead_threat else (lead_anomaly["distance_m"] if lead_anomaly else (lead_pred["time_to_conflict_s"] or 0.0)),
            "hazard_ttc_s": lead_threat["ttc_s"] if lead_threat else (lead_pred["time_to_conflict_s"] if lead_pred else None),
            "nominal_path_safe": not has_threat and lead_anomaly is None and (lead_pred is None or lead_pred["corridor_invasion_pct"] < 20),
            "selected_mode": planned_traj.behavior_mode.value,
            "selected_candidate_id": self.planner.last_selected_candidate_id,
            "selected_offset_m": selected_offset,
            "steer_command_deg": round(math.degrees(cmd.steering_angle_rad), 1),
            "safety_action": safe_traj.safety_action.value,
            "supervisor_gate": safe_traj.supervisor_gate_status,
            "is_rejected": safe_traj.is_rejected,
            "rejection_reason": safe_traj.rejection_reason or "NONE",
            "decision_explanation": self.planner.last_decision_explanation,
            "prediction_summary": lead_pred["explanation"] if lead_pred else ""
        }

        # Pack full telemetry payload
        self.latest_telemetry = {
            "timestamp": round(state.timestamp, 2),
            "step": self.step_count,
            "vehicle_id": "SIH26037-AV-01",
            "difficulty": self.difficulty.name,
            "perception": {
                "mode": self.perception_pipeline.mode.value,
                "active_tracks_count": len(actors_data),
                "sensor_health": perception_frame.sensor_health
            },
            "ego": {
                "x": round(state.pose.position.x, 2),
                "y": round(state.pose.position.y, 2),
                "heading_deg": round(math.degrees(state.pose.heading_rad), 1),
                "speed_mps": round(state.twist.speed_mps, 2),
                "speed_kph": round(state.twist.speed_mps * 3.6, 1),
                "steer_deg": round(math.degrees(cmd.steering_angle_rad), 1),
                "throttle_pct": round(cmd.throttle_pct, 1),
                "brake_pct": round(cmd.brake_pct, 1),
                "battery_soc": 95.0,
                "length_m": 4.2,
                "width_m": 1.8
            },
            "road": {
                "s": round(s_post, 2),
                "d": round(d_post, 2),
                "current_width_m": round(curr_width, 2),
                "left_edge_d_m": round(d_left, 2),
                "right_edge_d_m": round(d_right, 2),
                "current_margin_m": round(margin, 2),
                "min_corridor_margin_m": round(self.min_corridor_margin, 2),
                "is_ditch_breach": margin < 0.0,
                "status": "CHOKE_POINT" if curr_width < 3.8 else ("PASSING_BAY" if curr_width > 5.0 else "NOMINAL"),
                "polyline": road_polyline
            },
            "goal": {
                "x": round(gx, 2),
                "y": round(gy, 2),
                "yaw_deg": round(math.degrees(goal_yaw), 1),
                "distance_ahead_m": round(s_goal - s_curr, 1)
            },
            "actors": actors_data,
            "anomalies": anomalies_data,
            "predictions": predictions_data,
            "candidates": candidates_data,
            "collision_zones": collision_zones,
            "causal_event": causal_event,
            "trajectory": {
                "id": safe_traj.source_trajectory_id,
                "mode": planned_traj.behavior_mode.value,
                "safety_action": safe_traj.safety_action.value,
                "total_cost": round(planned_traj.total_cost, 2),
                "waypoints": [
                    {
                        "x": round(wp.x, 2),
                        "y": round(wp.y, 2),
                        "yaw_deg": round(math.degrees(wp.yaw_rad), 1),
                        "speed_mps": round(wp.speed_mps, 2)
                    }
                    for wp in safe_traj.waypoints
                ]
            },
            "status": {
                "is_running": self.is_running,
                "is_emergency_stop": self.is_emergency_stop,
                "is_completed": self.is_completed
            }
        }


def create_app(sim_engine: SimulationEngineState) -> FastAPI:
    app = FastAPI(title="SIH26037 Autonomous Driving System")

    @app.get("/health")
    async def health():
        return {
            "status": "HEALTHY",
            "vehicle_id": "SIH26037-AV-01",
            "perception_mode": sim_engine.perception_pipeline.mode.value,
            "predictor_mode": sim_engine.predictor.mode
        }

    @app.get("/telemetry")
    async def get_telemetry():
        return JSONResponse(content=sim_engine.latest_telemetry)

    @app.post("/simulation/toggle")
    async def toggle_simulation():
        sim_engine.is_running = not sim_engine.is_running
        return {"status": "TOGGLED", "is_running": sim_engine.is_running}

    @app.post("/simulation/reset")
    async def reset_simulation():
        sim_engine.reset()
        return {"status": "RESET"}

    @app.post("/simulation/emergency_stop")
    async def trigger_emergency_stop():
        sim_engine.is_emergency_stop = True
        return {"status": "EMERGENCY_STOP_TRIGGERED"}

    @app.post("/simulation/spawn_hazard")
    async def spawn_hazard_api(req: HazardSpawnRequest):
        sim_engine.spawn_hazard(req.hazard_type, req.distance_ahead_m)
        sim_engine.step()
        sim_engine.step()
        return {"status": "HAZARD_SPAWNED", "hazard_type": req.hazard_type}

    @app.post("/simulation/difficulty")
    async def set_difficulty_api(req: DifficultyRequest):
        sim_engine.set_difficulty(req.difficulty)
        return {"status": "DIFFICULTY_UPDATED", "difficulty": sim_engine.difficulty.name}

    @app.post("/simulation/perception_mode")
    async def set_perception_mode_api(req: PerceptionModeRequest):
        sim_engine.set_perception_mode(req.mode)
        return {"status": "PERCEPTION_MODE_UPDATED", "mode": sim_engine.perception_pipeline.mode.value}

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                sim_engine.step()
                await websocket.send_json(sim_engine.latest_telemetry)
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            pass

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard_html():
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIH26037 — Multi-Objective Adaptive Lattice Planner & 7-Score Decision Matrix</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #070b13; color: #e2e8f0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        canvas { background-color: #0a0f1d; border-radius: 0.75rem; border: 1px solid #1e293b; box-shadow: inset 0 0 30px rgba(0,0,0,0.7); }
        .glass-card { background: rgba(13, 20, 36, 0.88); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.75rem; }
        .chain-step { transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }
        .score-bar { height: 4px; border-radius: 2px; transition: width 0.2s ease-in-out; }
    </style>
</head>
<body class="p-3 md:p-6 max-w-[1600px] mx-auto space-y-4">

    <!-- Header & Interactive Controls -->
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-4 md:p-5">
        <div>
            <div class="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 mb-1">
                <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                SIH26037 • MULTI-OBJECTIVE ADAPTIVE LATTICE PLANNER
            </div>
            <h1 class="text-xl md:text-2xl font-black tracking-tight text-white flex items-center gap-3">
                Adaptive Spline Bundle & 7-Objective Decision Matrix
                <span id="badge-difficulty" class="text-xs px-2.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono font-bold">HARD TIER</span>
            </h1>
            <p class="text-xs text-slate-400 mt-0.5">
                Sense → Track → Forecast (60%/25%/15%) → 7-Score Lattice Evaluation (40% Col, 20% Clr, 15% Trav, 10% Prog, 5% Smth, 5% Dyn, 5% Unc) → Optimal Safe Trajectory Selection.
            </p>
        </div>

        <div class="flex flex-wrap items-center gap-2">
            <!-- Scenario Selector -->
            <select id="diff-select" onchange="changeDifficulty(this.value)" class="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-2 font-bold focus:ring-cyan-500 focus:border-cyan-500">
                <option value="EASY">Easy (Shoulder Auto)</option>
                <option value="MEDIUM">Medium (Boulder & Auto)</option>
                <option value="HARD" selected>Hard (Oncoming Tractor & Pedestrian)</option>
                <option value="EXTREME">Extreme (Wide Tractor & Darting Ped)</option>
            </select>

            <!-- Perception Mode Switcher -->
            <select id="perc-select" onchange="changePerceptionMode(this.value)" class="bg-cyan-950/80 border border-cyan-700 text-xs text-cyan-200 rounded-lg px-2.5 py-2 font-mono font-bold">
                <option value="NEURAL_IDD" selected>🤖 Perception: Neural IDD + EKF</option>
                <option value="GROUND_TRUTH">⚙️ Perception: Ground Truth Sim</option>
            </select>

            <!-- View Switcher -->
            <button onclick="toggleViewMode()" id="btn-view-mode" class="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 text-xs font-bold transition border border-cyan-500/30">
                🔭 View: Ego-Centric ADAS (30m Ahead)
            </button>

            <button onclick="toggleSim()" id="btn-toggle" class="px-3.5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition shadow">
                Pause
            </button>
            <button onclick="resetSim()" class="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold transition border border-slate-700">
                Reset
            </button>
            <button onclick="triggerEStop()" class="px-3.5 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold transition shadow animate-pulse">
                E-Stop
            </button>
        </div>
    </div>

    <!-- Live Causal Decision HUD Banner -->
    <div id="causal-banner" class="glass-card p-3.5 border border-cyan-500/30 bg-gradient-to-r from-slate-950 via-cyan-950/30 to-slate-950 space-y-2.5">
        <div class="flex items-center justify-between">
            <span class="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                End-to-End Perception, Multi-Modal Prediction & Causal Decision Chain
            </span>
            <span id="causal-summary-badge" class="text-xs font-mono font-bold px-3 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/40">
                PERCEPTION ACTIVE — CRUISING
            </span>
        </div>

        <!-- 5-Stage Causal Flow Stepper -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs font-mono">
            <div id="step-1" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">1. IDD Perception</span>
                <span id="hud-hazard" class="font-bold text-slate-300 block truncate">None Detected</span>
                <span id="hud-hazard-dist" class="text-[10px] text-slate-500">Track ID: --</span>
            </div>
            <div id="step-2" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">2. Motion Prediction</span>
                <span id="hud-ttc" class="font-bold text-slate-300 block truncate">Forecast: Clear</span>
                <span id="hud-risk" class="text-[10px] text-slate-500">Corridor Risk: <5%</span>
            </div>
            <div id="step-3" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">3. Candidate Lattice</span>
                <span id="hud-candidates" class="font-bold text-slate-300 block">28 Splines</span>
                <span id="hud-cand-status" class="text-[10px] text-emerald-400">7 Lat × 4 Spd</span>
            </div>
            <div id="step-4" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">4. Safety Supervisor</span>
                <span id="hud-gate" class="font-bold text-emerald-400 block truncate">PASSED (SAFE)</span>
                <span id="hud-offset" class="text-[10px] text-slate-400">Offset: 0.0m</span>
            </div>
            <div id="step-5" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">5. DBW Actuation</span>
                <span id="hud-steer-act" class="font-bold text-amber-400 block">Steer: 0.0°</span>
                <span id="hud-speed-act" class="text-[10px] text-slate-400">Speed: 21.6 km/h</span>
            </div>
        </div>
    </div>

    <!-- Main BEV Canvas Visualizer & Prediction Feed -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <!-- Main Canvas Area -->
        <div class="lg:col-span-3 glass-card p-4 space-y-3">
            <div class="flex flex-wrap items-center justify-between text-xs text-slate-400 px-1 gap-2">
                <div class="flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span class="font-bold uppercase tracking-wider text-slate-200">
                        Autonomous Driving Debug Canvas (BEV)
                    </span>
                    <span id="view-mode-tag" class="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">
                        EGO COCKPIT (30M AHEAD)
                    </span>
                    <span id="road-width-badge" class="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800">
                        WIDTH: 4.3m (NOMINAL)
                    </span>
                </div>

                <!-- Interactive Hazard Injection Bar -->
                <div class="flex items-center gap-1.5 flex-wrap">
                    <span class="text-[11px] text-slate-400 font-bold">Inject:</span>
                    <button onclick="spawnHazard('motorcycle')" class="px-2.5 py-1 rounded bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-[11px] font-bold transition">
                        🏍️ Motorcycle
                    </button>
                    <button onclick="spawnHazard('tractor')" class="px-2.5 py-1 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 text-[11px] font-bold transition">
                        🚜 Tractor
                    </button>
                    <button onclick="spawnHazard('pedestrian')" class="px-2.5 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 text-[11px] font-bold transition">
                        🚶 Villager
                    </button>
                    <button onclick="spawnHazard('cattle')" class="px-2.5 py-1 rounded bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-300 border border-yellow-500/40 text-[11px] font-bold transition">
                        🐄 Cattle
                    </button>
                    <button onclick="spawnHazard('auto')" class="px-2.5 py-1 rounded bg-orange-500/20 hover:bg-orange-500/30 text-orange-300 border border-orange-500/40 text-[11px] font-bold transition">
                        🛺 Auto
                    </button>
                    <button onclick="spawnHazard('pothole')" class="px-2.5 py-1 rounded bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/40 text-[11px] font-bold transition">
                        🕳️ Pothole
                    </button>
                    <button onclick="spawnHazard('waterlogged')" class="px-2.5 py-1 rounded bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 border border-sky-500/40 text-[11px] font-bold transition">
                        🌊 Flood Pool
                    </button>
                    <button onclick="spawnHazard('gravel')" class="px-2.5 py-1 rounded bg-orange-600/20 hover:bg-orange-600/30 text-orange-300 border border-orange-600/40 text-[11px] font-bold transition">
                        🧱 Gravel
                    </button>
                    <button onclick="spawnHazard('speed_bump')" class="px-2.5 py-1 rounded bg-yellow-600/20 hover:bg-yellow-600/30 text-yellow-300 border border-yellow-600/40 text-[11px] font-bold transition">
                        ⚠️ Speed Bump
                    </button>
                </div>
            </div>

            <!-- Canvas Element -->
            <div class="relative w-full overflow-hidden rounded-xl">
                <canvas id="simCanvas" width="1060" height="580" class="w-full h-auto block"></canvas>
            </div>

            <!-- Visual Legend & Explanation Bar -->
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2 text-[10px] text-slate-300 pt-2 border-t border-slate-800/80">
                <div class="flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-emerald-400 shadow-sm shadow-emerald-400"></span> ★ Selected Path</div>
                <div class="flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-cyan-400/40 border border-cyan-400"></span> Candidates (P1..P7)</div>
                <div class="flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-amber-400 border border-amber-400"></span> Forecast 60%</div>
                <div class="flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-rose-500 border border-rose-500"></span> Cut-In 25% ⚠️</div>
                <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-red-950 border border-red-500"></span> 🕳️ Potholes</div>
                <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-sky-950 border border-sky-400"></span> 🌊 Flooded Pool</div>
                <div class="flex items-center gap-1.5"><span class="w-3 h-1 bg-yellow-400"></span> Speed Bumps</div>
                <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full border border-yellow-400 text-yellow-400 font-mono text-[9px] flex items-center justify-center">🎯</span> Goal Horizon</div>
            </div>
        </div>

        <!-- Right Side: Probabilistic Predictions & Live Telemetry -->
        <div class="glass-card p-4 space-y-4">
            <!-- Telemetry Cards -->
            <div class="grid grid-cols-2 gap-2">
                <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <span class="text-[10px] text-slate-400 font-bold uppercase block">Speed</span>
                    <span id="kpi-speed" class="text-xl font-black text-cyan-400 font-mono">0.0</span>
                    <span class="text-[9px] text-slate-500 font-semibold">km/h</span>
                </div>
                <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <span class="text-[10px] text-slate-400 font-bold uppercase block">Steer (Stanley)</span>
                    <span id="kpi-steer" class="text-xl font-black text-amber-400 font-mono">0.0°</span>
                    <span class="text-[9px] text-slate-500 font-semibold">Front Wheel</span>
                </div>
                <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <span class="text-[10px] text-slate-400 font-bold uppercase block">Selected Traj</span>
                    <span id="kpi-traj-id" class="text-xl font-black text-emerald-300 font-mono">P4</span>
                    <span class="text-[9px] text-slate-500 font-semibold">Cost: 12.4</span>
                </div>
                <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <span class="text-[10px] text-slate-400 font-bold uppercase block">Ditch Margin</span>
                    <span id="kpi-margin" class="text-xl font-black text-purple-400 font-mono">2.15 m</span>
                    <span class="text-[9px] text-slate-500 font-semibold">Hard Invariant</span>
                </div>
            </div>

            <!-- Probabilistic Motion Prediction Panel -->
            <div class="space-y-2">
                <div class="flex items-center justify-between border-t border-slate-800 pt-2">
                    <h3 class="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-1.5">
                        <span>🧠</span> Motion Prediction (IDD)
                    </h3>
                    <span class="text-[10px] text-cyan-400 font-mono">Multi-Modal K=3</span>
                </div>
                
                <div id="predictions-list" class="space-y-2 text-xs max-h-56 overflow-y-auto pr-1">
                    <!-- Dynamically populated with multi-modal forecast cards -->
                </div>
            </div>

            <!-- Road Anomalies Feed (Potholes & Speed Bumps) -->
            <div class="space-y-1.5">
                <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300 pt-2 border-t border-slate-800">
                    Perceived Road Anomalies
                </h3>
                <div id="anomalies-list" class="space-y-1 text-[11px] font-mono max-h-24 overflow-y-auto pr-1">
                    <!-- Populated with potholes / humps -->
                </div>
            </div>
        </div>
    </div>

    <!-- Section: 🏆 ADAPTIVE TRAJECTORY SCORECARD & DECISION MATRIX -->
    <div class="glass-card p-4 space-y-3">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div>
                <h2 class="text-base font-black text-white flex items-center gap-2">
                    <span>🏆</span> ADAPTIVE TRAJECTORY EVALUATOR & SCORECARD
                    <span class="text-[11px] px-2.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-mono">
                        7-OBJECTIVE WEIGHTED MODEL
                    </span>
                </h2>
                <p class="text-xs text-slate-400 mt-0.5">
                    Real-time ranking of candidate splines across lateral Frenet offsets ($P_1 \dots P_7$) against Collision (40%), Clearance (20%), Traversability (15%), Progress (10%), Smoothness (5%), Dynamics (5%), and Uncertainty (5%).
                </p>
            </div>
            <div id="decision-summary-box" class="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/40 text-xs font-mono text-emerald-300 max-w-xl">
                ★ Evaluating candidates...
            </div>
        </div>

        <div class="overflow-x-auto">
            <table class="w-full text-left text-xs font-mono border-collapse">
                <thead>
                    <tr class="text-[10px] uppercase text-slate-400 border-b border-slate-800 bg-slate-900/50">
                        <th class="py-2.5 px-3">Candidate</th>
                        <th class="py-2.5 px-2">Offset (Δd)</th>
                        <th class="py-2.5 px-2">Speed</th>
                        <th class="py-2.5 px-2">Status</th>
                        <th class="py-2.5 px-2">Collision (40%)</th>
                        <th class="py-2.5 px-2">Clearance (20%)</th>
                        <th class="py-2.5 px-2">Traversability (15%)</th>
                        <th class="py-2.5 px-2">Progress (10%)</th>
                        <th class="py-2.5 px-2">Smooth (5%)</th>
                        <th class="py-2.5 px-2">Dyn (5%)</th>
                        <th class="py-2.5 px-2">Uncert (5%)</th>
                        <th class="py-2.5 px-2 font-bold text-white">Total Cost</th>
                        <th class="py-2.5 px-3">Decision Explanation</th>
                    </tr>
                </thead>
                <tbody id="scorecard-tbody" class="divide-y divide-slate-800/60">
                    <!-- Populated dynamically with candidate scores -->
                </tbody>
            </table>
        </div>
    </div>

    <!-- Script: WebSocket & High-Density Canvas Debug Renderer -->
    <script>
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        let currentData = null;
        let viewMode = 'ego';

        function toggleViewMode() {
            viewMode = viewMode === 'ego' ? 'road' : 'ego';
            const btn = document.getElementById('btn-view-mode');
            const tag = document.getElementById('view-mode-tag');
            if (viewMode === 'ego') {
                btn.innerText = '🔭 View: Ego-Centric ADAS (30m Ahead)';
                tag.innerText = 'EGO COCKPIT (30M AHEAD)';
            } else {
                btn.innerText = '🗺️ View: Panoramic Road Overview';
                tag.innerText = 'ROAD OVERVIEW (PANORAMA)';
            }
            if (currentData) renderScene(currentData);
        }

        const ws = new WebSocket(`ws://${window.location.host}/ws/telemetry`);
        ws.onmessage = (event) => {
            currentData = JSON.parse(event.data);
            updateUI(currentData);
            renderScene(currentData);
        };

        function updateUI(data) {
            if (!data || !data.ego) return;
            document.getElementById('kpi-speed').innerText = data.ego.speed_kph;
            document.getElementById('kpi-steer').innerText = `${data.ego.steer_deg}°`;
            document.getElementById('kpi-margin').innerText = `${data.road.current_margin_m} m`;

            const selectedCandId = (data.causal_event && data.causal_event.selected_candidate_id) || (data.candidates && data.candidates.find(c => c.is_selected) || {}).candidate_id || 'P4';
            const selectedCandCost = (data.trajectory && data.trajectory.total_cost) || 0.0;
            document.getElementById('kpi-traj-id').innerText = selectedCandId;
            document.getElementById('kpi-traj-id').nextElementSibling.innerText = `Cost: ${selectedCandCost.toFixed(1)}`;

            const widthBadge = document.getElementById('road-width-badge');
            if (data.road.status === 'CHOKE_POINT') {
                widthBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono bg-rose-950 text-rose-300 border border-rose-800 animate-pulse';
                widthBadge.innerText = `⚠️ CHOKE POINT: ${data.road.current_width_m}m`;
            } else if (data.road.status === 'PASSING_BAY') {
                widthBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-800';
                widthBadge.innerText = `PASSING BAY: ${data.road.current_width_m}m`;
            } else {
                widthBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800';
                widthBadge.innerText = `WIDTH: ${data.road.current_width_m}m (NOMINAL)`;
            }

            if (data.difficulty) {
                document.getElementById('badge-difficulty').innerText = `${data.difficulty} TIER`;
                document.getElementById('diff-select').value = data.difficulty;
            }

            if (data.perception && data.perception.mode) {
                document.getElementById('perc-select').value = data.perception.mode;
            }

            const btnToggle = document.getElementById('btn-toggle');
            btnToggle.innerText = data.status.is_running ? 'Pause' : 'Resume';

            // Update Causal HUD Stepper
            const ce = data.causal_event || {};
            const s1 = document.getElementById('step-1');
            const s2 = document.getElementById('step-2');
            const s3 = document.getElementById('step-3');
            const s4 = document.getElementById('step-4');
            const s5 = document.getElementById('step-5');

            if (ce.has_hazard) {
                s1.className = 'chain-step p-2 rounded-lg bg-amber-950/40 border border-amber-500/60 shadow-lg shadow-amber-900/20';
                document.getElementById('hud-hazard').innerHTML = `<span class="text-amber-400 font-bold">🚨 ${ce.hazard_id}</span>`;
                document.getElementById('hud-hazard-dist').innerText = `Range: ${ce.hazard_dist_m}m`;

                s2.className = 'chain-step p-2 rounded-lg bg-rose-950/40 border border-rose-500/60';
                document.getElementById('hud-ttc').innerHTML = `<span class="text-rose-400 font-bold">${ce.prediction_summary || 'Cut-in Risk Detected'}</span>`;
                document.getElementById('hud-risk').innerHTML = ce.nominal_path_safe ? '<span class="text-slate-400">Path: CLEAR</span>' : '<span class="text-rose-400 font-bold">Path: CONFLICT DETECTED</span>';

                s3.className = 'chain-step p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/60';
                document.getElementById('hud-candidates').innerText = `28 Evaluated`;
                document.getElementById('hud-cand-status').innerHTML = `<span class="text-cyan-300">Offset ${ce.selected_offset_m > 0 ? '+' : ''}${ce.selected_offset_m}m Safe</span>`;

                if (ce.is_rejected) {
                    s4.className = 'chain-step p-2 rounded-lg bg-rose-950/50 border border-rose-500 shadow-lg shadow-rose-950/50';
                    document.getElementById('hud-gate').innerHTML = `<span class="text-rose-400 font-bold">⛔ REJECTED: ${ce.rejection_reason}</span>`;
                } else if (ce.supervisor_gate === 'PASSED_WITH_SLOWDOWN') {
                    s4.className = 'chain-step p-2 rounded-lg bg-amber-950/50 border border-amber-500';
                    document.getElementById('hud-gate').innerHTML = `<span class="text-amber-300 font-bold">⚠️ PASSED (SLOWDOWN)</span>`;
                } else {
                    s4.className = 'chain-step p-2 rounded-lg bg-emerald-950/40 border border-emerald-500/60';
                    document.getElementById('hud-gate').innerHTML = `<span class="text-emerald-300 font-bold">🛡️ PASSED (SAFE: ${selectedCandId})</span>`;
                }
                document.getElementById('hud-offset').innerText = `Mode: ${ce.selected_mode} | Off: ${ce.selected_offset_m > 0 ? '+' : ''}${ce.selected_offset_m}m`;

                s5.className = 'chain-step p-2 rounded-lg bg-indigo-950/40 border border-indigo-500/60';
                document.getElementById('hud-steer-act').innerHTML = `<span class="text-amber-300 font-bold">Steer: ${ce.steer_command_deg}°</span>`;
                document.getElementById('hud-speed-act').innerText = `Speed: ${data.ego.speed_kph} km/h`;

                document.getElementById('causal-summary-badge').className = 'text-xs font-mono font-bold px-3 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-500/60 animate-pulse';
                document.getElementById('causal-summary-badge').innerText = `PROACTIVE AVOIDANCE: ${ce.hazard_id.toUpperCase()} (${ce.selected_mode})`;
            } else {
                [s1, s2, s3, s4, s5].forEach(el => el.className = 'chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800');
                document.getElementById('hud-hazard').innerText = 'None Detected';
                document.getElementById('hud-hazard-dist').innerText = 'Track ID: Clear';
                document.getElementById('hud-ttc').innerText = 'Forecast: Clear';
                document.getElementById('hud-risk').innerText = 'Corridor Risk: <5%';
                document.getElementById('hud-candidates').innerText = '28 Evaluated';
                document.getElementById('hud-cand-status').innerText = 'Center (0.0m) Clear';
                document.getElementById('hud-gate').innerHTML = `<span class="text-emerald-400">🛡️ PASSED (${selectedCandId})</span>`;
                document.getElementById('hud-offset').innerText = 'Offset: 0.0m';
                document.getElementById('hud-steer-act').innerText = `Steer: ${data.ego.steer_deg}°`;
                document.getElementById('hud-speed-act').innerText = `Speed: ${data.ego.speed_kph} km/h`;

                document.getElementById('causal-summary-badge').className = 'text-xs font-mono font-bold px-3 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/40';
                document.getElementById('causal-summary-badge').innerText = 'PERCEPTION ACTIVE — CRUISING';
            }

            // Update Decision Explanation Box
            const decBox = document.getElementById('decision-summary-box');
            if (ce.decision_explanation) {
                decBox.innerHTML = `★ <strong>${selectedCandId} Selected:</strong> ${ce.decision_explanation}`;
            }

            // Update Probabilistic Motion Prediction List
            const predList = document.getElementById('predictions-list');
            predList.innerHTML = '';
            (data.predictions || []).forEach(p => {
                const item = document.createElement('div');
                const isHigh = p.is_high_risk || p.corridor_invasion_pct >= 25;
                item.className = `p-2.5 rounded-lg border ${isHigh ? 'bg-rose-950/40 border-rose-500/60 shadow-md' : 'bg-slate-900 border-slate-800'} space-y-2`;
                
                const iconMap = {
                    'MOTORCYCLE': '🏍️',
                    'TRUCK': '🚜',
                    'AUTO_RICKSHAW': '🛺',
                    'PEDESTRIAN': '🚶',
                    'CATTLE_ANIMAL': '🐄',
                    'BUS': '🚌',
                    'CAR': '🚗'
                };
                const icon = iconMap[p.class] || '🚗';

                let modePillsHtml = '';
                (p.trajectories || []).forEach(t => {
                    const isCutIn = t.mode_name.includes('cut_in') || t.mode_name.includes('crossing');
                    const badgeColor = isCutIn ? (t.probability_pct >= 25 ? 'bg-rose-900/80 text-rose-200 border-rose-600 font-bold' : 'bg-amber-900/50 text-amber-200 border-amber-600') : 'bg-slate-800 text-slate-300 border-slate-700';
                    const modeLabel = t.mode_name.replace(/_/g, ' ');
                    modePillsHtml += `
                        <div class="flex items-center justify-between text-[10px] px-2 py-0.5 rounded ${badgeColor} border font-mono">
                            <span class="capitalize">${modeLabel}</span>
                            <span class="font-bold">${t.probability_pct}%</span>
                        </div>
                    `;
                });

                const riskBadge = p.corridor_invasion_pct > 0 
                    ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold ${p.corridor_invasion_pct >= 25 ? 'bg-rose-900 text-rose-200 animate-pulse' : 'bg-amber-900 text-amber-200'} font-mono">${p.corridor_invasion_pct}% Cut-in Risk</span>`
                    : `<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-300 font-mono">Path Clear</span>`;

                item.innerHTML = `
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-1.5 font-bold text-slate-200">
                            <span>${icon}</span>
                            <span class="font-mono text-xs">${p.id}</span>
                            <span class="text-[9px] px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">${p.intent}</span>
                        </div>
                        ${riskBadge}
                    </div>

                    <div class="space-y-1">
                        ${modePillsHtml}
                    </div>

                    <div class="text-[10px] text-slate-300 font-sans italic bg-slate-950/60 p-1.5 rounded border border-slate-800/80">
                        💬 "${p.explanation}"
                    </div>
                `;
                predList.appendChild(item);
            });

            // Update Anomalies List
            const anomList = document.getElementById('anomalies-list');
            anomList.innerHTML = '';
            (data.anomalies || []).forEach(an => {
                const row = document.createElement('div');
                row.className = 'flex items-center justify-between p-1.5 rounded bg-slate-900/80 border border-slate-800 text-[10px]';
                const icon = an.type === 'POTHOLE' ? '🕳️' : (an.type === 'SPEED_BUMP' ? '⚠️' : '🧱');
                const tagColor = an.type === 'POTHOLE' ? 'text-red-400' : 'text-yellow-400';
                row.innerHTML = `
                    <span class="${tagColor} font-bold">${icon} ${an.id}</span>
                    <span class="text-slate-400">@ ${an.distance_m}m (r: ${an.radius_m}m)</span>
                `;
                anomList.appendChild(row);
            });

            // Update 7-Objective Candidate Scorecard Table
            const tbody = document.getElementById('scorecard-tbody');
            tbody.innerHTML = '';

            // Group candidates by candidate_id to show primary representative or top candidates
            const cands = data.candidates || [];
            cands.forEach(c => {
                const tr = document.createElement('tr');
                const cb = c.cost_breakdown || {};
                
                let statusBadge = '';
                if (c.is_selected) {
                    statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold">★ OPTIMAL</span>';
                    tr.className = 'bg-emerald-950/30 font-bold border-l-4 border-emerald-400 text-slate-200';
                } else if (c.status_tag === 'SAFE') {
                    statusBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-800">FEASIBLE</span>';
                    tr.className = 'hover:bg-slate-900/60 text-slate-300';
                } else if (c.status_tag === 'UNSAFE_CLEARANCE') {
                    statusBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] bg-amber-950 text-amber-300 border border-amber-800">UNSAFE CLR</span>';
                    tr.className = 'hover:bg-slate-900/60 text-amber-200/80';
                } else if (c.status_tag === 'UNSAFE_TRAVERSABILITY') {
                    statusBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] bg-red-950 text-red-300 border border-red-800 font-bold">UNSAFE SURF</span>';
                    tr.className = 'bg-red-950/20 text-red-300/80';
                } else if (c.status_tag === 'COLLISION') {
                    statusBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] bg-rose-950 text-rose-300 border border-rose-800 font-bold">COLLISION</span>';
                    tr.className = 'bg-rose-950/20 text-rose-300/80';
                } else if (c.status_tag === 'DITCH_BREACH') {
                    statusBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] bg-purple-950 text-purple-300 border border-purple-800">DITCH BREACH</span>';
                    tr.className = 'bg-purple-950/20 text-purple-300/80';
                } else {
                    statusBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] bg-slate-900 text-slate-400 border border-slate-800">${c.status_tag}</span>`;
                    tr.className = 'text-slate-400';
                }

                const formatScore = (val) => {
                    const num = val || 0;
                    const color = num > 50 ? 'text-rose-400' : (num > 15 ? 'text-amber-400' : 'text-emerald-400');
                    return `<span class="${color}">${num.toFixed(1)}</span>`;
                };

                tr.innerHTML = `
                    <td class="py-2 px-3 font-bold text-cyan-300">${c.candidate_id}</td>
                    <td class="py-2 px-2">${c.offset > 0 ? '+' : ''}${c.offset.toFixed(1)}m</td>
                    <td class="py-2 px-2 text-slate-400">${c.speed_ratio_pct}%</td>
                    <td class="py-2 px-2">${statusBadge}</td>
                    <td class="py-2 px-2">${formatScore(cb.collision_safety)}</td>
                    <td class="py-2 px-2">${formatScore(cb.obstacle_clearance)}</td>
                    <td class="py-2 px-2">${formatScore(cb.road_traversability || cb.traversability)}</td>
                    <td class="py-2 px-2">${formatScore(cb.progress)}</td>
                    <td class="py-2 px-2">${formatScore(cb.path_smoothness)}</td>
                    <td class="py-2 px-2">${formatScore(cb.vehicle_dynamics)}</td>
                    <td class="py-2 px-2">${formatScore(cb.uncertainty)}</td>
                    <td class="py-2 px-2 font-bold ${c.cost > 900 ? 'text-rose-400' : 'text-emerald-300'}">${c.cost > 900 ? '999+' : c.cost.toFixed(1)}</td>
                    <td class="py-2 px-3 text-[11px] font-sans truncate max-w-xs text-slate-300" title="${c.explanation}">${c.explanation}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // ==========================================
        // HIGH-DENSITY UNSTRUCTURED ROAD BEV RENDERER
        // ==========================================
        function renderScene(data) {
            if (!data || !data.ego) return;
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            if (viewMode === 'ego') {
                renderEgoCockpitView(data, w, h);
            } else {
                renderRoadOverview(data, w, h);
            }
        }

        // ----------------------------------------------------
        // VIEW 1: EGO COCKPIT VIEW (Forward Lookahead 30m)
        // ----------------------------------------------------
        function renderEgoCockpitView(data, w, h) {
            const ego = data.ego;
            const egoX = ego.x;
            const egoY = ego.y;
            const egoHeading = ego.heading_deg * Math.PI / 180;

            const originX = w / 2;
            const originY = h - 90;
            const scale = 11.5;

            function worldToScreen(wx, wy) {
                const dx = wx - egoX;
                const dy = wy - egoY;
                const xEgo = dx * Math.cos(egoHeading) + dy * Math.sin(egoHeading);
                const yEgo = -dx * Math.sin(egoHeading) + dy * Math.cos(egoHeading);
                const sx = originX - yEgo * scale;
                const sy = originY - xEgo * scale;
                return { sx, sy, xEgo, yEgo };
            }

            // 1. Radar Grid & Dark Terrain Background
            ctx.fillStyle = '#0a0f1d';
            ctx.fillRect(0, 0, w, h);

            // Distance Metric Range Arcs (10m, 20m, 30m, 40m)
            [10, 20, 30, 40].forEach(r => {
                const arcR = r * scale;
                ctx.strokeStyle = r === 30 ? 'rgba(234, 179, 8, 0.35)' : 'rgba(6, 182, 212, 0.18)';
                ctx.lineWidth = r === 30 ? 1.5 : 1.0;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.arc(originX, originY, arcR, -Math.PI * 0.85, -Math.PI * 0.15);
                ctx.stroke();

                ctx.fillStyle = r === 30 ? '#fde047' : '#06b6d4';
                ctx.font = 'bold 10px monospace';
                ctx.fillText(`── ${r} m ──`, originX - 24, originY - arcR - 3);
            });

            // Lateral offset grid lines (-3m to +3m)
            [-3, -2, -1, 0, 1, 2, 3].forEach(lat => {
                const lineX = originX - lat * scale;
                ctx.strokeStyle = lat === 0 ? 'rgba(255, 255, 255, 0.15)' : 'rgba(148, 163, 184, 0.08)';
                ctx.lineWidth = lat === 0 ? 1.5 : 0.8;
                ctx.setLineDash(lat === 0 ? [5, 5] : [2, 6]);
                ctx.beginPath();
                ctx.moveTo(lineX, originY + 30);
                ctx.lineTo(lineX, originY - 42 * scale);
                ctx.stroke();
            });
            ctx.setLineDash([]);

            // 2. Unstructured Organic Road Surface
            const poly = data.road.polyline || [];
            if (poly.length > 2) {
                ctx.fillStyle = '#161e2e';
                ctx.beginPath();
                let started = false;
                poly.forEach(pt => {
                    const scr = worldToScreen(pt.lx, pt.ly);
                    if (!started) { ctx.moveTo(scr.sx, scr.sy); started = true; }
                    else { ctx.lineTo(scr.sx, scr.sy); }
                });
                for (let i = poly.length - 1; i >= 0; i--) {
                    const scr = worldToScreen(poly[i].rx, poly[i].ry);
                    ctx.lineTo(scr.sx, scr.sy);
                }
                ctx.closePath();
                ctx.fill();

                // Centerline
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([6, 6]);
                ctx.beginPath();
                poly.forEach((pt, i) => {
                    const scr = worldToScreen(pt.cx, pt.cy);
                    if (i === 0) ctx.moveTo(scr.sx, scr.sy);
                    else ctx.lineTo(scr.sx, scr.sy);
                });
                ctx.stroke();
                ctx.setLineDash([]);

                // Left Edge
                ctx.strokeStyle = '#eab308';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                poly.forEach((pt, i) => {
                    const scr = worldToScreen(pt.lx, pt.ly);
                    if (i === 0) ctx.moveTo(scr.sx, scr.sy);
                    else ctx.lineTo(scr.sx, scr.sy);
                });
                ctx.stroke();

                // Right Edge
                ctx.strokeStyle = '#eab308';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                poly.forEach((pt, i) => {
                    const scr = worldToScreen(pt.rx, pt.ry);
                    if (i === 0) ctx.moveTo(scr.sx, scr.sy);
                    else ctx.lineTo(scr.sx, scr.sy);
                });
                ctx.stroke();

                // Unpaved Shoulder Hazard Berm Hatching
                ctx.strokeStyle = 'rgba(234, 179, 8, 0.35)';
                ctx.lineWidth = 1.0;
                for (let i = 0; i < poly.length; i += 2) {
                    const lScr = worldToScreen(poly[i].lx, poly[i].ly);
                    const rScr = worldToScreen(poly[i].rx, poly[i].ry);
                    ctx.beginPath();
                    ctx.moveTo(lScr.sx, lScr.sy);
                    ctx.lineTo(lScr.sx - 8, lScr.sy - 4);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(rScr.sx, rScr.sy);
                    ctx.lineTo(rScr.sx + 8, rScr.sy - 4);
                    ctx.stroke();
                }
            }

            // 3. Road Anomalies (Potholes, Waterlogging, Speed Bumps, Gravel)
            (data.anomalies || []).forEach(an => {
                const scr = worldToScreen(an.x_world, an.y_world);
                const rPix = Math.max(8, an.radius_m * scale);

                if (an.type === 'POTHOLE') {
                    // Dark crater with red danger ring
                    const grad = ctx.createRadialGradient(scr.sx, scr.sy, 2, scr.sx, scr.sy, rPix);
                    grad.addColorStop(0, '#000000');
                    grad.addColorStop(0.7, '#450a0a');
                    grad.addColorStop(1, '#991b1b');
                    ctx.fillStyle = grad;
                    ctx.strokeStyle = '#ef4444';
                    ctx.lineWidth = 2.0;
                    ctx.beginPath();
                    ctx.arc(scr.sx, scr.sy, rPix, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();

                    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
                    ctx.lineWidth = 1.0;
                    ctx.beginPath();
                    ctx.arc(scr.sx, scr.sy, rPix * 0.5, 0, Math.PI * 2);
                    ctx.stroke();

                    ctx.fillStyle = '#fca5a5';
                    ctx.font = 'bold 8px monospace';
                    ctx.fillText(`🕳️ ${Math.round(an.depth_or_height_m * 100)}cm`, scr.sx + rPix + 3, scr.sy + 3);
                } else if (an.type === 'WATER_LOGGING' || an.type === 'WATERLOGGED') {
                    // Shimmering flooded blue pool with ripple rings
                    ctx.fillStyle = 'rgba(14, 165, 233, 0.45)';
                    ctx.strokeStyle = '#38bdf8';
                    ctx.lineWidth = 2.0;
                    ctx.beginPath();
                    ctx.arc(scr.sx, scr.sy, rPix, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();

                    ctx.strokeStyle = 'rgba(186, 230, 253, 0.7)';
                    ctx.lineWidth = 1.2;
                    ctx.beginPath();
                    ctx.arc(scr.sx, scr.sy, rPix * 0.6, 0, Math.PI * 2);
                    ctx.stroke();

                    ctx.fillStyle = '#bae6fd';
                    ctx.font = 'bold 8px monospace';
                    ctx.fillText(`🌊 WATER (${Math.round(an.depth_or_height_m * 100)}cm)`, scr.sx + rPix + 3, scr.sy + 3);
                } else if (an.type === 'SPEED_BUMP') {
                    ctx.strokeStyle = '#fde047';
                    ctx.lineWidth = 4.5;
                    ctx.beginPath();
                    ctx.moveTo(scr.sx - rPix * 1.5, scr.sy);
                    ctx.lineTo(scr.sx + rPix * 1.5, scr.sy);
                    ctx.stroke();

                    ctx.fillStyle = '#fef08a';
                    ctx.font = 'bold 8px monospace';
                    ctx.fillText(`⚠️ BUMP (+${Math.round(an.depth_or_height_m * 100)}cm)`, scr.sx + rPix * 1.5 + 4, scr.sy + 3);
                } else if (an.type === 'GRAVEL') {
                    ctx.fillStyle = 'rgba(249, 115, 22, 0.35)';
                    ctx.strokeStyle = '#f97316';
                    ctx.lineWidth = 1.8;
                    ctx.beginPath();
                    ctx.arc(scr.sx, scr.sy, rPix, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();

                    ctx.fillStyle = '#fdba74';
                    ctx.font = 'bold 8px monospace';
                    ctx.fillText('🧱 GRAVEL', scr.sx + rPix + 3, scr.sy + 3);
                }
            });

            // 4. Goal Direction Indicator (30m ahead lookahead horizon)
            if (data.goal) {
                const gScr = worldToScreen(data.goal.x, data.goal.y);
                ctx.strokeStyle = 'rgba(234, 179, 8, 0.35)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(originX, originY);
                ctx.lineTo(gScr.sx, gScr.sy);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.strokeStyle = '#eab308';
                ctx.fillStyle = 'rgba(234, 179, 8, 0.25)';
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                ctx.arc(gScr.sx, gScr.sy, 10, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(gScr.sx - 14, gScr.sy); ctx.lineTo(gScr.sx + 14, gScr.sy);
                ctx.moveTo(gScr.sx, gScr.sy - 14); ctx.lineTo(gScr.sx, gScr.sy + 14);
                ctx.stroke();

                ctx.fillStyle = '#fde047';
                ctx.font = 'bold 9px monospace';
                ctx.fillText(`🎯 GOAL: +${data.goal.distance_ahead_m}m`, gScr.sx + 14, gScr.sy + 3);
            }

            // 5. Candidate Ego Trajectories (7-Spline Bundle P1..P7)
            (data.candidates || []).forEach(cand => {
                if (!cand.waypoints || cand.waypoints.length === 0) return;
                ctx.beginPath();
                ctx.moveTo(originX, originY);

                cand.waypoints.forEach(wp => {
                    const scr = worldToScreen(wp.x, wp.y);
                    ctx.lineTo(scr.sx, scr.sy);
                });

                if (cand.is_selected) {
                    ctx.strokeStyle = '#10b981';
                    ctx.lineWidth = 4.5;
                    ctx.shadowColor = '#10b981';
                    ctx.shadowBlur = 12;
                    ctx.setLineDash([]);
                    ctx.stroke();
                    ctx.shadowBlur = 0;

                    cand.waypoints.forEach((wp, idx) => {
                        if (idx % 2 === 0) {
                            const scr = worldToScreen(wp.x, wp.y);
                            ctx.fillStyle = '#6ee7b7';
                            ctx.beginPath();
                            ctx.arc(scr.sx, scr.sy, 3.5, 0, Math.PI * 2);
                            ctx.fill();
                        }
                    });
                } else if (cand.is_feasible) {
                    ctx.strokeStyle = 'rgba(6, 182, 212, 0.35)';
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([4, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                } else {
                    ctx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([3, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            });

            // 6. Collision & Predicted Conflict Zones
            (data.collision_zones || []).forEach(cz => {
                const scr = worldToScreen(cz.x, cz.y);
                const rPix = cz.radius_m * scale;

                ctx.fillStyle = 'rgba(239, 68, 68, 0.28)';
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                ctx.arc(scr.sx, scr.sy, rPix, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.moveTo(scr.sx - 6, scr.sy - 6); ctx.lineTo(scr.sx + 6, scr.sy + 6);
                ctx.moveTo(scr.sx + 6, scr.sy - 6); ctx.lineTo(scr.sx - 6, scr.sy + 6);
                ctx.stroke();

                ctx.fillStyle = '#fca5a5';
                ctx.font = 'bold 9px monospace';
                ctx.fillText(`💥 ${cz.obstacle_id}`, scr.sx + rPix + 3, scr.sy + 3);
            });

            // 7. Multi-Modal Motion Predictions of Perceived Actors (60% / 25% / 15%)
            (data.predictions || []).forEach(pred => {
                (pred.trajectories || []).forEach(tr => {
                    if (tr.waypoints && tr.waypoints.length > 0) {
                        ctx.beginPath();
                        const s0 = worldToScreen(tr.waypoints[0].x, tr.waypoints[0].y);
                        ctx.moveTo(s0.sx, s0.sy);

                        tr.waypoints.forEach(pt => {
                            const scr = worldToScreen(pt.x, pt.y);
                            ctx.lineTo(scr.sx, scr.sy);
                        });

                        const isCutIn = tr.mode_name.includes('cut_in') || tr.mode_name.includes('crossing');
                        const isContinuation = tr.mode_name.includes('continuation');
                        
                        ctx.strokeStyle = isCutIn ? '#ef4444' : (isContinuation ? '#f59e0b' : '#38bdf8');
                        ctx.lineWidth = isCutIn ? 2.5 : 1.8;
                        ctx.setLineDash(isCutIn ? [5, 3] : [3, 3]);
                        ctx.stroke();
                        ctx.setLineDash([]);

                        // Expanding uncertainty ellipse at horizon
                        const lastPt = tr.waypoints[tr.waypoints.length - 1];
                        if (lastPt) {
                            const lScr = worldToScreen(lastPt.x, lastPt.y);
                            const rx = (lastPt.sigma_y || 0.4) * scale;
                            const ry = (lastPt.sigma_x || 0.6) * scale;

                            ctx.fillStyle = isCutIn ? 'rgba(239, 68, 68, 0.22)' : 'rgba(245, 158, 11, 0.18)';
                            ctx.strokeStyle = isCutIn ? '#ef4444' : '#f59e0b';
                            ctx.lineWidth = 1.0;
                            ctx.beginPath();
                            ctx.ellipse(lScr.sx, lScr.sy, rx, ry, 0, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.stroke();

                            // Probability Tag Badge
                            ctx.fillStyle = isCutIn ? '#fecaca' : '#fef3c7';
                            ctx.font = 'bold 9px monospace';
                            ctx.fillText(`${tr.probability_pct}%`, lScr.sx + rx + 3, lScr.sy + 3);
                        }
                    }
                });
            });

            // 8. Perceived Dynamic Obstacles (BBoxes + Velocity Arrows)
            (data.actors || []).forEach(act => {
                const scr = worldToScreen(act.x_world, act.y_world);
                const wPix = act.width_m * scale;
                const lPix = act.length_m * scale;

                ctx.save();
                ctx.translate(scr.sx, scr.sy);
                ctx.rotate(-act.yaw_deg * Math.PI / 180 + egoHeading);

                const iconMap = {
                    'MOTORCYCLE': '🏍️', 'TRUCK': '🚜', 'AUTO_RICKSHAW': '🛺',
                    'PEDESTRIAN': '🚶', 'CATTLE_ANIMAL': '🐄', 'BUS': '🚌', 'CAR': '🚗'
                };
                const icon = iconMap[act.class] || '🚗';

                ctx.fillStyle = act.ttc_s && act.ttc_s < 3.0 ? 'rgba(239, 68, 68, 0.65)' : 'rgba(245, 158, 11, 0.65)';
                ctx.strokeStyle = act.ttc_s && act.ttc_s < 3.0 ? '#ef4444' : '#f59e0b';
                ctx.lineWidth = 2.0;
                ctx.fillRect(-wPix / 2, -lPix / 2, wPix, lPix);
                ctx.strokeRect(-wPix / 2, -lPix / 2, wPix, lPix);

                ctx.restore();

                // Velocity Vector Arrow
                if (act.speed_mps > 0.3) {
                    const endX_world = act.x_world + act.vx_ego * 1.5;
                    const endY_world = act.y_world + act.vy_ego * 1.5;
                    const endScr = worldToScreen(endX_world, endY_world);

                    ctx.strokeStyle = '#f59e0b';
                    ctx.lineWidth = 2.0;
                    ctx.beginPath();
                    ctx.moveTo(scr.sx, scr.sy);
                    ctx.lineTo(endScr.sx, endScr.sy);
                    ctx.stroke();
                }

                // Actor Label Badge
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 10px monospace';
                const ttcDesc = act.ttc_s ? ` | TTC ${act.ttc_s}s` : '';
                ctx.fillText(`${icon} ${act.id} (${act.speed_kph}kph${ttcDesc})`, scr.sx + wPix / 2 + 5, scr.sy + 3);
            });

            // 9. Ego Vehicle Representation (Cockpit Bottom Origin)
            ctx.fillStyle = '#06b6d4';
            ctx.strokeStyle = '#22d3ee';
            ctx.lineWidth = 2.5;
            ctx.shadowColor = '#06b6d4';
            ctx.shadowBlur = 15;

            const egoW = 1.8 * scale;
            const egoL = 4.2 * scale;
            ctx.fillRect(originX - egoW / 2, originY - egoL * 0.8, egoW, egoL);
            ctx.strokeRect(originX - egoW / 2, originY - egoL * 0.8, egoW, egoL);
            ctx.shadowBlur = 0;

            // Front Wheels & Steer Angle Representation
            const steerRad = -ego.steer_deg * Math.PI / 180;
            const fwY = originY - egoL * 0.7;
            [-egoW / 2 - 2, egoW / 2 + 2].forEach(fx => {
                ctx.save();
                ctx.translate(originX + fx, fwY);
                ctx.rotate(steerRad);
                ctx.fillStyle = '#f59e0b';
                ctx.fillRect(-2, -6, 4, 12);
                ctx.restore();
            });

            // Headlights Beam
            ctx.fillStyle = 'rgba(34, 211, 238, 0.12)';
            ctx.beginPath();
            ctx.moveTo(originX - 10, originY - egoL * 0.8);
            ctx.lineTo(originX - 55, originY - egoL * 0.8 - 90);
            ctx.lineTo(originX + 55, originY - egoL * 0.8 - 90);
            ctx.lineTo(originX + 10, originY - egoL * 0.8);
            ctx.closePath();
            ctx.fill();

            // Ego Vehicle Label
            ctx.fillStyle = '#e2e8f0';
            ctx.font = 'bold 11px monospace';
            ctx.fillText('🚗 EGO (SIH26037)', originX - 45, originY + 25);
        }

        // ----------------------------------------------------
        // VIEW 2: PANORAMIC ROAD OVERVIEW
        // ----------------------------------------------------
        function renderRoadOverview(data, w, h) {
            ctx.fillStyle = '#0a0f1d';
            ctx.fillRect(0, 0, w, h);

            const scaleX = w / 260;
            const scaleY = 32;
            const midY = h / 2;

            const poly = data.road.polyline || [];
            if (poly.length > 2) {
                ctx.fillStyle = '#161e2e';
                ctx.beginPath();
                poly.forEach((pt, i) => {
                    const sx = pt.s * scaleX;
                    const sy = midY - pt.dl * scaleY;
                    if (i === 0) ctx.moveTo(sx, sy);
                    else ctx.lineTo(sx, sy);
                });
                for (let i = poly.length - 1; i >= 0; i--) {
                    const sx = poly[i].s * scaleX;
                    const sy = midY - poly[i].dr * scaleY;
                    ctx.lineTo(sx, sy);
                }
                ctx.closePath();
                ctx.fill();
            }

            // Ego Vehicle Marker
            const egoS = data.road.s;
            const egoD = data.road.d;
            const ex = egoS * scaleX;
            const ey = midY - egoD * scaleY;

            ctx.fillStyle = '#06b6d4';
            ctx.beginPath();
            ctx.arc(ex, ey, 7, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 10px monospace';
            ctx.fillText('🚗 EGO', ex + 10, ey + 4);
        }

        // Control API Triggers
        async function toggleSim() {
            await fetch('/simulation/toggle', { method: 'POST' });
        }
        async function resetSim() {
            await fetch('/simulation/reset', { method: 'POST' });
        }
        async function triggerEStop() {
            await fetch('/simulation/emergency_stop', { method: 'POST' });
        }
        async function spawnHazard(type) {
            await fetch('/simulation/spawn_hazard', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hazard_type: type, distance_ahead_m: 28.0 })
            });
        }
        async function changeDifficulty(diff) {
            await fetch('/simulation/difficulty', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ difficulty: diff })
            });
        }
        async function changePerceptionMode(mode) {
            await fetch('/simulation/perception_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            });
        }
    </script>
</body>
</html>
"""

    return app


sim_engine_global = SimulationEngineState(difficulty=DifficultyLevel.HARD)
app = create_app(sim_engine_global)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
