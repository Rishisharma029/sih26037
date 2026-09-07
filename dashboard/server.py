"""
SIH26037 Autonomous Driving Live BEV Visualizer & Telemetry Server (Port 5002)
Features:
1. High-Density Autonomous Driving Debug BEV (Road Corridor, Edges, Dynamic Safety Envelope,
   Obstacles with Velocity Arrows, Multi-Modal Predictions, 7-Candidate Ego Bundle,
   Selected Safe Trajectory, Collision Zones, Goal Direction & 30m Lookahead Grid).
2. Real-time Causal Decision HUD showing Sense -> Predict -> TTC -> Candidate Bundle -> Stanley Actuation.
3. Dual-View Mode Switcher: Ego-Centric ADAS Cockpit (Forward Up) vs Panoramic World Overview.
4. Interactive Hazard Injection (Tractor, Pedestrian, Auto) & Difficulty selector.
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
    TrackedObstacle, BoundingBox3D, PlannedTrajectory
)
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from scenarios.difficulty import DifficultyLevel
from vehicle_control.lateral_controller import StanleyLateralController
from vehicle_control.longitudinal_controller import LongitudinalPIDController
from collision_avoidance.safety_supervisor import SafetySupervisoryLayer
from perception.boundary_detector import FreeSpaceBoundaryDetector
from prediction.trajectory_predictor import TrajectoryPredictor
from planning.baseline_planner import BaselinePlanner
from coordinates import transform_actor_to_ego_tracked_obstacle, world_to_ego_2d

class HazardSpawnRequest(BaseModel):
    hazard_type: str = "tractor" # tractor | pedestrian | auto
    distance_ahead_m: float = 35.0

class DifficultyRequest(BaseModel):
    difficulty: str = "HARD" # EASY | MEDIUM | HARD | EXTREME

class SimulationEngineState:
    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.HARD):
        self.difficulty = difficulty
        self.scenario = UnmarkedVillageRoadScenario(difficulty=self.difficulty)
        self.lat_ctrl = StanleyLateralController(k_gain=1.4)
        self.lon_ctrl = LongitudinalPIDController(kp=22.0, ki=0.5, kd=2.0)
        self.supervisory = SafetySupervisoryLayer(aeb_ttc_threshold_s=1.0, replan_ttc_threshold_s=2.0, slowdown_ttc_threshold_s=4.0)
        self.planner = BaselinePlanner(horizon_seconds=3.0, dt=0.2)
        self.predictor = TrajectoryPredictor(horizon_seconds=3.0, dt=0.5, mode="ensemble")
        self.boundary_detector = FreeSpaceBoundaryDetector(default_width_m=4.3)
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

    def reset(self):
        self.scenario = UnmarkedVillageRoadScenario(difficulty=self.difficulty)
        self.is_emergency_stop = False
        self.is_completed = False
        self.step_count = 0
        self.min_corridor_margin = 2.0
        self.step()

    def spawn_hazard(self, hazard_type: str, dist_ahead: float = 35.0):
        if hazard_type == "tractor":
            self.scenario.spawn_oncoming_tractor(dist_ahead=dist_ahead, y=0.7, speed_mps=3.5)
        elif hazard_type == "pedestrian":
            self.scenario.spawn_crossing_pedestrian(dist_ahead=max(15.0, dist_ahead * 0.6), start_y=-2.0, speed_mps=1.4)
        elif hazard_type == "auto":
            self.scenario.spawn_parked_auto(dist_ahead=max(15.0, dist_ahead * 0.5), y=0.9)

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

        # 1. Standardized Perception: transform all actors to ego body frame (+X=forward, +Y=left)
        actors_data = []
        obstacles = []

        for a in self.scenario.env.actors:
            obs = transform_actor_to_ego_tracked_obstacle(
                actor_id=a.id,
                obstacle_class=a.obstacle_class,
                x_world=a.x,
                y_world=a.y,
                z_world=a.z,
                length_m=a.length_m,
                width_m=a.width_m,
                height_m=a.height_m,
                yaw_world_rad=a.yaw_rad,
                speed_mps=a.speed_mps,
                is_static=a.is_static,
                ego_pose=ego_state.pose,
                ego_twist=ego_state.twist,
                confidence=0.95
            )
            obstacles.append(obs)

            # Compute dynamic Time-To-Collision (TTC) in ego body frame
            ttc_eval = self.supervisory.ttc_calc.compute_ttc(ego_state, [obs])
            ttc_val = ttc_eval.min_ttc_seconds if ttc_eval.min_ttc_seconds < 100.0 else None

            # World and ego velocity vectors
            vx_world = a.speed_mps * math.cos(a.yaw_rad)
            vy_world = a.speed_mps * math.sin(a.yaw_rad)

            actors_data.append({
                "id": a.id,
                "class": a.obstacle_class.value,
                "x_world": round(a.x, 2),
                "y_world": round(a.y, 2),
                "z_world": round(a.z, 2),
                "length_m": round(a.length_m, 2),
                "width_m": round(a.width_m, 2),
                "height_m": round(a.height_m, 2),
                "yaw_world_deg": round(math.degrees(a.yaw_rad), 1),
                "x_ego": round(obs.bbox.center.x, 2), # Forward distance ahead (+X)
                "y_ego": round(obs.bbox.center.y, 2), # Lateral distance left (+Y) / right (-Y)
                "distance_m": round(obs.distance_m, 2), # Euclidean distance (always >= 0)
                "ttc_s": round(ttc_val, 2) if ttc_val is not None else None,
                "speed_mps": round(a.speed_mps, 2),
                "speed_kph": round(a.speed_mps * 3.6, 1),
                "vx_world": round(vx_world, 2),
                "vy_world": round(vy_world, 2),
                "vx_ego": round(obs.velocity.x, 2),
                "vy_ego": round(obs.velocity.y, 2),
                "is_static": a.is_static,
                "safety_radius_m": round(max(a.length_m, a.width_m) * 0.6 + 0.5, 2)
            })

        # 2. Road Corridor Detection
        corridor = self.boundary_detector.detect_corridor(
            timestamp=ego_state.timestamp,
            lookahead_m=45.0,
            step_m=3.0,
            current_s=s_curr,
            geometry=self.scenario.env.geometry
        )
        perception_frame = PerceptionOutput(
            timestamp=ego_state.timestamp,
            frame_id=self.step_count,
            obstacles=obstacles,
            drivable_corridor=corridor
        )

        # 3. Multi-Modal Motion Prediction
        pred_out = self.predictor.predict(perception_frame, ego_speed=ego_state.twist.speed_mps)

        # 4. Adaptive Path Planning (Discrete Lateral Offsets: -1.5m to +1.5m)
        planned_traj = self.planner.plan(
            ego_state=ego_state,
            perception=perception_frame,
            prediction=pred_out,
            target_cruise_speed_mps=self.target_speed_mps
        )

        # 5. Independent Collision Avoidance & Safety Layer Supervision
        safe_traj = self.supervisory.supervise(
            planned=planned_traj,
            ego_state=ego_state,
            perception=perception_frame,
            prediction=pred_out
        )

        if safe_traj.is_emergency_stop:
            self.is_emergency_stop = True

        # 6. Drive-By-Wire Vehicle Control Computation
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
            gear=GearMode.DRIVE,
            emergency_brake_active=self.is_emergency_stop or safe_traj.is_emergency_stop
        )

        # 7. Physics & Vehicle Dynamics Step
        state, raw_sensor = self.scenario.run_step(cmd)
        self.step_count += 1

        # 8. Precise Frenet Boundary & Ditch Margin Calculations
        s_post, d_post = self.scenario.env.geometry.cartesian_to_frenet(
            state.pose.position.x,
            state.pose.position.y,
            s_guess=s_curr
        )
        d_left, d_right = self.scenario.env.geometry.get_corridor_widths(s_post)
        margin = self.scenario.env.geometry.get_ditch_margin(s_post, d_post, vehicle_half_width=0.90)
        self.min_corridor_margin = min(self.min_corridor_margin, margin)

        if margin < 0.0:
            self.is_emergency_stop = True

        # 9. Format Multi-Modal Predictions (Forecasted paths with timestamps)
        predictions_data = []
        for agent in pred_out.agents:
            trajs_data = []
            for t in agent.trajectories:
                trajs_data.append({
                    "mode_name": t.mode_name,
                    "probability": round(t.probability, 2),
                    "collision_risk": round(t.collision_risk, 2),
                    "waypoints": [
                        {
                            "x": round(pt.position.x, 2),
                            "y": round(pt.position.y, 2),
                            "sigma_x": round(pt.sigma_x, 2),
                            "sigma_y": round(pt.sigma_y, 2),
                            "time_offset_s": round(pt.timestamp - ego_state.timestamp, 2)
                        }
                        for pt in t.waypoints
                    ]
                })
            predictions_data.append({
                "id": agent.id,
                "class": agent.obstacle_class.value,
                "intent": agent.primary_intent.value,
                "is_high_risk": agent.is_high_risk,
                "trajectories": trajs_data
            })

        # 10. Extract Collision Zones from Candidates
        collision_zones = []
        for cand in self.planner.last_candidates_summary:
            if not cand.get("is_feasible") and cand.get("collision_point"):
                cp = cand["collision_point"]
                collision_zones.append({
                    "x": cp["x"],
                    "y": cp["y"],
                    "radius_m": 1.6,
                    "obstacle_id": cand.get("collision_obstacle_id", "OBSTACLE"),
                    "offset_m": cand.get("offset", 0.0)
                })

        # 11. 30m Lookahead Goal Direction & Road Polyline
        s_goal = min(self.scenario.env.geometry.length_m - 2.0, s_curr + 30.0)
        gx, gy, goal_yaw = self.scenario.env.geometry.frenet_to_cartesian(s_goal, 0.0)

        road_polyline = []
        for s_idx in range(max(0, int(s_curr - 20)), min(int(self.scenario.env.geometry.length_m), int(s_curr + 65)), 2):
            cx, cy, _ = self.scenario.env.geometry.frenet_to_cartesian(float(s_idx), 0.0)
            dl, dr = self.scenario.env.geometry.get_corridor_widths(float(s_idx))
            lx, ly, _ = self.scenario.env.geometry.frenet_to_cartesian(float(s_idx), dl)
            rx, ry, _ = self.scenario.env.geometry.frenet_to_cartesian(float(s_idx), -dr)
            road_polyline.append({
                "s": float(s_idx),
                "cx": round(cx, 2), "cy": round(cy, 2),
                "lx": round(lx, 2), "ly": round(ly, 2),
                "rx": round(rx, 2), "ry": round(ry, 2)
            })

        # 12. Build Live Causal Decision Chain Data
        lead_threat = None
        for a in actors_data:
            if a["x_ego"] > 0 and (a["ttc_s"] is not None or a["distance_m"] < 35.0):
                if lead_threat is None or (a["ttc_s"] or 999) < (lead_threat["ttc_s"] or 999):
                    lead_threat = a

        has_threat = lead_threat is not None and ((lead_threat["ttc_s"] is not None and lead_threat["ttc_s"] < 5.0) or lead_threat["distance_m"] < 25.0)
        chosen_cand = next((c for c in self.planner.last_candidates_summary if c.get("is_selected")), None)
        center_cand = next((c for c in self.planner.last_candidates_summary if c.get("offset") == 0.0), None)

        causal_event = {
            "has_hazard": has_threat,
            "hazard_id": lead_threat["id"] if lead_threat else "None",
            "hazard_class": lead_threat["class"] if lead_threat else "None",
            "hazard_dist_m": lead_threat["distance_m"] if lead_threat else 999.0,
            "hazard_ttc_s": lead_threat["ttc_s"] if lead_threat else None,
            "nominal_path_safe": center_cand.get("is_feasible", True) if center_cand else True,
            "evaluated_candidates": len(self.planner.last_candidates_summary),
            "selected_offset_m": chosen_cand.get("offset", 0.0) if chosen_cand else 0.0,
            "selected_mode": planned_traj.behavior_mode.value,
            "safety_action": safe_traj.safety_action.value,
            "steer_command_deg": round(math.degrees(cmd.steering_angle_rad), 2),
            "action_summary": (
                f"🚨 {lead_threat['id'].upper()} IN PATH (@ {lead_threat['distance_m']}m, TTC {lead_threat['ttc_s']}s) → "
                f"Nominal: UNSAFE → Selected: {planned_traj.behavior_mode.value} ({chosen_cand['offset']:+.1f}m) → "
                f"Steering: {math.degrees(cmd.steering_angle_rad):+.1f}°"
                if has_threat and chosen_cand
                else "✅ PATH CLEAR — Cruising Nominal Centerline (Offset 0.0m)"
            )
        }

        self.latest_telemetry = {
            "timestamp": round(state.timestamp, 2),
            "step": self.step_count,
            "vehicle_id": "SIH26037-AV-01",
            "difficulty": self.difficulty.value,
            "ego": {
                "x": round(state.pose.position.x, 2),
                "y": round(state.pose.position.y, 2),
                "heading_deg": round(math.degrees(state.pose.heading_rad), 2),
                "speed_mps": round(state.twist.speed_mps, 2),
                "speed_kph": round(state.twist.speed_mps * 3.6, 1),
                "steer_deg": round(math.degrees(state.steer_angle_rad), 1),
                "throttle_pct": round(cmd.throttle_pct, 1),
                "brake_pct": round(cmd.brake_pct, 1),
                "battery_soc": round(state.battery_soc_pct, 1),
                "length_m": 4.5,
                "width_m": 1.8,
                "safety_buffer_lat_m": 0.6,
                "safety_buffer_lon_m": 1.2
            },
            "goal": {
                "x": round(gx, 2),
                "y": round(gy, 2),
                "s": round(s_goal, 1),
                "heading_deg": round(math.degrees(goal_yaw), 1),
                "distance_ahead_m": round(s_goal - s_curr, 1)
            },
            "road": {
                "s_curr": round(s_curr, 2),
                "corridor_left_m": round(d_left, 2),
                "corridor_right_m": round(d_right, 2),
                "current_margin_m": round(margin, 2),
                "min_margin_m": round(self.min_corridor_margin, 2),
                "polyline": road_polyline
            },
            "safety": {
                "safety_action": safe_traj.safety_action.value,
                "safety_status_reason": safe_traj.safety_status_reason,
                "min_ttc_s": round(safe_traj.min_ttc_seconds, 2) if safe_traj.min_ttc_seconds < 100.0 else None,
                "barrier_margin_m": round(safe_traj.barrier_margin_m, 2),
                "is_emergency_stop": safe_traj.is_emergency_stop,
                "replan_recommended": safe_traj.replan_recommended
            },
            "planning": {
                "behavior_mode": planned_traj.behavior_mode.value,
                "target_speed_kph": round(planned_traj.target_speed_mps * 3.6, 1),
                "trajectory_id": planned_traj.trajectory_id
            },
            "candidates": self.planner.last_candidates_summary,
            "collision_zones": collision_zones,
            "causal_event": causal_event,
            "status": {
                "is_running": self.is_running,
                "is_e_stop": self.is_emergency_stop,
                "target_speed_kph": round(self.target_speed_mps * 3.6, 1)
            },
            "actors": actors_data,
            "predictions": predictions_data,
            "high_risk_agent_ids": pred_out.high_risk_agent_ids
        }

sim_engine = SimulationEngineState()

def create_app() -> FastAPI:
    app = FastAPI(title="SIH26037 Autonomous Mobility Stack", version="0.1.0")

    @app.on_event("startup")
    async def start_sim_loop():
        async def background_loop():
            while True:
                sim_engine.step()
                await asyncio.sleep(0.05) # 20 Hz
        asyncio.create_task(background_loop())

    @app.get("/health")
    async def health():
        return {
            "status": "UP",
            "service": "SIH26037 Autonomous Mobility Service",
            "vehicle_id": "SIH26037-AV-01",
            "scene": "Unmarked Indian Village Road (Phase 1)",
            "difficulty": sim_engine.difficulty.value,
            "e_stop": sim_engine.is_emergency_stop,
            "is_running": sim_engine.is_running
        }

    @app.get("/telemetry")
    async def get_telemetry():
        return sim_engine.latest_telemetry

    @app.post("/emergency_stop")
    async def trigger_emergency_stop():
        sim_engine.is_emergency_stop = True
        return {"status": "EMERGENCY_STOP_TRIGGERED", "e_stop": True}

    @app.post("/simulation/toggle")
    async def toggle_simulation():
        sim_engine.is_running = not sim_engine.is_running
        return {"is_running": sim_engine.is_running}

    @app.post("/simulation/reset")
    async def reset_simulation():
        sim_engine.reset()
        return {"status": "RESET_SUCCESSFUL"}

    @app.post("/simulation/spawn_hazard")
    async def spawn_hazard_api(req: HazardSpawnRequest):
        sim_engine.spawn_hazard(req.hazard_type, req.distance_ahead_m)
        return {"status": "HAZARD_SPAWNED", "type": req.hazard_type}

    @app.post("/simulation/difficulty")
    async def set_difficulty_api(req: DifficultyRequest):
        sim_engine.set_difficulty(req.difficulty)
        return {"status": "DIFFICULTY_SET", "difficulty": sim_engine.difficulty.value}

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
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
    <title>SIH26037 — Autonomous Driving Debug BEV & Causal Chain Visualizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #070b13; color: #e2e8f0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        canvas { background-color: #0a0f1d; border-radius: 0.75rem; border: 1px solid #1e293b; box-shadow: inset 0 0 30px rgba(0,0,0,0.7); }
        .glass-card { background: rgba(13, 20, 36, 0.88); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.75rem; }
        .chain-step { transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }
        .badge-glow { box-shadow: 0 0 12px rgba(6, 182, 212, 0.35); }
    </style>
</head>
<body class="p-3 md:p-6 max-w-[1550px] mx-auto space-y-4">

    <!-- Header & Interactive Controls -->
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-4 md:p-5">
        <div>
            <div class="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 mb-1">
                <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                SIH26037 • AUTONOMOUS DRIVING DEBUG VIEW (BEV)
            </div>
            <h1 class="text-xl md:text-2xl font-black tracking-tight text-white flex items-center gap-3">
                Adaptive Path Planning & Collision Avoidance
                <span id="badge-difficulty" class="text-xs px-2.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono font-bold">HARD TIER</span>
            </h1>
            <p class="text-xs text-slate-400 mt-0.5">
                Full Debug Pipeline: Road Corridor • Boundary Margins • Obstacle Bounding Boxes & Velocity Vectors • Multi-Modal Predictions • 7 Candidate Trajectories • Selected Safe Path • Safety Envelope • Collision Zones • Goal Vector.
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
                Real-Time Sense-Predict-Plan Causal Arbiter
            </span>
            <span id="causal-summary-badge" class="text-xs font-mono font-bold px-3 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/40">
                CRUISING NOMINAL PATH
            </span>
        </div>

        <!-- 5-Stage Causal Flow Stepper -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs font-mono">
            <div id="step-1" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">1. Hazard Incursion</span>
                <span id="hud-hazard" class="font-bold text-slate-300 block truncate">None Detected</span>
                <span id="hud-hazard-dist" class="text-[10px] text-slate-500">Range: --</span>
            </div>
            <div id="step-2" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">2. Intent & TTC</span>
                <span id="hud-ttc" class="font-bold text-slate-300 block">TTC > 4.0s (Safe)</span>
                <span id="hud-risk" class="text-[10px] text-slate-500">Nominal: VALID</span>
            </div>
            <div id="step-3" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">3. Candidate Bundle</span>
                <span id="hud-candidates" class="font-bold text-slate-300 block">7 Evaluated</span>
                <span id="hud-cand-status" class="text-[10px] text-emerald-400">Center (0.0m) Clear</span>
            </div>
            <div id="step-4" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">4. Safe Selection</span>
                <span id="hud-mode" class="font-bold text-cyan-400 block">CRUISE</span>
                <span id="hud-offset" class="text-[10px] text-slate-400">Offset: 0.0m</span>
            </div>
            <div id="step-5" class="chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">5. DBW Actuation</span>
                <span id="hud-steer-act" class="font-bold text-amber-400 block">Steer: 0.0°</span>
                <span id="hud-speed-act" class="text-[10px] text-slate-400">Speed: 21.6 km/h</span>
            </div>
        </div>
    </div>

    <!-- Main BEV Canvas Visualizer & Threat Feed -->
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
                </div>

                <!-- Interactive Hazard Injection Bar -->
                <div class="flex items-center gap-1.5">
                    <span class="text-[11px] text-slate-400 font-bold">Inject Hazard:</span>
                    <button onclick="spawnHazard('tractor')" class="px-2.5 py-1 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 text-[11px] font-bold transition">
                        🚜 Oncoming Tractor
                    </button>
                    <button onclick="spawnHazard('pedestrian')" class="px-2.5 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 text-[11px] font-bold transition">
                        🚶 Crossing Villager
                    </button>
                    <button onclick="spawnHazard('auto')" class="px-2.5 py-1 rounded bg-orange-500/20 hover:bg-orange-500/30 text-orange-300 border border-orange-500/40 text-[11px] font-bold transition">
                        🛺 Parked Auto
                    </button>
                </div>
            </div>

            <!-- Canvas Element -->
            <div class="relative w-full overflow-hidden rounded-xl">
                <canvas id="simCanvas" width="1020" height="560" class="w-full h-auto block"></canvas>
            </div>

            <!-- Visual Legend & Explanation Bar -->
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2 text-[10px] text-slate-300 pt-2 border-t border-slate-800/80">
                <div class="flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-emerald-400 shadow-sm shadow-emerald-400"></span> Selected Trajectory</div>
                <div class="flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-cyan-400/40 border border-cyan-400"></span> Candidate Splines (7)</div>
                <div class="flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-rose-500"></span> Blocked Trajectory</div>
                <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-rose-500/40 border border-rose-500"></span> Collision Zones</div>
                <div class="flex items-center gap-1.5"><span class="w-3 h-1 rounded bg-cyan-400/30 border border-cyan-400"></span> Safety Envelope</div>
                <div class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-amber-400"></span> Velocity Vector (→)</div>
                <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full border border-yellow-400 text-yellow-400 font-mono text-[9px] flex items-center justify-center">🎯</span> Goal Horizon</div>
            </div>
        </div>

        <!-- Right Side: Threat Radar & Live KPIs -->
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
                    <span class="text-[10px] text-slate-400 font-bold uppercase block">Ditch Margin</span>
                    <span id="kpi-margin" class="text-xl font-black text-purple-400 font-mono">2.15 m</span>
                    <span class="text-[9px] text-slate-500 font-semibold">Hard Invariant</span>
                </div>
                <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <span class="text-[10px] text-slate-400 font-bold uppercase block">Safety Tier</span>
                    <span id="kpi-safety" class="text-sm font-black text-emerald-400 font-mono block mt-1">NONE</span>
                    <span class="text-[9px] text-slate-500 font-semibold">Arbiter State</span>
                </div>
            </div>

            <!-- Threat Radar Feed -->
            <div class="space-y-2">
                <div class="flex items-center justify-between border-t border-slate-800 pt-2">
                    <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300">
                        Perception & Velocity Tracking
                    </h3>
                    <span class="text-[10px] text-cyan-400 font-mono">20 Hz Track</span>
                </div>
                
                <div id="actors-list" class="space-y-2 text-xs max-h-48 overflow-y-auto pr-1">
                    <!-- Dynamically populated -->
                </div>
            </div>

            <!-- Lattice Candidates Breakdown -->
            <div class="space-y-1.5">
                <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300 pt-2 border-t border-slate-800">
                    7 Candidate Splines Evaluated
                </h3>
                <div id="candidates-list" class="space-y-1 text-[11px] font-mono max-h-40 overflow-y-auto pr-1">
                    <!-- Populated with all 7 offset evaluations -->
                </div>
            </div>
        </div>
    </div>

    <!-- Script: WebSocket & High-Density Canvas Debug Renderer -->
    <script>
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        let currentData = null;
        let viewMode = 'ego'; // 'ego' (30m ahead top-down) or 'road' (world horizontal track)

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
            document.getElementById('kpi-safety').innerText = data.safety.safety_action;

            if (data.difficulty) {
                document.getElementById('badge-difficulty').innerText = `${data.difficulty} TIER`;
                document.getElementById('diff-select').value = data.difficulty;
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
                document.getElementById('hud-ttc').innerHTML = `<span class="text-rose-400 font-bold">TTC: ${ce.hazard_ttc_s || '<1.0'}s</span>`;
                document.getElementById('hud-risk').innerHTML = ce.nominal_path_safe ? '<span class="text-slate-400">Nominal: CLEAR</span>' : '<span class="text-rose-400 font-bold">Nominal: UNSAFE (Blocked)</span>';

                s3.className = 'chain-step p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/60';
                document.getElementById('hud-candidates').innerText = `7 Candidates Evaluated`;
                document.getElementById('hud-cand-status').innerHTML = `<span class="text-cyan-300">Offset ${ce.selected_offset_m > 0 ? '+' : ''}${ce.selected_offset_m}m Safe</span>`;

                s4.className = 'chain-step p-2 rounded-lg bg-emerald-950/40 border border-emerald-500/60';
                document.getElementById('hud-mode').innerText = ce.selected_mode;
                document.getElementById('hud-offset').innerText = `Chosen: ${ce.selected_offset_m > 0 ? '+' : ''}${ce.selected_offset_m}m`;

                s5.className = 'chain-step p-2 rounded-lg bg-indigo-950/40 border border-indigo-500/60';
                document.getElementById('hud-steer-act').innerHTML = `<span class="text-amber-300 font-bold">Steer: ${ce.steer_command_deg}°</span>`;
                document.getElementById('hud-speed-act').innerText = `Speed: ${data.ego.speed_kph} km/h`;

                document.getElementById('causal-summary-badge').className = 'text-xs font-mono font-bold px-3 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-500/60 animate-pulse';
                document.getElementById('causal-summary-badge').innerText = `AVOIDING: ${ce.hazard_id.toUpperCase()} (${ce.selected_mode})`;
            } else {
                [s1, s2, s3, s4, s5].forEach(el => el.className = 'chain-step p-2 rounded-lg bg-slate-900/90 border border-slate-800');
                document.getElementById('hud-hazard').innerText = 'None Detected';
                document.getElementById('hud-hazard-dist').innerText = 'Range: Clear';
                document.getElementById('hud-ttc').innerText = 'TTC > 4.0s (Safe)';
                document.getElementById('hud-risk').innerText = 'Nominal: VALID';
                document.getElementById('hud-candidates').innerText = '7 Evaluated';
                document.getElementById('hud-cand-status').innerText = 'Center (0.0m) Clear';
                document.getElementById('hud-mode').innerText = 'CRUISE';
                document.getElementById('hud-offset').innerText = 'Offset: 0.0m';
                document.getElementById('hud-steer-act').innerText = `Steer: ${data.ego.steer_deg}°`;
                document.getElementById('hud-speed-act').innerText = `Speed: ${data.ego.speed_kph} km/h`;

                document.getElementById('causal-summary-badge').className = 'text-xs font-mono font-bold px-3 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/40';
                document.getElementById('causal-summary-badge').innerText = 'CRUISING NOMINAL PATH';
            }

            // Update Actors List with Velocity Vectors
            const listEl = document.getElementById('actors-list');
            listEl.innerHTML = '';
            (data.actors || []).forEach(a => {
                const item = document.createElement('div');
                item.className = 'p-2 rounded-lg bg-slate-900 border border-slate-800 space-y-1';
                const aheadText = a.x_ego >= 0 ? `+${a.x_ego}m ahead` : `${a.x_ego}m behind`;
                const latText = a.y_ego >= 0 ? `+${a.y_ego}m L` : `${a.y_ego}m R`;
                const ttcBadge = a.ttc_s ? `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${a.ttc_s < 2.0 ? 'bg-rose-900/80 text-rose-300 animate-pulse' : 'bg-amber-900/40 text-amber-300'}">TTC: ${a.ttc_s}s</span>` : '';
                item.innerHTML = `
                    <div class="flex items-center justify-between font-semibold">
                        <span class="text-slate-200 font-mono">${a.id}</span>
                        <div class="flex items-center gap-1">
                            ${ttcBadge}
                            <span class="text-[10px] px-1.5 py-0.2 rounded ${a.is_static ? 'bg-slate-800 text-slate-400' : 'bg-cyan-900/50 text-cyan-300'}">${a.class}</span>
                        </div>
                    </div>
                    <div class="flex justify-between text-[10px] text-slate-400 font-mono">
                        <span>Range: <b class="text-slate-200">${a.distance_m}m</b> (${aheadText}, ${latText})</span>
                        <span>Vel: <b class="text-amber-300">${a.speed_mps}m/s (${a.speed_kph}kph)</b></span>
                    </div>
                `;
                listEl.appendChild(item);
            });

            // Update Candidates Breakdown
            const candList = document.getElementById('candidates-list');
            candList.innerHTML = '';
            (data.candidates || []).forEach(c => {
                const row = document.createElement('div');
                row.className = `flex justify-between px-2 py-1 rounded text-[10px] ${
                    c.is_selected 
                        ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-500/50 font-bold'
                        : c.is_feasible
                            ? 'bg-slate-900/60 text-slate-400'
                            : 'bg-rose-950/30 text-rose-400/80 line-through'
                }`;
                const offsetLabel = `${c.offset > 0 ? '+' : ''}${c.offset.toFixed(1)}m`;
                const statusTag = c.is_selected ? '★ SELECTED' : c.is_feasible ? 'FEASIBLE' : c.rejection_reason;
                row.innerHTML = `<span>Offset ${offsetLabel}</span><span>${statusTag}</span>`;
                candList.appendChild(row);
            });
        }

        // ==========================================
        // HIGH-DENSITY AUTONOMOUS DRIVING BEV RENDERER
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

            // Ego origin on canvas (centered horizontally, lower-middle vertically)
            const originX = w / 2;
            const originY = h - 90;
            const scale = 11.5; // pixels per meter (35m forward lookahead)

            // Helper to transform world coordinates (wx, wy) into Canvas Screen Coordinates
            function worldToScreen(wx, wy) {
                const dx = wx - egoX;
                const dy = wy - egoY;
                // Rotate into ego body frame (+X forward, +Y left)
                const xEgo = dx * Math.cos(egoHeading) + dy * Math.sin(egoHeading);
                const yEgo = -dx * Math.sin(egoHeading) + dy * Math.cos(egoHeading);
                // Canvas mapping: forward (+xEgo) is UP (-Y), left (+yEgo) is LEFT (-X)
                const sx = originX - yEgo * scale;
                const sy = originY - xEgo * scale;
                return { sx, sy, xEgo, yEgo };
            }

            // 1. Radar Grid & Dark Background
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

                // Range Badge
                ctx.fillStyle = r === 30 ? '#fde047' : '#06b6d4';
                ctx.font = 'bold 10px monospace';
                ctx.fillText(`── ${r} m ──`, originX - 24, originY - arcR - 3);
            });

            // Lateral offset grid lines (-3m, -2m, -1m, 0, +1m, +2m, +3m)
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

            // 2. Road Corridor (Drivable polygon & Road Edges with Ditch Berms)
            const poly = data.road.polyline || [];
            if (poly.length > 2) {
                // Shaded asphalt surface
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

                // Road Centerline
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
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

                // Left Ditch Boundary Edge with Hazard Striping
                ctx.strokeStyle = '#eab308';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                poly.forEach((pt, i) => {
                    const scr = worldToScreen(pt.lx, pt.ly);
                    if (i === 0) ctx.moveTo(scr.sx, scr.sy);
                    else ctx.lineTo(scr.sx, scr.sy);
                });
                ctx.stroke();

                // Right Ditch Boundary Edge
                ctx.strokeStyle = '#eab308';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                poly.forEach((pt, i) => {
                    const scr = worldToScreen(pt.rx, pt.ry);
                    if (i === 0) ctx.moveTo(scr.sx, scr.sy);
                    else ctx.lineTo(scr.sx, scr.sy);
                });
                ctx.stroke();
            }

            // 3. Goal Direction Indicator (30m ahead lookahead horizon)
            if (data.goal) {
                const gScr = worldToScreen(data.goal.x, data.goal.y);
                // Target line from ego to goal
                ctx.strokeStyle = 'rgba(234, 179, 8, 0.35)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(originX, originY);
                ctx.lineTo(gScr.sx, gScr.sy);
                ctx.stroke();
                ctx.setLineDash([]);

                // Goal Target Crosshair
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

            // 4. Candidate Ego Trajectories (7-Spline Bundle)
            (data.candidates || []).forEach(cand => {
                if (!cand.waypoints || cand.waypoints.length === 0) return;
                ctx.beginPath();
                ctx.moveTo(originX, originY);

                cand.waypoints.forEach(wp => {
                    const scr = worldToScreen(wp.x, wp.y);
                    ctx.lineTo(scr.sx, scr.sy);
                });

                if (cand.is_selected) {
                    ctx.strokeStyle = '#10b981'; // Vibrant glowing green
                    ctx.lineWidth = 4.0;
                    ctx.shadowColor = '#10b981';
                    ctx.shadowBlur = 10;
                    ctx.setLineDash([]);
                    ctx.stroke();
                    ctx.shadowBlur = 0;

                    // Waypoint Speed Profile Markers
                    cand.waypoints.forEach((wp, idx) => {
                        if (idx % 3 === 0) {
                            const scr = worldToScreen(wp.x, wp.y);
                            ctx.fillStyle = '#6ee7b7';
                            ctx.beginPath();
                            ctx.arc(scr.sx, scr.sy, 3.5, 0, Math.PI * 2);
                            ctx.fill();
                        }
                    });
                } else if (cand.is_feasible) {
                    ctx.strokeStyle = 'rgba(6, 182, 212, 0.40)'; // Cyan for clear alternatives
                    ctx.lineWidth = 1.8;
                    ctx.setLineDash([4, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                } else {
                    ctx.strokeStyle = 'rgba(239, 68, 68, 0.55)'; // Dashed red for collision
                    ctx.lineWidth = 1.8;
                    ctx.setLineDash([3, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            });

            // 5. Collision Zones (Interference Overlaps)
            (data.collision_zones || []).forEach(cz => {
                const scr = worldToScreen(cz.x, cz.y);
                const rPix = cz.radius_m * scale;

                // Pulsating Red Hazard Disk
                ctx.fillStyle = 'rgba(239, 68, 68, 0.28)';
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                ctx.arc(scr.sx, scr.sy, rPix, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                // ✖ Collision Marker
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.moveTo(scr.sx - 6, scr.sy - 6); ctx.lineTo(scr.sx + 6, scr.sy + 6);
                ctx.moveTo(scr.sx + 6, scr.sy - 6); ctx.lineTo(scr.sx - 6, scr.sy + 6);
                ctx.stroke();

                ctx.fillStyle = '#fca5a5';
                ctx.font = 'bold 9px monospace';
                ctx.fillText(`💥 COLLISION (${cz.obstacle_id})`, scr.sx + rPix + 3, scr.sy + 3);
            });

            // 6. Multi-Modal Motion Predictions of Obstacles
            (data.predictions || []).forEach(pred => {
                (pred.trajectories || []).forEach(tr => {
                    if (tr.waypoints && tr.waypoints.length > 0) {
                        ctx.beginPath();
                        const s0 = worldToScreen(tr.waypoints[0].x, tr.waypoints[0].y);
                        ctx.moveTo(s0.sx, s0.sy);
                        tr.waypoints.forEach(wp => {
                            const scr = worldToScreen(wp.x, wp.y);
                            ctx.lineTo(scr.sx, scr.sy);
                        });

                        ctx.strokeStyle = tr.collision_risk > 0.4 ? 'rgba(244, 63, 94, 0.85)' : 'rgba(245, 158, 11, 0.55)';
                        ctx.lineWidth = tr.collision_risk > 0.4 ? 2.5 : 1.5;
                        ctx.setLineDash([3, 3]);
                        ctx.stroke();
                        ctx.setLineDash([]);

                        // Time Horizon Waypoint Dots (+1.0s, +2.0s, +3.0s)
                        tr.waypoints.forEach(wp => {
                            if (Math.abs(wp.time_offset_s - 1.0) < 0.15 || Math.abs(wp.time_offset_s - 2.0) < 0.15 || Math.abs(wp.time_offset_s - 3.0) < 0.15) {
                                const scr = worldToScreen(wp.x, wp.y);
                                ctx.fillStyle = tr.collision_risk > 0.4 ? '#f43f5e' : '#fbbf24';
                                ctx.beginPath();
                                ctx.arc(scr.sx, scr.sy, 3, 0, Math.PI * 2);
                                ctx.fill();

                                ctx.font = '8px monospace';
                                ctx.fillText(`+${wp.time_offset_s}s`, scr.sx + 5, scr.sy + 3);
                            }
                        });
                    }
                });
            });

            // 7. Obstacles & Velocity Vectors
            (data.actors || []).forEach(a => {
                const scr = worldToScreen(a.x_world, a.y_world);
                const relHeading = (a.yaw_world_deg - ego.heading_deg) * Math.PI / 180;

                ctx.save();
                ctx.translate(scr.sx, scr.sy);
                ctx.rotate(-relHeading); // Rotate to obstacle relative heading in body frame

                const lengthPix = a.length_m * scale;
                const widthPix = a.width_m * scale;

                if (a.class === 'TRUCK' || a.id.includes('tractor')) {
                    // Agricultural Tractor (Amber/Yellow with big wheels)
                    ctx.fillStyle = '#f59e0b';
                    ctx.fillRect(-widthPix/2, -lengthPix/2, widthPix, lengthPix);
                    ctx.fillStyle = '#0f172a';
                    // Rear Big Wheels
                    ctx.fillRect(-widthPix/2 - 4, -lengthPix/2 + 2, 4, lengthPix * 0.4);
                    ctx.fillRect(widthPix/2, -lengthPix/2 + 2, 4, lengthPix * 0.4);
                    // Front Wheels
                    ctx.fillRect(-widthPix/2 - 2, lengthPix/2 - lengthPix * 0.35, 3, lengthPix * 0.3);
                    ctx.fillRect(widthPix/2 - 1, lengthPix/2 - lengthPix * 0.35, 3, lengthPix * 0.3);
                    // Cab Roof
                    ctx.fillStyle = '#78350f';
                    ctx.fillRect(-widthPix/3, -lengthPix/4, widthPix * 0.66, lengthPix * 0.4);

                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 9px sans-serif';
                    ctx.fillText('🚜 TRACTOR', -widthPix/2 + 2, 2);
                } else if (a.class === 'PEDESTRIAN' || a.id.includes('pedestrian') || a.id.includes('villager')) {
                    // Pedestrian Circle with Step Direction
                    ctx.fillStyle = '#f43f5e';
                    ctx.beginPath();
                    ctx.arc(0, 0, 7, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 8px sans-serif';
                    ctx.fillText('🚶 PED', -10, -10);
                } else if (a.class === 'AUTO_RICKSHAW' || a.id.includes('auto')) {
                    // Auto-Rickshaw (3-Wheeler)
                    ctx.fillStyle = '#ea580c';
                    ctx.fillRect(-widthPix/2, -lengthPix/2, widthPix, lengthPix);
                    ctx.fillStyle = '#fde047';
                    ctx.fillRect(-widthPix/2, -lengthPix/2, widthPix, lengthPix * 0.3); // Yellow canopy front
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 8px sans-serif';
                    ctx.fillText('🛺 AUTO', -widthPix/2 + 3, 2);
                } else {
                    // Boulder / Debris
                    ctx.fillStyle = '#64748b';
                    ctx.beginPath();
                    ctx.arc(0, 0, Math.max(6, widthPix/2), 0, Math.PI * 2);
                    ctx.fill();
                }

                ctx.restore();

                // Velocity Vector Arrow (proportional to speed)
                if (a.speed_mps > 0.2) {
                    const vLenPix = a.speed_mps * 3.5 * scale * 0.2; // Arrow length scaled to speed
                    // Velocity direction in screen coordinates
                    const vAngle = Math.atan2(a.vy_world, a.vx_world) - egoHeading;
                    const vEndSx = scr.sx + vLenPix * Math.sin(vAngle);
                    const vEndSy = scr.sy - vLenPix * Math.cos(vAngle);

                    ctx.strokeStyle = '#fbbf24';
                    ctx.lineWidth = 2.5;
                    ctx.beginPath();
                    ctx.moveTo(scr.sx, scr.sy);
                    ctx.lineTo(vEndSx, vEndSy);
                    ctx.stroke();

                    // Arrowhead
                    ctx.fillStyle = '#fbbf24';
                    ctx.beginPath();
                    ctx.arc(vEndSx, vEndSy, 3.5, 0, Math.PI * 2);
                    ctx.fill();

                    // Speed Badge
                    ctx.fillStyle = '#fef08a';
                    ctx.font = 'bold 9px monospace';
                    ctx.fillText(`${a.speed_mps}m/s (${a.speed_kph}kph)`, vEndSx + 5, vEndSy - 2);
                }

                // Range Tag
                ctx.fillStyle = '#e2e8f0';
                ctx.font = 'bold 9px monospace';
                const aheadTxt = a.x_ego >= 0 ? `+${a.x_ego}m` : `${a.x_ego}m`;
                ctx.fillText(`${a.id} (${aheadTxt})`, scr.sx + 12, scr.sy + 10);
            });

            // 8. Ego Vehicle & Dynamic Safety Envelope
            // Dynamic Safety Envelope (Expanded footprint bubble around ego)
            const envLatPix = (ego.width_m/2 + ego.safety_buffer_lat_m) * scale;
            const envLonPix = (ego.length_m/2 + ego.safety_buffer_lon_m) * scale;
            const isDanger = data.safety.is_emergency_stop || (data.causal_event && data.causal_event.has_hazard && data.causal_event.hazard_ttc_s && data.causal_event.hazard_ttc_s < 2.0);
            const isCaution = data.causal_event && data.causal_event.has_hazard;

            ctx.strokeStyle = isDanger ? '#ef4444' : (isCaution ? '#f59e0b' : '#06b6d4');
            ctx.fillStyle = isDanger ? 'rgba(239, 68, 68, 0.15)' : (isCaution ? 'rgba(245, 158, 11, 0.10)' : 'rgba(6, 182, 212, 0.08)');
            ctx.lineWidth = 1.8;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.roundRect(originX - envLatPix, originY - envLonPix, envLatPix * 2, envLonPix * 2, 8);
            ctx.fill();
            ctx.stroke();
            ctx.setLineDash([]);

            // Ego Body (Facing straight UP in ego cockpit view)
            const egoLPix = ego.length_m * scale;
            const egoWPix = ego.width_m * scale;

            // Headlight beams
            const grad = ctx.createLinearGradient(originX, originY - egoLPix/2, originX, originY - egoLPix/2 - 40);
            grad.addColorStop(0, 'rgba(254, 240, 138, 0.4)');
            grad.addColorStop(1, 'rgba(254, 240, 138, 0.0)');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.moveTo(originX - egoWPix/2 + 2, originY - egoLPix/2);
            ctx.lineTo(originX - egoWPix/2 - 15, originY - egoLPix/2 - 45);
            ctx.lineTo(originX + egoWPix/2 + 15, originY - egoLPix/2 - 45);
            ctx.lineTo(originX + egoWPix/2 - 2, originY - egoLPix/2);
            ctx.closePath();
            ctx.fill();

            // Car Body
            ctx.fillStyle = '#06b6d4';
            ctx.fillRect(originX - egoWPix/2, originY - egoLPix/2, egoWPix, egoLPix);
            ctx.fillStyle = '#0891b2';
            ctx.fillRect(originX - egoWPix/3, originY - egoLPix/4, egoWPix * 0.66, egoLPix * 0.5); // Roof

            // Front Steered Wheels (Deflected by Stanley controller angle)
            const steerRad = -ego.steer_deg * Math.PI / 180;
            ctx.fillStyle = '#0f172a';

            // Left Front Wheel
            ctx.save();
            ctx.translate(originX - egoWPix/2 - 2, originY - egoLPix/3);
            ctx.rotate(steerRad);
            ctx.fillRect(-2, -5, 4, 10);
            ctx.restore();

            // Right Front Wheel
            ctx.save();
            ctx.translate(originX + egoWPix/2 + 2, originY - egoLPix/3);
            ctx.rotate(steerRad);
            ctx.fillRect(-2, -5, 4, 10);
            ctx.restore();

            // Fixed Rear Wheels
            ctx.fillRect(originX - egoWPix/2 - 3, originY + egoLPix/4, 4, 10);
            ctx.fillRect(originX + egoWPix/2 - 1, originY + egoLPix/4, 4, 10);

            // Ego Badge
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 9px sans-serif';
            ctx.fillText('🚗 EGO AV', originX - 22, originY + 3);
        }

        // ----------------------------------------------------
        // VIEW 2: ROAD OVERVIEW (PANORAMIC HORIZONTAL)
        // ----------------------------------------------------
        function renderRoadOverview(data, w, h) {
            const egoX = data.ego.x;
            const egoY = data.ego.y;
            const scale = 14;
            const offsetX = 220 - egoX * scale;
            const centerY = h / 2;

            ctx.fillStyle = '#0a0f1d';
            ctx.fillRect(0, 0, w, h);

            // Road surface
            const poly = data.road.polyline || [];
            if (poly.length > 2) {
                ctx.fillStyle = '#1e293b';
                ctx.beginPath();
                poly.forEach((pt, i) => {
                    const sx = pt.cx * scale + offsetX;
                    const sy = centerY - pt.ly * scale;
                    if (i === 0) ctx.moveTo(sx, sy);
                    else ctx.lineTo(sx, sy);
                });
                for (let i = poly.length - 1; i >= 0; i--) {
                    const sx = poly[i].cx * scale + offsetX;
                    const sy = centerY - poly[i].ry * scale;
                    ctx.lineTo(sx, sy);
                }
                ctx.closePath();
                ctx.fill();

                // Road Edges
                ctx.strokeStyle = '#eab308';
                ctx.lineWidth = 2.0;
                ctx.stroke();
            }

            // Candidates
            (data.candidates || []).forEach(cand => {
                if (!cand.waypoints || cand.waypoints.length === 0) return;
                ctx.beginPath();
                ctx.moveTo(egoX * scale + offsetX, centerY - egoY * scale);
                cand.waypoints.forEach(wp => {
                    ctx.lineTo(wp.x * scale + offsetX, centerY - wp.y * scale);
                });

                if (cand.is_selected) {
                    ctx.strokeStyle = '#10b981';
                    ctx.lineWidth = 3.5;
                    ctx.stroke();
                } else if (cand.is_feasible) {
                    ctx.strokeStyle = 'rgba(6, 182, 212, 0.35)';
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([4, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                } else {
                    ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([2, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            });

            // Actors
            (data.actors || []).forEach(a => {
                const ax = a.x_world * scale + offsetX;
                const ay = centerY - a.y_world * scale;
                ctx.fillStyle = a.class === 'TRUCK' ? '#f59e0b' : (a.class === 'PEDESTRIAN' ? '#f43f5e' : '#ea580c');
                ctx.beginPath();
                ctx.arc(ax, ay, 9, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 9px monospace';
                ctx.fillText(a.id, ax - 12, ay - 12);
            });

            // Ego
            const ex = egoX * scale + offsetX;
            const ey = centerY - egoY * scale;
            ctx.fillStyle = '#06b6d4';
            ctx.fillRect(ex - 22, ey - 10, 44, 20);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 9px sans-serif';
            ctx.fillText('EGO', ex - 10, ey + 3);
        }

        async function toggleSim() {
            await fetch('/simulation/toggle', { method: 'POST' });
        }

        async function resetSim() {
            await fetch('/simulation/reset', { method: 'POST' });
        }

        async function triggerEStop() {
            await fetch('/emergency_stop', { method: 'POST' });
        }

        async function spawnHazard(type) {
            await fetch('/simulation/spawn_hazard', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hazard_type: type, distance_ahead_m: 35.0 })
            });
        }

        async function changeDifficulty(level) {
            await fetch('/simulation/difficulty', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ difficulty: level })
            });
        }
    </script>
</body>
</html>
"""

    return app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=5002)
