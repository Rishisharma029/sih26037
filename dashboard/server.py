"""
SIH26037 Live Telemetry & Simulation Visualizer Server (Port 5002)
Provides:
1. Live Interactive HTML5 Canvas Bird's-Eye-View (BEV) visualizer with Candidate Path Bundle rendering.
2. Real-time Causal Decision HUD showing the entire Sense-Predict-Plan-Safe-Act chain.
3. Real-time telemetry streaming over WebSockets and REST APIs.
4. Interactive Hazard Spawner (Oncoming Tractor, Crossing Villager, Parked Auto) and Difficulty selector.
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
from coordinates import transform_actor_to_ego_tracked_obstacle

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

            actors_data.append({
                "id": a.id,
                "class": a.obstacle_class.value,
                "x_world": round(a.x, 2),
                "y_world": round(a.y, 2),
                "x_ego": round(obs.bbox.center.x, 2), # Forward distance ahead (+X)
                "y_ego": round(obs.bbox.center.y, 2), # Lateral distance left (+Y) / right (-Y)
                "distance_m": round(obs.distance_m, 2), # Euclidean distance (always >= 0)
                "ttc_s": round(ttc_val, 2) if ttc_val is not None else None,
                "speed_kph": round(a.speed_mps * 3.6, 1),
                "yaw_deg": round(math.degrees(a.yaw_rad), 1),
                "is_static": a.is_static
            })

        # 2. Road Corridor Detection
        corridor = self.boundary_detector.detect_corridor(
            timestamp=ego_state.timestamp,
            lookahead_m=40.0,
            step_m=4.0,
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

        # 9. Format Predictions
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
                            "sigma_y": round(pt.sigma_y, 2)
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

        # 10. Build Live Causal Decision Chain Data
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
                "speed_kph": round(state.twist.speed_mps * 3.6, 1),
                "steer_deg": round(math.degrees(state.steer_angle_rad), 1),
                "throttle_pct": round(cmd.throttle_pct, 1),
                "brake_pct": round(cmd.brake_pct, 1),
                "battery_soc": round(state.battery_soc_pct, 1)
            },
            "road": {
                "corridor_left_m": round(d_left, 2),
                "corridor_right_m": round(d_right, 2),
                "current_margin_m": round(margin, 2),
                "min_margin_m": round(self.min_corridor_margin, 2)
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
    <title>SIH26037 — Reactive Collision Avoidance & Causal Decision Chain</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0b0f19; color: #e2e8f0; font-family: ui-sans-serif, system-ui, sans-serif; }
        canvas { background-color: #111827; border-radius: 0.75rem; border: 1px solid #1e293b; }
        .glass-card { background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.75rem; }
        .chain-step { transition: all 0.2s ease-in-out; }
    </style>
</head>
<body class="p-4 md:p-6 max-w-7xl mx-auto space-y-5">

    <!-- Header & Interactive Controls -->
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-5">
        <div>
            <div class="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 mb-1.5">
                <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                SIH26037 • VISIBLY REACTIVE AUTONOMOUS VEHICLE
            </div>
            <h1 class="text-xl md:text-2xl font-black tracking-tight text-white flex items-center gap-3">
                Unmarked Village Road — Reactive Hazard Avoidance
                <span id="badge-difficulty" class="text-xs px-2.5 py-1 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono font-bold">HARD TIER</span>
            </h1>
            <p class="text-xs text-slate-400 mt-0.5">
                Causal Pipeline: Hazard Detection → Trajectory Prediction → TTC Invalidation → Candidate Bundle Evaluation → Stanley Steering Deflection.
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
    <div id="causal-banner" class="glass-card p-4 border border-cyan-500/30 bg-gradient-to-r from-slate-900 via-cyan-950/40 to-slate-900 space-y-3">
        <div class="flex items-center justify-between">
            <span class="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                End-to-End Causal Chain Engine
            </span>
            <span id="causal-summary-badge" class="text-xs font-mono font-bold px-2.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/40">
                CRUISING NOMINAL PATH
            </span>
        </div>

        <!-- 5-Stage Causal Flow Stepper -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-2.5 text-xs font-mono">
            <div id="step-1" class="chain-step p-2.5 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">1. Hazard Incursion</span>
                <span id="hud-hazard" class="font-bold text-slate-300 block truncate">None Detected</span>
                <span id="hud-hazard-dist" class="text-[10px] text-slate-500">Range: --</span>
            </div>
            <div id="step-2" class="chain-step p-2.5 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">2. TTC & Intent</span>
                <span id="hud-ttc" class="font-bold text-slate-300 block">TTC > 4.0s (Safe)</span>
                <span id="hud-risk" class="text-[10px] text-slate-500">Nominal: VALID</span>
            </div>
            <div id="step-3" class="chain-step p-2.5 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">3. Candidate Bundle</span>
                <span id="hud-candidates" class="font-bold text-slate-300 block">7 Evaluated</span>
                <span id="hud-cand-status" class="text-[10px] text-emerald-400">Center (0.0m) Clear</span>
            </div>
            <div id="step-4" class="chain-step p-2.5 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">4. Safe Selection</span>
                <span id="hud-mode" class="font-bold text-cyan-400 block">CRUISE</span>
                <span id="hud-offset" class="text-[10px] text-slate-400">Offset: 0.0m</span>
            </div>
            <div id="step-5" class="chain-step p-2.5 rounded-lg bg-slate-900/90 border border-slate-800">
                <span class="text-[10px] text-slate-500 font-sans block uppercase">5. DBW Actuation</span>
                <span id="hud-steer-act" class="font-bold text-amber-400 block">Steer: 0.0°</span>
                <span id="hud-speed-act" class="text-[10px] text-slate-400">Speed: 21.6 km/h</span>
            </div>
        </div>
    </div>

    <!-- Live Telemetry KPI Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Speed</span>
            <span id="kpi-speed" class="text-2xl font-black text-cyan-400 font-mono">0.0</span>
            <span class="text-[10px] text-slate-500 font-semibold">km/h</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Stanley Steer</span>
            <span id="kpi-steer" class="text-2xl font-black text-amber-400 font-mono">0.0°</span>
            <span class="text-[10px] text-slate-500 font-semibold">Front Wheel Angle</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Distance Traveled</span>
            <span id="kpi-dist" class="text-2xl font-black text-emerald-400 font-mono">0.0 m</span>
            <span class="text-[10px] text-slate-500 font-semibold">Station along road</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Ditch Margin</span>
            <span id="kpi-margin" class="text-2xl font-black text-purple-400 font-mono">2.15 m</span>
            <span class="text-[10px] text-slate-500 font-semibold">Safe Limit >= 0.15m</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Safety Action</span>
            <span id="kpi-safety" class="text-sm font-black text-emerald-400 font-mono block mt-1">NONE</span>
            <span class="text-[10px] text-slate-500 font-semibold">Supervisory Arbiter</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Behavior Mode</span>
            <span id="kpi-mode" class="text-sm font-black text-cyan-400 font-mono block mt-1">CRUISE</span>
            <span class="text-[10px] text-slate-500 font-semibold">Lattice Decision</span>
        </div>
    </div>

    <!-- Main BEV Canvas Visualizer -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div class="lg:col-span-3 glass-card p-4 space-y-3">
            <div class="flex flex-wrap items-center justify-between text-xs text-slate-400 px-1 gap-2">
                <span class="font-bold uppercase tracking-wider flex items-center gap-1.5 text-slate-300">
                    <span class="w-2 h-2 rounded-full bg-emerald-400"></span> Live BEV Renderer with Candidate Trajectory Bundle
                </span>
                <!-- Interactive Hazard Injection Bar -->
                <div class="flex items-center gap-2">
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

            <canvas id="simCanvas" width="920" height="440" class="w-full h-auto"></canvas>

            <div class="flex flex-wrap items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80 gap-3">
                <div class="flex flex-wrap items-center gap-3">
                    <span class="inline-flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-emerald-400"></span> Selected Safe Path</span>
                    <span class="inline-flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-rose-500"></span> Blocked Candidate (Collision)</span>
                    <span class="inline-flex items-center gap-1.5"><span class="w-3 h-1.5 rounded bg-cyan-500/40"></span> Candidate Alternatives</span>
                    <span class="inline-flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded bg-amber-500"></span> Tractor</span>
                    <span class="inline-flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded bg-rose-500"></span> Pedestrian</span>
                </div>
                <div class="text-slate-500 font-mono">
                    Candidate Offsets: [-1.5m ... +1.5m]
                </div>
            </div>
        </div>

        <!-- Right Side: Detected Objects & Candidate Evaluator Feed -->
        <div class="glass-card p-4 space-y-4">
            <div class="flex items-center justify-between">
                <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Live Radar & Threat Feed
                </h3>
                <span class="text-[10px] text-cyan-400 font-mono">20 Hz Sense</span>
            </div>
            
            <div id="actors-list" class="space-y-2 text-xs max-h-60 overflow-y-auto pr-1">
                <!-- Dynamically populated -->
            </div>

            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300 pt-2 border-t border-slate-800">
                Lattice Candidates Breakdown
            </h3>
            <div id="candidates-list" class="space-y-1 text-[11px] font-mono">
                <!-- Populated with all 7 offset evaluations -->
            </div>
        </div>
    </div>

    <!-- Script: WebSocket & Canvas Renderer -->
    <script>
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        let currentData = null;

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
            document.getElementById('kpi-dist').innerText = `${data.ego.x} m`;
            document.getElementById('kpi-margin').innerText = `${data.road.current_margin_m} m`;
            document.getElementById('kpi-safety').innerText = data.safety.safety_action;
            document.getElementById('kpi-mode').innerText = data.planning.behavior_mode;

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
                s1.className = 'chain-step p-2.5 rounded-lg bg-amber-950/40 border border-amber-500/60 shadow-lg shadow-amber-900/20';
                document.getElementById('hud-hazard').innerHTML = `<span class="text-amber-400">🚨 ${ce.hazard_id}</span>`;
                document.getElementById('hud-hazard-dist').innerText = `Range: ${ce.hazard_dist_m}m`;

                s2.className = 'chain-step p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/60';
                document.getElementById('hud-ttc').innerHTML = `<span class="text-rose-400 font-bold">TTC: ${ce.hazard_ttc_s || '<1.0'}s</span>`;
                document.getElementById('hud-risk').innerHTML = ce.nominal_path_safe ? '<span class="text-slate-400">Nominal: CLEAR</span>' : '<span class="text-rose-400 font-bold">Nominal: UNSAFE (Blocked)</span>';

                s3.className = 'chain-step p-2.5 rounded-lg bg-cyan-950/40 border border-cyan-500/60';
                document.getElementById('hud-candidates').innerText = `7 Candidates Evaluated`;
                document.getElementById('hud-cand-status').innerHTML = `<span class="text-cyan-300">Offset ${ce.selected_offset_m > 0 ? '+' : ''}${ce.selected_offset_m}m Safe</span>`;

                s4.className = 'chain-step p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/60';
                document.getElementById('hud-mode').innerText = ce.selected_mode;
                document.getElementById('hud-offset').innerText = `Chosen: ${ce.selected_offset_m > 0 ? '+' : ''}${ce.selected_offset_m}m`;

                s5.className = 'chain-step p-2.5 rounded-lg bg-indigo-950/40 border border-indigo-500/60';
                document.getElementById('hud-steer-act').innerHTML = `<span class="text-amber-300 font-bold">Steer: ${ce.steer_command_deg}°</span>`;
                document.getElementById('hud-speed-act').innerText = `Speed: ${data.ego.speed_kph} km/h`;

                document.getElementById('causal-summary-badge').className = 'text-xs font-mono font-bold px-2.5 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-500/60 animate-pulse';
                document.getElementById('causal-summary-badge').innerText = `AVOIDING: ${ce.hazard_id.toUpperCase()} (${ce.selected_mode})`;
            } else {
                [s1, s2, s3, s4, s5].forEach(el => el.className = 'chain-step p-2.5 rounded-lg bg-slate-900/90 border border-slate-800');
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

                document.getElementById('causal-summary-badge').className = 'text-xs font-mono font-bold px-2.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/40';
                document.getElementById('causal-summary-badge').innerText = 'CRUISING NOMINAL PATH';
            }

            // Update Actors List
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
                        <span class="text-slate-200">${a.id}</span>
                        <div class="flex items-center gap-1">
                            ${ttcBadge}
                            <span class="text-[10px] px-1.5 py-0.2 rounded ${a.is_static ? 'bg-slate-800 text-slate-400' : 'bg-cyan-900/50 text-cyan-300'}">${a.class}</span>
                        </div>
                    </div>
                    <div class="flex justify-between text-[10px] text-slate-400">
                        <span>Range: <b class="text-slate-200 font-mono">${a.distance_m}m</b> (${aheadText}, ${latText})</span>
                        <span>Speed: <b class="text-slate-200 font-mono">${a.speed_kph} km/h</b></span>
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

        function renderScene(data) {
            if (!data || !data.ego) return;
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            // Follow ego vehicle horizontally
            const egoX = data.ego.x;
            const egoY = data.ego.y;
            const scale = 15; // pixels per meter
            const offsetX = 200 - egoX * scale;
            const centerY = h / 2;

            // 1. Draw Grass/Verge Background
            ctx.fillStyle = '#111827';
            ctx.fillRect(0, 0, w, h);

            // 2. Draw Unmarked Road Surface
            ctx.fillStyle = '#1e293b';
            ctx.beginPath();
            for (let x = -20; x < 260; x += 2) {
                const dL = 2.15 + 0.15 * Math.sin(x * 0.08);
                const screenX = x * scale + offsetX;
                const screenY = centerY - dL * scale;
                if (x === -20) ctx.moveTo(screenX, screenY);
                else ctx.lineTo(screenX, screenY);
            }
            for (let x = 258; x >= -20; x -= 2) {
                const dR = -(2.15 + 0.18 * Math.cos(x * 0.07));
                const screenX = x * scale + offsetX;
                const screenY = centerY - dR * scale;
                ctx.lineTo(screenX, screenY);
            }
            ctx.closePath();
            ctx.fill();

            // 3. Draw Irregular Shoulder & Ditch Lines
            ctx.strokeStyle = '#475569';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);

            // 4. Draw Ego Range Distance Arcs (+10m, +25m, +40m)
            const egoScreenX = egoX * scale + offsetX;
            const egoScreenY = centerY - egoY * scale;

            [10, 25, 40].forEach(r => {
                ctx.strokeStyle = 'rgba(6, 182, 212, 0.20)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 5]);
                ctx.beginPath();
                ctx.arc(egoScreenX, egoScreenY, r * scale, -Math.PI / 3, Math.PI / 3);
                ctx.stroke();
                ctx.fillStyle = 'rgba(6, 182, 212, 0.45)';
                ctx.font = '9px monospace';
                ctx.fillText(`+${r}m`, egoScreenX + r * scale + 3, egoScreenY - 4);
            });
            ctx.setLineDash([]);

            // 5. Draw Candidate Trajectory Bundle (Green = chosen, Red = blocked, Cyan = clear alternatives)
            (data.candidates || []).forEach(cand => {
                if (!cand.waypoints || cand.waypoints.length === 0) return;
                ctx.beginPath();
                ctx.moveTo(egoScreenX, egoScreenY);

                cand.waypoints.forEach(pt => {
                    const px = pt.x * scale + offsetX;
                    const py = centerY - pt.y * scale;
                    ctx.lineTo(px, py);
                });

                if (cand.is_selected) {
                    ctx.strokeStyle = '#10b981'; // Vibrant Green for selected
                    ctx.lineWidth = 3.5;
                    ctx.shadowColor = '#10b981';
                    ctx.shadowBlur = 8;
                    ctx.setLineDash([]);
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                } else if (cand.is_feasible) {
                    ctx.strokeStyle = 'rgba(6, 182, 212, 0.35)'; // Cyan for clear alternatives
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([4, 3]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                } else {
                    ctx.strokeStyle = 'rgba(239, 68, 68, 0.50)'; // Red for collision/blocked
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([2, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // Draw collision X marker at end of blocked path
                    const lastPt = cand.waypoints[cand.waypoints.length - 1];
                    if (lastPt) {
                        const lx = lastPt.x * scale + offsetX;
                        const ly = centerY - lastPt.y * scale;
                        ctx.strokeStyle = '#ef4444';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(lx - 4, ly - 4); ctx.lineTo(lx + 4, ly + 4);
                        ctx.moveTo(lx + 4, ly - 4); ctx.lineTo(lx - 4, ly + 4);
                        ctx.stroke();
                    }
                }
            });

            // 6. Draw Multi-Modal Predictions (Forecasted paths of actors)
            (data.predictions || []).forEach(pred => {
                (pred.trajectories || []).forEach(tr => {
                    if (tr.waypoints && tr.waypoints.length > 0) {
                        ctx.beginPath();
                        const startPt = tr.waypoints[0];
                        ctx.moveTo(startPt.x * scale + offsetX, centerY - startPt.y * scale);
                        tr.waypoints.forEach(wp => {
                            ctx.lineTo(wp.x * scale + offsetX, centerY - wp.y * scale);
                        });
                        ctx.strokeStyle = tr.collision_risk > 0.4 ? 'rgba(244, 63, 94, 0.6)' : 'rgba(245, 158, 11, 0.4)';
                        ctx.lineWidth = 1.5;
                        ctx.setLineDash([3, 3]);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                });
            });

            // 7. Draw Actors (Tractor, Pedestrian, Auto, Boulder)
            (data.actors || []).forEach(a => {
                const ax = a.x_world * scale + offsetX;
                const ay = centerY - a.y_world * scale;

                if (a.class === 'TRUCK' || a.id.includes('tractor')) {
                    // Draw Agricultural Tractor (Amber / Yellow body with large wheels)
                    ctx.fillStyle = '#f59e0b';
                    ctx.fillRect(ax - 26, ay - 14, 52, 28);
                    ctx.fillStyle = '#1e293b';
                    ctx.fillRect(ax - 24, ay - 16, 12, 4); // Rear big wheels
                    ctx.fillRect(ax - 24, ay + 12, 12, 4);
                    ctx.fillRect(ax + 12, ay - 14, 8, 3); // Front wheels
                    ctx.fillRect(ax + 12, ay + 11, 8, 3);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 9px sans-serif';
                    ctx.fillText('TRACTOR', ax - 20, ay + 3);
                } else if (a.class === 'PEDESTRIAN' || a.id.includes('pedestrian') || a.id.includes('villager')) {
                    // Crossing Villager (Rose Dot)
                    ctx.fillStyle = '#f43f5e';
                    ctx.beginPath();
                    ctx.arc(ax, ay, 7, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 8px sans-serif';
                    ctx.fillText('PED', ax - 8, ay - 9);
                } else if (a.class === 'AUTO_RICKSHAW' || a.id.includes('auto')) {
                    // Parked Auto-Rickshaw (Orange Box)
                    ctx.fillStyle = '#ea580c';
                    ctx.fillRect(ax - 16, ay - 9, 32, 18);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 8px sans-serif';
                    ctx.fillText('AUTO', ax - 11, ay + 3);
                } else {
                    // Roadside Boulder / Debris
                    ctx.fillStyle = '#64748b';
                    ctx.beginPath();
                    ctx.arc(ax, ay, 9, 0, Math.PI * 2);
                    ctx.fill();
                }

                // Range tag
                ctx.fillStyle = 'rgba(226, 232, 240, 0.9)';
                ctx.font = '9px monospace';
                const tag = `${a.x_ego >= 0 ? '+' : ''}${a.x_ego}m`;
                ctx.fillText(tag, ax - 12, ay - 17);
            });

            // 8. Draw Ego Autonomous Vehicle with Steerable Front Wheels
            ctx.save();
            ctx.translate(egoScreenX, egoScreenY);
            ctx.rotate(-data.ego.heading_deg * Math.PI / 180);

            // Vehicle Body (Cyan)
            ctx.fillStyle = '#06b6d4';
            ctx.fillRect(-26, -11, 52, 22);
            ctx.fillStyle = '#0891b2';
            ctx.fillRect(-12, -9, 24, 18); // Cabin roof

            // Headlights
            ctx.fillStyle = '#fef08a';
            ctx.fillRect(25, -9, 3, 5);
            ctx.fillRect(25, 4, 3, 5);

            // Front Steered Wheels
            const steerRad = -data.ego.steer_deg * Math.PI / 180;
            ctx.fillStyle = '#0f172a';

            // Front Left Wheel
            ctx.save();
            ctx.translate(18, -12);
            ctx.rotate(steerRad);
            ctx.fillRect(-5, -2, 10, 4);
            ctx.restore();

            // Front Right Wheel
            ctx.save();
            ctx.translate(18, 12);
            ctx.rotate(steerRad);
            ctx.fillRect(-5, -2, 10, 4);
            ctx.restore();

            // Rear Fixed Wheels
            ctx.fillRect(-20, -14, 10, 4);
            ctx.fillRect(-20, 10, 10, 4);

            ctx.restore();
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
