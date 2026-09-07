"""
SIH26037 Live Telemetry & Simulation Visualizer Server (Port 5002)
Provides:
1. Live Interactive HTML5 Canvas Bird's-Eye-View (BEV) visualizer of the Unmarked Indian Village Road scene.
2. Real-time telemetry streaming over WebSockets and REST APIs.
3. Closed-loop physics stepping and multi-agent interaction.
4. Seamless integration with the CampusOS Genova platform.
"""
import sys
import os
import math
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure sih26037 package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interfaces import (
    VehicleTelemetry, Pose3D, Twist3D, BehaviorMode,
    SafetyAction, ControlCommand, SafeTrajectory, TrajectoryPoint,
    Point3D, Vector3D, GearMode, PerceptionOutput, FreeSpaceCorridor,
    TrackedObstacle, BoundingBox3D
)
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from vehicle_control.lateral_controller import StanleyLateralController
from vehicle_control.longitudinal_controller import LongitudinalPIDController
from collision_avoidance.emergency_brake import EmergencyBrakeSupervisory
from perception.boundary_detector import FreeSpaceBoundaryDetector
from prediction.trajectory_predictor import TrajectoryPredictor

class SimulationEngineState:
    def __init__(self):
        self.scenario = UnmarkedVillageRoadScenario()
        self.lat_ctrl = StanleyLateralController(k_gain=1.4)
        self.lon_ctrl = LongitudinalPIDController(kp=22.0, ki=0.5, kd=2.0)
        self.supervisory = EmergencyBrakeSupervisory(aeb_ttc_threshold_s=0.85)
        self.predictor = TrajectoryPredictor(horizon_seconds=3.0, dt=0.5, mode="ensemble")
        self.is_running = True
        self.target_speed_mps = 6.0
        self.is_emergency_stop = False
        self.step_count = 0
        self.min_corridor_margin = 2.0
        self.latest_telemetry: Dict[str, Any] = {}

    def reset(self):
        self.scenario = UnmarkedVillageRoadScenario()
        self.is_emergency_stop = False
        self.step_count = 0
        self.min_corridor_margin = 2.0

    def step(self):
        if not self.is_running:
            return

        dt = self.scenario.dt
        ego_state = self.scenario.simulator.state
        s_curr = ego_state.pose.position.x

        # 1. Sample road geometry ahead to build candidate trajectory
        waypoints = []
        for i in range(1, 12):
            s_ahead = s_curr + i * 2.2
            rx, ry, ryaw = self.scenario.env.geometry.get_centerline_point(s_ahead)
            waypoints.append(TrajectoryPoint(
                timestamp=ego_state.timestamp + i * 0.15,
                x=rx,
                y=ry,
                yaw_rad=ryaw,
                speed_mps=0.0 if self.is_emergency_stop else self.target_speed_mps
            ))

        safe_traj = SafeTrajectory(
            timestamp=ego_state.timestamp,
            source_trajectory_id="village_live_traj",
            waypoints=waypoints,
            safety_action=SafetyAction.EMERGENCY_BRAKE if self.is_emergency_stop else SafetyAction.NONE,
            is_emergency_stop=self.is_emergency_stop,
            barrier_margin_m=2.5,
            min_ttc_seconds=999.0
        )

        # 2. Control computation
        steer = self.lat_ctrl.compute_steering(ego_state, safe_traj)
        throttle, brake = self.lon_ctrl.compute_throttle_brake(ego_state, safe_traj, dt=dt)

        if self.is_emergency_stop:
            throttle = 0.0
            brake = 100.0

        cmd = ControlCommand(
            timestamp=ego_state.timestamp,
            steering_angle_rad=steer,
            throttle_pct=throttle,
            brake_pct=brake,
            gear=GearMode.DRIVE,
            emergency_brake_active=self.is_emergency_stop
        )

        # 3. Physics step
        state, raw_sensor = self.scenario.run_step(cmd)
        self.step_count += 1

        # 4. Corridor bounds & metrics
        d_left, d_right = self.scenario.env.geometry.get_corridor_widths(state.pose.position.x)
        margin_left = d_left - state.pose.position.y
        margin_right = state.pose.position.y - d_right
        margin = min(margin_left, margin_right)
        self.min_corridor_margin = min(self.min_corridor_margin, margin)

        # 5. Transform all Actors to standard Ego Vehicle Coordinates
        actors_data = []
        obstacles = []
        from coordinates import transform_actor_to_ego_tracked_obstacle

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
                ego_pose=state.pose,
                ego_twist=state.twist,
                confidence=0.95
            )
            obstacles.append(obs)

            # Compute dynamic Time-To-Collision (TTC) in ego body frame
            ttc_eval = self.supervisory.ttc_calc.compute_ttc(state, [obs])
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

        # Run Phase 5 Motion Prediction
        perception_frame = PerceptionOutput(
            timestamp=state.timestamp,
            frame_id=self.step_count,
            obstacles=obstacles,
            drivable_corridor=FreeSpaceCorridor(
                timestamp=state.timestamp,
                boundary_points=[],
                average_width_m=float(d_left - d_right)
            )
        )
        pred_out = self.predictor.predict(perception_frame, ego_speed=state.twist.speed_mps)

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

        self.latest_telemetry = {
            "timestamp": round(state.timestamp, 2),
            "step": self.step_count,
            "vehicle_id": "SIH26037-AV-01",
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
    <title>SIH26037 — Autonomous Vehicle Live Simulation</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0b0f19; color: #e2e8f0; font-family: ui-sans-serif, system-ui, sans-serif; }
        canvas { background-color: #111827; border-radius: 0.75rem; border: 1px solid #1e293b; }
        .glass-card { background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.75rem; }
    </style>
</head>
<body class="p-4 md:p-6 max-w-7xl mx-auto space-y-6">

    <!-- Header -->
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-5">
        <div>
            <div class="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 mb-1.5">
                <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                SIH26037 • CLOSED-LOOP SIMULATOR
            </div>
            <h1 class="text-xl md:text-2xl font-black tracking-tight text-white">
                Unmarked Indian Village Road — Live Traversal
            </h1>
            <p class="text-xs text-slate-400 mt-0.5">
                Phase 1 Benchmark: Zero lane markings, irregular ditch verges, static boulder & parked auto, oncoming tractor, crossing pedestrian.
            </p>
        </div>

        <div class="flex items-center gap-2.5">
            <button onclick="toggleSim()" id="btn-toggle" class="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition shadow">
                Pause Simulation
            </button>
            <button onclick="resetSim()" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold transition border border-slate-700">
                Reset Scene
            </button>
            <button onclick="triggerEStop()" class="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold transition shadow animate-pulse">
                Emergency Stop (E-Stop)
            </button>
        </div>
    </div>

    <!-- Live Telemetry KPI Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Speed</span>
            <span id="kpi-speed" class="text-2xl font-black text-cyan-400 font-mono">0.0</span>
            <span class="text-[10px] text-slate-500 font-semibold">km/h (Target 21.6)</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Steering Angle</span>
            <span id="kpi-steer" class="text-2xl font-black text-amber-400 font-mono">0.0°</span>
            <span class="text-[10px] text-slate-500 font-semibold">Stanley Front Axle</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Distance Traveled</span>
            <span id="kpi-dist" class="text-2xl font-black text-emerald-400 font-mono">0.0 m</span>
            <span class="text-[10px] text-slate-500 font-semibold">Total Road 200m</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Ditch Margin</span>
            <span id="kpi-margin" class="text-2xl font-black text-purple-400 font-mono">2.15 m</span>
            <span class="text-[10px] text-slate-500 font-semibold">Safe >= 0.5m</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Throttle / Brake</span>
            <span id="kpi-throttle" class="text-2xl font-black text-blue-400 font-mono">0% / 0%</span>
            <span class="text-[10px] text-slate-500 font-semibold">PID Demand</span>
        </div>
        <div class="glass-card p-3.5">
            <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Battery SoC</span>
            <span id="kpi-battery" class="text-2xl font-black text-teal-400 font-mono">96.5%</span>
            <span class="text-[10px] text-slate-500 font-semibold">Nominal 400V</span>
        </div>
    </div>

    <!-- Main Bird's-Eye-View (BEV) Simulation Canvas -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div class="lg:col-span-3 glass-card p-4 space-y-3">
            <div class="flex items-center justify-between text-xs text-slate-400 px-1">
                <span class="font-bold uppercase tracking-wider flex items-center gap-1.5 text-slate-300">
                    <span class="w-2 h-2 rounded-full bg-emerald-400"></span> Live Bird's-Eye-View (BEV) Road Renderer
                </span>
                <span>Zoom: Follow Ego Vehicle • Resolution: 0.05m/px</span>
            </div>
            <canvas id="simCanvas" width="900" height="420" class="w-full h-auto"></canvas>
            <div class="flex flex-wrap items-center justify-between text-[11px] text-slate-400 pt-1">
                <div class="flex items-center gap-4">
                    <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-cyan-500"></span> Ego Vehicle</span>
                    <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-amber-500"></span> Oncoming Tractor</span>
                    <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-rose-500"></span> Crossing Pedestrian</span>
                    <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-orange-600"></span> Parked Auto-Rickshaw</span>
                    <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-slate-600"></span> Roadside Boulder</span>
                </div>
                <div class="text-slate-500">
                    Irregular Ditch Boundaries Active
                </div>
            </div>
        </div>

        <!-- Right Side: Live Actors Radar Feed -->
        <div class="glass-card p-4 space-y-4">
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300">
                Detected Objects & Actors
            </h3>
            <div id="actors-list" class="space-y-2.5 text-xs">
                <!-- Dynamically populated -->
            </div>
            <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400 space-y-1">
                <div class="font-bold text-slate-300 uppercase text-[10px]">CampusOS Genova Bridge</div>
                <div>Status: <span class="text-emerald-400 font-semibold">Active & Broadcasting</span></div>
                <div>WebSocket: <span class="font-mono text-cyan-400">ws://localhost:5002/ws/telemetry</span></div>
                <div>REST Health: <span class="font-mono text-cyan-400">/health</span></div>
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
            document.getElementById('kpi-throttle').innerText = `${data.ego.throttle_pct}% / ${data.ego.brake_pct}%`;
            document.getElementById('kpi-battery').innerText = `${data.ego.battery_soc}%`;

            const btnToggle = document.getElementById('btn-toggle');
            btnToggle.innerText = data.status.is_running ? 'Pause Simulation' : 'Resume Simulation';
            btnToggle.className = data.status.is_running 
                ? 'px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition shadow'
                : 'px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold transition shadow';

            // Update Actors List with standard Ego Coordinates
            const listEl = document.getElementById('actors-list');
            listEl.innerHTML = '';
            (data.actors || []).forEach(a => {
                const item = document.createElement('div');
                item.className = 'p-2.5 rounded-lg bg-slate-900 border border-slate-800 space-y-1.5';
                const aheadText = a.x_ego >= 0 ? `+${a.x_ego}m ahead` : `${a.x_ego}m behind`;
                const latText = a.y_ego >= 0 ? `+${a.y_ego}m L` : `${a.y_ego}m R`;
                const ttcBadge = a.ttc_s ? `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${a.ttc_s < 1.5 ? 'bg-rose-900/60 text-rose-300 animate-pulse' : 'bg-amber-900/40 text-amber-300'}">TTC: ${a.ttc_s}s</span>` : '';
                item.innerHTML = `
                    <div class="flex items-center justify-between font-semibold">
                        <span class="text-slate-200">${a.id}</span>
                        <div class="flex items-center gap-1">
                            ${ttcBadge}
                            <span class="text-[10px] px-1.5 py-0.5 rounded ${a.is_static ? 'bg-slate-800 text-slate-400' : 'bg-cyan-900/50 text-cyan-300'}">${a.class}</span>
                        </div>
                    </div>
                    <div class="flex justify-between text-[11px] text-slate-400">
                        <span>Range: <b class="text-slate-200 font-mono">${a.distance_m}m</b> (<span class="text-cyan-400">${aheadText}</span>, <span class="text-purple-400">${latText}</span>)</span>
                        <span>Speed: <b class="text-slate-200 font-mono">${a.speed_kph} km/h</b></span>
                    </div>
                `;
                listEl.appendChild(item);
            });
        }

        function renderScene(data) {
            if (!data || !data.ego) return;
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            // Center view on ego vehicle with lookahead
            const egoX = data.ego.x;
            const scale = 14; // pixels per meter
            const offsetX = 180 - egoX * scale;
            const centerY = h / 2;

            // 1. Draw Grass/Verge Background
            ctx.fillStyle = '#1e293b';
            ctx.fillRect(0, 0, w, h);

            // 2. Draw Road Pavement (without painted lane markers)
            ctx.fillStyle = '#334155';
            ctx.beginPath();
            for (let x = -20; x < 250; x += 2) {
                const dL = 2.15 + 0.15 * Math.sin(x * 0.08);
                const screenX = x * scale + offsetX;
                const screenY = centerY - dL * scale;
                if (x === -20) ctx.moveTo(screenX, screenY);
                else ctx.lineTo(screenX, screenY);
            }
            for (let x = 248; x >= -20; x -= 2) {
                const dR = -(2.15 + 0.18 * Math.cos(x * 0.07));
                const screenX = x * scale + offsetX;
                const screenY = centerY - dR * scale;
                ctx.lineTo(screenX, screenY);
            }
            ctx.closePath();
            ctx.fill();

            // 3. Draw Irregular Shoulder & Ditch Lines
            ctx.strokeStyle = '#64748b';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);

            // 4. Draw Ego Range Distance Arcs (+10m, +25m, +50m ahead)
            const egoScreenX = egoX * scale + offsetX;
            const egoScreenY = centerY - data.ego.y * scale;

            [10, 25, 50].forEach(r => {
                ctx.strokeStyle = 'rgba(6, 182, 212, 0.25)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 6]);
                ctx.beginPath();
                ctx.arc(egoScreenX, egoScreenY, r * scale, -Math.PI / 3, Math.PI / 3);
                ctx.stroke();
                ctx.fillStyle = 'rgba(6, 182, 212, 0.4)';
                ctx.font = '9px monospace';
                ctx.fillText(`+${r}m`, egoScreenX + r * scale + 3, egoScreenY - 4);
            });
            ctx.setLineDash([]);

            // 5. Draw Road Anomalies (Pothole at x=88m)
            const potScreenX = 88 * scale + offsetX;
            const potScreenY = centerY - 0.3 * scale;
            ctx.fillStyle = '#0f172a';
            ctx.beginPath();
            ctx.arc(potScreenX, potScreenY, 0.45 * scale, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#475569';
            ctx.stroke();

            // 6. Draw Actors
            (data.actors || []).forEach(a => {
                const ax = a.x_world * scale + offsetX;
                const ay = centerY - a.y_world * scale;

                if (a.id === 'oncoming_tractor') {
                    ctx.fillStyle = '#f59e0b';
                    ctx.fillRect(ax - 28, ay - 14, 56, 28);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = '10px sans-serif';
                    ctx.fillText('TRACTOR', ax - 24, ay + 4);
                } else if (a.id === 'crossing_pedestrian') {
                    ctx.fillStyle = '#f43f5e';
                    ctx.beginPath();
                    ctx.arc(ax, ay, 7, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#ffffff';
                    ctx.font = '9px sans-serif';
                    ctx.fillText('PED', ax - 9, ay - 10);
                } else if (a.id === 'parked_auto_rickshaw') {
                    ctx.fillStyle = '#ea580c';
                    ctx.fillRect(ax - 18, ay - 9, 36, 18);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = '9px sans-serif';
                    ctx.fillText('AUTO', ax - 12, ay + 3);
                } else if (a.id === 'roadside_boulder') {
                    ctx.fillStyle = '#475569';
                    ctx.beginPath();
                    ctx.arc(ax, ay, 10, 0, Math.PI * 2);
                    ctx.fill();
                }

                // Metric Tag above actor (Forward distance & lateral offset)
                ctx.fillStyle = 'rgba(226, 232, 240, 0.85)';
                ctx.font = '9px monospace';
                const tag = `${a.x_ego >= 0 ? '+' : ''}${a.x_ego}m (${a.distance_m}m)`;
                ctx.fillText(tag, ax - 15, ay - 16);
            });

            // 7. Draw Ego Autonomous Vehicle
            ctx.save();
            ctx.translate(egoScreenX, egoScreenY);
            ctx.rotate(-data.ego.heading_deg * Math.PI / 180);

            // Vehicle Body (Cyan)
            ctx.fillStyle = '#06b6d4';
            ctx.fillRect(-30, -12, 60, 24);
            ctx.fillStyle = '#0891b2';
            ctx.fillRect(-15, -10, 30, 20);

            // Headlights
            ctx.fillStyle = '#fef08a';
            ctx.fillRect(28, -10, 4, 6);
            ctx.fillRect(28, 4, 4, 6);

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
    </script>
</body>
</html>
"""

    return app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=5002)
