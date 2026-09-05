"""FastAPI telemetry server compatible with CampusOS Genova portal (Port 5002)."""
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from interfaces import VehicleTelemetry, Pose3D, Twist3D, BehaviorMode, SafetyAction

def create_app() -> FastAPI:
    app = FastAPI(title="SIH26037 Autonomous Mobility Service", version="0.1.0")

    latest_telemetry = VehicleTelemetry(
        timestamp=0.0,
        vehicle_id="SIH26037-AV-01",
        pose=Pose3D(),
        twist=Twist3D(),
        battery_soc=96.5,
        behavior_mode=BehaviorMode.CRUISE,
        safety_action=SafetyAction.NONE,
        current_speed_kph=0.0,
        steering_angle_deg=0.0,
        min_ttc_seconds=999.0,
        is_e_stop_active=False
    )

    @app.get("/health")
    async def health():
        return {
            "status": "UP",
            "service": "SIH26037 Autonomous Mobility Stack",
            "vehicle_id": latest_telemetry.vehicle_id,
            "e_stop": latest_telemetry.is_e_stop_active
        }

    @app.get("/telemetry")
    async def get_telemetry():
        return latest_telemetry.model_dump()

    @app.post("/emergency_stop")
    async def trigger_emergency_stop():
        latest_telemetry.is_e_stop_active = True
        latest_telemetry.safety_action = SafetyAction.EMERGENCY_BRAKE
        latest_telemetry.behavior_mode = BehaviorMode.EMERGENCY_STOP
        return {"status": "EMERGENCY_STOP_TRIGGERED"}

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(latest_telemetry.model_dump())
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            pass

    return app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=5002)
