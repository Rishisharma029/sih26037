"""ROS2 WebSocket Bridge Client matching CampusOS port 9090."""
from typing import Dict, Any

class CampusOSBridgeClient:
    """Provides direct compatibility with rosbridge_suite (Port 9090)."""
    def __init__(self, ws_url: str = "ws://localhost:9090"):
        self.ws_url = ws_url
        self.connected = False

    def serialize_odometry(self, x: float, y: float, yaw: float, speed: float) -> dict:
        """Encodes into standard ROS2 nav_msgs/Odometry format."""
        return {
            "op": "publish",
            "topic": "/vehicle/odom",
            "msg": {
                "pose": {"pose": {"position": {"x": x, "y": y, "z": 0.0}}},
                "twist": {"twist": {"linear": {"x": speed, "y": 0.0, "z": 0.0}}}
            }
        }
