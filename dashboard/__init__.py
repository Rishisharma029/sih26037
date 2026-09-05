"""Dashboard & Telemetry server for SIH26037."""
from .server import create_app
from .bridge import CampusOSBridgeClient

__all__ = ["create_app", "CampusOSBridgeClient"]
