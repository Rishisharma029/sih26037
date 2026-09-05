"""
Real-world Indian Road Geographic Registry.
Maps the 5 SIH hallmark validation scenarios to real coordinates in India with
satellite imagery metadata, road dimensions, surface conditions, and actor profiles.
"""
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field

class RealWorldScenarioLocation(BaseModel):
    scenario_id: str
    name: str
    location_name: str
    state: str
    latitude: float
    longitude: float
    elevation_m: float
    road_type: str
    carriageway_width_m: float
    shoulder_type: str
    painted_markings: bool
    typical_hazards: List[str]
    traffic_density: str
    satellite_zoom_level: int = 18

INDIAN_SCENARIO_REGISTRY: Dict[str, RealWorldScenarioLocation] = {
    "SCENARIO_1_VILLAGE": RealWorldScenarioLocation(
        scenario_id="SCENARIO_1",
        name="Unmarked Village Road",
        location_name="Khed Shivapur Rural Bypass, Pune District",
        state="Maharashtra",
        latitude=18.3541,
        longitude=73.8423,
        elevation_m=642.0,
        road_type="Single-lane rural carriageway",
        carriageway_width_m=4.2,
        shoulder_type="Unpaved mud shoulder with open irrigation ditch",
        painted_markings=False,
        typical_hazards=["Agricultural tractor", "Ditch drop-offs", "Roadside banyan trees", "Potholes"],
        traffic_density="Low to moderate mixed-speed",
        satellite_zoom_level=19
    ),
    "SCENARIO_2_JUNCTION": RealWorldScenarioLocation(
        scenario_id="SCENARIO_2",
        name="Busy Urban Intersection Without Signals",
        location_name="Silk Board - Marathahalli Outer Feeder Cross",
        state="Karnataka (Bengaluru)",
        latitude=12.9176,
        longitude=77.6238,
        elevation_m=895.0,
        road_type="Multi-arm unsignalized urban junction",
        carriageway_width_m=8.5,
        shoulder_type="Curbed perimeter with high pedestrian spillover",
        painted_markings=False,
        typical_hazards=["Perpendicular auto-rickshaw crossing", "Motorcycle filtering", "Darting pedestrians"],
        traffic_density="High density chaotic swarm",
        satellite_zoom_level=19
    ),
    "SCENARIO_3_HIGHWAY": RealWorldScenarioLocation(
        scenario_id="SCENARIO_3",
        name="Highway Merge with Slow Cut-in",
        location_name="NH-48 Jaipur-Delhi Highway Unregulated Merge, Kotputli",
        state="Rajasthan",
        latitude=27.2154,
        longitude=75.9812,
        elevation_m=340.0,
        road_type="Semi-divided rural highway",
        carriageway_width_m=7.2,
        shoulder_type="Sandy gravel shoulder with gradual incline",
        painted_markings=False,
        typical_hazards=["Overloaded 3-wheeler cutting in at 25 km/h", "High closing velocity differential"],
        traffic_density="High speed differential mixed flow",
        satellite_zoom_level=18
    ),
    "SCENARIO_4_MARKET": RealWorldScenarioLocation(
        scenario_id="SCENARIO_4",
        name="Dense Market Road",
        location_name="Chandni Chowk - Sadar Bazaar Street, Old Delhi",
        state="Delhi NCR",
        latitude=28.6506,
        longitude=77.2303,
        elevation_m=216.0,
        road_type="Commercial heritage market corridor",
        carriageway_width_m=4.8,
        shoulder_type="Occupied by vendor stalls and hand pushcarts",
        painted_markings=False,
        typical_hazards=["Stationary fruit pushcarts", "Continuous pedestrian swarms", "Bicycles"],
        traffic_density="Extreme pedestrian & micro-mobility density",
        satellite_zoom_level=20
    ),
    "SCENARIO_5_CATTLE": RealWorldScenarioLocation(
        scenario_id="SCENARIO_5",
        name="Sudden Cattle Roadblock on Blind Curve",
        location_name="SH-87 Varanasi-Ghazipur Highway Bend",
        state="Uttar Pradesh",
        latitude=25.3176,
        longitude=83.0062,
        elevation_m=81.0,
        road_type="State highway curved carriageway (R=40m)",
        carriageway_width_m=6.4,
        shoulder_type="Overgrown foliage verge with obstructed line-of-sight",
        painted_markings=False,
        typical_hazards=["Resting stray cattle in lane center", "Blind curvature visual occlusion"],
        traffic_density="Moderate with unexpected roadblocks",
        satellite_zoom_level=19
    )
}
