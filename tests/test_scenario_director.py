"""
Test Suite: Step 12 - Scenario Director & Interactive Benchmark Injector
Verifies:
1. Reconfiguration of simulation state to all 5 benchmark scenarios (Village, Junction, Highway, Market, Cattle).
2. Dynamic hazard injection package overrides (Tractor, Pedestrian, Boulder, Pothole, Flooded, Cattle, Motorcycle).
3. Difficulty scaling (EASY, MEDIUM, HARD, EXTREME).
4. FastAPI POST /simulation/direct_scenario endpoint execution and JSON response.
5. Zero-latency autonomous drive auto-start.
"""
import pytest
from fastapi.testclient import TestClient

from dashboard.server import (
    SimulationEngineState,
    ScenarioDirectorRequest,
    create_app
)
from scenarios.difficulty import DifficultyLevel
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from scenarios.scenario_unsignalled_junction import UnsignalledJunctionScenario
from scenarios.scenario_highway_cutin import HighwayCutInScenario
from scenarios.scenario_dense_market import DenseMarketScenario
from scenarios.scenario_cattle_crossing import CattleCrossingScenario


@pytest.fixture
def sim_engine():
    return SimulationEngineState(difficulty=DifficultyLevel.HARD)


@pytest.fixture
def client(sim_engine):
    app = create_app(sim_engine)
    return TestClient(app)


def test_scenario_director_village_custom_hazards(sim_engine):
    """Verify directing village scenario with custom hazard combination."""
    sim_engine.direct_scenario(
        scenario_type="village",
        diff_name="EXTREME",
        hazards=["tractor", "pothole", "pedestrian"],
        auto_start=True
    )
    assert isinstance(sim_engine.scenario, UnmarkedVillageRoadScenario)
    assert sim_engine.difficulty == DifficultyLevel.EXTREME
    assert sim_engine.is_running is True
    assert len(sim_engine.scenario.env.actors) == 2  # tractor + pedestrian
    assert len(sim_engine.scenario.env.anomalies) == 1  # pothole
    assert sim_engine.latest_telemetry is not None


def test_scenario_director_all_5_benchmarks(sim_engine):
    """Verify switching across all 5 benchmark scenarios."""
    scenarios_to_test = [
        ("village", UnmarkedVillageRoadScenario),
        ("junction", UnsignalledJunctionScenario),
        ("highway", HighwayCutInScenario),
        ("market", DenseMarketScenario),
        ("cattle", CattleCrossingScenario)
    ]
    for sc_name, sc_class in scenarios_to_test:
        sim_engine.direct_scenario(scenario_type=sc_name, diff_name="MEDIUM", auto_start=True)
        assert isinstance(sim_engine.scenario, sc_class)
        assert sim_engine.difficulty == DifficultyLevel.MEDIUM
        assert sim_engine.is_running is True
        assert sim_engine.latest_telemetry["ego"]["speed_mps"] >= 0.0


def test_scenario_director_full_hazard_package(sim_engine):
    """Verify injecting all hazard types simultaneously."""
    all_hazards = [
        "tractor", "pedestrian", "auto", "boulder",
        "pothole", "waterlogged", "motorcycle", "cattle",
        "gravel", "speed_bump"
    ]
    sim_engine.direct_scenario(
        scenario_type="village",
        diff_name="HARD",
        hazards=all_hazards,
        auto_start=True
    )
    # Check that both dynamic actors and road anomalies are populated
    assert len(sim_engine.scenario.env.actors) >= 4
    assert len(sim_engine.scenario.env.anomalies) >= 4
    
    # Step forward and ensure simulation runs without exception
    sim_engine.step()
    assert "causal_event" in sim_engine.latest_telemetry
    assert "decision" in sim_engine.latest_telemetry


def test_fastapi_direct_scenario_endpoint(client):
    """Verify POST /simulation/direct_scenario returns 200 and correctly updates engine."""
    payload = {
        "scenario_type": "junction",
        "difficulty": "EXTREME",
        "hazards": ["tractor", "motorcycle", "pothole"],
        "auto_start": True
    }
    response = client.post("/simulation/direct_scenario", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SCENARIO_DIRECTED"
    assert data["scenario"] == "junction"
    assert data["difficulty"] == "EXTREME"
    assert data["hazards_count"] == 3
    assert data["is_running"] is True
