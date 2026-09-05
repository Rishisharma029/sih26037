"""End-to-end integration test for benchmark scenarios."""
from scenarios.scenario_cattle_crossing import CattleCrossingScenario
from scenarios.scenario_highway_cutin import HighwayCutInScenario
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from interfaces import ControlCommand

def test_scenario_step_execution():
    scenario = CattleCrossingScenario()
    cmd = ControlCommand(timestamp=0.0, steering_angle_rad=0.0, throttle_pct=25.0, brake_pct=0.0)
    state, raw = scenario.run_step(cmd)
    assert state.twist.speed_mps >= 0.0
    assert len(scenario.history_states) == 1

def test_scenario_population():
    s1 = UnmarkedVillageRoadScenario()
    assert len(s1.env.obstacles) == 1

    s2 = HighwayCutInScenario()
    assert len(s2.env.obstacles) == 1
