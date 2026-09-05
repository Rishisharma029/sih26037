"""Comprehensive Integration Tests for 5 Hallmark Indian Scenarios across 4 Difficulty Tiers."""
import pytest
from interfaces import ControlCommand
from scenarios.difficulty import DifficultyLevel
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from scenarios.scenario_unsignalled_junction import UnsignalledJunctionScenario
from scenarios.scenario_highway_cutin import HighwayCutInScenario
from scenarios.scenario_dense_market import DenseMarketScenario
from scenarios.scenario_cattle_crossing import CattleCrossingScenario
from scenarios.benchmark_suite import execute_scenario_episode


@pytest.mark.parametrize("scenario_cls", [
    UnmarkedVillageRoadScenario,
    UnsignalledJunctionScenario,
    HighwayCutInScenario,
    DenseMarketScenario,
    CattleCrossingScenario
])
@pytest.mark.parametrize("diff", [
    DifficultyLevel.EASY,
    DifficultyLevel.MEDIUM,
    DifficultyLevel.HARD,
    DifficultyLevel.EXTREME
])
def test_all_scenarios_and_difficulties_instantiation(scenario_cls, diff):
    """Verify that all 5 scenarios cleanly instantiate across Easy, Medium, Hard, Extreme."""
    scenario = scenario_cls(difficulty=diff)
    assert scenario.name is not None
    assert scenario.difficulty == diff
    assert len(scenario.env.actors) >= 1

    # Verify execution of 1 step
    cmd = ControlCommand(timestamp=0.0, steering_angle_rad=0.0, throttle_pct=20.0, brake_pct=0.0)
    state, _ = scenario.run_step(cmd)
    assert state.twist.speed_mps >= 0.0


def test_cattle_crossing_extreme_closed_loop_episode():
    """Verify closed-loop episode execution on extreme cattle crossing scenario."""
    result = execute_scenario_episode(
        CattleCrossingScenario,
        difficulty=DifficultyLevel.EXTREME,
        target_speed_mps=5.0
    )
    assert result["scenario_name"] == "05_cattle_crossing"
    assert result["difficulty"] == "EXTREME"
    assert result["collisions"] == 0
    assert result["min_corridor_margin_m"] > 0.15
