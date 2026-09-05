"""Comparative Head-to-Head Evaluator: Baseline Planner vs Adaptive Planner."""
import time
import math
from typing import Dict, List, Tuple
from scenarios.difficulty import DifficultyLevel
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario
from scenarios.scenario_unsignalled_junction import UnsignalledJunctionScenario
from scenarios.scenario_highway_cutin import HighwayCutInScenario
from scenarios.scenario_dense_market import DenseMarketScenario
from scenarios.scenario_cattle_crossing import CattleCrossingScenario
from planning import AdaptiveLatticePlanner, BaselinePlanner
from simulation.closed_loop_pipeline import ClosedLoopAutonomyPipeline
from .evaluator import BenchmarkHarness
from .metrics import FullEvaluationResult
from .reporter import ScorecardReporter


class ComparativeBenchmarkSuite:
    """Executes identical scenarios under both Baseline and Adaptive planners,

    measuring comparative deltas in safety, clearance, latency, smoothness, and stability.
    """

    def __init__(self):
        self.harness = BenchmarkHarness()
        self.scenarios = [
            ("01_village_road", UnmarkedVillageRoadScenario),
            ("02_uncontrolled_intersection", UnsignalledJunctionScenario),
            ("03_highway_merge", HighwayCutInScenario),
            ("04_dense_market", DenseMarketScenario),
            ("05_cattle_crossing", CattleCrossingScenario),
        ]

    def run_episode(
        self,
        scenario_cls,
        difficulty: DifficultyLevel,
        planner_type: str,
        duration_s: float = 6.0,
        dt: float = 0.05
    ) -> FullEvaluationResult:
        """Run a single closed-loop scenario episode with designated planner."""
        sc = scenario_cls(difficulty=difficulty, duration_seconds=duration_s, dt=dt)
        env = sc.env

        if planner_type == "Adaptive Planner":
            planner = AdaptiveLatticePlanner()
        else:
            planner = BaselinePlanner()

        pipeline = ClosedLoopAutonomyPipeline(
            environment=env,
            planner=planner,
            dt=dt,
            plan_dt=0.20
        )

        steps = int(duration_s / dt)
        states = []
        clearances = []
        ttcs = []
        latencies_ms = []

        for _ in range(steps):
            t_start = time.perf_counter_ns()
            step_res = pipeline.step()
            t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
            latencies_ms.append(t_elapsed_ms)

            state = step_res["ego_state"]
            states.append(state)
            clearances.append(step_res["safety_status"]["min_clearance_m"])
            ttcs.append(step_res["safety_status"]["min_ttc_s"])

        return self.harness.evaluate(
            scenario_name=sc.name,
            difficulty=difficulty.value,
            planner_type=planner_type,
            states=states,
            min_clearance_records=clearances,
            ttc_records=ttcs,
            latencies_ms=latencies_ms,
            aeb_triggers=pipeline.safety_supervisor.aeb_trigger_count,
            emergency_replans=pipeline.safety_supervisor.emergency_replan_count,
            target_distance_m=45.0
        )

    def run_comparative_matrix(self) -> Tuple[List[FullEvaluationResult], List[FullEvaluationResult]]:
        """Run all 5 scenarios across Baseline vs Adaptive and return paired results."""
        baseline_results: List[FullEvaluationResult] = []
        adaptive_results: List[FullEvaluationResult] = []

        print("=" * 80)
        print("SIH26037 HEAD-TO-HEAD BENCHMARK: Baseline Planner VS Adaptive Planner")
        print("=" * 80)

        for name, cls in self.scenarios:
            for diff in [DifficultyLevel.MEDIUM, DifficultyLevel.EXTREME]:
                print(f"--> Evaluating {name} [{diff.value}]...")

                # Baseline
                res_base = self.run_episode(cls, diff, planner_type="Baseline Planner")
                baseline_results.append(res_base)

                # Adaptive
                res_adapt = self.run_episode(cls, diff, planner_type="Adaptive Planner")
                adaptive_results.append(res_adapt)

                print(f"    Baseline Score: {res_base.mission.composite_score:.1f} | Min Clear: {res_base.safety.min_clearance_m:.2f}m | AEB: {res_base.computational.aeb_intervention_count}")
                print(f"    Adaptive Score: {res_adapt.mission.composite_score:.1f} | Min Clear: {res_adapt.safety.min_clearance_m:.2f}m | AEB: {res_adapt.computational.aeb_intervention_count}")

        return baseline_results, adaptive_results


def main():
    suite = ComparativeBenchmarkSuite()
    base_res, adapt_res = suite.run_comparative_matrix()
    report_md = ScorecardReporter.generate_comparative_markdown(base_res, adapt_res)

    output_path = "docs/EVALUATION_REPORT.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n[SUCCESS] Comparative Evaluation Report generated at:", output_path)


if __name__ == "__main__":
    main()
