"""Comprehensive Unit & Scenario Tests for Phase 6 Adaptive Lattice Planner."""
import math
import pytest

from interfaces import (
    EgoVehicleState, Pose3D, Point3D, Twist3D, Vector3D,
    PerceptionOutput, FreeSpaceCorridor, CorridorBoundaryPoint, PredictionOutput,
    PredictedAgent, PredictedTrajectory, PredictedTrajectoryPoint,
    ObstacleClass, MotionIntent, BehaviorMode, TrackedObstacle, BoundingBox3D
)
from planning.frenet_lattice import FrenetLatticeGenerator, QuinticPolynomial
from planning.cost_evaluator import TrajectoryCostEvaluator, TrajectoryCostScore
from planning.local_planner import AdaptiveLatticePlanner


def _create_mock_ego(x: float = 0.0, y: float = 0.0, speed: float = 6.0) -> EgoVehicleState:
    return EgoVehicleState(
        timestamp=10.0,
        pose=Pose3D(position=Point3D(x=x, y=y, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=speed),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )


def test_quintic_polynomial_boundary_conditions():
    """Verify that quintic polynomial satisfies exact start and end boundary conditions."""
    poly = QuinticPolynomial(
        xs=0.0, vxs=5.0, axs=0.0,
        xe=20.0, vxe=8.0, axe=0.0,
        T=3.0
    )
    # Start conditions
    assert pytest.approx(poly.calc_point(0.0), abs=1e-3) == 0.0
    assert pytest.approx(poly.calc_first_derivative(0.0), abs=1e-3) == 5.0
    assert pytest.approx(poly.calc_second_derivative(0.0), abs=1e-3) == 0.0

    # End conditions
    assert pytest.approx(poly.calc_point(3.0), abs=1e-2) == 20.0
    assert pytest.approx(poly.calc_first_derivative(3.0), abs=1e-2) == 8.0
    assert pytest.approx(poly.calc_second_derivative(3.0), abs=1e-2) == 0.0


def test_lattice_generator_sampling_diversity():
    """Verify lattice generator produces candidates covering multiple lateral offsets and speeds."""
    gen = FrenetLatticeGenerator(dt=0.2)
    ego = _create_mock_ego(speed=6.0)
    candidates = gen.sample_candidates(ego, target_cruise_speed_mps=8.0)

    assert len(candidates) >= 28
    # Check that lateral targets span left and right
    targets = set(c.target_d for c in candidates)
    assert -1.0 in targets
    assert 0.0 in targets
    assert 1.0 in targets
    # Check candidate IDs follow P1, P2...
    assert candidates[0].candidate_id == "P1"


def test_7_objective_cost_evaluator_weights_and_breakdown():
    """Verify 7-objective cost evaluator weights normalize to 1.0 and cost breakdown contains all 7 metrics."""
    evaluator = TrajectoryCostEvaluator(
        w_collision=0.40,
        w_clearance=0.20,
        w_traversability=0.15,
        w_progress=0.10,
        w_smoothness=0.05,
        w_dynamics=0.05,
        w_uncertainty=0.05
    )
    assert pytest.approx(
        evaluator.w_collision + evaluator.w_clearance + evaluator.w_traversability +
        evaluator.w_progress + evaluator.w_smoothness + evaluator.w_dynamics + evaluator.w_uncertainty,
        abs=1e-5
    ) == 1.0

    gen = FrenetLatticeGenerator(dt=0.2)
    ego = _create_mock_ego(speed=6.0)
    candidates = gen.sample_candidates(ego, target_cruise_speed_mps=6.0)
    
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.5)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=3.0, agents=[])

    cand = candidates[0]
    score = evaluator.evaluate(cand, perc, pred, target_cruise_speed_mps=6.0)

    assert score.is_feasible
    assert "collision_safety" in score.cost_breakdown
    assert "obstacle_clearance" in score.cost_breakdown
    assert "road_traversability" in score.cost_breakdown
    assert "progress" in score.cost_breakdown
    assert "path_smoothness" in score.cost_breakdown
    assert "vehicle_dynamics" in score.cost_breakdown
    assert "uncertainty" in score.cost_breakdown
    assert score.status_tag == "SAFE"


def test_explainable_rejection_tags():
    """Verify candidate evaluation tags infeasible trajectories with explicit human-readable reasons."""
    evaluator = TrajectoryCostEvaluator()
    gen = FrenetLatticeGenerator(dt=0.2)
    ego = _create_mock_ego(speed=6.0)
    candidates = gen.sample_candidates(ego, target_cruise_speed_mps=6.0)

    # 1. Test DITCH_BREACH: road is extremely narrow (1.5m total width -> left edge = 0.75m, car half width = 0.90m)
    corridor_narrow = FreeSpaceCorridor(
        timestamp=10.0,
        boundary_points=[CorridorBoundaryPoint(s=float(s), d_left=1.0, d_right=-1.0) for s in range(0, 30, 2)],
        average_width_m=2.0
    )
    perc_narrow = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=corridor_narrow
    )
    pred_empty = PredictionOutput(timestamp=10.0, horizon_seconds=3.0, agents=[])

    cand_left = next(c for c in candidates if c.target_d == 1.5)
    score_ditch = evaluator.evaluate(cand_left, perc_narrow, pred_empty, target_cruise_speed_mps=6.0)
    assert not score_ditch.is_feasible
    assert score_ditch.status_tag == "DITCH_BREACH"
    assert "violates safety invariant" in score_ditch.explanation

    # 2. Test COLLISION: Obstacle directly on candidate waypoint
    obs_collision = TrackedObstacle(
        id="blocking_tractor",
        obstacle_class=ObstacleClass.TRUCK,
        confidence=0.98,
        bbox=BoundingBox3D(center=Point3D(x=10.0, y=0.0, z=0.5), size=Vector3D(x=3.0, y=1.8, z=2.0)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=10.0,
        is_static=True
    )
    pred_tractor = PredictedAgent(
        id="blocking_tractor",
        obstacle_class=ObstacleClass.TRUCK,
        primary_intent=MotionIntent.STATIONARY,
        trajectories=[
            PredictedTrajectory(
                probability=1.0,
                mode_name="stationary",
                waypoints=[
                    PredictedTrajectoryPoint(
                        timestamp=10.0 + step * 0.2,
                        position=Point3D(x=10.0, y=0.0, z=0.5),
                        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                        yaw_rad=0.0,
                        sigma_x=0.1,
                        sigma_y=0.1
                    )
                    for step in range(1, 16)
                ]
            )
        ]
    )
    perc_col = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs_collision],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=7.0)
    )
    pred_col = PredictionOutput(timestamp=10.0, horizon_seconds=3.0, agents=[pred_tractor])

    cand_center = next(c for c in candidates if c.target_d == 0.0 and c.target_v > 4.0)
    score_col = evaluator.evaluate(cand_center, perc_col, pred_col, target_cruise_speed_mps=6.0)
    assert not score_col.is_feasible
    assert score_col.status_tag == "COLLISION"
    assert "Collision risk" in score_col.explanation


def test_cruise_nominal_planning():
    """Verify nominal planning in free road selects straight cruise mode."""
    planner = AdaptiveLatticePlanner(horizon_seconds=2.5, dt=0.2)
    ego = _create_mock_ego(speed=6.0)
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.5)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.5, agents=[])

    plan = planner.plan(ego, perc, pred, target_cruise_speed_mps=8.0)

    assert plan.is_feasible
    assert plan.behavior_mode == BehaviorMode.CRUISE
    # Selected path should stay centered near y=0
    assert abs(plan.waypoints[-1].y) <= 0.6
    assert plan.waypoints[-1].speed_mps >= 5.0
    assert len(planner.last_scored_candidates) > 0


def test_obstacle_avoidance_nudge():
    """Verify planner generates a lateral nudge trajectory around a stationary obstacle in lane center."""
    planner = AdaptiveLatticePlanner(horizon_seconds=2.5, dt=0.2)
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)

    # Obstacle blocking center lane at x=10m, y=0.0m
    obs_lead = TrackedObstacle(
        id="parked_auto",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=10.0, y=0.0, z=0.5), size=Vector3D(x=2.5, y=1.4, z=1.6)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=10.0,
        is_static=True
    )

    # Multi-modal prediction: stationary track at x=10, y=0
    pred_agent = PredictedAgent(
        id="parked_auto",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        primary_intent=MotionIntent.STATIONARY,
        trajectories=[
            PredictedTrajectory(
                probability=1.0,
                mode_name="stationary",
                waypoints=[
                    PredictedTrajectoryPoint(
                        timestamp=10.0 + step * 0.2,
                        position=Point3D(x=10.0, y=0.0, z=0.5),
                        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                        yaw_rad=0.0,
                        sigma_x=0.1,
                        sigma_y=0.1
                    )
                    for step in range(1, 16)
                ]
            )
        ]
    )

    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1,
        obstacles=[obs_lead],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.5)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.5, agents=[pred_agent])

    plan = planner.plan(ego, perc, pred, target_cruise_speed_mps=6.0)

    assert plan.is_feasible
    # Planner must choose to nudge left or right around the parked auto (not straight collision)
    assert plan.behavior_mode in [BehaviorMode.NUDGE_LEFT, BehaviorMode.NUDGE_RIGHT]
    assert abs(plan.waypoints[-1].y) >= 0.4
    assert planner.last_selected_candidate_id is not None


def test_cost_evaluator_uncertainty_penalty():
    """Verify cost evaluator penalizes trajectories encroaching on high-uncertainty actor regions."""
    evaluator = TrajectoryCostEvaluator()
    gen = FrenetLatticeGenerator(dt=0.2)
    ego = _create_mock_ego(speed=6.0)
    candidates = gen.sample_candidates(ego, target_cruise_speed_mps=6.0)

    # Actor at x=10, y=1.0 with large lateral sigma_y = 0.8m
    agent_uncertain = PredictedAgent(
        id="moto_weaving",
        obstacle_class=ObstacleClass.MOTORCYCLE,
        primary_intent=MotionIntent.ERRATIC_SWERVE,
        trajectories=[
            PredictedTrajectory(
                probability=0.8,
                mode_name="nudge",
                waypoints=[
                    PredictedTrajectoryPoint(
                        timestamp=10.0 + step * 0.2,
                        position=Point3D(x=10.0, y=1.0, z=0.5),
                        velocity=Vector3D(x=2.0, y=0.2, z=0.0),
                        yaw_rad=0.0,
                        sigma_x=0.5,
                        sigma_y=0.8
                    )
                    for step in range(1, 16)
                ]
            )
        ]
    )

    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.5)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=2.5, agents=[agent_uncertain])

    cand_near = next(c for c in candidates if c.target_d == 1.0 and c.target_v > 4.0)
    cand_far = next(c for c in candidates if c.target_d == -1.0 and c.target_v > 4.0)

    score_near = evaluator.evaluate(cand_near, perc, pred, target_cruise_speed_mps=6.0)
    score_far = evaluator.evaluate(cand_far, perc, pred, target_cruise_speed_mps=6.0)

    assert score_near.cost_breakdown["uncertainty"] > score_far.cost_breakdown["uncertainty"]
    assert score_near.total_cost > score_far.total_cost


def test_baseline_planner_candidate_offsets_clear():
    """Verify BaselinePlanner selects center cruise (0.0m offset) when road is clear."""
    from planning.baseline_planner import BaselinePlanner
    planner = BaselinePlanner(horizon_seconds=3.0, dt=0.2)
    ego = _create_mock_ego(speed=6.0)
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=7.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=3.0, agents=[])
    
    plan = planner.plan(ego, perc, pred, target_cruise_speed_mps=6.0)
    assert plan.is_feasible
    assert plan.behavior_mode == BehaviorMode.CRUISE
    assert abs(plan.waypoints[-1].y) < 0.2
    assert len(plan.waypoints) == 15


def test_baseline_planner_candidate_offsets_nudge_left():
    """Verify BaselinePlanner selects positive offset (NUDGE_LEFT) when center is blocked."""
    from planning.baseline_planner import BaselinePlanner
    planner = BaselinePlanner(horizon_seconds=3.0, dt=0.2)
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)
    
    # Center blocked at x=15, y=0.0 by motorcycle
    obs_center = TrackedObstacle(
        id="block_center",
        obstacle_class=ObstacleClass.MOTORCYCLE,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=15.0, y=0.0, z=0.5), size=Vector3D(x=1.8, y=0.6, z=1.2)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=15.0,
        is_static=True
    )
    # Also block right side at x=15, y=-1.0
    obs_right = TrackedObstacle(
        id="block_right",
        obstacle_class=ObstacleClass.STATIC_DEBRIS,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=15.0, y=-1.0, z=0.3), size=Vector3D(x=1.5, y=0.8, z=0.6)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=15.0,
        is_static=True
    )
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=[obs_center, obs_right],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=7.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=3.0, agents=[])
    
    plan = planner.plan(ego, perc, pred, target_cruise_speed_mps=6.0)
    assert plan.is_feasible
    assert plan.behavior_mode == BehaviorMode.NUDGE_LEFT
    assert plan.waypoints[-1].y > 0.4


def test_baseline_planner_candidate_offsets_emergency_stop():
    """Verify BaselinePlanner triggers emergency stop when road is completely wall-blocked."""
    from planning.baseline_planner import BaselinePlanner
    planner = BaselinePlanner(horizon_seconds=3.0, dt=0.2)
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)
    
    wall_obstacles = [
        TrackedObstacle(
            id=f"wall_{i}",
            obstacle_class=ObstacleClass.STATIC_DEBRIS,
            confidence=0.9,
            bbox=BoundingBox3D(center=Point3D(x=10.0, y=y_val, z=0.5), size=Vector3D(x=2.0, y=1.2, z=1.0)),
            velocity=Vector3D(x=0.0, y=0.0, z=0.0),
            distance_m=10.0,
            is_static=True
        )
        for i, y_val in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0])
    ]
    perc = PerceptionOutput(
        timestamp=10.0, frame_id=1, obstacles=wall_obstacles,
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=7.0)
    )
    pred = PredictionOutput(timestamp=10.0, horizon_seconds=3.0, agents=[])
    
    plan = planner.plan(ego, perc, pred, target_cruise_speed_mps=6.0)
    assert plan.behavior_mode == BehaviorMode.EMERGENCY_STOP
    assert plan.waypoints[-1].speed_mps == 0.0
