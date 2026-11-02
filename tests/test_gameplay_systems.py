"""Comprehensive gameplay and system integration tests for Kessler AI.

Tests:
1. Player controls and physics dynamics.
2. Collision detection (bullet-asteroid, ship-asteroid, shield/hull damage).
3. Threat calculation and collision metrics.
4. A* global route planning and obstacle avoidance.
5. Safe zone arrival and docking mechanics.
6. Emergency collision avoidance override.
7. AI controller integration with live game state.
8. Game state transitions (menu, play, pause, victory, game over, restart).
"""
import math
import numpy as np
import pytest

from kessler_mpc.debris import Asteroid, DebrisManager
from kessler_mpc.game import GameMode, GameState, KesslerGameApp
from kessler_mpc.kessler_ai import AIBehaviorState, KesslerAIController
from kessler_mpc.pathfinding import AStarPlanner, WaypointNavigator
from kessler_mpc.physics import PhysicsEngine
from kessler_mpc.player import PlayerShip
from kessler_mpc.safe_zone import SafeZone
from kessler_mpc.settings import GameSettings
from kessler_mpc.spaceship import Spaceship
from kessler_mpc.threat_assessment import ThreatEvaluator


# =============================================================================
# 1. Player Movement and Controls
# =============================================================================
def test_player_thrust_and_drag_physics():
    player = PlayerShip(position=(500.0, 400.0), heading=0.0)
    assert player.speed == 0.0

    # Apply forward thrust for 0.5s
    dt = 0.1
    for _ in range(5):
        is_thrusting, nozzle = player.update_physics(thrust_input=480.0, turn_input=0.0, dt=dt)
        assert is_thrusting

    # Ship should have positive forward speed
    assert player.speed > 0.0
    assert player.x > 500.0

    # Coast with no thrust -> drag slows ship down
    prev_spd = player.speed
    for _ in range(5):
        player.update_physics(thrust_input=0.0, turn_input=0.0, dt=dt)
    assert player.speed < prev_spd


def test_player_boost_mechanic():
    player = PlayerShip(position=(500.0, 400.0), heading=0.0)
    assert player.boost_energy == 100.0

    # Boost active -> consumes boost energy
    player.update_physics(thrust_input=480.0, turn_input=0.0, dt=0.2, boost_active=True)
    assert player.is_boosting
    assert player.boost_energy < 100.0

    # Boost off -> recharges boost energy
    boost_val = player.boost_energy
    player.update_physics(thrust_input=0.0, turn_input=0.0, dt=0.2, boost_active=False)
    assert not player.is_boosting
    assert player.boost_energy > boost_val


def test_player_turning():
    player = PlayerShip(position=(500.0, 400.0), heading=90.0)
    # Turn clockwise (+180 deg/s) for 0.5s -> heading increases by 90 deg -> 180 deg
    player.update_physics(thrust_input=0.0, turn_input=180.0, dt=0.5)
    assert player.heading == pytest.approx(180.0, abs=1.0)


# =============================================================================
# 2. Collision Detection & Resolution
# =============================================================================
def test_bullet_asteroid_collision_and_splitting():
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0))
    ship = Spaceship(position=(200.0, 200.0), heading=0.0)

    # Spawn size 2 asteroid at (300, 200)
    ast = debris_mgr.spawn_asteroid(position=(300.0, 200.0), velocity=(0.0, 0.0), size=2)
    assert len(debris_mgr.asteroids) == 1

    # Spawn bullet right next to it
    bullet = ship.fire()
    assert bullet is not None
    bullet.x = 295.0
    bullet.y = 200.0

    events = physics.check_bullet_asteroid_collisions([ship], debris_mgr)
    assert len(events) == 1
    assert events[0].event_type == "bullet_asteroid"
    # Size 2 splits into 3 size 1 children
    assert len(debris_mgr.asteroids) == 3
    assert ship.asteroids_destroyed == 1
    assert ship.shots_hit == 1


def test_ship_asteroid_collision_and_shield_damage():
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0))
    ship = Spaceship(position=(400.0, 400.0), heading=0.0)
    assert ship.shields == 100.0
    assert ship.health == 100.0

    # Asteroid right on top of ship
    ast = debris_mgr.spawn_asteroid(position=(405.0, 400.0), velocity=(-50.0, 0.0), size=2)

    events = physics.check_ship_asteroid_collisions([ship], debris_mgr)
    assert len(events) == 1
    assert events[0].event_type == "ship_asteroid"
    # Shields should have absorbed damage
    assert ship.shields < 100.0
    assert ship.health == 100.0  # Hull protected while shields up


def test_ship_death_and_lives_decrement():
    ship = Spaceship(position=(400.0, 400.0))
    ship.shields = 0.0
    ship.health = 20.0
    ship.lives = 3

    died = ship.apply_damage(50.0)
    assert died
    assert ship.lives == 2
    assert ship.health == 0.0


# =============================================================================
# 3. Threat Assessment & Calculations
# =============================================================================
def test_threat_evaluator_metrics():
    evaluator = ThreatEvaluator()
    ctrl = KesslerAIController()

    ship_state = {
        "position": (500.0, 400.0),
        "velocity": (0.0, 0.0),
        "speed": 0.0,
        "heading": 0.0,
        "radius": 20.0,
        "lives": 3,
    }
    # Asteroid moving head-on toward ship
    game_state = {
        "map_size": (1000.0, 800.0),
        "frame_rate": 60.0,
        "time": 0.0,
        "asteroids": [
            {
                "position": (700.0, 400.0),
                "velocity": (-100.0, 0.0),
                "speed": 100.0,
                "size": 3,
                "radius": 50.0,
                "mass": 200.0,
            }
        ],
    }

    world = ctrl._build_world(ship_state, game_state)
    raw_th, threat_list = evaluator.assess_threats(world)

    assert len(threat_list) == 1
    t = threat_list[0]
    assert t.distance == pytest.approx(200.0)
    assert t.relative_speed == pytest.approx(100.0)
    assert t.time_to_closest_approach > 0.0
    assert t.overall_threat_score > 0.0


# =============================================================================
# 4. A* Global Route Planning & Waypoint Navigation
# =============================================================================
def test_astar_planner_finds_path():
    planner = AStarPlanner(map_w=1000.0, map_h=800.0, cell_size=25.0)

    # Empty debris field
    pos = np.zeros((0, 2))
    vel = np.zeros((0, 2))
    rad = np.zeros(0)

    path = planner.plan_path(
        start_pos=(100.0, 100.0),
        goal_pos=(800.0, 700.0),
        asteroid_positions=pos,
        asteroid_velocities=vel,
        asteroid_radii=rad,
    )

    assert len(path) >= 2
    assert path[0] == (100.0, 100.0)
    assert path[-1] == (800.0, 700.0)


def test_astar_avoids_large_obstacle():
    planner = AStarPlanner(map_w=1000.0, map_h=800.0, cell_size=25.0)

    # Place a large stationary obstacle directly between (100, 400) and (900, 400)
    pos = np.array([[500.0, 400.0]])
    vel = np.array([[0.0, 0.0]])
    rad = np.array([80.0])

    path = planner.plan_path(
        start_pos=(100.0, 400.0),
        goal_pos=(900.0, 400.0),
        asteroid_positions=pos,
        asteroid_velocities=vel,
        asteroid_radii=rad,
    )

    assert len(path) >= 3
    # Check that path does not pass directly through the obstacle center (500, 400)
    for px, py in path:
        dist_to_obs = math.hypot(px - 500.0, py - 400.0)
        assert dist_to_obs > 70.0, f"Waypoint ({px}, {py}) too close to obstacle!"


def test_waypoint_navigator_progression():
    nav = WaypointNavigator(waypoint_reach_radius=30.0)
    path = [(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)]
    nav.set_path(path)

    # At start, target is first waypoint
    t1 = nav.get_current_target((100.0, 100.0))
    # Since ship is at (100, 100), reach_radius (30) advances target to next waypoint
    assert t1 == (200.0, 100.0)

    # When ship arrives near (200, 100), target advances to (300, 100)
    t2 = nav.get_current_target((195.0, 100.0))
    assert t2 == (300.0, 100.0)
    assert nav.is_finished


# =============================================================================
# 5. Safe Zone Objective & Docking Mechanics
# =============================================================================
def test_safe_zone_docking_and_victory():
    sz = SafeZone(center=(800.0, 200.0), radius=60.0, required_dock_time=2.0)

    # Ship outside safe zone
    assert not sz.contains_point((100.0, 100.0))
    assert sz.distance_to_perimeter((100.0, 100.0)) > 500.0

    # Ship enters safe zone
    ship_id = 1
    ship_pos = {ship_id: (810.0, 200.0)}
    assert sz.contains_point(ship_pos[ship_id])

    # Update for 1 second -> 50% docked
    sz.update(1.0, ship_pos)
    assert sz.get_dock_progress(ship_id) == pytest.approx(0.5, abs=0.05)
    assert not sz.is_ship_docked(ship_id)

    # Update for another 1.2 seconds -> 100% docked (Victory condition)
    sz.update(1.2, ship_pos)
    assert sz.get_dock_progress(ship_id) == 1.0
    assert sz.is_ship_docked(ship_id)


# =============================================================================
# 6. Emergency Collision Avoidance Override
# =============================================================================
def test_emergency_collision_avoidance_trigger():
    ctrl = KesslerAIController()

    ship_state = {
        "position": (500.0, 400.0),
        "velocity": (0.0, 0.0),
        "speed": 0.0,
        "heading": 0.0,
        "radius": 20.0,
        "lives": 3,
        "thrust_range": (-480.0, 480.0),
        "turn_rate_range": (-180.0, 180.0),
        "max_speed": 240.0,
        "drag": 80.0,
    }
    # High-speed asteroid barreling toward ship: distance 80, speed 150 -> TTC = 0.53s (< 0.75s critical)
    game_state = {
        "map_size": (1000.0, 800.0),
        "frame_rate": 60.0,
        "time": 0.0,
        "asteroids": [
            {
                "position": (580.0, 400.0),
                "velocity": (-150.0, 0.0),
                "speed": 150.0,
                "size": 4,
                "radius": 80.0,
                "mass": 600.0,
            }
        ],
    }

    thrust, turn, fire, boost = ctrl.get_actions(ship_state, game_state)

    # Must activate emergency collision avoidance override
    assert ctrl.telemetry.emergency_override
    assert ctrl.telemetry.state == AIBehaviorState.EMERGENCY_AVOIDANCE
    # Thrust or turn executed to evade
    assert abs(thrust) > 0.0 or abs(turn) > 0.0


# =============================================================================
# 7. AI Live Simulation Integration
# =============================================================================
def test_ai_live_simulation_integration():
    ctrl = KesslerAIController()

    ship_state = {
        "position": (200.0, 600.0),
        "velocity": (50.0, 0.0),
        "speed": 50.0,
        "heading": 45.0,
        "radius": 20.0,
        "lives": 3,
    }
    game_state = {
        "map_size": (1000.0, 800.0),
        "frame_rate": 60.0,
        "time": 1.0,
        "asteroids": [
            {"position": (400.0, 400.0), "velocity": (20.0, -10.0), "size": 2, "radius": 30.0, "mass": 75.0},
            {"position": (700.0, 500.0), "velocity": (-30.0, 15.0), "size": 3, "radius": 50.0, "mass": 200.0},
        ],
    }

    thrust, turn, fire, boost = ctrl.get_actions(ship_state, game_state)

    # Valid physical outputs
    assert -480.0 <= thrust <= 480.0
    assert -180.0 <= turn <= 180.0
    assert isinstance(fire, bool)
    assert isinstance(boost, bool)

    # Telemetry populated
    assert len(ctrl.telemetry.astar_path) > 0
    assert len(ctrl.telemetry.mpc_trajectory) > 0
    assert ctrl.telemetry.risk_multiplier >= 1.0


# =============================================================================
# 8. Game State Transitions & Session Reset
# =============================================================================
def test_game_app_state_transitions():
    app = KesslerGameApp(settings=GameSettings(sfx_enabled=False))

    # Initial state is MENU
    assert app.state == GameState.MENU

    # Start Player Mode
    app.start_game(GameMode.PLAYER)
    assert app.state == GameState.PLAYING
    assert app.player_ship is not None
    assert app.player_ship.is_alive
    assert app.player_ship.lives == 3

    # Pause and Resume
    app.set_state(GameState.PAUSED)
    assert app.state == GameState.PAUSED
    app.set_state(GameState.PLAYING)
    assert app.state == GameState.PLAYING

    # Trigger Game Over
    app.player_ship.lives = 0
    app.player_ship.is_alive = False
    app._update(0.016)
    assert app.state == GameState.GAME_OVER

    # Restart Match
    app.start_game(GameMode.AI_DEMO)
    assert app.state == GameState.PLAYING
    assert app.ai_ship is not None
    assert app.ai_ship.is_alive
    assert app.ai_ship.lives == 3
