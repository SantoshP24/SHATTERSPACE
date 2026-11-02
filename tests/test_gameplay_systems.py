"""Comprehensive gameplay and system integration tests for Kessler AI.

Tests:
1. Player controls and physics dynamics.
2. Collision detection (bullet-asteroid, ship-asteroid, shield/hull damage).
3. Threat calculation and collision metrics.
4. A* global route planning and obstacle avoidance.
5. Safe zone arrival and docking mechanics.
6. Emergency collision avoidance override (EVADE state).
7. Autonomous AI targeting, lead intercept, and behavior states (ATTACK, EVADE, REPOSITION, SURVIVE).
8. Scoring prevention of duplicate points and safe asteroid spawns.
9. Countdown timer expiration (VICTORY) and zero health (DEFEAT).
10. Game state transitions and session reset.
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
    assert ship.score == 100  # Size 2 awards 100 points


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
    # Shields absorb damage first
    assert ship.shields < 100.0
    assert ship.damage_taken > 0.0


def test_ship_death_when_health_reaches_zero():
    ship = Spaceship(position=(400.0, 400.0))
    ship.shields = 0.0
    ship.health = 20.0

    died = ship.apply_damage(50.0)
    assert died
    assert ship.health == 0.0
    assert ship.damage_taken == 50.0


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
        "health": 100.0,
    }
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
    for px, py in path:
        dist_to_obs = math.hypot(px - 500.0, py - 400.0)
        assert dist_to_obs > 70.0, f"Waypoint ({px}, {py}) too close to obstacle!"


def test_waypoint_navigator_progression():
    nav = WaypointNavigator(waypoint_reach_radius=30.0)
    path = [(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)]
    nav.set_path(path)

    t1 = nav.get_current_target((100.0, 100.0))
    assert t1 == (200.0, 100.0)

    t2 = nav.get_current_target((195.0, 100.0))
    assert t2 == (300.0, 100.0)
    assert nav.is_finished


# =============================================================================
# 5. Safe Zone Objective & Arrival
# =============================================================================
def test_safe_zone_docking_and_victory():
    sz = SafeZone(center=(800.0, 200.0), radius=60.0, required_dock_time=2.0)

    assert not sz.contains_point((100.0, 100.0))
    assert sz.distance_to_perimeter((100.0, 100.0)) > 500.0

    ship_id = 1
    ship_pos = {ship_id: (810.0, 200.0)}
    assert sz.contains_point(ship_pos[ship_id])

    sz.update(1.0, ship_pos)
    assert sz.get_dock_progress(ship_id) == pytest.approx(0.5, abs=0.05)
    assert not sz.is_ship_docked(ship_id)

    sz.update(1.2, ship_pos)
    assert sz.get_dock_progress(ship_id) == 1.0
    assert sz.is_ship_docked(ship_id)


# =============================================================================
# 6. Emergency Collision Avoidance Override (EVADE State)
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
        "health": 100.0,
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

    # Must activate EVADE state
    assert ctrl.telemetry.emergency_override
    assert ctrl.telemetry.state == AIBehaviorState.EVADE
    assert abs(thrust) > 0.0 or abs(turn) > 0.0


# =============================================================================
# 7. AI Behavior States: ATTACK, SURVIVE, REPOSITION
# =============================================================================
def test_ai_behavior_state_survive_on_low_health():
    ctrl = KesslerAIController()

    # Low health ship (health = 25) with nearby threat
    ship_state = {
        "position": (500.0, 400.0),
        "velocity": (0.0, 0.0),
        "speed": 0.0,
        "heading": 0.0,
        "radius": 20.0,
        "lives": 1,
        "health": 25.0,
    }
    game_state = {
        "map_size": (1000.0, 800.0),
        "frame_rate": 60.0,
        "time": 1.0,
        "asteroids": [
            {"position": (650.0, 400.0), "velocity": (0.0, 20.0), "size": 3, "radius": 50.0, "mass": 200.0},
        ],
    }

    ctrl.get_actions(ship_state, game_state)
    assert ctrl.telemetry.state == AIBehaviorState.SURVIVE
    assert ctrl.telemetry.risk_multiplier >= 1.5


def test_ai_behavior_state_attack_when_safe():
    ctrl = KesslerAIController()

    # Healthy ship facing an asteroid at medium range (distance 300)
    ship_state = {
        "position": (500.0, 400.0),
        "velocity": (0.0, 0.0),
        "speed": 0.0,
        "heading": 0.0,
        "radius": 20.0,
        "lives": 3,
        "health": 100.0,
        "can_fire": True,
        "bullets_remaining": 100,
    }
    game_state = {
        "map_size": (1000.0, 800.0),
        "frame_rate": 60.0,
        "time": 1.0,
        "asteroids": [
            {"position": (800.0, 400.0), "velocity": (0.0, 0.0), "size": 1, "radius": 15.0, "mass": 25.0},
        ],
    }

    ctrl.get_actions(ship_state, game_state)
    assert ctrl.telemetry.state == AIBehaviorState.ATTACK


# =============================================================================
# 8. Scoring & Safe Spawning
# =============================================================================
def test_no_duplicate_scoring_for_same_asteroid():
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0))
    ast = debris_mgr.spawn_asteroid(position=(200.0, 200.0), velocity=(0.0, 0.0), size=2)

    # First award gives full points (100)
    first_pts = debris_mgr.award_points(ast)
    assert first_pts == 100

    # Second award for same asteroid uid yields 0 points
    second_pts = debris_mgr.award_points(ast)
    assert second_pts == 0


def test_safe_asteroid_spawn_respects_clearance():
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0))
    ship_pos = (500.0, 400.0)

    # Spawn 10 asteroids with avoid_pos
    for _ in range(10):
        ast = debris_mgr.spawn_asteroid(avoid_pos=ship_pos, min_distance=220.0)
        dist = math.hypot(ast.x - ship_pos[0], ast.y - ship_pos[1])
        assert dist >= 220.0, f"Asteroid spawned too close to ship: {dist:.1f} < 220.0"


# =============================================================================
# 9. Countdown Timer Expiration (VICTORY) and Zero Health (DEFEAT)
# =============================================================================
def test_countdown_timer_expiration_triggers_victory():
    # Set a short 2.0s countdown timer
    settings = GameSettings(sfx_enabled=False, countdown_duration=2.0)
    app = KesslerGameApp(settings=settings)
    app.start_game(GameMode.AUTONOMOUS)

    assert app.state == GameState.PLAYING
    assert app.countdown_remaining == 2.0

    # Advance 1.0s -> timer is 1.0s
    app._update(1.0)
    assert app.countdown_remaining == pytest.approx(1.0, abs=0.05)
    assert app.state == GameState.PLAYING

    # Advance 1.1s -> countdown reaches 0 -> VICTORY!
    app._update(1.1)
    assert app.countdown_remaining == 0.0
    assert app.state == GameState.VICTORY


def test_zero_health_triggers_immediate_defeat():
    settings = GameSettings(sfx_enabled=False, countdown_duration=180.0)
    app = KesslerGameApp(settings=settings)
    app.start_game(GameMode.AUTONOMOUS)

    assert app.state == GameState.PLAYING
    assert app.ai_ship is not None
    assert app.ai_ship.health == 100.0

    # Deplete health to zero
    app.ai_ship.shields = 0.0
    app.ai_ship.apply_damage(100.0)
    assert app.ai_ship.health == 0.0

    app._update(0.016)
    assert app.state == GameState.DEFEAT


# =============================================================================
# 10. Game State Transitions & Reset
# =============================================================================
def test_game_app_state_transitions():
    app = KesslerGameApp(settings=GameSettings(sfx_enabled=False))

    assert app.state == GameState.MENU

    app.start_game(GameMode.AUTONOMOUS)
    assert app.state == GameState.PLAYING
    assert app.ai_ship is not None
    assert app.ai_ship.is_alive
    assert app.ai_ship.health == 100.0

    app.set_state(GameState.PAUSED)
    assert app.state == GameState.PAUSED
    app.set_state(GameState.PLAYING)
    assert app.state == GameState.PLAYING

    # Restart
    app.start_game(GameMode.AUTONOMOUS)
    assert app.state == GameState.PLAYING
    assert app.countdown_remaining == 180.0
