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


# =============================================================================
# 11. Boundary Constraints & Screen-Wrapping Removal
# =============================================================================
def test_spaceship_does_not_wrap_around_edges():
    """Verify ship cannot escape arena or wrap across borders."""
    ship = Spaceship(position=(990.0, 400.0), heading=0.0, map_size=(1000.0, 800.0))
    ship.speed = 200.0

    # Advance forward toward right edge (x=1000)
    for _ in range(15):
        ship.update_physics(thrust_input=480.0, turn_input=0.0, dt=0.05)

    # Position must be clamped at map_w - radius (1000 - 20 = 980), NEVER wrapping to 0
    assert ship.x <= 980.0
    assert ship.x >= 970.0
    assert ship.speed == 0.0  # Speed cancelled against wall

    # Test top border (heading 270 deg = straight up toward y=0)
    ship.y = 15.0
    ship.heading = 270.0
    ship.speed = 200.0
    for _ in range(10):
        ship.update_physics(thrust_input=480.0, turn_input=0.0, dt=0.05)

    # Must be clamped at radius (20.0), never wrapping to 800
    assert ship.y >= 20.0
    assert ship.y <= 25.0


def test_asteroid_wall_bounce_no_wrapping():
    """Verify asteroids bounce elastically off perimeter and never wrap across edges."""
    mgr = DebrisManager(map_size=(1000.0, 800.0), wall_bounce=True)
    # Spawn asteroid near right boundary traveling right (+vx)
    ast = mgr.spawn_asteroid(position=(960.0, 400.0), velocity=(120.0, 0.0), size=1)
    orig_vx = ast.vx
    assert orig_vx > 0

    # Update through wall impact
    for _ in range(10):
        ast.update(0.05)

    # Asteroid must have reversed its vx and stayed within [0, 1000]
    assert ast.vx < 0
    assert ast.x <= (1000.0 - ast.radius)
    assert ast.x > 800.0  # Did not wrap to near 0


def test_bullet_despawns_at_boundaries():
    """Verify projectiles leaving playable arena are removed without wrapping."""
    ship = Spaceship(position=(990.0, 400.0), heading=0.0, map_size=(1000.0, 800.0))
    bullet = ship.fire()
    assert bullet is not None
    assert len(ship.bullets) == 1

    # Update physics with bullet passing x=1000
    ship.update_physics(thrust_input=0.0, turn_input=0.0, dt=0.1)

    # Bullet exceeded map_w -> despawned
    assert len(ship.bullets) == 0


# =============================================================================
# 12. Boundary-Aware AI Navigation & Corner Escape
# =============================================================================
def test_ai_corner_escape_maneuver():
    """Verify AI detects entrapment in corner and actively navigates toward center."""
    controller = KesslerAIController(map_size=(1000.0, 800.0))
    ship_dict = {
        "position": (50.0, 50.0),  # Deep top-left corner
        "velocity": (-10.0, -10.0),
        "speed": 14.0,
        "heading": 225.0,  # Facing directly into corner walls
        "radius": 20.0,
        "health": 100.0,
        "shields": 100.0,
        "lives": 3,
        "can_fire": True,
        "thrust_range": (-480.0, 480.0),
        "turn_rate_range": (-180.0, 180.0),
        "max_speed": 240.0,
        "drag": 80.0,
    }
    game_dict = {
        "map_size": (1000.0, 800.0),
        "frame_rate": 60.0,
        "time": 1.0,
        "asteroids": [],
    }

    thrust, turn, fire, boost = controller.get_actions(ship_dict, game_dict)

    # AI must turn toward arena center (45 deg is center from (50, 50))
    assert controller.telemetry.state in (AIBehaviorState.REPOSITION, AIBehaviorState.SURVIVE)
    # The turn must steer toward center
    assert turn != 0.0


def test_ai_wall_avoidance_trajectory():
    """Verify MPC planner rejects or penalizes candidate trajectories exiting arena."""
    from kessler_mpc.planner import MPCPlanner
    from kessler_mpc.config import make_config
    from kessler_mpc.world import World
    from kessler_mpc.dynamics import ShipParams
    from kessler_mpc.threat import assess
    from kessler_mpc.risk import AdaptiveRisk

    cfg = make_config(wrap_boundaries=False)
    planner = MPCPlanner(cfg)

    params = ShipParams(
        thrust_min=-480.0, thrust_max=480.0, turn_max=180.0,
        max_speed=240.0, drag=80.0, radius=20.0,
        map_w=1000.0, map_h=800.0, frame_dt=1.0/60.0
    )
    # Ship at right wall edge facing wall
    world = World(
        params=params, pos=np.array([970.0, 400.0]), vel=np.array([100.0, 0.0]),
        speed=100.0, heading=0.0, lives=3, respawning=False, respawn_left=0.0,
        can_fire=True, bullets_remaining=250, time=0.0,
        ast_pos=np.zeros((0, 2)), ast_vel=np.zeros((0, 2)), ast_size=np.zeros(0),
        ast_rad=np.zeros(0), ast_mass=np.zeros(0), other_pos=np.zeros((0, 2)),
        other_vel=np.zeros((0, 2)), other_rad=np.zeros(0), bul_pos=np.zeros((0, 2)),
        bul_vel=np.zeros((0, 2)), bul_mass=np.zeros(0), random_splits=False,
        health=100.0, shields=100.0,
    )
    th = assess(world, cfg)
    risk = AdaptiveRisk(cfg).update(world.lives, world.health)
    seqs = np.zeros((1, 15, 2), dtype=np.int8)
    seqs[0, :, 0] = 2  # Full forward thrust toward wall

    ev = planner.evaluate(world, th, None, risk, seqs)
    # Forward candidate leaves arena bounds -> collides=True
    assert ev["collides"][0]


# =============================================================================
# 13. Settings Persistence & Game Parameter Customization
# =============================================================================
def test_settings_save_and_load_persistence(tmp_path):
    """Verify settings can be modified, saved to JSON, and reloaded accurately."""
    test_file = str(tmp_path / "custom_settings.json")
    s = GameSettings()
    s.ship_max_speed = 360.0
    s.ship_max_health = 250.0
    s.ship_starting_lives = 5
    s.initial_asteroids = 12
    s.asteroid_split_enabled = False
    s.countdown_duration = 240.0

    assert s.save_to_file(test_file)

    loaded = GameSettings.load_from_file(test_file)
    assert loaded.ship_max_speed == 360.0
    assert loaded.ship_max_health == 250.0
    assert loaded.ship_starting_lives == 5
    assert loaded.initial_asteroids == 12
    assert loaded.asteroid_split_enabled is False
    assert loaded.countdown_duration == 240.0


def test_settings_validation_and_clamping():
    """Verify safe limits are enforced on invalid or extreme numbers."""
    s = GameSettings()
    s.ship_max_speed = -50.0
    s.ship_max_health = 99999.0
    s.ship_starting_lives = 0
    s.countdown_duration = -10.0

    s.validate_and_clamp()
    assert s.ship_max_speed == 80.0
    assert s.ship_max_health == 500.0
    assert s.ship_starting_lives == 1
    assert s.countdown_duration == 10.0


def test_settings_affect_spaceship_gameplay():
    """Verify custom settings affect actual spaceship creation and physics."""
    custom = GameSettings(
        ship_max_speed=320.0,
        ship_thrust_max=650.0,
        ship_turn_max=240.0,
        ship_max_health=175.0,
        ship_starting_lives=4,
        bullet_speed=950.0,
        bullet_damage=80.0,
    )
    ship = Spaceship(settings=custom)

    assert ship.params.max_speed == 320.0
    assert ship.params.thrust_max == 650.0
    assert ship.params.turn_max == 240.0
    assert ship.max_health == 175.0
    assert ship.health == 175.0
    assert ship.lives == 4
    assert ship.bullet_speed == 950.0
    assert ship.bullet_damage == 80.0


def test_settings_split_disabled():
    """Verify disabling asteroid splitting prevents fragmentation on destruction."""
    mgr = DebrisManager(map_size=(1000.0, 800.0), split_enabled=False)
    ast = mgr.spawn_asteroid(position=(500.0, 400.0), velocity=(0.0, 0.0), size=3)
    assert len(mgr.asteroids) == 1

    children = mgr.split_asteroid(ast)
    assert len(children) == 0
    assert len(mgr.asteroids) == 0  # Destroyed without splitting


def test_settings_menu_ui_interactions():
    """Verify SettingsMenuUI correctly synchronizes and applies changes."""
    settings = GameSettings()
    from kessler_mpc.ui import SettingsMenuUI

    ui = SettingsMenuUI(settings)
    assert ui.current_tab == "SPACESHIP"

    # Switch tab to ASTEROIDS
    ui.set_tab("ASTEROIDS")
    assert ui.current_tab == "ASTEROIDS"

    # Modify slider value and apply
    ui.asteroid_sliders["initial_asteroids"].set_value(11.0)
    ui.apply_to_settings()

    assert settings.initial_asteroids == 11
    assert "CONFIG SAVED" in ui.status_msg


def test_difficulty_presets_and_progression():
    """Verify all 5 difficulty presets (CADET, PILOT, VETERAN, ELITE, NIGHTMARE) configure escalating challenge."""
    from kessler_mpc.settings import (
        DifficultyLevel,
        DIFFICULTY_DESCRIPTIONS,
        DIFFICULTY_PRESETS,
    )

    levels = list(DifficultyLevel)
    assert [d.name for d in levels] == ["CADET", "PILOT", "VETERAN", "ELITE", "NIGHTMARE"]
    assert [d.value for d in levels] == ["Cadet", "Pilot", "Veteran", "Elite", "Nightmare"]

    # Verify descriptions exist
    for lvl in levels:
        assert lvl in DIFFICULTY_DESCRIPTIONS
        assert len(DIFFICULTY_DESCRIPTIONS[lvl]) > 0

    # Verify escalating difficulty parameters
    settings = GameSettings()
    prev_speed = 0.0
    prev_asteroids = 0

    for lvl in levels:
        settings.apply_difficulty_preset(lvl)
        assert settings.difficulty == lvl
        assert settings.asteroid_speed_mult > prev_speed
        assert settings.initial_asteroids >= prev_asteroids
        prev_speed = settings.asteroid_speed_mult
        prev_asteroids = settings.initial_asteroids

    # Test legacy aliases
    assert DifficultyLevel.CASUAL == DifficultyLevel.CADET
    assert DifficultyLevel.STANDARD == DifficultyLevel.PILOT
    assert DifficultyLevel.HARDCORE == DifficultyLevel.VETERAN
    assert DifficultyLevel.KESSLER_SYNDROME == DifficultyLevel.NIGHTMARE

    # Test from_str and from_dict legacy mappings
    assert DifficultyLevel.from_str("casual") == DifficultyLevel.CADET
    assert DifficultyLevel.from_str("standard") == DifficultyLevel.PILOT
    assert DifficultyLevel.from_str("hardcore") == DifficultyLevel.VETERAN
    assert DifficultyLevel.from_str("kessler") == DifficultyLevel.NIGHTMARE
    assert DifficultyLevel.from_str("kessler_syndrome") == DifficultyLevel.NIGHTMARE


def test_shatterspace_game_branding_and_menu_labels():
    """Verify game branding and exact main menu button labels."""
    from kessler_mpc.settings import GAME_TITLE, GAME_SUBTITLE, GAME_TAGLINE, DifficultyLevel

    assert GAME_TITLE == "SHATTERSPACE"
    assert GAME_SUBTITLE == "DEEP SPACE SURVIVAL"
    assert GAME_TAGLINE == "One ship. Endless asteroids. Stay alive."

    settings = GameSettings(difficulty=DifficultyLevel.CADET)
    app = KesslerGameApp(settings=settings)

    button_texts = [btn.text for btn in app.main_menu_buttons]
    assert button_texts == [
        "START GAME",
        "MANUAL PILOT",
        "PLAYER + AI",
        "DIFFICULTY: CADET",
        "GAME SETTINGS",
        "CONTROLS",
        "QUIT GAME",
    ]

    # Test cycling difficulty from main menu
    app._cycle_difficulty_action()
    assert app.settings.difficulty == DifficultyLevel.PILOT
    assert app.main_menu_buttons[3].text == "DIFFICULTY: PILOT"

    app._cycle_difficulty_action()
    assert app.settings.difficulty == DifficultyLevel.VETERAN
    assert app.main_menu_buttons[3].text == "DIFFICULTY: VETERAN"

    app._cycle_difficulty_action()
    assert app.settings.difficulty == DifficultyLevel.ELITE
    assert app.main_menu_buttons[3].text == "DIFFICULTY: ELITE"

    app._cycle_difficulty_action()
    assert app.settings.difficulty == DifficultyLevel.NIGHTMARE
    assert app.main_menu_buttons[3].text == "DIFFICULTY: NIGHTMARE"

    app._cycle_difficulty_action()
    assert app.settings.difficulty == DifficultyLevel.CADET
    assert app.main_menu_buttons[3].text == "DIFFICULTY: CADET"


def test_game_app_step_execution():
    """Verify app.step() executes a single frame, updates physics/AI, and respects termination."""
    settings = GameSettings(screen_width=1000, screen_height=800, countdown_duration=180.0)
    app = KesslerGameApp(settings=settings)
    app.start_game(GameMode.AUTONOMOUS)

    assert app.is_running is True
    # Step single frame
    running = app.step(0.016)
    assert running is True
    assert app.ai_ship is not None
    assert app.countdown_remaining < 180.0

    # Test quit termination
    app.quit()
    assert app.is_running is False
    assert app.step(0.016) is False


def test_game_app_run_async_loop():
    """Verify asynchronous game loop execution yields control every frame and runs AI without freezing."""
    settings = GameSettings(screen_width=1000, screen_height=800, countdown_duration=180.0)
    app = KesslerGameApp(settings=settings)
    app.start_game(GameMode.AUTONOMOUS)

    coro = app.run_async()
    frame_yields = 0

    # Advance 15 frames by driving the async coroutine (verifies each frame yields via asyncio.sleep(0))
    try:
        for _ in range(15):
            coro.send(None)
            frame_yields += 1
            assert app.is_running is True
    except StopIteration:
        pass

    assert frame_yields == 15
    assert app.is_running is True
    # Verify AI controller updated telemetry during async loop
    assert app.ai_controller.telemetry is not None
    assert app.ai_controller.telemetry.state is not None

    # Clean termination
    app.quit()
    try:
        coro.send(None)
    except StopIteration:
        pass
    coro.close()


def test_main_async_entry_point_coroutine():
    """Verify main.main() is an async coroutine function required for Pygbag WebAssembly deployment."""
    import inspect
    import main

    assert inspect.iscoroutinefunction(main.main), "main.main must be an async coroutine function for Pygbag"
    assert hasattr(main, "main_sync"), "main.py must provide main_sync for desktop compatibility"


