"""Comprehensive unit and integration tests for Kessler AI advanced features:
1. Spaceship Health & Lives System (damage by asteroid size, invulnerability, respawn, health regen).
2. Asteroid Health & Fragmentation System (Small 1 HP, Medium 2 HP, Large 3 HP, fragment count).
3. Special Red Penalty Asteroids (spawning, max active, opposite-edge exit, shooting penalty, collision penalty, AI exclusion & avoidance).
4. AI Target Selection, Threat Assessment, MPC Trajectory, and Adaptive Risk Behavior.
5. Integration & Settings Persistence.
"""
import math
import os
import numpy as np
import pytest

from kessler_mpc.debris import Asteroid, DebrisManager, RedPenaltyAsteroid
from kessler_mpc.geometry import lead_intercept
from kessler_mpc.kessler_ai import AIBehaviorState, KesslerAIController
from kessler_mpc.physics import CollisionEvent, PhysicsEngine
from kessler_mpc.risk import AdaptiveRisk
from kessler_mpc.settings import DifficultyLevel, GameSettings
from kessler_mpc.spaceship import Bullet, Spaceship
from kessler_mpc.targeting import TargetSelector
from kessler_mpc.threat import assess
from kessler_mpc.world import World, read_world


# =============================================================================
# 1. SPACESHIP HEALTH AND LIVES TESTS
# =============================================================================
def test_large_asteroid_collision_damage():
    """Large asteroid (size 3) collision inflicts 40 HP damage."""
    settings = GameSettings(damage_large_asteroid=40.0)
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)
    ship = Spaceship(position=(500.0, 400.0), settings=settings)
    assert ship.health == 100.0

    ast = debris_mgr.spawn_asteroid(position=(505.0, 400.0), size=3)
    events = physics.check_ship_asteroid_collisions([ship], debris_mgr)

    assert len(events) == 1
    assert events[0].event_type == "ship_asteroid"
    assert ship.health == 60.0  # 100 - 40
    assert ship.is_invulnerable
    assert ship.collision_invuln_timer > 0.0


def test_medium_asteroid_collision_damage():
    """Medium asteroid (size 2) collision inflicts 25 HP damage."""
    settings = GameSettings(damage_medium_asteroid=25.0)
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)
    ship = Spaceship(position=(500.0, 400.0), settings=settings)

    ast = debris_mgr.spawn_asteroid(position=(505.0, 400.0), size=2)
    events = physics.check_ship_asteroid_collisions([ship], debris_mgr)

    assert len(events) == 1
    assert ship.health == 75.0  # 100 - 25


def test_small_asteroid_collision_damage():
    """Small asteroid (size 1) collision inflicts 10 HP damage."""
    settings = GameSettings(damage_small_asteroid=10.0)
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)
    ship = Spaceship(position=(500.0, 400.0), settings=settings)

    ast = debris_mgr.spawn_asteroid(position=(505.0, 400.0), size=1)
    events = physics.check_ship_asteroid_collisions([ship], debris_mgr)

    assert len(events) == 1
    assert ship.health == 90.0  # 100 - 10


def test_multiple_collisions_during_invulnerability():
    """Collisions occurring during collision invulnerability do not deduct health repeatedly."""
    settings = GameSettings(normal_collision_invulnerability=1.5)
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)
    ship = Spaceship(position=(500.0, 400.0), settings=settings)

    # First impact
    ast1 = debris_mgr.spawn_asteroid(position=(505.0, 400.0), size=2)
    physics.check_ship_asteroid_collisions([ship], debris_mgr)
    initial_health = ship.health
    assert ship.is_invulnerable

    # Second asteroid hits immediately while still invulnerable
    ast2 = debris_mgr.spawn_asteroid(position=(ship.x + 2.0, ship.y), size=3)
    events = physics.check_ship_asteroid_collisions([ship], debris_mgr)
    assert len(events) == 0  # Ignored due to invulnerability
    assert ship.health == initial_health  # No extra damage taken


def test_ship_death_and_respawn():
    """When health reaches 0, lives decrement by 1 and ship respawns with full HP & invulnerability."""
    settings = GameSettings(respawn_invulnerability=3.0)
    ship = Spaceship(position=(300.0, 300.0), settings=settings)
    assert ship.lives == 3

    # Fatal blow
    died = ship.apply_damage(120.0)
    assert died
    assert ship.lives == 2
    assert ship.is_alive

    ship.respawn()
    assert ship.health == 100.0
    assert ship.is_respawning
    assert ship.is_invulnerable
    assert ship.respawn_timer == 3.0


def test_ship_game_over_when_lives_exhausted():
    """When the ship runs out of lives, it dies permanently."""
    ship = Spaceship(position=(300.0, 300.0), lives=1)
    died = ship.apply_damage(100.0)
    assert died
    assert ship.lives == 0
    assert not ship.is_alive


def test_passive_health_regeneration():
    """Passive health regeneration gradually restores ship hull when enabled."""
    settings = GameSettings(health_regen_enabled=True, health_regen_rate=10.0)
    ship = Spaceship(position=(500.0, 400.0), settings=settings)
    ship.health = 50.0

    # Advance physics 1.0 second
    ship.update_physics(thrust_input=0.0, turn_input=0.0, dt=1.0)
    assert ship.health == pytest.approx(60.0, abs=0.1)


# =============================================================================
# 2. ASTEROID HEALTH AND FRAGMENTATION TESTS
# =============================================================================
def test_small_asteroid_destroyed_after_one_hit():
    """Small asteroid (1 HP) is destroyed after 1 hit and produces 0 fragments."""
    settings = GameSettings(bullet_damage=1.0, asteroid_health_small=1.0)
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)
    ship = Spaceship(position=(200.0, 200.0), settings=settings)

    ast = debris_mgr.spawn_asteroid(position=(250.0, 200.0), velocity=(0.0, 0.0), size=1)
    assert ast.health == 1.0

    bullet = ship.fire()
    assert bullet is not None
    bullet.x = 245.0
    bullet.y = 200.0

    events = physics.check_bullet_asteroid_collisions([ship], debris_mgr)
    assert len(events) == 1
    assert len(debris_mgr.asteroids) == 0  # Destroyed
    assert ship.asteroids_destroyed == 1


def test_medium_asteroid_fragmentation_after_two_hits():
    """Medium asteroid (2 HP) survives 1 hit, and splits after the 2nd hit."""
    settings = GameSettings(bullet_damage=1.0, asteroid_health_medium=2.0, fragment_count_min=2, fragment_count_max=2)
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)
    ship = Spaceship(position=(200.0, 200.0), settings=settings)

    ast = debris_mgr.spawn_asteroid(position=(250.0, 200.0), velocity=(0.0, 0.0), size=2)
    assert ast.health == 2.0

    # First hit: health decreases to 1.0, not destroyed
    bullet1 = ship.fire()
    bullet1.x = 245.0
    bullet1.y = 200.0
    events1 = physics.check_bullet_asteroid_collisions([ship], debris_mgr)
    assert len(events1) == 1
    assert ast.health == 1.0
    assert len(debris_mgr.asteroids) == 1

    # Second hit: health reaches 0, splits into 2 small fragments
    ship.fire_cooldown = 0.0
    bullet2 = ship.fire()
    bullet2.x = 245.0
    bullet2.y = 200.0
    events2 = physics.check_bullet_asteroid_collisions([ship], debris_mgr)
    assert len(events2) == 1
    assert len(debris_mgr.asteroids) == 2  # 2 small children
    assert all(a.size == 1 for a in debris_mgr.asteroids)


def test_fragment_count_configuration():
    """Fragment count respects min and max configuration."""
    settings2 = GameSettings(fragment_count_min=2, fragment_count_max=2)
    mgr2 = DebrisManager(map_size=(1000.0, 800.0), settings=settings2)
    ast2 = mgr2.spawn_asteroid(position=(300.0, 300.0), size=3)
    children2 = mgr2.split_asteroid(ast2)
    assert len(children2) == 2

    settings3 = GameSettings(fragment_count_min=3, fragment_count_max=3)
    mgr3 = DebrisManager(map_size=(1000.0, 800.0), settings=settings3)
    ast3 = mgr3.spawn_asteroid(position=(300.0, 300.0), size=3)
    children3 = mgr3.split_asteroid(ast3)
    assert len(children3) == 3


# =============================================================================
# 3. SPECIAL RED PENALTY ASTEROID TESTS
# =============================================================================
def test_red_asteroid_spawning_and_visual_properties():
    """Red penalty asteroid spawns with distinct red coloring, radius, and speed."""
    settings = GameSettings(red_asteroid_speed=120.0, red_asteroid_radius=25.0)
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)

    red = debris_mgr.spawn_red_asteroid(entry_edge=0)  # Top edge
    assert red is not None
    assert red.is_penalty
    assert red.radius == 25.0
    assert red.speed == pytest.approx(120.0, rel=1e-3)
    assert red.base_color == (235, 25, 25)
    assert len(debris_mgr.red_asteroids) == 1


def test_red_asteroid_max_active_enforcement():
    """DebrisManager enforces maximum active red asteroids."""
    settings = GameSettings(red_asteroid_max_active=1)
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)

    red1 = debris_mgr.spawn_red_asteroid()
    assert red1 is not None
    assert len(debris_mgr.red_asteroids) == 1

    # Second spawn attempt must be rejected because max active is 1
    red2 = debris_mgr.spawn_red_asteroid()
    assert red2 is None
    assert len(debris_mgr.red_asteroids) == 1


def test_red_asteroid_opposite_edge_exit():
    """Red asteroids travel toward opposite edge and despawn once out of bounds."""
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0))
    # Place a red asteroid near the bottom edge moving downward
    red = RedPenaltyAsteroid(
        position=(500.0, 790.0),
        velocity=(0.0, 100.0),
        map_size=(1000.0, 800.0),
    )
    debris_mgr.asteroids.append(red)
    assert len(debris_mgr.red_asteroids) == 1

    # Advance 1.0 second: moves to y = 890 (past H + pad)
    debris_mgr.update(1.0)
    assert red.is_out_of_bounds()
    assert len(debris_mgr.red_asteroids) == 0  # Cleanly pruned


def test_bullet_hit_red_asteroid_penalty_and_no_fragments():
    """CRITICAL: Shooting a red asteroid removes it immediately, applies -5 penalty once, and creates NO fragments."""
    settings = GameSettings(red_asteroid_shooting_penalty=5)
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)
    ship = Spaceship(position=(300.0, 300.0), settings=settings)
    ship.score = 50

    red = debris_mgr.spawn_red_asteroid(entry_edge=0)
    red.x = 400.0
    red.y = 300.0

    bullet = ship.fire()
    assert bullet is not None
    bullet.x = 395.0
    bullet.y = 300.0

    events = physics.check_bullet_asteroid_collisions([ship], debris_mgr)
    assert len(events) == 1
    assert events[0].event_type == "bullet_red_asteroid"
    assert events[0].penalty == 5

    # Score deducted by 5
    assert ship.score == 45
    # Red asteroid removed immediately
    assert len(debris_mgr.asteroids) == 0
    # No kills or asteroid destruction points awarded
    assert ship.asteroids_destroyed == 0


def test_ship_collision_with_red_asteroid():
    """Ship colliding with red asteroid deducts 10 penalty once, removes it, and does not deduct ship health."""
    settings = GameSettings(red_asteroid_collision_penalty=10)
    physics = PhysicsEngine(map_size=(1000.0, 800.0))
    debris_mgr = DebrisManager(map_size=(1000.0, 800.0), settings=settings)
    ship = Spaceship(position=(500.0, 400.0), settings=settings)
    ship.score = 30
    initial_health = ship.health
    initial_lives = ship.lives

    red = RedPenaltyAsteroid(
        position=(505.0, 400.0),
        velocity=(0.0, 0.0),
        map_size=(1000.0, 800.0),
    )
    debris_mgr.asteroids.append(red)

    events = physics.check_ship_asteroid_collisions([ship], debris_mgr)
    assert len(events) == 1
    assert events[0].event_type == "ship_red_asteroid"
    assert events[0].penalty == 10

    # Score deducted by 10
    assert ship.score == 20
    # Red asteroid removed immediately
    assert len(debris_mgr.asteroids) == 0
    # Ship health and lives preserved
    assert ship.health == initial_health
    assert ship.lives == initial_lives


# =============================================================================
# 4. AI TARGET SELECTION AND AVOIDANCE TESTS
# =============================================================================
def test_ai_target_selector_never_targets_red_asteroid():
    """AI TargetSelector strictly excludes red penalty asteroids from target selection."""
    ai = KesslerAIController()
    selector = ai.selector

    # Create world with 1 normal asteroid and 1 red penalty asteroid
    ship_state = {
        "id": 0,
        "position": (500.0, 400.0),
        "velocity": (0.0, 0.0),
        "speed": 0.0,
        "heading": 0.0,
        "lives": 3,
        "health": 100.0,
        "shields": 100.0,
        "is_respawning": False,
        "can_fire": True,
        "bullets_remaining": 100,
    }
    game_state = {
        "map_size": (1000.0, 800.0),
        "frame_rate": 60.0,
        "time": 10.0,
        "asteroids": [
            {"position": (550.0, 400.0), "velocity": (-20.0, 0.0), "size": 2, "radius": 30.0, "mass": 75.0, "is_penalty": True},
            {"position": (650.0, 400.0), "velocity": (0.0, 0.0), "size": 1, "radius": 15.0, "mass": 25.0, "is_penalty": False},
        ],
    }

    world = ai._build_world(ship_state, game_state)
    raw_threats, _ = ai.threat_eval.assess_threats(world)
    target = selector.select(world, raw_threats)

    assert target is not None
    # Must target normal asteroid at index 1, NEVER the red asteroid at index 0
    assert target.idx == 1


def test_ai_fire_control_avoids_shooting_red_asteroid():
    """FireControl rejects firing if the bullet trajectory would hit a red penalty asteroid."""
    ai = KesslerAIController()
    # Red asteroid right in front of ship
    ship_state = {
        "id": 0,
        "position": (500.0, 400.0),
        "velocity": (0.0, 0.0),
        "speed": 0.0,
        "heading": 0.0,  # facing right
        "lives": 3,
        "health": 100.0,
        "shields": 100.0,
        "is_respawning": False,
        "can_fire": True,
        "bullets_remaining": 100,
    }
    game_state = {
        "map_size": (1000.0, 800.0),
        "frame_rate": 60.0,
        "time": 5.0,
        "asteroids": [
            {"position": (580.0, 400.0), "velocity": (0.0, 0.0), "size": 2, "radius": 25.0, "mass": 80.0, "is_penalty": True},
        ],
    }
    world = ai._build_world(ship_state, game_state)
    shot = ai.fire._shot(world, thrust=0.0, turn=0.0, aim_fraction=0.9)
    # Must return None so weapon will not fire at red hazard
    assert shot is None


def test_ai_predictive_aiming_lead_intercept():
    """Predictive aiming accurately calculates intercept time and leading point."""
    p_ship = np.array([100.0, 100.0])
    p_ast = np.array([500.0, 100.0])
    v_ast = np.array([0.0, 120.0])
    v_bullet = 800.0

    t_hit = lead_intercept(p_ship, p_ast, v_ast, v_bullet)
    assert np.isfinite(t_hit)
    assert t_hit > 0.0

    aim_pos = p_ast + v_ast * t_hit
    bullet_dist = np.linalg.norm(aim_pos - p_ship)
    assert bullet_dist == pytest.approx(v_bullet * t_hit, rel=1e-4)


def test_adaptive_risk_model_scaling():
    """Risk multiplier rho(q) scales inverse-proportionally with health and lives."""
    risk = AdaptiveRisk({"adaptive_risk": True, "initial_lives": 3, "lives_fraction_tolerance": 0.02,
                         "low_lives_fraction": 0.33, "mid_lives_fraction": 0.66,
                         "low_lives_risk_multiplier": 2.0, "aim_radius_fraction": (0.9, 0.65, 0.45)})

    # Full health & lives: level 0, mult 1.0
    s0 = risk.update(lives=3, health=100.0)
    assert s0.level == 0
    assert s0.multiplier == 1.0

    # Low health (25 HP): level 2, mult 2.0
    s2 = risk.update(lives=3, health=25.0)
    assert s2.level == 2
    assert s2.multiplier == 2.0


# =============================================================================
# 5. INTEGRATION AND SETTINGS PERSISTENCE TESTS
# =============================================================================
def test_all_settings_persist_and_load(tmp_path):
    """Test full configuration serialization, validation clamping, and JSON persistence."""
    test_file = str(tmp_path / "test_settings.json")
    custom = GameSettings(
        ship_max_speed=320.0,
        ship_thrust_max=640.0,
        ship_turn_max=220.0,
        ship_max_health=160.0,
        ship_starting_lives=4,
        damage_large_asteroid=45.0,
        damage_medium_asteroid=30.0,
        damage_small_asteroid=12.0,
        normal_collision_invulnerability=1.8,
        respawn_invulnerability=3.5,
        health_regen_enabled=True,
        health_regen_rate=8.0,
        asteroid_health_small=1.0,
        asteroid_health_medium=2.0,
        asteroid_health_large=3.0,
        fragment_count_min=2,
        fragment_count_max=3,
        bullet_damage=1.0,
        red_asteroid_enabled=True,
        red_asteroid_max_active=2,
        red_asteroid_spawn_probability=0.08,
        red_asteroid_spawn_interval=6.0,
        red_asteroid_speed=110.0,
        red_asteroid_radius=28.0,
        red_asteroid_shooting_penalty=8,
        red_asteroid_collision_penalty=15,
        threat_weight_risk=0.55,
        threat_weight_urgency=0.30,
        threat_weight_geometry=0.15,
        target_switch_threshold=0.20,
    )
    saved = custom.save_to_file(test_file)
    assert saved

    loaded = GameSettings.load_from_file(test_file)
    assert loaded.ship_max_speed == 320.0
    assert loaded.ship_max_health == 160.0
    assert loaded.ship_starting_lives == 4
    assert loaded.damage_large_asteroid == 45.0
    assert loaded.damage_medium_asteroid == 30.0
    assert loaded.normal_collision_invulnerability == 1.8
    assert loaded.health_regen_enabled is True
    assert loaded.red_asteroid_max_active == 2
    assert loaded.red_asteroid_shooting_penalty == 8
    assert loaded.red_asteroid_collision_penalty == 15
    assert loaded.threat_weight_risk == 0.55
    assert loaded.target_switch_threshold == 0.20


def test_ship_design_configurations_and_drawing():
    """Verify all 3 sci-fi ship designs can be initialized and rendered onto a surface."""
    import pygame
    pygame.init()
    surf = pygame.Surface((200, 200))

    for design in ["Interceptor", "Scout", "Vanguard"]:
        settings = GameSettings(ship_design=design)
        ship = Spaceship(position=(100.0, 100.0), settings=settings, ship_design=design)
        assert ship.ship_design == design
        # Draw without raising exceptions
        ship.draw(surf)
        # Advance physics and draw again
        ship.update_physics(100.0, 30.0, 0.016)
        ship.draw(surf)


def test_settings_ship_design_persistence(tmp_path):
    """Verify ship design parameter persists to settings.json."""
    test_file = str(tmp_path / "settings_design.json")
    s = GameSettings(ship_design="Scout")
    assert s.save_to_file(test_file)

    loaded = GameSettings.load_from_file(test_file)
    assert loaded.ship_design == "Scout"
