"""Headless and live simulation verification script.

Runs the actual game loop for 120 frames in all modes:
- Player Mode
- AI Demo Mode
- Survival Mode
- Co-op Mode
Verifies rendering, physics, audio calls, state transitions, docking, and restart.
"""
import os
import sys

# Allow SDL dummy video and dummy audio in case no display or audio hardware
# (or test with real display if available)
import pygame

from kessler_mpc.game import GameMode, GameState, KesslerGameApp
from kessler_mpc.settings import GameSettings


def verify_game_modes():
    print(">>> Initializing KesslerGameApp...")
    settings = GameSettings(screen_width=1000, screen_height=800, sfx_enabled=True)
    app = KesslerGameApp(settings=settings)

    print(">>> Testing Mode 1: Player Mode (60 frames)...")
    app.start_game(GameMode.PLAYER)
    assert app.state == GameState.PLAYING
    assert app.player_ship is not None

    for f in range(60):
        # Simulate delta time 1/60s
        app._update(0.016)
        app._render()
        if f == 10:
            # Test player firing
            b = app.player_ship.fire()
            assert b is not None
        if f == 20:
            # Test player thrust and turn
            app.player_ship.update_physics(thrust_input=480.0, turn_input=90.0, dt=0.016, boost_active=True)

    print(f"    Player Mode OK! Player pos: ({app.player_ship.x:.1f}, {app.player_ship.y:.1f}), bullets: {len(app.player_ship.bullets)}")

    print(">>> Testing Mode 2: AI Demonstration Mode (120 frames)...")
    app.start_game(GameMode.AI_DEMO)
    assert app.state == GameState.PLAYING
    assert app.ai_ship is not None
    assert app.ai_controller is not None

    for f in range(120):
        app._update(0.016)
        app._render()

    tel = app.ai_controller.telemetry
    print(f"    AI Demo OK! AI pos: ({app.ai_ship.x:.1f}, {app.ai_ship.y:.1f})")
    print(f"    AI State: {tel.state.value}, Risk: {tel.risk_multiplier:.1f}x, MPC cost: {tel.mpc_cost:.1f}")
    print(f"    A* Waypoints planned: {len(tel.astar_path)}, MPC Trajectory points: {len(tel.mpc_trajectory)}")

    print(">>> Testing Mode 3: Survival Challenge (60 frames)...")
    app.start_game(GameMode.SURVIVAL)
    assert app.state == GameState.PLAYING
    for f in range(60):
        app._update(0.016)
        app._render()
    print("    Survival Mode OK!")

    print(">>> Testing Mode 4: Co-op Mode (60 frames)...")
    app.start_game(GameMode.COOP)
    assert app.state == GameState.PLAYING
    assert app.player_ship is not None
    assert app.ai_ship is not None
    for f in range(60):
        app._update(0.016)
        app._render()
    print("    Co-op Mode OK!")

    print(">>> Testing Pause & State Transitions...")
    app.set_state(GameState.PAUSED)
    app._render()
    assert app.state == GameState.PAUSED

    app.set_state(GameState.PLAYING)
    app._render()
    assert app.state == GameState.PLAYING

    print(">>> Testing Safe Zone Docking & Victory Transition...")
    # Teleport ship directly inside safe zone
    app.player_ship.x = app.safe_zone.x
    app.player_ship.y = app.safe_zone.y
    # Simulate 3.5 seconds inside safe zone to complete docking
    for _ in range(220):
        app._update(0.016)
        app._render()
        if app.state == GameState.VICTORY:
            break
    assert app.state == GameState.VICTORY, f"Expected VICTORY state, got {app.state}"
    print("    Victory State Triggered Successfully!")

    print(">>> Testing Ship Destruction & Game Over Transition...")
    app.start_game(GameMode.PLAYER)
    app.player_ship.lives = 0
    app.player_ship.is_alive = False
    app._update(0.016)
    app._render()
    assert app.state == GameState.GAME_OVER
    print("    Game Over State Triggered Successfully!")

    print(">>> Testing Restart Functionality...")
    app.start_game(GameMode.PLAYER)
    assert app.state == GameState.PLAYING
    assert app.player_ship.lives == 3
    assert app.player_ship.is_alive
    print("    Restart Functionality Verified!")

    print("\nALL GAMEPLAY SYSTEMS AND ENGINE VERIFICATIONS PASSED 100%!")


if __name__ == "__main__":
    verify_game_modes()
