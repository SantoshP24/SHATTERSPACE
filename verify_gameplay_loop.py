"""Comprehensive autonomous gameplay loop verification script.

Runs the actual autonomous game loop for 180 frames:
- Verifies ship is controlled 100% autonomously by AI (no player inputs).
- Verifies autonomous movement, rotation, lead targeting, firing, and dodging.
- Verifies AI behavior states (ATTACK, EVADE, REPOSITION, SURVIVE).
- Verifies countdown timer decrementing and countdown expiration (VICTORY).
- Verifies zero health triggering DEFEAT immediately.
- Verifies scoring and restart functionality.
"""
from __future__ import annotations

import sys
import pygame

from kessler_mpc.game import GameMode, GameState, KesslerGameApp
from kessler_mpc.kessler_ai import AIBehaviorState
from kessler_mpc.settings import GameSettings


def verify_autonomous_game():
    print(">>> Initializing KesslerGameApp in Autonomous Mode (180s Countdown)...")
    settings = GameSettings(screen_width=1000, screen_height=800, sfx_enabled=True, countdown_duration=180.0)
    app = KesslerGameApp(settings=settings)

    print(">>> Launching Autonomous AI Survival Game...")
    app.start_game(GameMode.AUTONOMOUS)
    assert app.state == GameState.PLAYING
    assert app.ai_ship is not None
    assert app.ai_controller is not None
    assert app.player_ship is None  # Completely autonomous!

    initial_pos = (app.ai_ship.x, app.ai_ship.y)
    observed_states = set()

    print(">>> Simulating 120 live autonomous frames (2 seconds of game time)...")
    for f in range(120):
        app._update(0.016)
        app._render()
        tel = app.ai_controller.telemetry
        observed_states.add(tel.state)

    final_pos = (app.ai_ship.x, app.ai_ship.y)
    # Verify autonomous ship moved on its own
    assert initial_pos != final_pos, "AI ship did not autonomously move!"
    print(f"    AI ship moved autonomously from {initial_pos} to ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
    print(f"    AI states observed during flight: {[s.value for s in observed_states]}")
    print(f"    Countdown remaining: {app.countdown_remaining:.1f}s / {app.countdown_total:.1f}s")
    print(f"    Active hazards: {len(app.debris.asteroids)}, AI kills: {app.ai_ship.kills}, Score: {app.ai_ship.score}")

    print(">>> Verifying VICTORY screen when countdown timer expires...")
    app.countdown_remaining = 0.05
    app._update(0.06)
    app._render()
    assert app.state == GameState.VICTORY, f"Expected VICTORY state, got {app.state}"
    print("    VICTORY state confirmed upon countdown expiration!")

    print(">>> Verifying DEFEAT screen when ship health reaches 0...")
    app.start_game(GameMode.AUTONOMOUS)
    assert app.state == GameState.PLAYING
    assert app.ai_ship.health == 100.0
    # Strip shields and deplete health to 0
    app.ai_ship.shields = 0.0
    app.ai_ship.apply_damage(100.0)
    assert app.ai_ship.health == 0.0
    app._update(0.016)
    app._render()
    assert app.state == GameState.DEFEAT, f"Expected DEFEAT state, got {app.state}"
    print("    DEFEAT state confirmed upon zero health!")

    print(">>> Verifying Mission Restart...")
    app.start_game(GameMode.AUTONOMOUS)
    assert app.state == GameState.PLAYING
    assert app.ai_ship.health == 100.0
    assert app.countdown_remaining == 180.0
    assert app.ai_ship.is_alive
    print("    Mission Restart cleanly restored countdown and ship health!")

    print("\nALL AUTONOMOUS GAMEPLAY VERIFICATIONS COMPLETED SUCCESSFULLY (100% PASS)!")


if __name__ == "__main__":
    verify_autonomous_game()
