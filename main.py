"""Kessler AI: The Playable Autonomous Space Survival Game.

Main entry point. Launches the interactive Pygame application.
Features fully autonomous spaceship control with MPC trajectory planning,
A* route generation, threat ranking, countdown timer survival, and emergency collision override.
"""
from __future__ import annotations

import argparse
import sys

from kessler_mpc.game import GameMode, KesslerGameApp
from kessler_mpc.settings import DifficultyLevel, GameSettings


def parse_args():
    parser = argparse.ArgumentParser(description="Kessler AI: Autonomous Asteroid Survival Game")
    parser.add_argument(
        "--mode",
        choices=["ai", "player", "coop"],
        default="ai",
        help="Initial game mode (ai: Autonomous Survival [default], player: Manual Pilot, coop: Dual Mode)",
    )
    parser.add_argument(
        "--timer",
        type=float,
        default=180.0,
        help="Countdown survival timer in seconds (default: 180.0)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["casual", "standard", "hardcore", "kessler"],
        default="standard",
        help="Difficulty preset level",
    )
    parser.add_argument("--width", type=int, default=1000, help="Window display width (default: 1000)")
    parser.add_argument("--height", type=int, default=800, help="Window display height (default: 800)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    parser.add_argument("--no-sound", action="store_true", help="Mute procedural audio effects")
    parser.add_argument("--no-overlay", action="store_true", help="Start with AI telemetry overlay hidden")
    return parser.parse_args()


def main():
    args = parse_args()

    diff_map = {
        "casual": DifficultyLevel.CASUAL,
        "standard": DifficultyLevel.STANDARD,
        "hardcore": DifficultyLevel.HARDCORE,
        "kessler": DifficultyLevel.KESSLER_SYNDROME,
    }

    settings = GameSettings(
        screen_width=args.width,
        screen_height=args.height,
        fullscreen=args.fullscreen,
        sfx_enabled=not args.no_sound,
        difficulty=diff_map.get(args.difficulty, DifficultyLevel.STANDARD),
        countdown_duration=args.timer,
        show_debug_overlay=not args.no_overlay,
    )

    app = KesslerGameApp(settings=settings)

    # Launch into selected mode (defaulting to Autonomous AI)
    mode_map = {
        "ai": GameMode.AUTONOMOUS,
        "player": GameMode.PLAYER,
        "coop": GameMode.COOP,
    }
    app.start_game(mode_map.get(args.mode, GameMode.AUTONOMOUS))

    app.run()


if __name__ == "__main__":
    main()
