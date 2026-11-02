"""Kessler AI: The Playable Space Survival Game.

Main entry point. Launches the interactive Pygame application.
Supports player survival, autonomous Kessler AI demo mode, endless challenge, and co-op.
"""
from __future__ import annotations

import argparse
import sys

from kessler_mpc.game import GameMode, KesslerGameApp
from kessler_mpc.settings import DifficultyLevel, GameSettings


def parse_args():
    parser = argparse.ArgumentParser(description="Kessler AI: Orbital Survival Game")
    parser.add_argument(
        "--mode",
        choices=["player", "ai", "survival", "coop"],
        default="player",
        help="Initial game mode to launch into directly (player, ai, survival, coop)",
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
        show_debug_overlay=not args.no_overlay,
    )

    app = KesslerGameApp(settings=settings)

    # If launched directly into a specific mode, start right away
    mode_map = {
        "player": GameMode.PLAYER,
        "ai": GameMode.AI_DEMO,
        "survival": GameMode.SURVIVAL,
        "coop": GameMode.COOP,
    }
    if args.mode in mode_map:
        app.start_game(mode_map[args.mode])

    app.run()


if __name__ == "__main__":
    main()
