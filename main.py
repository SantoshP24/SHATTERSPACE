"""SHATTERSPACE: Deep Space Survival.

Main entry point. Launches the interactive Pygame application.
Features fully autonomous spaceship control with MPC trajectory planning,
A* route generation, threat ranking, countdown timer survival, and emergency collision override.
"""
from __future__ import annotations

import argparse
import sys

from kessler_mpc.game import GameMode, KesslerGameApp
from kessler_mpc.settings import (
    GAME_SUBTITLE,
    GAME_TAGLINE,
    GAME_TITLE,
    DifficultyLevel,
    GameSettings,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"{GAME_TITLE}: {GAME_SUBTITLE} - \"{GAME_TAGLINE}\""
    )
    parser.add_argument(
        "--mode",
        choices=["ai", "player", "coop"],
        default="ai",
        help="Initial game mode (ai: Start Game / Autonomous AI [default], player: Manual Pilot, coop: Player + AI)",
    )
    parser.add_argument(
        "--timer",
        type=float,
        default=180.0,
        help="Countdown survival timer in seconds (default: 180.0)",
    )
    parser.add_argument(
        "--difficulty",
        choices=[
            "cadet",
            "pilot",
            "veteran",
            "elite",
            "nightmare",
            "casual",
            "standard",
            "hardcore",
            "kessler",
        ],
        default=None,
        help="Difficulty preset level: cadet, pilot, veteran, elite, nightmare",
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
        "cadet": DifficultyLevel.CADET,
        "pilot": DifficultyLevel.PILOT,
        "veteran": DifficultyLevel.VETERAN,
        "elite": DifficultyLevel.ELITE,
        "nightmare": DifficultyLevel.NIGHTMARE,
        "casual": DifficultyLevel.CASUAL,
        "standard": DifficultyLevel.STANDARD,
        "hardcore": DifficultyLevel.HARDCORE,
        "kessler": DifficultyLevel.KESSLER_SYNDROME,
    }

    # Load persistent user settings as foundation, then apply any explicit CLI flags
    settings = GameSettings.load_from_file()

    if args.width != 1000:
        settings.screen_width = args.width
    if args.height != 800:
        settings.screen_height = args.height
    if args.fullscreen:
        settings.fullscreen = True
    if args.no_sound:
        settings.sfx_enabled = False
    if args.no_overlay:
        settings.show_debug_overlay = False
    if args.timer != 180.0:
        settings.countdown_duration = args.timer
    if args.difficulty is not None:
        selected_diff = diff_map.get(args.difficulty.lower(), DifficultyLevel.CADET)
        settings.apply_difficulty_preset(selected_diff)

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
