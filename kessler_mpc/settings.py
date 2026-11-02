"""Game settings, difficulty presets, and configuration options for Kessler AI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


class DifficultyLevel(Enum):
    CASUAL = "Casual"
    STANDARD = "Standard"
    HARDCORE = "Hardcore"
    KESSLER_SYNDROME = "Kessler Syndrome"


@dataclass
class DifficultyPreset:
    initial_asteroids: int
    asteroid_speed_mult: float
    spawn_interval_s: float
    max_asteroids: int
    damage_mult: float


DIFFICULTY_PRESETS: Dict[DifficultyLevel, DifficultyPreset] = {
    DifficultyLevel.CASUAL: DifficultyPreset(
        initial_asteroids=4,
        asteroid_speed_mult=0.75,
        spawn_interval_s=18.0,
        max_asteroids=12,
        damage_mult=0.7,
    ),
    DifficultyLevel.STANDARD: DifficultyPreset(
        initial_asteroids=6,
        asteroid_speed_mult=1.0,
        spawn_interval_s=12.0,
        max_asteroids=20,
        damage_mult=1.0,
    ),
    DifficultyLevel.HARDCORE: DifficultyPreset(
        initial_asteroids=9,
        asteroid_speed_mult=1.3,
        spawn_interval_s=8.0,
        max_asteroids=28,
        damage_mult=1.4,
    ),
    DifficultyLevel.KESSLER_SYNDROME: DifficultyPreset(
        initial_asteroids=14,
        asteroid_speed_mult=1.55,
        spawn_interval_s=5.0,
        max_asteroids=38,
        damage_mult=1.7,
    ),
}


@dataclass
class GameSettings:
    # Display
    screen_width: int = 1000
    screen_height: int = 800
    target_fps: int = 60
    fullscreen: bool = False

    # Audio
    master_volume: float = 0.8
    sfx_enabled: bool = True

    # Graphics & Visuals
    particle_density: str = "High"  # "Low", "Medium", "High"
    screen_shake_enabled: bool = True

    # Gameplay
    difficulty: DifficultyLevel = DifficultyLevel.STANDARD

    # Debug Overlay Toggles
    show_debug_overlay: bool = True
    show_astar_path: bool = True
    show_mpc_trajectories: bool = True
    show_threat_vectors: bool = True
    show_detection_radius: bool = True
    show_target_reticle: bool = True

    def get_difficulty_preset(self) -> DifficultyPreset:
        return DIFFICULTY_PRESETS[self.difficulty]

    def cycle_difficulty(self) -> DifficultyLevel:
        levels = list(DifficultyLevel)
        idx = (levels.index(self.difficulty) + 1) % len(levels)
        self.difficulty = levels[idx]
        return self.difficulty
