"""Game settings, difficulty presets, and configuration options for SHATTERSPACE.

Supports persistent JSON storage, validation clamps, and live configuration
for Spaceship, Asteroid, and Simulation parameters.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

# =============================================================================
# CENTRAL GAME BRANDING & IDENTITY
# =============================================================================
GAME_TITLE: str = "SHATTERSPACE"
GAME_SUBTITLE: str = "DEEP SPACE SURVIVAL"
GAME_TAGLINE: str = "One ship. Endless asteroids. Stay alive."


class DifficultyLevel(Enum):
    CADET = "Cadet"
    PILOT = "Pilot"
    VETERAN = "Veteran"
    ELITE = "Elite"
    NIGHTMARE = "Nightmare"

    @classmethod
    def from_str(cls, val: str) -> "DifficultyLevel":
        if isinstance(val, cls):
            return val
        s = str(val).strip().lower().replace(" ", "_")
        legacy_map = {
            "casual": cls.CADET,
            "standard": cls.PILOT,
            "hardcore": cls.VETERAN,
            "kessler": cls.NIGHTMARE,
            "kessler_syndrome": cls.NIGHTMARE,
        }
        if s in legacy_map:
            return legacy_map[s]
        for member in cls:
            if member.name.lower() == s or member.value.lower() == s:
                return member
        return cls.CADET


# Backward compatibility aliases for existing code and test references
DifficultyLevel.CASUAL = DifficultyLevel.CADET          # type: ignore
DifficultyLevel.STANDARD = DifficultyLevel.PILOT        # type: ignore
DifficultyLevel.HARDCORE = DifficultyLevel.VETERAN      # type: ignore
DifficultyLevel.KESSLER_SYNDROME = DifficultyLevel.NIGHTMARE  # type: ignore


DIFFICULTY_DESCRIPTIONS: Dict[DifficultyLevel, str] = {
    DifficultyLevel.CADET: "A forgiving introduction to space survival.",
    DifficultyLevel.PILOT: "A balanced challenge.",
    DifficultyLevel.VETERAN: "Faster threats and less room for mistakes.",
    DifficultyLevel.ELITE: "Intense asteroid encounters.",
    DifficultyLevel.NIGHTMARE: "The ultimate survival test.",
}


@dataclass
class DifficultyPreset:
    initial_asteroids: int
    asteroid_speed_mult: float
    spawn_interval_s: float
    max_asteroids: int
    damage_mult: float


DIFFICULTY_PRESETS: Dict[DifficultyLevel, DifficultyPreset] = {
    DifficultyLevel.CADET: DifficultyPreset(
        initial_asteroids=4,
        asteroid_speed_mult=0.75,
        spawn_interval_s=18.0,
        max_asteroids=12,
        damage_mult=0.7,
    ),
    DifficultyLevel.PILOT: DifficultyPreset(
        initial_asteroids=6,
        asteroid_speed_mult=1.0,
        spawn_interval_s=12.0,
        max_asteroids=20,
        damage_mult=1.0,
    ),
    DifficultyLevel.VETERAN: DifficultyPreset(
        initial_asteroids=8,
        asteroid_speed_mult=1.2,
        spawn_interval_s=9.0,
        max_asteroids=26,
        damage_mult=1.25,
    ),
    DifficultyLevel.ELITE: DifficultyPreset(
        initial_asteroids=11,
        asteroid_speed_mult=1.4,
        spawn_interval_s=6.5,
        max_asteroids=32,
        damage_mult=1.5,
    ),
    DifficultyLevel.NIGHTMARE: DifficultyPreset(
        initial_asteroids=15,
        asteroid_speed_mult=1.65,
        spawn_interval_s=4.5,
        max_asteroids=40,
        damage_mult=1.8,
    ),
}


SETTINGS_FILE = "settings.json"


@dataclass
class GameSettings:
    # --- Display & Engine ---
    screen_width: int = 1000
    screen_height: int = 800
    target_fps: int = 60
    fullscreen: bool = False

    # --- Audio ---
    master_volume: float = 0.8
    sfx_enabled: bool = True

    # --- Graphics & Visuals ---
    particle_density: str = "High"  # "Low", "Medium", "High"
    screen_shake_enabled: bool = True

    # --- Gameplay & Mission Settings ---
    difficulty: DifficultyLevel = DifficultyLevel.CADET
    countdown_duration: float = 180.0
    difficulty_progression: bool = True
    ai_aggression: str = "Balanced"  # "Defensive", "Balanced", "Aggressive"
    adaptive_risk: bool = True

    # --- Spaceship Settings ---
    ship_design: str = "Interceptor"  # "Interceptor", "Scout", "Vanguard"
    ship_max_speed: float = 240.0
    ship_thrust_max: float = 480.0
    ship_turn_max: float = 180.0
    ship_max_health: float = 100.0
    ship_starting_lives: int = 3
    ship_fire_rate: float = 7.0  # shots per second
    bullet_speed: float = 800.0
    bullet_damage: float = 1.0
    collision_safety_margin: float = 35.0
    damage_large_asteroid: float = 40.0
    damage_medium_asteroid: float = 25.0
    damage_small_asteroid: float = 10.0
    normal_collision_invulnerability: float = 1.5
    respawn_invulnerability: float = 3.0
    health_regen_enabled: bool = False
    health_regen_rate: float = 5.0

    # --- Asteroid Settings ---
    initial_asteroids: int = 6
    max_asteroids: int = 20
    asteroid_speed_mult: float = 1.0
    asteroid_speed_variation: float = 0.25
    spawn_interval_s: float = 12.0
    asteroid_health: float = 50.0  # Base health for size 3; scaled by size
    asteroid_health_small: float = 1.0
    asteroid_health_medium: float = 2.0
    asteroid_health_large: float = 3.0
    fragment_count_min: int = 3
    fragment_count_max: int = 3
    fragment_speed_mult: float = 1.0
    score_size_1: int = 200
    score_size_2: int = 100
    score_size_3: int = 50
    score_size_4: int = 25
    asteroid_split_enabled: bool = True
    asteroid_wall_bounce: bool = True

    # --- Red Penalty Asteroid Settings ---
    red_asteroid_enabled: bool = True
    red_asteroid_max_active: int = 1
    red_asteroid_spawn_probability: float = 0.05
    red_asteroid_spawn_interval: float = 5.0
    red_asteroid_speed: float = 100.0
    red_asteroid_radius: float = 25.0
    red_asteroid_health: float = 1.0
    red_asteroid_shooting_penalty: int = 5
    red_asteroid_collision_penalty: int = 10

    # --- AI Settings ---
    threat_weight_risk: float = 0.50
    threat_weight_urgency: float = 0.35
    threat_weight_geometry: float = 0.15
    weight_target_threat: float = 0.35
    weight_strategic_value: float = 0.25
    weight_hit_prob: float = 0.25
    weight_frag_risk: float = 0.10
    weight_attack_effort: float = 0.05
    target_switch_threshold: float = 0.15
    adaptive_risk_beta: float = 1.5

    # --- Debug Overlay Toggles ---
    show_debug_overlay: bool = True
    show_astar_path: bool = True
    show_mpc_trajectories: bool = True
    show_threat_vectors: bool = True
    show_detection_radius: bool = True
    show_target_reticle: bool = True

    @property
    def weapon_cooldown(self) -> float:
        """Weapon cooldown interval in seconds derived from fire rate."""
        return 1.0 / max(0.5, self.ship_fire_rate)

    @weapon_cooldown.setter
    def weapon_cooldown(self, val: float) -> None:
        val = max(0.04, min(2.0, float(val)))
        self.ship_fire_rate = 1.0 / val

    def get_difficulty_preset(self) -> DifficultyPreset:
        preset = DIFFICULTY_PRESETS[self.difficulty]
        # Allow user custom settings to override preset when customized
        return DifficultyPreset(
            initial_asteroids=self.initial_asteroids,
            asteroid_speed_mult=self.asteroid_speed_mult * preset.asteroid_speed_mult,
            spawn_interval_s=max(2.0, self.spawn_interval_s * (preset.spawn_interval_s / 12.0)),
            max_asteroids=self.max_asteroids,
            damage_mult=preset.damage_mult,
        )

    def cycle_difficulty(self) -> DifficultyLevel:
        levels = list(DifficultyLevel)
        idx = (levels.index(self.difficulty) + 1) % len(levels)
        self.apply_difficulty(levels[idx])
        return self.difficulty

    def apply_difficulty(self, diff: DifficultyLevel) -> None:
        self.difficulty = diff
        p = DIFFICULTY_PRESETS[diff]
        self.initial_asteroids = p.initial_asteroids
        self.asteroid_speed_mult = p.asteroid_speed_mult
        self.spawn_interval_s = p.spawn_interval_s
        self.max_asteroids = p.max_asteroids

    # Alias for apply_difficulty
    apply_difficulty_preset = apply_difficulty

    def cycle_ai_aggression(self) -> str:
        options = ["Defensive", "Balanced", "Aggressive"]
        idx = (options.index(self.ai_aggression) + 1) % len(options)
        self.ai_aggression = options[idx]
        return self.ai_aggression

    def get_score_table(self) -> Dict[int, int]:
        return {
            1: self.score_size_1,
            2: self.score_size_2,
            3: self.score_size_3,
            4: self.score_size_4,
        }

    def validate_and_clamp(self) -> None:
        """Enforce strict numerical boundaries and safe limits on all settings."""
        self.screen_width = int(max(640, min(2560, self.screen_width)))
        self.screen_height = int(max(480, min(1440, self.screen_height)))
        self.target_fps = int(max(30, min(144, self.target_fps)))
        self.countdown_duration = float(max(10.0, min(600.0, self.countdown_duration)))

        # Ship limits
        if self.ship_design not in ("Interceptor", "Scout", "Vanguard"):
            self.ship_design = "Interceptor"
        self.ship_max_speed = float(max(80.0, min(600.0, self.ship_max_speed)))
        self.ship_thrust_max = float(max(150.0, min(1200.0, self.ship_thrust_max)))
        self.ship_turn_max = float(max(60.0, min(360.0, self.ship_turn_max)))
        self.ship_max_health = float(max(10.0, min(500.0, self.ship_max_health)))
        self.ship_starting_lives = int(max(1, min(10, self.ship_starting_lives)))
        self.ship_fire_rate = float(max(1.0, min(25.0, self.ship_fire_rate)))
        self.bullet_speed = float(max(200.0, min(1600.0, self.bullet_speed)))
        self.bullet_damage = float(max(0.5, min(300.0, self.bullet_damage)))
        self.collision_safety_margin = float(max(10.0, min(120.0, self.collision_safety_margin)))
        self.damage_large_asteroid = float(max(5.0, min(200.0, self.damage_large_asteroid)))
        self.damage_medium_asteroid = float(max(5.0, min(150.0, self.damage_medium_asteroid)))
        self.damage_small_asteroid = float(max(2.0, min(100.0, self.damage_small_asteroid)))
        self.normal_collision_invulnerability = float(max(0.2, min(5.0, self.normal_collision_invulnerability)))
        self.respawn_invulnerability = float(max(0.5, min(10.0, self.respawn_invulnerability)))
        self.health_regen_rate = float(max(0.5, min(50.0, self.health_regen_rate)))

        # Asteroid limits
        self.initial_asteroids = int(max(1, min(30, self.initial_asteroids)))
        self.max_asteroids = int(max(self.initial_asteroids, min(60, self.max_asteroids)))
        self.asteroid_speed_mult = float(max(0.2, min(3.5, self.asteroid_speed_mult)))
        self.asteroid_speed_variation = float(max(0.0, min(1.0, self.asteroid_speed_variation)))
        self.spawn_interval_s = float(max(1.5, min(60.0, self.spawn_interval_s)))
        self.asteroid_health = float(max(0.5, min(300.0, self.asteroid_health)))
        self.asteroid_health_small = float(max(0.5, min(50.0, self.asteroid_health_small)))
        self.asteroid_health_medium = float(max(0.5, min(100.0, self.asteroid_health_medium)))
        self.asteroid_health_large = float(max(0.5, min(200.0, self.asteroid_health_large)))
        self.fragment_count_min = int(max(1, min(5, self.fragment_count_min)))
        self.fragment_count_max = int(max(self.fragment_count_min, min(6, self.fragment_count_max)))
        self.fragment_speed_mult = float(max(0.2, min(3.0, self.fragment_speed_mult)))
        self.score_size_1 = int(max(5, min(2000, self.score_size_1)))
        self.score_size_2 = int(max(5, min(1000, self.score_size_2)))
        self.score_size_3 = int(max(5, min(500, self.score_size_3)))
        self.score_size_4 = int(max(1, min(250, self.score_size_4)))

        # Red Penalty Asteroid limits
        self.red_asteroid_max_active = int(max(1, min(5, self.red_asteroid_max_active)))
        self.red_asteroid_spawn_probability = float(max(0.01, min(1.0, self.red_asteroid_spawn_probability)))
        self.red_asteroid_spawn_interval = float(max(1.0, min(30.0, self.red_asteroid_spawn_interval)))
        self.red_asteroid_speed = float(max(20.0, min(300.0, self.red_asteroid_speed)))
        self.red_asteroid_radius = float(max(10.0, min(60.0, self.red_asteroid_radius)))
        self.red_asteroid_health = float(max(0.5, min(50.0, self.red_asteroid_health)))
        self.red_asteroid_shooting_penalty = int(max(1, min(100, self.red_asteroid_shooting_penalty)))
        self.red_asteroid_collision_penalty = int(max(1, min(200, self.red_asteroid_collision_penalty)))

        # AI limits
        self.threat_weight_risk = float(max(0.0, min(1.0, self.threat_weight_risk)))
        self.threat_weight_urgency = float(max(0.0, min(1.0, self.threat_weight_urgency)))
        self.threat_weight_geometry = float(max(0.0, min(1.0, self.threat_weight_geometry)))
        self.target_switch_threshold = float(max(0.0, min(1.0, self.target_switch_threshold)))
        self.adaptive_risk_beta = float(max(0.1, min(5.0, self.adaptive_risk_beta)))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["difficulty"] = self.difficulty.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GameSettings:
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if "difficulty" in filtered and isinstance(filtered["difficulty"], str):
            diff_str = filtered["difficulty"].strip().lower().replace(" ", "_")
            alias_map = {
                "cadet": DifficultyLevel.CADET,
                "casual": DifficultyLevel.CADET,
                "pilot": DifficultyLevel.PILOT,
                "standard": DifficultyLevel.PILOT,
                "veteran": DifficultyLevel.VETERAN,
                "hardcore": DifficultyLevel.VETERAN,
                "elite": DifficultyLevel.ELITE,
                "nightmare": DifficultyLevel.NIGHTMARE,
                "kessler": DifficultyLevel.NIGHTMARE,
                "kessler_syndrome": DifficultyLevel.NIGHTMARE,
            }
            filtered["difficulty"] = alias_map.get(diff_str, DifficultyLevel.CADET)

        settings = cls(**filtered)
        settings.validate_and_clamp()
        return settings

    def save_to_file(self, filepath: str = SETTINGS_FILE) -> bool:
        """Persist settings to local JSON file."""
        try:
            self.validate_and_clamp()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings to {filepath}: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str = SETTINGS_FILE) -> GameSettings:
        """Load settings from JSON file or return defaults if file absent or corrupt."""
        if not os.path.exists(filepath):
            s = cls()
            s.save_to_file(filepath)
            return s
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error loading {filepath}, falling back to defaults: {e}")
            return cls()

    def reset_to_defaults(self) -> None:
        """Reset all parameters to factory defaults."""
        defaults = GameSettings()
        for k, v in asdict(defaults).items():
            if k == "difficulty":
                setattr(self, k, DifficultyLevel.CADET)
            else:
                setattr(self, k, v)
