"""Procedural sound effects synthesizer for Kessler AI survival game.

Generates retro-modern sci-fi sound effects directly in memory using NumPy and
Pygame Mixer, requiring NO external audio files. Gracefully handles systems
without sound hardware.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pygame


class SoundManager:
    """Manages procedural audio effects and ambient space tones."""

    def __init__(self, sample_rate: int = 44100, enabled: bool = True, master_volume: float = 0.7):
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.master_volume = master_volume
        self.sfx_volume = 0.8
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.initialized = False

        if not self.enabled:
            return

        try:
            if not pygame.mixer.get_init():
                # Initialize mixer with 16-bit signed stereo
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2, buffer=1024)
            self.initialized = True
            self._generate_all_sfx()
        except Exception:
            # Audio device might be busy, absent, or sandbox-restricted
            self.initialized = False

    def _make_sound(self, mono_samples: np.ndarray) -> Optional[pygame.mixer.Sound]:
        """Convert float numpy array (-1.0 to 1.0) into a stereo Pygame Sound."""
        if not self.initialized:
            return None
        try:
            # Clamp and convert to 16-bit signed integers
            clamped = np.clip(mono_samples, -1.0, 1.0)
            int16_samples = (clamped * 32767).astype(np.int16)
            # Duplicate to stereo (N, 2)
            stereo = np.column_stack([int16_samples, int16_samples])
            return pygame.sndarray.make_sound(stereo)
        except Exception:
            return None

    def _generate_all_sfx(self) -> None:
        """Synthesize all game sound effects procedurally."""
        sr = self.sample_rate

        # 1. Laser Blast (Frequency chirp 900 Hz -> 220 Hz)
        duration = 0.14
        t = np.linspace(0, duration, int(sr * duration), False)
        freq = np.linspace(900, 220, len(t))
        phase = 2.0 * np.pi * np.cumsum(freq) / sr
        laser = 0.5 * np.sin(phase) * (1.0 - t / duration) ** 1.5
        s = self._make_sound(laser)
        if s:
            self.sounds["laser"] = s

        # 2. Small Explosion (Filtered white noise burst)
        duration = 0.35
        n_samples = int(sr * duration)
        noise = np.random.uniform(-1.0, 1.0, n_samples)
        t = np.linspace(0, duration, n_samples, False)
        env = np.exp(-12.0 * t)
        small_boom = noise * env * 0.7
        s = self._make_sound(small_boom)
        if s:
            self.sounds["explosion_small"] = s

        # 3. Large Explosion (Deep sub-bass + noise decay)
        duration = 0.8
        n_samples = int(sr * duration)
        noise = np.random.uniform(-1.0, 1.0, n_samples)
        t = np.linspace(0, duration, n_samples, False)
        env = np.exp(-5.0 * t)
        sub = np.sin(2.0 * np.pi * 55.0 * t) * np.exp(-3.0 * t)
        big_boom = (noise * 0.6 + sub * 0.7) * env
        s = self._make_sound(big_boom)
        if s:
            self.sounds["explosion_large"] = s

        # 4. Boost Surge (Resonant rising sweep)
        duration = 0.4
        t = np.linspace(0, duration, int(sr * duration), False)
        freq = 120.0 + 350.0 * (t / duration) ** 2
        phase = 2.0 * np.pi * np.cumsum(freq) / sr
        boost = 0.4 * np.sin(phase) * np.sin(np.pi * t / duration)
        s = self._make_sound(boost)
        if s:
            self.sounds["boost"] = s

        # 5. Shield Hit (Metallic ping with warble)
        duration = 0.25
        t = np.linspace(0, duration, int(sr * duration), False)
        ping = 0.4 * np.sin(2.0 * np.pi * 750.0 * t + 0.3 * np.sin(2.0 * np.pi * 40.0 * t)) * np.exp(-14.0 * t)
        s = self._make_sound(ping)
        if s:
            self.sounds["shield_hit"] = s

        # 6. Alarm Beep (Urgent collision warning)
        duration = 0.15
        t = np.linspace(0, duration, int(sr * duration), False)
        alarm = 0.35 * np.sign(np.sin(2.0 * np.pi * 880.0 * t)) * np.sin(np.pi * t / duration)
        s = self._make_sound(alarm)
        if s:
            self.sounds["alarm"] = s

        # 7. Safe Zone Beacon / Docking Chime
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), False)
        chime = (
            0.3 * np.sin(2.0 * np.pi * 523.25 * t)  # C5
            + 0.3 * np.sin(2.0 * np.pi * 659.25 * t)  # E5
            + 0.3 * np.sin(2.0 * np.pi * 783.99 * t)  # G5
        ) * np.exp(-4.0 * t)
        s = self._make_sound(chime)
        if s:
            self.sounds["docking"] = s

        # 8. Victory Fanfare
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), False)
        # Four notes: C5, E5, G5, C6
        n1 = np.where((t >= 0.0) & (t < 0.2), np.sin(2.0 * np.pi * 523.25 * t), 0.0)
        n2 = np.where((t >= 0.2) & (t < 0.4), np.sin(2.0 * np.pi * 659.25 * t), 0.0)
        n3 = np.where((t >= 0.4) & (t < 0.6), np.sin(2.0 * np.pi * 783.99 * t), 0.0)
        n4 = np.where(t >= 0.6, np.sin(2.0 * np.pi * 1046.50 * t) * np.exp(-3.0 * (t - 0.6)), 0.0)
        victory = (n1 + n2 + n3 + n4) * 0.45
        s = self._make_sound(victory)
        if s:
            self.sounds["victory"] = s

        # 9. Game Over Tones
        duration = 1.2
        t = np.linspace(0, duration, int(sr * duration), False)
        m1 = np.where((t >= 0.0) & (t < 0.35), np.sin(2.0 * np.pi * 392.00 * t), 0.0)  # G4
        m2 = np.where((t >= 0.35) & (t < 0.7), np.sin(2.0 * np.pi * 349.23 * t), 0.0)  # F4
        m3 = np.where(t >= 0.7, np.sin(2.0 * np.pi * 293.66 * t) * np.exp(-2.5 * (t - 0.7)), 0.0)  # D4
        game_over = (m1 + m2 + m3) * 0.45
        s = self._make_sound(game_over)
        if s:
            self.sounds["game_over"] = s

        # 10. UI Click
        duration = 0.04
        t = np.linspace(0, duration, int(sr * duration), False)
        click = 0.3 * np.sin(2.0 * np.pi * 1200.0 * t) * (1.0 - t / duration)
        s = self._make_sound(click)
        if s:
            self.sounds["ui_click"] = s

    def play(self, sound_name: str) -> None:
        """Play a named sound effect if audio is active."""
        if not self.initialized or not self.enabled:
            return
        sound = self.sounds.get(sound_name)
        if sound:
            try:
                sound.set_volume(self.master_volume * self.sfx_volume)
                sound.play()
            except Exception:
                pass

    def set_master_volume(self, vol: float) -> None:
        self.master_volume = max(0.0, min(1.0, vol))

    def toggle_mute(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled
