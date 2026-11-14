"""Visual effects, multi-layer starfield, and particle systems for Kessler AI.

Provides high-performance procedural rendering for space environments,
thruster plumes, explosions, shockwaves, shield ripples, and camera screen-shake.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pygame


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float
    max_life: float
    size: float
    fade: bool = True


@dataclass
class Shockwave:
    x: float
    y: float
    radius: float
    max_radius: float
    color: Tuple[int, int, int]
    life: float
    max_life: float
    thickness: int = 2


class ParticleSystem:
    """Manages active particle effects and shockwaves."""

    def __init__(self, max_particles: int = 1500):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self.shockwaves: List[Shockwave] = []

    def clear(self) -> None:
        self.particles.clear()
        self.shockwaves.clear()

    def emit_thruster(
        self,
        muzzle_pos: Tuple[float, float],
        heading_deg: float,
        is_boost: bool = False,
        count: int = 2,
    ) -> None:
        """Emit rocket thrust particles behind the ship."""
        rad = math.radians(heading_deg + 180.0)  # Opposite to heading
        spread = 0.35 if not is_boost else 0.18
        speed_base = 120.0 if not is_boost else 280.0

        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            angle = rad + random.uniform(-spread, spread)
            spd = speed_base * random.uniform(0.6, 1.3)
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd

            if is_boost:
                # Cyan / electric blue flame
                color = random.choice([
                    (50, 220, 255),
                    (100, 240, 255),
                    (200, 255, 255),
                    (0, 150, 255),
                ])
                life = random.uniform(0.12, 0.25)
                size = random.uniform(3.0, 5.5)
            else:
                # Orange / amber rocket plume
                color = random.choice([
                    (255, 120, 30),
                    (255, 200, 50),
                    (255, 80, 20),
                    (240, 240, 120),
                ])
                life = random.uniform(0.10, 0.20)
                size = random.uniform(2.5, 4.0)

            self.particles.append(
                Particle(
                    x=muzzle_pos[0],
                    y=muzzle_pos[1],
                    vx=vx,
                    vy=vy,
                    color=color,
                    life=life,
                    max_life=life,
                    size=size,
                )
            )

    def emit_explosion(
        self,
        pos: Tuple[float, float],
        count: int = 35,
        color_palette: Optional[List[Tuple[int, int, int]]] = None,
        speed_mult: float = 1.0,
    ) -> None:
        """Emit fiery debris sparks and expanding fireball particles."""
        palette = color_palette or [
            (255, 255, 220),
            (255, 180, 40),
            (255, 90, 20),
            (200, 40, 10),
            (160, 160, 160),
        ]

        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            angle = random.uniform(0, 2 * math.pi)
            spd = random.uniform(30.0, 220.0) * speed_mult
            life = random.uniform(0.25, 0.75)
            color = random.choice(palette)
            self.particles.append(
                Particle(
                    x=pos[0],
                    y=pos[1],
                    vx=math.cos(angle) * spd,
                    vy=math.sin(angle) * spd,
                    color=color,
                    life=life,
                    max_life=life,
                    size=random.uniform(2.0, 5.0),
                )
            )

        # Trigger shockwave ring
        self.shockwaves.append(
            Shockwave(
                x=pos[0],
                y=pos[1],
                radius=4.0,
                max_radius=50.0 * speed_mult,
                color=(255, 200, 100),
                life=0.35,
                max_life=0.35,
                thickness=2,
            )
        )

    def emit_shield_ripple(self, pos: Tuple[float, float], radius: float = 25.0) -> None:
        """Emit glowing cyan shield ripple."""
        self.shockwaves.append(
            Shockwave(
                x=pos[0],
                y=pos[1],
                radius=radius * 0.7,
                max_radius=radius * 1.3,
                color=(80, 220, 255),
                life=0.25,
                max_life=0.25,
                thickness=3,
            )
        )

    def emit_spark(self, pos: Tuple[float, float], normal_angle: float) -> None:
        """Emit ricochet / bullet impact sparks."""
        for _ in range(8):
            if len(self.particles) >= self.max_particles:
                break
            ang = normal_angle + random.uniform(-0.8, 0.8)
            spd = random.uniform(60.0, 180.0)
            self.particles.append(
                Particle(
                    x=pos[0],
                    y=pos[1],
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd,
                    color=(255, 255, 150),
                    life=random.uniform(0.1, 0.25),
                    max_life=0.25,
                    size=random.uniform(1.5, 3.0),
                )
            )

    def update(self, dt: float) -> None:
        """Update particle physics and shockwaves."""
        dead_p = []
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= 0.96  # Drag
            p.vy *= 0.96
            p.life -= dt
            if p.life <= 0:
                dead_p.append(p)

        for p in dead_p:
            self.particles.remove(p)

        dead_s = []
        for s in self.shockwaves:
            s.life -= dt
            progress = 1.0 - (s.life / s.max_life)
            s.radius = s.max_radius * progress
            if s.life <= 0:
                dead_s.append(s)

        for s in dead_s:
            self.shockwaves.remove(s)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw particles and shockwaves."""
        for p in self.particles:
            alpha = max(0.0, min(1.0, p.life / p.max_life))
            size = max(1.0, p.size * alpha)
            r, g, b = p.color
            color = (int(r * alpha), int(g * alpha), int(b * alpha))
            pygame.draw.circle(surface, color, (int(p.x), int(p.y)), max(1, int(size)))

        for s in self.shockwaves:
            alpha = max(0.0, min(1.0, s.life / s.max_life))
            r, g, b = s.color
            color = (int(r * alpha), int(g * alpha), int(b * alpha))
            rad = max(1, int(s.radius))
            thick = max(1, int(s.thickness * alpha))
            pygame.draw.circle(surface, color, (int(s.x), int(s.y)), rad, thick)


class ScreenShake:
    """Camera screen shake generator."""

    def __init__(self):
        self.trauma: float = 0.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0

    def add_trauma(self, amount: float) -> None:
        self.trauma = min(1.0, self.trauma + amount)

    def update(self, dt: float) -> None:
        if self.trauma > 0.0:
            self.trauma = max(0.0, self.trauma - dt * 1.5)
            # Shake intensity is quadratic with trauma
            shake = self.trauma**2
            max_offset = 14.0 * shake
            self.offset_x = random.uniform(-max_offset, max_offset)
            self.offset_y = random.uniform(-max_offset, max_offset)
        else:
            self.offset_x = 0.0
            self.offset_y = 0.0


class Starfield:
    """Multi-layer parallax starfield with distant nebula gas and deep-space celestial elements."""

    def __init__(self, width: int = 1000, height: int = 800):
        self.width = width
        self.height = height

        # Layer 1: very distant, dim pin-prick stars
        self.stars_layer1 = [
            (random.uniform(0, width), random.uniform(0, height), random.uniform(0.25, 0.65))
            for _ in range(200)
        ]
        # Layer 2: medium stars with colour temperature variation
        self.stars_layer2 = [
            (
                random.uniform(0, width),
                random.uniform(0, height),
                random.uniform(1.0, 1.6),
                random.choice([
                    (200, 215, 255),   # cool blue-white
                    (255, 245, 220),   # warm white
                    (200, 240, 255),   # ice blue
                    (255, 220, 190),   # yellow dwarf
                ]),
                random.uniform(0, 2 * math.pi),  # twinkle phase
                random.uniform(1.5, 3.5),         # twinkle speed
            )
            for _ in range(80)
        ]
        # Layer 3: a handful of brighter stars with a subtle cross flare
        self.stars_layer3 = [
            (
                random.uniform(0, width),
                random.uniform(0, height),
                random.uniform(1.8, 2.8),
                random.choice([(230, 245, 255), (255, 230, 170), (175, 225, 255)]),
            )
            for _ in range(18)
        ]

        # Nebula clouds — pre-rendered once at startup, very restrained alpha
        self.nebula_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        self._generate_nebulae()

        # Distant planet silhouette in upper-right
        self.planet_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        self._generate_planet()

        self.time: float = 0.0

    def _generate_nebulae(self) -> None:
        """Soft, low-opacity gas clouds — dark space palette."""
        nebulae = [
            (self.width * 0.20, self.height * 0.28, 220, (22, 12, 48), 9),   # deep violet
            (self.width * 0.72, self.height * 0.65, 180, (10, 28, 48), 8),   # midnight blue
            (self.width * 0.50, self.height * 0.80, 160, (38, 15, 24), 7),   # dark crimson hint
        ]
        for cx, cy, max_r, rgb, max_alpha in nebulae:
            steps = range(max_r, 8, -18)
            for r in steps:
                # alpha peaks at the edge, fades to centre (gas cloud feel)
                t = 1.0 - r / max_r
                alpha = int(max_alpha * math.sin(t * math.pi))
                pygame.draw.circle(self.nebula_surf, (*rgb, max(0, alpha)), (int(cx), int(cy)), r)

    def _generate_planet(self) -> None:
        """Distant gas giant — muted, unobtrusive."""
        cx = int(self.width * 0.87)
        cy = int(self.height * 0.14)
        r = 52
        # Dark body with slight atmosphere fringe
        pygame.draw.circle(self.planet_surf, (18, 26, 42, 180), (cx, cy), r)
        # Lighter crescent on lit side
        pygame.draw.circle(self.planet_surf, (38, 56, 82, 160), (cx - 6, cy - 7), r - 5)
        # Ring — thin, translucent
        rect = pygame.Rect(cx - 82, cy - 14, 164, 28)
        pygame.draw.ellipse(self.planet_surf, (68, 95, 125, 55), rect, 2)

    def update(self, dt: float) -> None:
        self.time += dt

    def draw(self, surface: pygame.Surface, cam_x: float = 0.0, cam_y: float = 0.0) -> None:
        """Render background starfield and celestial elements."""
        # Deep space — very dark blue-black
        surface.fill((6, 8, 15))

        # Nebulae and planet (static layers)
        surface.blit(self.nebula_surf, (0, 0))
        surface.blit(self.planet_surf, (0, 0))

        # Layer 1 — single pixel dim stars, no parallax offset for perf
        for x, y, brightness in self.stars_layer1:
            val = int(255 * brightness)
            px = int(x) % self.width
            py = int(y) % self.height
            surface.set_at((px, py), (val, val, int(val * 0.95)))

        # Layer 2 — small circles with gentle twinkle
        for x, y, size, color, phase, speed in self.stars_layer2:
            px = (x - cam_x * 0.08) % self.width
            py = (y - cam_y * 0.08) % self.height
            twinkle = 0.75 + 0.25 * math.sin(self.time * speed + phase)
            c = (
                max(0, min(255, int(color[0] * twinkle))),
                max(0, min(255, int(color[1] * twinkle))),
                max(0, min(255, int(color[2] * twinkle))),
            )
            pygame.draw.circle(surface, c, (int(px), int(py)), max(1, int(size)))

        # Layer 3 — brighter stars, subtle cross diffraction spike
        for x, y, size, color in self.stars_layer3:
            px = (x - cam_x * 0.16) % self.width
            py = (y - cam_y * 0.16) % self.height
            ix, iy = int(px), int(py)
            pygame.draw.circle(surface, color, (ix, iy), max(1, int(size)))
            dim = (color[0] // 5, color[1] // 5, color[2] // 5)
            pygame.draw.line(surface, dim, (ix - 4, iy), (ix + 4, iy), 1)
            pygame.draw.line(surface, dim, (ix, iy - 4), (ix, iy + 4), 1)
