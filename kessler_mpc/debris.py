"""Asteroid and space debris management for Kessler AI survival game.

Implements procedural rock silhouettes, realistic rotation, exact engine fragmentation
(3 child asteroids splitting with +/-15 deg fan out), and debris state adapters.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pygame


# Size mapping matching KesslerGame / Asteroid dynamics
SIZE_RADII = {1: 15.0, 2: 30.0, 3: 50.0, 4: 80.0}
SIZE_MASS = {1: 25.0, 2: 75.0, 3: 200.0, 4: 600.0}
SIZE_SCORES = {1: 100, 2: 50, 3: 25, 4: 10}


class Asteroid:
    """Individual asteroid / debris object with procedural polygon geometry."""

    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        size: int = 3,
        map_size: Tuple[float, float] = (1000.0, 800.0),
    ):
        self.x = float(position[0])
        self.y = float(position[1])
        self.vx = float(velocity[0])
        self.vy = float(velocity[1])
        self.size = max(1, min(4, int(size)))
        self.map_w, self.map_h = map_size

        self.radius = SIZE_RADII.get(self.size, 20.0 * self.size)
        self.mass = SIZE_MASS.get(self.size, 50.0 * self.size)
        self.score_value = SIZE_SCORES.get(self.size, 20)

        # Rotation and visual styling
        self.rotation_angle = random.uniform(0, 360.0)
        self.rot_speed = random.uniform(-45.0, 45.0)  # deg/s

        # Generate unique jagged polygon offsets
        self.num_vertices = random.randint(9, 14)
        self.poly_offsets = []
        for i in range(self.num_vertices):
            ang = (2.0 * math.pi * i) / self.num_vertices
            # Jitter radius by 15-25%
            r_jitter = self.radius * random.uniform(0.78, 1.22)
            self.poly_offsets.append((ang, r_jitter))

        # Asteroid color tone (dark charcoal to dusty slate)
        shade = random.randint(75, 115)
        tint = random.choice([(0, 0, 0), (8, 4, 0), (0, 4, 8), (4, 4, 4)])
        self.base_color = (
            min(255, shade + tint[0]),
            min(255, shade + tint[1]),
            min(255, shade + tint[2]),
        )
        self.highlight_color = (
            min(255, self.base_color[0] + 35),
            min(255, self.base_color[1] + 35),
            min(255, self.base_color[2] + 35),
        )

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def update(self, dt: float) -> None:
        """Advance asteroid position and toroidal wrapping."""
        self.x = (self.x + self.vx * dt) % self.map_w
        self.y = (self.y + self.vy * dt) % self.map_h
        self.rotation_angle = (self.rotation_angle + self.rot_speed * dt) % 360.0

    def draw(self, surface: pygame.Surface) -> None:
        """Draw jagged textured rock polygon."""
        rot_rad = math.radians(self.rotation_angle)
        points = []
        for base_ang, r in self.poly_offsets:
            ang = base_ang + rot_rad
            px = self.x + math.cos(ang) * r
            py = self.y + math.sin(ang) * r
            points.append((px, py))

        if len(points) >= 3:
            # Main rock body
            pygame.draw.polygon(surface, self.base_color, points)
            # Outline
            pygame.draw.polygon(surface, self.highlight_color, points, 2)

            # Craters on larger asteroids
            if self.size >= 2:
                crater_r = max(2, int(self.radius * 0.22))
                cx = int(self.x + math.cos(rot_rad + 0.6) * self.radius * 0.38)
                cy = int(self.y + math.sin(rot_rad + 0.6) * self.radius * 0.38)
                dark_crater = (
                    max(0, self.base_color[0] - 25),
                    max(0, self.base_color[1] - 25),
                    max(0, self.base_color[2] - 25),
                )
                pygame.draw.circle(surface, dark_crater, (cx, cy), crater_r)
                pygame.draw.circle(surface, self.highlight_color, (cx, cy), crater_r, 1)

    def to_dict(self) -> dict:
        """Dictionary representation matching kesslergame / world adapter."""
        return {
            "position": (self.x, self.y),
            "velocity": (self.vx, self.vy),
            "speed": self.speed,
            "size": self.size,
            "radius": self.radius,
            "mass": self.mass,
        }


class DebrisManager:
    """Manages spawning, tracking, and fragmentation of all active asteroids."""

    def __init__(self, map_size: Tuple[float, float] = (1000.0, 800.0)):
        self.map_w, self.map_h = map_size
        self.asteroids: List[Asteroid] = []

    def clear(self) -> None:
        self.asteroids.clear()

    def spawn_asteroid(
        self,
        position: Optional[Tuple[float, float]] = None,
        velocity: Optional[Tuple[float, float]] = None,
        size: int = 3,
    ) -> Asteroid:
        """Spawn a new asteroid, defaulting to random edges if position is omitted."""
        if position is None:
            # Spawn along one of the 4 outer screen borders
            edge = random.randint(0, 3)
            if edge == 0:  # Top
                x = random.uniform(0, self.map_w)
                y = 5.0
            elif edge == 1:  # Bottom
                x = random.uniform(0, self.map_w)
                y = self.map_h - 5.0
            elif edge == 2:  # Left
                x = 5.0
                y = random.uniform(0, self.map_h)
            else:  # Right
                x = self.map_w - 5.0
                y = random.uniform(0, self.map_h)
            position = (x, y)

        if velocity is None:
            # Aim generally inward toward the center arena
            target_x = self.map_w * 0.5 + random.uniform(-200, 200)
            target_y = self.map_h * 0.5 + random.uniform(-200, 200)
            angle = math.atan2(target_y - position[1], target_x - position[0])
            spd = random.uniform(40.0, 110.0)
            velocity = (math.cos(angle) * spd, math.sin(angle) * spd)

        ast = Asteroid(position=position, velocity=velocity, size=size, map_size=(self.map_w, self.map_h))
        self.asteroids.append(ast)
        return ast

    def populate_scenario(self, count: int = 6, seed: Optional[int] = None) -> None:
        """Create a balanced initial set of asteroids."""
        if seed is not None:
            random.seed(seed)

        self.clear()
        sizes = [4, 3, 3, 2, 2, 1]
        for i in range(count):
            size = sizes[i % len(sizes)]
            # Place away from central ship spawn (500, 400)
            while True:
                x = random.uniform(60, self.map_w - 60)
                y = random.uniform(60, self.map_h - 60)
                if math.hypot(x - 500.0, y - 400.0) > 160.0:
                    break
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(45.0, 95.0)
            self.spawn_asteroid(position=(x, y), velocity=(math.cos(ang) * spd, math.sin(ang) * spd), size=size)

    def split_asteroid(self, asteroid: Asteroid, split_angle_deg: float = 15.0) -> List[Asteroid]:
        """Split asteroid into 3 children with +/-15 deg fan out if size > 1."""
        if asteroid in self.asteroids:
            self.asteroids.remove(asteroid)

        if asteroid.size <= 1:
            return []

        children = []
        new_size = asteroid.size - 1
        parent_spd = asteroid.speed
        parent_ang = math.atan2(asteroid.vy, asteroid.vx)

        # 3 children fan out at -15 deg, 0 deg, +15 deg
        angles = [
            parent_ang - math.radians(split_angle_deg),
            parent_ang,
            parent_ang + math.radians(split_angle_deg),
        ]

        for ang in angles:
            # Slight random speed variance for natural movement
            spd = max(30.0, parent_spd * random.uniform(0.9, 1.15))
            vx = math.cos(ang) * spd
            vy = math.sin(ang) * spd
            # Child spawn slightly offset along fan angle
            cx = asteroid.x + math.cos(ang) * (asteroid.radius * 0.4)
            cy = asteroid.y + math.sin(ang) * (asteroid.radius * 0.4)
            child = Asteroid(position=(cx, cy), velocity=(vx, vy), size=new_size, map_size=(self.map_w, self.map_h))
            self.asteroids.append(child)
            children.append(child)

        return children

    def update(self, dt: float) -> None:
        for a in self.asteroids:
            a.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        for a in self.asteroids:
            a.draw(surface)

    def get_dicts(self) -> List[dict]:
        return [a.to_dict() for a in self.asteroids]
