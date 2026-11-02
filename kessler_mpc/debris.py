"""Asteroid and space debris management for Kessler AI survival game.

Implements procedural rock silhouettes, realistic rotation, exact engine fragmentation
(3 child asteroids splitting with +/-15 deg fan out), unique asteroid tracking to prevent
duplicate scoring, and safe spawning to prevent unavoidable collisions.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pygame

# Size mapping matching KesslerGame / Asteroid dynamics
SIZE_RADII = {1: 15.0, 2: 30.0, 3: 50.0, 4: 80.0}
SIZE_MASS = {1: 25.0, 2: 75.0, 3: 200.0, 4: 600.0}
DEFAULT_SIZE_SCORES = {1: 200, 2: 100, 3: 50, 4: 25}


class Asteroid:
    """Individual asteroid / debris object with procedural polygon geometry and unique ID."""

    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        size: int = 3,
        map_size: Tuple[float, float] = (1000.0, 800.0),
        uid: int = 0,
        score_table: Optional[Dict[int, int]] = None,
    ):
        self.uid = uid
        self.x = float(position[0])
        self.y = float(position[1])
        self.vx = float(velocity[0])
        self.vy = float(velocity[1])
        self.size = max(1, min(4, int(size)))
        self.map_w, self.map_h = map_size

        scores = score_table or DEFAULT_SIZE_SCORES
        self.radius = SIZE_RADII.get(self.size, 20.0 * self.size)
        self.mass = SIZE_MASS.get(self.size, 50.0 * self.size)
        self.score_value = scores.get(self.size, 25)

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
            "id": self.uid,
            "position": (self.x, self.y),
            "velocity": (self.vx, self.vy),
            "speed": self.speed,
            "size": self.size,
            "radius": self.radius,
            "mass": self.mass,
        }


class DebrisManager:
    """Manages spawning, tracking, scoring, and fragmentation of all active asteroids."""

    def __init__(
        self,
        map_size: Tuple[float, float] = (1000.0, 800.0),
        score_table: Optional[Dict[int, int]] = None,
    ):
        self.map_w, self.map_h = map_size
        self.score_table = score_table or dict(DEFAULT_SIZE_SCORES)
        self.asteroids: List[Asteroid] = []
        self._next_uid: int = 1
        self.scored_uids: Set[int] = set()

    def clear(self) -> None:
        self.asteroids.clear()
        self.scored_uids.clear()
        self._next_uid = 1

    def award_points(self, asteroid: Asteroid) -> int:
        """Award points once per asteroid; return 0 if already scored."""
        if asteroid.uid in self.scored_uids:
            return 0
        self.scored_uids.add(asteroid.uid)
        return self.score_table.get(asteroid.size, 25)

    def spawn_asteroid(
        self,
        position: Optional[Tuple[float, float]] = None,
        velocity: Optional[Tuple[float, float]] = None,
        size: int = 3,
        avoid_pos: Optional[Tuple[float, float]] = None,
        min_distance: float = 220.0,
    ) -> Asteroid:
        """Spawn an asteroid, guaranteeing safe distance and non-instant-collision trajectories."""
        uid = self._next_uid
        self._next_uid += 1

        if position is None:
            # Try up to 10 candidates along perimeter until safe distance from avoid_pos
            best_pos = None
            for _ in range(12):
                edge = random.randint(0, 3)
                if edge == 0:  # Top
                    cand = (random.uniform(0, self.map_w), 5.0)
                elif edge == 1:  # Bottom
                    cand = (random.uniform(0, self.map_w), self.map_h - 5.0)
                elif edge == 2:  # Left
                    cand = (5.0, random.uniform(0, self.map_h))
                else:  # Right
                    cand = (self.map_w - 5.0, random.uniform(0, self.map_h))

                if avoid_pos is not None:
                    d = math.hypot(cand[0] - avoid_pos[0], cand[1] - avoid_pos[1])
                    if d >= min_distance:
                        best_pos = cand
                        break
                else:
                    best_pos = cand
                    break

            position = best_pos or (5.0, 5.0)

        if velocity is None:
            # Aim generally inward, but divert away from direct line to ship
            target_x = self.map_w * 0.5 + random.uniform(-250, 250)
            target_y = self.map_h * 0.5 + random.uniform(-250, 250)

            if avoid_pos is not None:
                # If target point is too close to ship, offset it
                if math.hypot(target_x - avoid_pos[0], target_y - avoid_pos[1]) < min_distance:
                    target_x = (target_x + 300.0) % self.map_w
                    target_y = (target_y + 300.0) % self.map_h

            angle = math.atan2(target_y - position[1], target_x - position[0])
            spd = random.uniform(40.0, 105.0)
            velocity = (math.cos(angle) * spd, math.sin(angle) * spd)

        ast = Asteroid(
            position=position,
            velocity=velocity,
            size=size,
            map_size=(self.map_w, self.map_h),
            uid=uid,
            score_table=self.score_table,
        )
        self.asteroids.append(ast)
        return ast

    def populate_scenario(
        self,
        count: int = 6,
        seed: Optional[int] = None,
        avoid_pos: Tuple[float, float] = (500.0, 400.0),
    ) -> None:
        """Create a balanced initial set of asteroids spaced safely away from ship."""
        if seed is not None:
            random.seed(seed)

        self.clear()
        sizes = [4, 3, 3, 2, 2, 1]
        for i in range(count):
            size = sizes[i % len(sizes)]
            # Place at safe distance from ship spawn
            while True:
                x = random.uniform(60, self.map_w - 60)
                y = random.uniform(60, self.map_h - 60)
                if math.hypot(x - avoid_pos[0], y - avoid_pos[1]) > 180.0:
                    break
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(45.0, 90.0)
            self.spawn_asteroid(
                position=(x, y),
                velocity=(math.cos(ang) * spd, math.sin(ang) * spd),
                size=size,
                avoid_pos=avoid_pos,
            )

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
            spd = max(30.0, parent_spd * random.uniform(0.92, 1.12))
            vx = math.cos(ang) * spd
            vy = math.sin(ang) * spd
            cx = asteroid.x + math.cos(ang) * (asteroid.radius * 0.4)
            cy = asteroid.y + math.sin(ang) * (asteroid.radius * 0.4)

            uid = self._next_uid
            self._next_uid += 1

            child = Asteroid(
                position=(cx, cy),
                velocity=(vx, vy),
                size=new_size,
                map_size=(self.map_w, self.map_h),
                uid=uid,
                score_table=self.score_table,
            )
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
