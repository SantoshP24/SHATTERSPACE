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
        health: Optional[float] = None,
        wall_bounce: bool = True,
        is_penalty: bool = False,
        size_health_map: Optional[Dict[int, float]] = None,
    ):
        self.uid = uid
        self.x = float(position[0])
        self.y = float(position[1])
        self.vx = float(velocity[0])
        self.vy = float(velocity[1])
        self.size = max(1, min(4, int(size)))
        self.map_w, self.map_h = map_size
        self.wall_bounce = wall_bounce
        self.is_penalty = is_penalty

        scores = score_table or DEFAULT_SIZE_SCORES
        self.radius = SIZE_RADII.get(self.size, 20.0 * self.size)
        self.mass = SIZE_MASS.get(self.size, 50.0 * self.size)
        self.score_value = scores.get(self.size, 25)

        # Health configuration: default Small (1)=1 HP, Medium (2)=2 HP, Large (3)=3 HP, XL (4)=4 HP
        if health is not None:
            self.max_health = float(health)
        elif size_health_map is not None and self.size in size_health_map:
            self.max_health = float(size_health_map[self.size])
        else:
            self.max_health = float(self.size)
        self.health = self.max_health

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

        # Asteroid color — size-keyed palettes for clear visual differentiation
        # Size 1 (small): lighter, dusty tan/grey
        # Size 2 (medium): mid charcoal with blue-grey tint
        # Size 3 (large): dark slate / greenish grey
        # Size 4 (huge): very dark near-black with cold blue sheen
        size_base_ranges = {
            1: (95, 125, (5, 3, 0)),     # warm dusty
            2: (75, 105, (0, 2, 6)),     # cool grey-blue
            3: (60,  90, (0, 4, 2)),     # dark slate-green
            4: (45,  70, (0, 2, 8)),     # very dark cold
        }
        lo, hi, tint_base = size_base_ranges.get(self.size, (75, 110, (0, 0, 0)))
        shade = random.randint(lo, hi)
        tint = tuple(random.randint(0, t + 4) for t in tint_base)
        self.base_color = (
            min(255, shade + tint[0]),
            min(255, shade + tint[1]),
            min(255, shade + tint[2]),
        )
        self.highlight_color = (
            min(255, self.base_color[0] + 28),
            min(255, self.base_color[1] + 28),
            min(255, self.base_color[2] + 28),
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
        """Advance asteroid position with elastic wall boundary bounce (no wrapping)."""
        if self.wall_bounce:
            self.x += self.vx * dt
            self.y += self.vy * dt

            # Horizontal boundary bounce
            if self.x - self.radius <= 0 and self.vx < 0:
                self.vx = -self.vx
                self.x = self.radius
            elif self.x + self.radius >= self.map_w and self.vx > 0:
                self.vx = -self.vx
                self.x = self.map_w - self.radius

            # Vertical boundary bounce
            if self.y - self.radius <= 0 and self.vy < 0:
                self.vy = -self.vy
                self.y = self.radius
            elif self.y + self.radius >= self.map_h and self.vy > 0:
                self.vy = -self.vy
                self.y = self.map_h - self.radius
        else:
            self.x = (self.x + self.vx * dt) % self.map_w
            self.y = (self.y + self.vy * dt) % self.map_h

        self.rotation_angle = (self.rotation_angle + self.rot_speed * dt) % 360.0

    def draw(self, surface: pygame.Surface) -> None:
        """Draw jagged rock asteroid with size-appropriate detail level."""
        rot_rad = math.radians(self.rotation_angle)
        points = []
        for base_ang, r in self.poly_offsets:
            ang = base_ang + rot_rad
            px = self.x + math.cos(ang) * r
            py = self.y + math.sin(ang) * r
            points.append((px, py))

        if len(points) < 3:
            return

        # Main rock body
        pygame.draw.polygon(surface, self.base_color, points)

        # Size-based detail level
        if self.size >= 3:
            # Large: add inner face panel (brighter inner polygon)
            inner_pts = []
            inner_scale = 0.62
            for base_ang, r in self.poly_offsets:
                ang = base_ang + rot_rad
                px = self.x + math.cos(ang) * r * inner_scale
                py = self.y + math.sin(ang) * r * inner_scale
                inner_pts.append((px, py))
            panel_col = tuple(min(255, c + 18) for c in self.base_color)
            pygame.draw.polygon(surface, panel_col, inner_pts)

        # Outline
        pygame.draw.polygon(surface, self.highlight_color, points, 1)

        # Craters — larger asteroids only
        if self.size >= 2:
            # Primary crater
            cr = max(2, int(self.radius * 0.20))
            cxc = int(self.x + math.cos(rot_rad + 0.6) * self.radius * 0.36)
            cyc = int(self.y + math.sin(rot_rad + 0.6) * self.radius * 0.36)
            dark = tuple(max(0, c - 28) for c in self.base_color)
            pygame.draw.circle(surface, dark, (cxc, cyc), cr)
            pygame.draw.circle(surface, self.highlight_color, (cxc, cyc), cr, 1)

            if self.size >= 3:
                # Secondary smaller crater
                cr2 = max(1, int(self.radius * 0.12))
                cxc2 = int(self.x + math.cos(rot_rad + 2.2) * self.radius * 0.50)
                cyc2 = int(self.y + math.sin(rot_rad + 2.2) * self.radius * 0.50)
                pygame.draw.circle(surface, dark, (cxc2, cyc2), cr2)

        # Health damage cracks: draw thin split lines when asteroid is damaged
        if self.health < self.max_health and self.size >= 2:
            dmg_frac = 1.0 - self.health / max(1.0, self.max_health)
            if dmg_frac > 0.3:
                crack_col = tuple(min(255, c + 40) for c in self.base_color)
                # One crack radiating from centre
                ang_crack = rot_rad + 1.1
                cx_end = int(self.x + math.cos(ang_crack) * self.radius * 0.65)
                cy_end = int(self.y + math.sin(ang_crack) * self.radius * 0.65)
                pygame.draw.line(surface, crack_col, (int(self.x), int(self.y)), (cx_end, cy_end), 1)

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
            "health": self.health,
            "max_health": self.max_health,
            "is_penalty": getattr(self, "is_penalty", False),
        }


class RedPenaltyAsteroid(Asteroid):
    """Special environmental hazard: distinct glowing red asteroid.

    Spawns from one screen edge and travels toward the opposite edge.
    Deducts penalty when shot or collided with; disappears immediately without fragments.
    """

    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        map_size: Tuple[float, float] = (1000.0, 800.0),
        uid: int = 0,
        radius: float = 25.0,
        health: float = 1.0,
    ):
        super().__init__(
            position=position,
            velocity=velocity,
            size=2,
            map_size=map_size,
            uid=uid,
            health=health,
            wall_bounce=False,
            is_penalty=True,
        )
        self.radius = float(radius)
        self.mass = 80.0
        # Distinct Crimson / Neon Red color scheme
        self.base_color = (235, 25, 25)
        self.highlight_color = (255, 95, 95)
        self.glow_color = (255, 45, 45)

    def is_out_of_bounds(self) -> bool:
        """Check if red asteroid has fully exited the opposite arena edge."""
        pad = self.radius * 2.5
        return (
            self.x < -pad
            or self.x > self.map_w + pad
            or self.y < -pad
            or self.y > self.map_h + pad
        )

    def update(self, dt: float) -> None:
        """Advance straight across the map without boundary bounce."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.rotation_angle = (self.rotation_angle + self.rot_speed * dt) % 360.0

    def draw(self, surface: pygame.Surface) -> None:
        """Draw crimson rock with pulsating glowing red aura and warning core."""
        # Outer pulsating glow aura
        t = pygame.time.get_ticks() if pygame.get_init() else 0
        pulse = math.sin(t * 0.009) * 3.5
        glow_radius = int(self.radius * 1.35 + pulse)
        glow_surf = pygame.Surface((glow_radius * 2 + 6, glow_radius * 2 + 6), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surf,
            (*self.glow_color, 75),
            (glow_radius + 3, glow_radius + 3),
            glow_radius,
            2,
        )
        surface.blit(glow_surf, (int(self.x - glow_radius - 3), int(self.y - glow_radius - 3)))

        # Base jagged rock polygon
        super().draw(surface)

        # Danger indicator core
        pygame.draw.circle(surface, (255, 220, 220), (int(self.x), int(self.y)), 4)


class DebrisManager:
    """Manages spawning, tracking, scoring, and fragmentation of all active asteroids."""

    def __init__(
        self,
        map_size: Tuple[float, float] = (1000.0, 800.0),
        score_table: Optional[Dict[int, int]] = None,
        split_enabled: bool = True,
        wall_bounce: bool = True,
        base_health: float = 50.0,
        settings: Optional[Any] = None,
    ):
        self.map_w, self.map_h = map_size
        self.settings = settings
        self.score_table = score_table or dict(DEFAULT_SIZE_SCORES)
        self.split_enabled = split_enabled
        self.wall_bounce = wall_bounce
        self.base_health = float(base_health)

        # Size-based health map
        if settings is not None:
            self.size_health_map = {
                1: getattr(settings, "asteroid_health_small", 1.0),
                2: getattr(settings, "asteroid_health_medium", 2.0),
                3: getattr(settings, "asteroid_health_large", 3.0),
                4: 4.0,
            }
            self.fragment_count_min = getattr(settings, "fragment_count_min", 3)
            self.fragment_count_max = getattr(settings, "fragment_count_max", 3)
            self.fragment_speed_mult = getattr(settings, "fragment_speed_mult", 1.0)
            self.red_enabled = getattr(settings, "red_asteroid_enabled", True)
            self.red_max_active = getattr(settings, "red_asteroid_max_active", 1)
            self.red_spawn_prob = getattr(settings, "red_asteroid_spawn_probability", 0.05)
            self.red_spawn_interval = getattr(settings, "red_asteroid_spawn_interval", 5.0)
            self.red_speed = getattr(settings, "red_asteroid_speed", 100.0)
            self.red_radius = getattr(settings, "red_asteroid_radius", 25.0)
        else:
            self.size_health_map = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
            self.fragment_count_min = 3
            self.fragment_count_max = 3
            self.fragment_speed_mult = 1.0
            self.red_enabled = True
            self.red_max_active = 1
            self.red_spawn_prob = 0.05
            self.red_spawn_interval = 5.0
            self.red_speed = 100.0
            self.red_radius = 25.0

        self.asteroids: List[Asteroid] = []
        self._next_uid: int = 1
        self.scored_uids: Set[int] = set()
        self.red_spawn_timer: float = 0.0

    @property
    def normal_asteroids(self) -> List[Asteroid]:
        return [a for a in self.asteroids if not getattr(a, "is_penalty", False)]

    @property
    def red_asteroids(self) -> List[RedPenaltyAsteroid]:
        return [a for a in self.asteroids if getattr(a, "is_penalty", False)]

    def clear(self) -> None:
        self.asteroids.clear()
        self.scored_uids.clear()
        self._next_uid = 1
        self.red_spawn_timer = 0.0

    def award_points(self, asteroid: Asteroid) -> int:
        """Award points once per asteroid; return 0 if already scored or is red penalty."""
        if getattr(asteroid, "is_penalty", False):
            return 0
        if asteroid.uid in self.scored_uids:
            return 0
        self.scored_uids.add(asteroid.uid)
        return self.score_table.get(asteroid.size, 25)

    def spawn_red_asteroid(self, entry_edge: Optional[int] = None) -> Optional[RedPenaltyAsteroid]:
        """Spawn a special red penalty asteroid entering from one edge toward opposite edge."""
        if len(self.red_asteroids) >= self.red_max_active:
            return None

        uid = self._next_uid
        self._next_uid += 1
        r = self.red_radius
        edge = entry_edge if entry_edge is not None else random.randint(0, 3)

        if edge == 0:  # Top -> Bottom
            x = random.uniform(50.0, self.map_w - 50.0)
            y = -r
            target_x = random.uniform(50.0, self.map_w - 50.0)
            target_y = self.map_h + r
        elif edge == 1:  # Bottom -> Top
            x = random.uniform(50.0, self.map_w - 50.0)
            y = self.map_h + r
            target_x = random.uniform(50.0, self.map_w - 50.0)
            target_y = -r
        elif edge == 2:  # Left -> Right
            x = -r
            y = random.uniform(50.0, self.map_h - 50.0)
            target_x = self.map_w + r
            target_y = random.uniform(50.0, self.map_h - 50.0)
        else:  # Right -> Left
            x = self.map_w + r
            y = random.uniform(50.0, self.map_h - 50.0)
            target_x = -r
            target_y = random.uniform(50.0, self.map_h - 50.0)

        angle = math.atan2(target_y - y, target_x - x)
        vx = math.cos(angle) * self.red_speed
        vy = math.sin(angle) * self.red_speed

        red_hp = getattr(self.settings, "red_asteroid_health", 1.0) if self.settings else 1.0
        red_ast = RedPenaltyAsteroid(
            position=(x, y),
            velocity=(vx, vy),
            map_size=(self.map_w, self.map_h),
            uid=uid,
            radius=r,
            health=red_hp,
        )
        self.asteroids.append(red_ast)
        return red_ast

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
            health=self.base_health if self.settings is None else self.size_health_map.get(size, float(size)),
            wall_bounce=self.wall_bounce,
            size_health_map=self.size_health_map,
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

    def split_asteroid(
        self,
        asteroid: Asteroid,
        split_angle_deg: float = 15.0,
        avoid_pos: Optional[Tuple[float, float]] = None,
    ) -> List[Asteroid]:
        """Split asteroid into 2-3 children with +/-15 deg fan out if size > 1 and splitting enabled.
        
        Red penalty asteroids never split.
        """
        if asteroid in self.asteroids:
            self.asteroids.remove(asteroid)

        if getattr(asteroid, "is_penalty", False) or not self.split_enabled or asteroid.size <= 1:
            return []

        children = []
        new_size = asteroid.size - 1
        parent_spd = asteroid.speed
        parent_ang = math.atan2(asteroid.vy, asteroid.vx)

        # Configurable child count (2 to 3 per specification)
        count = random.randint(self.fragment_count_min, self.fragment_count_max)
        if count == 2:
            angles = [
                parent_ang - math.radians(split_angle_deg),
                parent_ang + math.radians(split_angle_deg),
            ]
        elif count == 3:
            angles = [
                parent_ang - math.radians(split_angle_deg),
                parent_ang,
                parent_ang + math.radians(split_angle_deg),
            ]
        else:
            angles = [
                parent_ang + (i - (count - 1) / 2.0) * math.radians(split_angle_deg)
                for i in range(count)
            ]

        for ang in angles:
            spd = max(30.0, parent_spd * self.fragment_speed_mult * random.uniform(0.92, 1.12))
            vx = math.cos(ang) * spd
            vy = math.sin(ang) * spd
            cx = asteroid.x + math.cos(ang) * (asteroid.radius * 0.4)
            cy = asteroid.y + math.sin(ang) * (asteroid.radius * 0.4)

            # Avoid spawning fragments inside the ship
            if avoid_pos is not None:
                d = math.hypot(cx - avoid_pos[0], cy - avoid_pos[1])
                safe_d = 25.0 + SIZE_RADII.get(new_size, 15.0)
                if d < safe_d and d > 1e-4:
                    nx = (cx - avoid_pos[0]) / d
                    ny = (cy - avoid_pos[1]) / d
                    cx = avoid_pos[0] + nx * (safe_d + 5.0)
                    cy = avoid_pos[1] + ny * (safe_d + 5.0)

            uid = self._next_uid
            self._next_uid += 1

            child_hp = self.size_health_map.get(new_size, float(new_size))
            child = Asteroid(
                position=(cx, cy),
                velocity=(vx, vy),
                size=new_size,
                map_size=(self.map_w, self.map_h),
                uid=uid,
                score_table=self.score_table,
                health=child_hp,
                wall_bounce=self.wall_bounce,
                size_health_map=self.size_health_map,
            )
            self.asteroids.append(child)
            children.append(child)

        return children

    def update(self, dt: float) -> None:
        """Advance all active debris, prune out-of-bounds red asteroids, and handle red asteroid spawns."""
        for a in list(self.asteroids):
            a.update(dt)
            # Remove red asteroids that have crossed the opposite boundary
            if getattr(a, "is_penalty", False) and getattr(a, "is_out_of_bounds", lambda: False)():
                if a in self.asteroids:
                    self.asteroids.remove(a)

        # Periodic random check to spawn special red penalty asteroid
        if self.red_enabled and len(self.red_asteroids) < self.red_max_active:
            self.red_spawn_timer += dt
            if self.red_spawn_timer >= self.red_spawn_interval:
                self.red_spawn_timer = 0.0
                if random.random() < self.red_spawn_prob:
                    self.spawn_red_asteroid()

    def draw(self, surface: pygame.Surface) -> None:
        for a in self.asteroids:
            a.draw(surface)

    def get_dicts(self) -> List[dict]:
        return [a.to_dict() for a in self.asteroids]
