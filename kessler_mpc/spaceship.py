"""Spaceship physics and rendering base for Kessler AI survival game.

Matches the validated ship dynamics from dynamics.py while adding shields,
health, afterburner boost, laser weapons, and animated ship sprites.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pygame

from .dynamics import ShipParams, speed_step, step


class Bullet:
    """Laser projectile fired by player or AI ship."""

    def __init__(
        self,
        position: Tuple[float, float],
        heading_deg: float,
        speed: float = 800.0,
        owner_id: int = 0,
        lifespan: float = 1.4,
    ):
        self.x = float(position[0])
        self.y = float(position[1])
        rad = math.radians(heading_deg)
        self.vx = math.cos(rad) * speed
        self.vy = math.sin(rad) * speed
        self.heading = heading_deg
        self.owner_id = owner_id
        self.life = lifespan
        self.max_life = lifespan
        self.radius = 3.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    def update(self, dt: float) -> bool:
        """Advance bullet. Returns False if expired."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surface: pygame.Surface, color: Tuple[int, int, int] = (100, 240, 255)) -> None:
        """Render elongated laser bolt."""
        rad = math.radians(self.heading)
        tail_x = self.x - math.cos(rad) * 12.0
        tail_y = self.y - math.sin(rad) * 12.0
        pygame.draw.line(surface, color, (int(tail_x), int(tail_y)), (int(self.x), int(self.y)), 3)
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), 2)


class Spaceship:
    """Base spaceship with Newtonian flight, shields, health, boost, and weapons."""

    def __init__(
        self,
        ship_id: int = 0,
        position: Tuple[float, float] = (500.0, 400.0),
        heading: float = 90.0,
        map_size: Tuple[float, float] = (1000.0, 800.0),
        is_ai: bool = False,
    ):
        self.ship_id = ship_id
        self.is_ai = is_ai
        self.map_w, self.map_h = map_size

        # Dynamics parameters matching engine
        self.params = ShipParams(
            thrust_min=-480.0,
            thrust_max=480.0,
            turn_max=180.0,
            max_speed=240.0,
            drag=80.0,
            radius=20.0,
            map_w=self.map_w,
            map_h=self.map_h,
            frame_dt=1.0 / 60.0,
        )

        # State
        self.x = float(position[0])
        self.y = float(position[1])
        self.heading = float(heading)
        self.speed = 0.0

        # Health & Shields
        self.max_health = 100.0
        self.health = 100.0
        self.max_shields = 100.0
        self.shields = 100.0
        self.shield_recharge_rate = 18.0  # points/s
        self.time_since_last_hit = 10.0
        self.lives = 3
        self.is_alive = True
        self.is_respawning = False
        self.respawn_timer = 0.0
        self.respawn_duration = 3.0

        # Boost afterburner
        self.boost_energy = 100.0  # 0..100%
        self.boost_consume_rate = 45.0  # % per sec
        self.boost_recharge_rate = 22.0  # % per sec
        self.is_boosting = False

        # Weapons
        self.fire_cooldown = 0.0
        self.fire_rate = 0.14  # seconds between shots
        self.bullets: List[Bullet] = []
        self.bullets_remaining = 250
        self.shots_fired = 0
        self.shots_hit = 0

        # Score & Stats
        self.score = 0
        self.asteroids_destroyed = 0
        self.deaths = 0

        # Visual styling
        if is_ai:
            self.hull_color = (230, 70, 70)       # Crimson / Gold
            self.accent_color = (255, 200, 50)
            self.shield_color = (255, 120, 200)
        else:
            self.hull_color = (40, 160, 255)      # Cyan / Cobalt Blue
            self.accent_color = (180, 230, 255)
            self.shield_color = (80, 220, 255)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        rad = math.radians(self.heading)
        return self.speed * np.array([math.cos(rad), math.sin(rad)], dtype=float)

    def respawn(self, spawn_pos: Optional[Tuple[float, float]] = None) -> None:
        """Reset ship position and grant respawn invulnerability."""
        if spawn_pos is not None:
            self.x, self.y = float(spawn_pos[0]), float(spawn_pos[1])
        else:
            self.x, self.y = self.map_w * 0.5, self.map_h * 0.5

        self.speed = 0.0
        self.heading = 90.0
        self.health = self.max_health
        self.shields = self.max_shields
        self.boost_energy = 100.0
        self.is_respawning = True
        self.respawn_timer = self.respawn_duration

    def apply_damage(self, amount: float) -> bool:
        """Damage shields first, then hull. Returns True if ship died."""
        if self.is_respawning or not self.is_alive:
            return False

        self.time_since_last_hit = 0.0

        if self.shields > 0:
            absorbed = min(self.shields, amount)
            self.shields -= absorbed
            amount -= absorbed

        if amount > 0:
            self.health -= amount

        if self.health <= 0:
            self.health = 0.0
            self.deaths += 1
            self.lives -= 1
            if self.lives <= 0:
                self.is_alive = False
            return True
        return False

    def fire(self) -> Optional[Bullet]:
        """Attempt to spawn a bullet from the ship's forward nose."""
        if self.fire_cooldown > 0 or not self.is_alive:
            return None

        # Firing cancels respawn invulnerability (per engine rules)
        self.is_respawning = False

        rad = math.radians(self.heading)
        muzzle_x = self.x + math.cos(rad) * (self.params.radius + 4.0)
        muzzle_y = self.y + math.sin(rad) * (self.params.radius + 4.0)

        bullet = Bullet(
            position=(muzzle_x, muzzle_y),
            heading_deg=self.heading,
            speed=800.0,
            owner_id=self.ship_id,
        )
        self.bullets.append(bullet)
        self.fire_cooldown = self.fire_rate
        self.shots_fired += 1
        return bullet

    def update_physics(
        self,
        thrust_input: float,
        turn_input: float,
        dt: float,
        boost_active: bool = False,
    ) -> Tuple[bool, Tuple[float, float]]:
        """Advance ship dynamics using the exact engine speed_step & step math."""
        # Handle weapon cooldown
        if self.fire_cooldown > 0:
            self.fire_cooldown = max(0.0, self.fire_cooldown - dt)

        # Handle respawn timer
        if self.is_respawning:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0:
                self.is_respawning = False

        # Handle shield recharge
        self.time_since_last_hit += dt
        if self.time_since_last_hit > 2.5 and self.shields < self.max_shields:
            self.shields = min(self.max_shields, self.shields + self.shield_recharge_rate * dt)

        # Handle boost mechanics
        self.is_boosting = False
        applied_thrust = float(np.clip(thrust_input, self.params.thrust_min, self.params.thrust_max))
        applied_turn = float(np.clip(turn_input, -self.params.turn_max, self.params.turn_max))

        if boost_active and self.boost_energy > 5.0 and applied_thrust > 0:
            self.is_boosting = True
            self.boost_energy = max(0.0, self.boost_energy - self.boost_consume_rate * dt)
            applied_thrust *= 1.85
        else:
            self.boost_energy = min(100.0, self.boost_energy + self.boost_recharge_rate * dt)

        # Advance physics using the engine's step function
        new_pos, new_spd, new_head = step(
            pos=np.array([[self.x, self.y]], dtype=float),
            speed=np.array([self.speed], dtype=float),
            heading=np.array([self.heading], dtype=float),
            thrust=np.array([applied_thrust], dtype=float),
            turn=np.array([applied_turn], dtype=float),
            p=self.params,
            dt=dt,
            wrap=True,
        )

        self.x = float(new_pos[0, 0])
        self.y = float(new_pos[0, 1])
        self.speed = float(new_spd[0])
        self.heading = float(new_head[0])

        # Update active bullets
        dead_b = []
        for b in self.bullets:
            if not b.update(dt):
                dead_b.append(b)
            # Remove bullets leaving map bounds (bullets don't wrap)
            elif b.x < 0 or b.x > self.map_w or b.y < 0 or b.y > self.map_h:
                dead_b.append(b)

        for b in dead_b:
            self.bullets.remove(b)

        # Return nozzle position for thruster particle generation
        rad = math.radians(self.heading + 180.0)
        nozzle = (self.x + math.cos(rad) * 14.0, self.y + math.sin(rad) * 14.0)
        is_thrusting = applied_thrust > 10.0 or self.is_boosting
        return is_thrusting, nozzle

    def draw(self, surface: pygame.Surface) -> None:
        """Render ship geometry, cockpit glass, and shield aura."""
        if not self.is_alive:
            return

        rad = math.radians(self.heading)
        r = self.params.radius

        # Ship polygon vertices (Sleek sci-fi wedge)
        nose = (self.x + math.cos(rad) * (r * 1.3), self.y + math.sin(rad) * (r * 1.3))
        left_wing = (
            self.x + math.cos(rad + 2.45) * (r * 1.15),
            self.y + math.sin(rad + 2.45) * (r * 1.15),
        )
        engine_notch = (self.x - math.cos(rad) * (r * 0.5), self.y - math.sin(rad) * (r * 0.5))
        right_wing = (
            self.x + math.cos(rad - 2.45) * (r * 1.15),
            self.y + math.sin(rad - 2.45) * (r * 1.15),
        )

        # Blink effect when respawn-invulnerable
        if self.is_respawning and int(self.respawn_timer * 10) % 2 == 0:
            alpha = 100
        else:
            alpha = 255

        # Hull polygon
        pygame.draw.polygon(surface, self.hull_color, [nose, left_wing, engine_notch, right_wing])
        pygame.draw.polygon(surface, self.accent_color, [nose, left_wing, engine_notch, right_wing], 2)

        # Cockpit canopy glass
        cockpit_pos = (self.x + math.cos(rad) * (r * 0.35), self.y + math.sin(rad) * (r * 0.35))
        pygame.draw.circle(surface, (200, 255, 255), (int(cockpit_pos[0]), int(cockpit_pos[1])), 4)

        # Shield bubble
        if self.shields > 10.0 or self.is_respawning:
            shield_alpha = int(90 * (self.shields / self.max_shields)) if not self.is_respawning else 140
            shield_surf = pygame.Surface((int(r * 3.0), int(r * 3.0)), pygame.SRCALPHA)
            color = (*self.shield_color, shield_alpha)
            pygame.draw.circle(shield_surf, color, (int(r * 1.5), int(r * 1.5)), int(r * 1.35), 2)
            surface.blit(shield_surf, (int(self.x - r * 1.5), int(self.y - r * 1.5)))

        # Draw bullets
        for b in self.bullets:
            b.draw(surface)

    def to_dict(self) -> dict:
        """Dictionary adapter for World / KesslerMPC."""
        return {
            "id": self.ship_id,
            "position": (self.x, self.y),
            "velocity": (self.velocity[0], self.velocity[1]),
            "speed": self.speed,
            "heading": self.heading,
            "radius": self.params.radius,
            "lives": self.lives,
            "health": self.health,
            "shields": self.shields,
            "max_speed": self.params.max_speed,
            "drag": self.params.drag,
            "thrust_range": (self.params.thrust_min, self.params.thrust_max),
            "turn_rate_range": (-self.params.turn_max, self.params.turn_max),
            "can_fire": self.fire_cooldown <= 0,
            "bullets_remaining": self.bullets_remaining,
            "is_respawning": self.is_respawning,
            "respawn_time_left": self.respawn_timer,
        }
