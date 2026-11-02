"""Physics engine and collision resolution for Kessler AI survival game.

Handles:
- Ship-asteroid elastic collision response and shield/hull damage calculation.
- Bullet-asteroid ray/circle intersection and fragmentation triggers.
- Ship-ship collisions for co-op/dual gameplay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .debris import Asteroid, DebrisManager
from .spaceship import Bullet, Spaceship


@dataclass
class CollisionEvent:
    event_type: str  # "ship_asteroid", "bullet_asteroid", "ship_ship"
    pos: Tuple[float, float]
    intensity: float = 1.0
    ship_id: Optional[int] = None
    asteroid_size: int = 1


class PhysicsEngine:
    """Detects and resolves physical interactions between game entities."""

    def __init__(self, map_size: Tuple[float, float] = (1000.0, 800.0)):
        self.map_w, self.map_h = map_size

    def check_bullet_asteroid_collisions(
        self,
        ships: List[Spaceship],
        debris_mgr: DebrisManager,
    ) -> List[CollisionEvent]:
        """Check all active bullets against asteroids."""
        events: List[CollisionEvent] = []

        for ship in ships:
            dead_bullets = []
            for b in ship.bullets:
                bullet_consumed = False
                for a in list(debris_mgr.asteroids):
                    # Distance check
                    dist = math.hypot(b.x - a.x, b.y - a.y)
                    if dist <= (b.radius + a.radius):
                        # Hit detected!
                        bullet_consumed = True
                        ship.shots_hit += 1
                        ship.score += a.score_value
                        ship.asteroids_destroyed += 1

                        events.append(
                            CollisionEvent(
                                event_type="bullet_asteroid",
                                pos=(b.x, b.y),
                                intensity=float(a.size),
                                ship_id=ship.ship_id,
                                asteroid_size=a.size,
                            )
                        )

                        # Split or destroy asteroid
                        debris_mgr.split_asteroid(a)
                        break

                if bullet_consumed:
                    dead_bullets.append(b)

            for b in dead_bullets:
                if b in ship.bullets:
                    ship.bullets.remove(b)

        return events

    def check_ship_asteroid_collisions(
        self,
        ships: List[Spaceship],
        debris_mgr: DebrisManager,
    ) -> List[CollisionEvent]:
        """Check all active ships against asteroids."""
        events: List[CollisionEvent] = []

        for ship in ships:
            if not ship.is_alive or ship.is_respawning:
                continue

            ship_vel = ship.velocity

            for a in list(debris_mgr.asteroids):
                dist = math.hypot(ship.x - a.x, ship.y - a.y)
                r_sum = ship.params.radius + a.radius

                if dist <= r_sum:
                    # Normal vector pointing from asteroid to ship
                    if dist > 1e-4:
                        nx = (ship.x - a.x) / dist
                        ny = (ship.y - a.y) / dist
                    else:
                        nx, ny = 1.0, 0.0

                    # Relative velocity
                    rel_vx = ship_vel[0] - a.vx
                    rel_vy = ship_vel[1] - a.vy
                    rel_speed = math.hypot(rel_vx, rel_vy)

                    # Damage proportional to size and impact speed
                    damage = 25.0 * a.size + 0.12 * rel_speed
                    ship_died = ship.apply_damage(damage)

                    # Physical impulse bounce: push ship and asteroid apart
                    push = 80.0 + 15.0 * a.size
                    ship.x = (ship.x + nx * 12.0) % self.map_w
                    ship.y = (ship.y + ny * 12.0) % self.map_h
                    ship.speed = max(-100.0, min(100.0, ship.speed * 0.4))

                    # Asteroid deflects slightly
                    a.vx -= nx * (push / a.mass) * 15.0
                    a.vy -= ny * (push / a.mass) * 15.0

                    events.append(
                        CollisionEvent(
                            event_type="ship_asteroid",
                            pos=((ship.x + a.x) * 0.5, (ship.y + a.y) * 0.5),
                            intensity=min(3.0, damage / 30.0),
                            ship_id=ship.ship_id,
                            asteroid_size=a.size,
                        )
                    )

                    # Smaller asteroids shatter on ship impact
                    if a.size <= 2:
                        debris_mgr.split_asteroid(a)

                    if ship_died:
                        if ship.lives > 0:
                            ship.respawn()
                    break

        return events

    def check_ship_ship_collisions(self, ships: List[Spaceship]) -> List[CollisionEvent]:
        """Check collisions between multiple ships."""
        events: List[CollisionEvent] = []
        n = len(ships)
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = ships[i], ships[j]
                if not s1.is_alive or not s2.is_alive:
                    continue

                dist = math.hypot(s1.x - s2.x, s1.y - s2.y)
                r_sum = s1.params.radius + s2.params.radius

                if dist <= r_sum:
                    nx = (s1.x - s2.x) / max(dist, 1e-4)
                    ny = (s1.y - s2.y) / max(dist, 1e-4)

                    s1.x = (s1.x + nx * 5.0) % self.map_w
                    s1.y = (s1.y + ny * 5.0) % self.map_h
                    s2.x = (s2.x - nx * 5.0) % self.map_w
                    s2.y = (s2.y - ny * 5.0) % self.map_h

                    s1.apply_damage(10.0)
                    s2.apply_damage(10.0)

                    events.append(
                        CollisionEvent(
                            event_type="ship_ship",
                            pos=((s1.x + s2.x) * 0.5, (s1.y + s2.y) * 0.5),
                            intensity=1.0,
                        )
                    )
        return events
