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
        damage: float = 50.0,
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
        self.damage = float(damage)

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
        """Render a slim plasma bolt: bright core with a short trailing tail."""
        rad = math.radians(self.heading)
        tail_x = self.x - math.cos(rad) * 9.0
        tail_y = self.y - math.sin(rad) * 9.0
        # Tail — slightly dimmer
        tail_col = (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60))
        pygame.draw.line(surface, tail_col, (int(tail_x), int(tail_y)), (int(self.x), int(self.y)), 2)
        # Bright head
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), 2)


class Spaceship:
    """Base spaceship with Newtonian flight, shields, health, boost, and weapons."""

    def __init__(
        self,
        ship_id: int = 0,
        position: Tuple[float, float] = (500.0, 400.0),
        heading: float = 90.0,
        map_size: Tuple[float, float] = (1000.0, 800.0),
        is_ai: bool = False,
        settings: Optional[Any] = None,
        max_speed: Optional[float] = None,
        thrust_max: Optional[float] = None,
        turn_max: Optional[float] = None,
        max_health: Optional[float] = None,
        lives: Optional[int] = None,
        fire_rate: Optional[float] = None,
        bullet_speed: Optional[float] = None,
        bullet_damage: Optional[float] = None,
        collision_safety_margin: Optional[float] = None,
        ship_design: Optional[str] = None,
    ):
        self.ship_id = ship_id
        self.is_ai = is_ai
        self.map_w, self.map_h = map_size
        self.ship_design = ship_design or (getattr(settings, "ship_design", "Interceptor") if settings else "Interceptor")

        # Resolve dynamics & attributes from settings or parameters
        spd_cap = max_speed or (getattr(settings, "ship_max_speed", 240.0) if settings else 240.0)
        thrust_cap = thrust_max or (getattr(settings, "ship_thrust_max", 480.0) if settings else 480.0)
        turn_cap = turn_max or (getattr(settings, "ship_turn_max", 180.0) if settings else 180.0)

        # Dynamics parameters matching engine
        self.params = ShipParams(
            thrust_min=-thrust_cap,
            thrust_max=thrust_cap,
            turn_max=turn_cap,
            max_speed=spd_cap,
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
        resolved_health = max_health or (getattr(settings, "ship_max_health", 100.0) if settings else 100.0)
        self.max_health = float(resolved_health)
        self.health = float(resolved_health)
        self.respawn_health = float(resolved_health)
        self.max_shields = 100.0
        self.shields = 100.0
        self.shield_recharge_rate = 18.0  # points/s
        self.time_since_last_hit = 10.0
        self.lives = lives if lives is not None else (getattr(settings, "ship_starting_lives", 3) if settings else 3)
        self.is_alive = True
        self.is_respawning = False
        self.respawn_timer = 0.0
        self.respawn_duration = getattr(settings, "respawn_invulnerability", 3.0) if settings else 3.0
        self.collision_invuln_timer = 0.0
        self.normal_collision_invulnerability = (
            getattr(settings, "normal_collision_invulnerability", 1.5) if settings else 1.5
        )
        self.health_regen_enabled = getattr(settings, "health_regen_enabled", False) if settings else False
        self.health_regen_rate = getattr(settings, "health_regen_rate", 5.0) if settings else 5.0

        # Boost afterburner
        self.boost_energy = 100.0  # 0..100%
        self.boost_consume_rate = 45.0  # % per sec
        self.boost_recharge_rate = 22.0  # % per sec
        self.is_boosting = False

        # Weapons & Projectiles
        self.fire_cooldown = 0.0
        if fire_rate is not None:
            self.fire_rate = float(fire_rate)
        elif settings is not None:
            self.fire_rate = float(getattr(settings, "weapon_cooldown", 0.14))
        else:
            self.fire_rate = 0.14

        self.bullet_speed = bullet_speed or (getattr(settings, "bullet_speed", 800.0) if settings else 800.0)
        self.bullet_damage = bullet_damage or (getattr(settings, "bullet_damage", 50.0) if settings else 50.0)
        self.collision_safety_margin = collision_safety_margin or (
            getattr(settings, "collision_safety_margin", 35.0) if settings else 35.0
        )
        self.bullets: List[Bullet] = []
        self.bullets_remaining = 250
        self.shots_fired = 0
        self.shots_hit = 0

        # Score & Stats
        self.score = 0
        self.asteroids_destroyed = 0
        self.damage_taken = 0.0
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
    def kills(self) -> int:
        return self.asteroids_destroyed

    @property
    def accuracy(self) -> float:
        return (self.shots_hit / max(1, self.shots_fired))

    @property
    def is_invulnerable(self) -> bool:
        """Returns True if the ship is immune to collision damage (respawn or post-hit)."""
        return self.is_respawning or self.collision_invuln_timer > 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        rad = math.radians(self.heading)
        return self.speed * np.array([math.cos(rad), math.sin(rad)], dtype=float)

    def respawn(self, spawn_pos: Optional[Tuple[float, float]] = None) -> None:
        """Reset ship position, restore configured respawn health, and grant respawn invulnerability."""
        if spawn_pos is not None:
            self.x, self.y = float(spawn_pos[0]), float(spawn_pos[1])
        else:
            self.x, self.y = self.map_w * 0.5, self.map_h * 0.5

        self.speed = 0.0
        self.heading = 90.0
        self.health = float(self.respawn_health)
        self.shields = self.max_shields
        self.boost_energy = 100.0
        self.is_respawning = True
        self.respawn_timer = self.respawn_duration
        self.collision_invuln_timer = 0.0

    def apply_damage(self, amount: float) -> bool:
        """Apply damage directly to health and shields. Returns True if ship died."""
        if self.is_invulnerable or not self.is_alive:
            return False

        self.time_since_last_hit = 0.0
        amt = float(amount)
        self.damage_taken += amt

        # Damage shields if present
        if self.shields > 0:
            self.shields = max(0.0, self.shields - amt)

        # Direct health reduction per formula: SHIP_HP_NEW = max(0, SHIP_HP_OLD - DAMAGE)
        self.health = max(0.0, self.health - amt)

        # Grant normal collision invulnerability period to prevent multiple collision triggers
        self.collision_invuln_timer = self.normal_collision_invulnerability

        if self.health <= 0.0:
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
            speed=self.bullet_speed,
            owner_id=self.ship_id,
            damage=self.bullet_damage,
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

        # Handle collision invulnerability timer
        if self.collision_invuln_timer > 0:
            self.collision_invuln_timer = max(0.0, self.collision_invuln_timer - dt)

        # Handle respawn timer
        if self.is_respawning:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0:
                self.is_respawning = False

        # Handle optional health regeneration
        if self.health_regen_enabled and self.is_alive and self.health < self.max_health:
            self.health = min(self.max_health, self.health + self.health_regen_rate * dt)

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

        # Advance physics using the engine's step function (NO SCREEN WRAPPING)
        new_pos, new_spd, new_head = step(
            pos=np.array([[self.x, self.y]], dtype=float),
            speed=np.array([self.speed], dtype=float),
            heading=np.array([self.heading], dtype=float),
            thrust=np.array([applied_thrust], dtype=float),
            turn=np.array([applied_turn], dtype=float),
            p=self.params,
            dt=dt,
            wrap=False,
        )

        px = float(new_pos[0, 0])
        py = float(new_pos[0, 1])
        spd = float(new_spd[0])
        head = float(new_head[0])

        # Enforce arena boundary collision constraints (no wrapping)
        r = self.params.radius
        rad = math.radians(head)
        vx = math.cos(rad) * spd
        vy = math.sin(rad) * spd

        if px < r:
            px = r
            if vx < 0:
                spd = 0.0
        elif px > self.map_w - r:
            px = self.map_w - r
            if vx > 0:
                spd = 0.0

        if py < r:
            py = r
            if vy < 0:
                spd = 0.0
        elif py > self.map_h - r:
            py = self.map_h - r
            if vy > 0:
                spd = 0.0

        self.x = px
        self.y = py
        self.speed = spd
        self.heading = head

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
        """Render detailed sci-fi spacecraft with hull panels, wings, engine nozzles, and cockpit."""
        if not self.is_alive:
            return

        rad = math.radians(self.heading)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        # Perpendicular (port/starboard) direction
        perp_r = rad + math.pi * 0.5
        cos_p, sin_p = math.cos(perp_r), math.sin(perp_r)
        r = self.params.radius
        cx, cy = self.x, self.y

        # --- Invulnerability blink ---
        is_invuln = self.is_invulnerable
        blink_on = not (is_invuln and int((self.respawn_timer + self.collision_invuln_timer) * 10) % 2 == 0)
        if not blink_on:
            # Draw bullets even when blinking
            for b in self.bullets:
                b.draw(surface)
            return

        # -----------------------------------------------------------
        # Helper: offset a point forward (fwd) and sideways (side)
        def pt(fwd: float, side: float) -> Tuple[float, float]:
            return (cx + cos_r * fwd + cos_p * side, cy + sin_r * fwd + sin_p * side)

        # Colour scheme
        h1 = self.hull_color         # primary hull
        h2 = self.accent_color       # accent / trim
        # Darker underside / panel shade
        shade = tuple(max(0, int(c * 0.55)) for c in h1)
        mid   = tuple(min(255, int(c * 0.80)) for c in h1)

        # -----------------------------------------------------------
        # Draw according to selected ship design
        design = getattr(self, "ship_design", "Interceptor")
        if design == "Scout":
            self._draw_scout(surface, pt, r, h1, h2, shade, mid)
        elif design == "Vanguard":
            self._draw_vanguard(surface, pt, r, h1, h2, shade, mid)
        else:
            self._draw_interceptor(surface, pt, r, h1, h2, shade, mid)

        # -----------------------------------------------------------
        # Shield bubble (only when active)
        if self.shields > 10.0 or is_invuln:
            shield_alpha = int(70 * (self.shields / self.max_shields)) if not is_invuln else 110
            sr = int(r * 1.5)
            shield_surf = pygame.Surface((sr * 2, sr * 2), pygame.SRCALPHA)
            pygame.draw.circle(shield_surf, (*self.shield_color, shield_alpha), (sr, sr), sr, 2)
            surface.blit(shield_surf, (int(cx - sr), int(cy - sr)))

        # Draw bullets
        for b in self.bullets:
            b.draw(surface)

    def _draw_interceptor(self, surface, pt, r, h1, h2, shade, mid) -> None:
        """Sleek swept-wing delta interceptor with dorsal spine and twin engine glow."""
        # Engine exhaust glow
        nozzle_l = pt(-r * 0.70, +r * 0.42)
        nozzle_r = pt(-r * 0.70, -r * 0.42)
        nozzle_c = pt(-r * 0.90, 0.0)
        glow_alpha = 55 if not self.is_boosting else 95
        engine_surf = pygame.Surface((int(r * 2.4), int(r * 2.4)), pygame.SRCALPHA)
        ec = int(r * 1.2)
        engine_color = (80, 160, 255) if not self.is_boosting else (100, 230, 255)
        pygame.draw.circle(engine_surf, (*engine_color, glow_alpha), (ec, ec), int(r * 0.55))
        for nx, ny in [nozzle_l, nozzle_r, nozzle_c]:
            surface.blit(engine_surf, (int(nx - r * 1.2), int(ny - r * 1.2)))

        # Wings
        wing_l = [
            pt(r * 0.55, r * 0.08),
            pt(-r * 0.15, r * 1.05),
            pt(-r * 0.70, r * 0.42),
            pt(-r * 0.35, r * 0.10),
        ]
        wing_r = [
            pt(r * 0.55, -r * 0.08),
            pt(-r * 0.15, -r * 1.05),
            pt(-r * 0.70, -r * 0.42),
            pt(-r * 0.35, -r * 0.10),
        ]
        pygame.draw.polygon(surface, shade, wing_l)
        pygame.draw.polygon(surface, shade, wing_r)
        pygame.draw.line(surface, h2, (int(wing_l[0][0]), int(wing_l[0][1])), (int(wing_l[1][0]), int(wing_l[1][1])), 2)
        pygame.draw.line(surface, h2, (int(wing_r[0][0]), int(wing_r[0][1])), (int(wing_r[1][0]), int(wing_r[1][1])), 2)

        # Fuselage
        fuselage = [
            pt(r * 1.10, 0.0),
            pt(r * 0.70, r * 0.26),
            pt(-r * 0.30, r * 0.32),
            pt(-r * 0.75, r * 0.16),
            pt(-r * 0.90, 0.0),
            pt(-r * 0.75, -r * 0.16),
            pt(-r * 0.30, -r * 0.32),
            pt(r * 0.70, -r * 0.26),
        ]
        pygame.draw.polygon(surface, h1, fuselage)

        # Dorsal panel stripe
        panel_stripe = [
            pt(r * 1.00, 0.0),
            pt(r * 0.65, r * 0.14),
            pt(-r * 0.25, r * 0.18),
            pt(-r * 0.70, r * 0.08),
            pt(-r * 0.70, -r * 0.08),
            pt(-r * 0.25, -r * 0.18),
            pt(r * 0.65, -r * 0.14),
        ]
        pygame.draw.polygon(surface, mid, panel_stripe)
        pygame.draw.polygon(surface, h2, fuselage, 1)

        # Cockpit canopy
        cp = pt(r * 0.42, 0.0)
        cockpit_col = (160, 230, 255) if not self.is_ai else (255, 210, 140)
        canopy_surf = pygame.Surface((22, 14), pygame.SRCALPHA)
        pygame.draw.ellipse(canopy_surf, (*cockpit_col, 200), (0, 0, 22, 14))
        pygame.draw.ellipse(canopy_surf, (255, 255, 255, 100), (3, 2, 8, 5))
        surface.blit(canopy_surf, (int(cp[0]) - 11, int(cp[1]) - 7))

        # Nozzles
        nozzle_inner = (40, 80, 140) if not self.is_boosting else (60, 160, 240)
        nozzle_rim = (100, 160, 220) if not self.is_boosting else (140, 220, 255)
        for np_pos in [nozzle_l, nozzle_r, nozzle_c]:
            nx, ny = int(np_pos[0]), int(np_pos[1])
            pygame.draw.circle(surface, nozzle_inner, (nx, ny), 4)
            pygame.draw.circle(surface, nozzle_rim, (nx, ny), 4, 1)

    def _draw_scout(self, surface, pt, r, h1, h2, shade, mid) -> None:
        """Agile forward-canted recon craft with needle nose and twin wing strakes."""
        # Engine exhaust glow
        nozzle_l = pt(-r * 0.82, +r * 0.32)
        nozzle_r = pt(-r * 0.82, -r * 0.32)
        glow_alpha = 60 if not self.is_boosting else 100
        engine_surf = pygame.Surface((int(r * 2.2), int(r * 2.2)), pygame.SRCALPHA)
        ec = int(r * 1.1)
        engine_color = (60, 220, 200) if not self.is_boosting else (120, 255, 240)
        pygame.draw.circle(engine_surf, (*engine_color, glow_alpha), (ec, ec), int(r * 0.50))
        for nx, ny in [nozzle_l, nozzle_r]:
            surface.blit(engine_surf, (int(nx - r * 1.1), int(ny - r * 1.1)))

        # Forward-canted wings
        wing_l = [
            pt(r * 0.15, r * 0.18),
            pt(r * 0.40, r * 1.12),
            pt(-r * 0.45, r * 0.85),
            pt(-r * 0.50, r * 0.28),
        ]
        wing_r = [
            pt(r * 0.15, -r * 0.18),
            pt(r * 0.40, -r * 1.12),
            pt(-r * 0.45, -r * 0.85),
            pt(-r * 0.50, -r * 0.28),
        ]
        pygame.draw.polygon(surface, shade, wing_l)
        pygame.draw.polygon(surface, shade, wing_r)
        pygame.draw.line(surface, h2, (int(wing_l[0][0]), int(wing_l[0][1])), (int(wing_l[1][0]), int(wing_l[1][1])), 2)
        pygame.draw.line(surface, h2, (int(wing_r[0][0]), int(wing_r[0][1])), (int(wing_r[1][0]), int(wing_r[1][1])), 2)

        # Fuselage (sharp arrow / dart)
        fuselage = [
            pt(r * 1.25, 0.0),          # long needle tip
            pt(r * 0.80, r * 0.16),
            pt(r * 0.20, r * 0.24),
            pt(-r * 0.80, r * 0.32),
            pt(-r * 0.65, 0.0),         # recessed center tail
            pt(-r * 0.80, -r * 0.32),
            pt(r * 0.20, -r * 0.24),
            pt(r * 0.80, -r * 0.16),
        ]
        pygame.draw.polygon(surface, h1, fuselage)

        # Dorsal stripe
        dorsal = [
            pt(r * 1.15, 0.0),
            pt(r * 0.70, r * 0.09),
            pt(-r * 0.40, r * 0.12),
            pt(-r * 0.60, 0.0),
            pt(-r * 0.40, -r * 0.12),
            pt(r * 0.70, -r * 0.09),
        ]
        pygame.draw.polygon(surface, mid, dorsal)
        pygame.draw.polygon(surface, h2, fuselage, 1)

        # Large panoramic bubble canopy
        cp = pt(r * 0.38, 0.0)
        cockpit_col = (130, 240, 220) if not self.is_ai else (255, 200, 100)
        canopy_surf = pygame.Surface((24, 16), pygame.SRCALPHA)
        pygame.draw.ellipse(canopy_surf, (*cockpit_col, 210), (0, 0, 24, 16))
        pygame.draw.ellipse(canopy_surf, (255, 255, 255, 120), (4, 3, 9, 5))
        surface.blit(canopy_surf, (int(cp[0]) - 12, int(cp[1]) - 8))

        # Twin engine nozzles
        nozzle_col = (30, 100, 110) if not self.is_boosting else (80, 220, 220)
        for np_pos in [nozzle_l, nozzle_r]:
            nx, ny = int(np_pos[0]), int(np_pos[1])
            pygame.draw.circle(surface, nozzle_col, (nx, ny), 4)
            pygame.draw.circle(surface, h2, (nx, ny), 4, 1)

    def _draw_vanguard(self, surface, pt, r, h1, h2, shade, mid) -> None:
        """Heavy angular armored gunship with faceted plating and broad dual nacelles."""
        # Heavy dual drive glow
        nozzle_l = pt(-r * 0.85, +r * 0.50)
        nozzle_r = pt(-r * 0.85, -r * 0.50)
        glow_alpha = 70 if not self.is_boosting else 115
        engine_surf = pygame.Surface((int(r * 2.6), int(r * 2.6)), pygame.SRCALPHA)
        ec = int(r * 1.3)
        engine_color = (255, 140, 50) if not self.is_boosting else (255, 210, 80)
        pygame.draw.circle(engine_surf, (*engine_color, glow_alpha), (ec, ec), int(r * 0.65))
        for nx, ny in [nozzle_l, nozzle_r]:
            surface.blit(engine_surf, (int(nx - r * 1.3), int(ny - r * 1.3)))

        # Outboard armored nacelles
        nacelle_l = [
            pt(r * 0.35, r * 0.55),
            pt(r * 0.10, r * 1.05),
            pt(-r * 0.85, r * 0.70),
            pt(-r * 0.85, r * 0.35),
            pt(-r * 0.20, r * 0.40),
        ]
        nacelle_r = [
            pt(r * 0.35, -r * 0.55),
            pt(r * 0.10, -r * 1.05),
            pt(-r * 0.85, -r * 0.70),
            pt(-r * 0.85, -r * 0.35),
            pt(-r * 0.20, -r * 0.40),
        ]
        pygame.draw.polygon(surface, shade, nacelle_l)
        pygame.draw.polygon(surface, shade, nacelle_r)
        pygame.draw.polygon(surface, h2, nacelle_l, 1)
        pygame.draw.polygon(surface, h2, nacelle_r, 1)

        # Heavy faceted main hull
        hull = [
            pt(r * 1.05, 0.0),          # blunt prow
            pt(r * 0.80, r * 0.35),     # port forward facet
            pt(r * 0.10, r * 0.45),
            pt(-r * 0.75, r * 0.30),
            pt(-r * 0.85, 0.0),
            pt(-r * 0.75, -r * 0.30),
            pt(r * 0.10, -r * 0.45),
            pt(r * 0.80, -r * 0.35),
        ]
        pygame.draw.polygon(surface, h1, hull)

        # Center armor plate
        center_plate = [
            pt(r * 0.90, 0.0),
            pt(r * 0.65, r * 0.20),
            pt(-r * 0.40, r * 0.22),
            pt(-r * 0.70, 0.0),
            pt(-r * 0.40, -r * 0.22),
            pt(r * 0.65, -r * 0.20),
        ]
        pygame.draw.polygon(surface, mid, center_plate)
        pygame.draw.polygon(surface, h2, hull, 1)

        # Armored cockpit slit
        cp = pt(r * 0.48, 0.0)
        cockpit_col = (255, 180, 80) if not self.is_ai else (255, 100, 100)
        slit_surf = pygame.Surface((18, 8), pygame.SRCALPHA)
        pygame.draw.rect(slit_surf, (*cockpit_col, 220), (0, 0, 18, 8), border_radius=2)
        surface.blit(slit_surf, (int(cp[0]) - 9, int(cp[1]) - 4))

        # Heavy main drive nozzles
        for np_pos in [nozzle_l, nozzle_r]:
            nx, ny = int(np_pos[0]), int(np_pos[1])
            pygame.draw.circle(surface, (60, 45, 30), (nx, ny), 5)
            pygame.draw.circle(surface, (200, 130, 40), (nx, ny), 5, 1)

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
            "is_invulnerable": self.is_invulnerable,
            "collision_invuln_left": self.collision_invuln_timer,
        }
