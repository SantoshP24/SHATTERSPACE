"""Player spaceship controller and input handler for Kessler AI survival game.

Supports:
- WASD or Arrow Keys for standard thruster / rotational navigation.
- Mouse aiming with crosshair aiming support.
- Spacebar and Left Mouse Click for laser cannons.
- Left/Right Shift for Afterburner Boost with cooldown.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pygame

from .geometry import wrap_angle_deg
from .spaceship import Spaceship


class PlayerShip(Spaceship):
    """Interactive player-controlled spaceship."""

    def __init__(
        self,
        ship_id: int = 1,
        position: Tuple[float, float] = (500.0, 400.0),
        heading: float = 90.0,
        map_size: Tuple[float, float] = (1000.0, 800.0),
        settings: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            ship_id=ship_id,
            position=position,
            heading=heading,
            map_size=map_size,
            is_ai=False,
            settings=settings,
            **kwargs,
        )
        self.mouse_aim_enabled: bool = True

    def process_input(
        self,
        keys,
        mouse_pos: Tuple[int, int],
        mouse_buttons: Tuple[bool, bool, bool],
        dt: float,
    ) -> Tuple[float, float, bool, bool]:
        """Convert keyboard and mouse state into (thrust, turn, fire, boost)."""
        thrust = 0.0
        turn = 0.0
        fire = False
        boost = False

        # Forward / Reverse thrust
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            thrust = self.params.thrust_max
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            thrust = self.params.thrust_min

        # Keyboard turning
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            turn -= self.params.turn_max
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            turn += self.params.turn_max

        # Mouse aiming (overrides keyboard turning if active mouse movement)
        if self.mouse_aim_enabled and (turn == 0.0):
            mx, my = mouse_pos
            dx = mx - self.x
            dy = my - self.y
            dist_sq = dx * dx + dy * dy
            if dist_sq > 25.0:  # Deadzone around center of ship
                target_heading = math.degrees(math.atan2(dy, dx)) % 360.0
                turn_err = wrap_angle_deg(target_heading - self.heading)
                # Proportional turn rate up to maximum
                turn = float(np.clip(turn_err / dt, -self.params.turn_max, self.params.turn_max))

        # Boost (Left Shift or Right Shift)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            boost = True

        # Fire weapons (Spacebar or Left Mouse Button)
        if keys[pygame.K_SPACE] or mouse_buttons[0]:
            fire = True

        return thrust, turn, fire, boost
