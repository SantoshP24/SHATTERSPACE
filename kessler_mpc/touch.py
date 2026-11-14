"""Browser-compatible responsive virtual touch controller for mobile and tablet gameplay.

Provides on-screen virtual joystick and touch action buttons (Fire, Boost, Pause)
with multi-touch support for touchscreens while remaining unobtrusive during desktop play
and completely inactive during autonomous AI mode.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import pygame


class VirtualJoystick:
    """Analog virtual joystick supporting smooth 360-degree directional navigation."""

    def __init__(self, center: Tuple[float, float], base_radius: float = 65.0, knob_radius: float = 28.0):
        self.center_x, self.center_y = center
        self.base_radius = base_radius
        self.knob_radius = knob_radius

        self.knob_x, self.knob_y = self.center_x, self.center_y
        self.touch_id: Optional[int] = None
        self.thrust: float = 0.0  # -1.0 (reverse) to +1.0 (forward)
        self.turn: float = 0.0    # -1.0 (left) to +1.0 (right)

    def handle_touch_down(self, touch_id: int, pos: Tuple[float, float]) -> bool:
        if self.touch_id is not None:
            return False
        x, y = pos
        dx = x - self.center_x
        dy = y - self.center_y
        dist = math.hypot(dx, dy)
        # Accept touch within 1.35x base radius to be forgiving for thumb placement
        if dist <= self.base_radius * 1.35:
            self.touch_id = touch_id
            self.update_knob(x, y)
            return True
        return False

    def handle_touch_motion(self, touch_id: int, pos: Tuple[float, float]) -> bool:
        if self.touch_id == touch_id:
            self.update_knob(pos[0], pos[1])
            return True
        return False

    def handle_touch_up(self, touch_id: int) -> bool:
        if self.touch_id == touch_id:
            self.reset()
            return True
        return False

    def update_knob(self, x: float, y: float) -> None:
        dx = x - self.center_x
        dy = y - self.center_y
        dist = math.hypot(dx, dy)

        if dist > self.base_radius:
            dx = (dx / dist) * self.base_radius
            dy = (dy / dist) * self.base_radius
            dist = self.base_radius

        self.knob_x = self.center_x + dx
        self.knob_y = self.center_y + dy

        # Normalized values:
        # Thrust: dragging up (-dy) is forward thrust, dragging down (+dy) is reverse
        norm_y = -dy / self.base_radius
        # Turn: dragging right (+dx) is turning right, dragging left (-dx) is turning left
        norm_x = dx / self.base_radius

        deadzone = 0.12
        if abs(norm_y) < deadzone:
            norm_y = 0.0
        if abs(norm_x) < deadzone:
            norm_x = 0.0

        self.thrust = float(norm_y)
        self.turn = float(norm_x)

    def reset(self) -> None:
        self.touch_id = None
        self.knob_x, self.knob_y = self.center_x, self.center_y
        self.thrust = 0.0
        self.turn = 0.0

    def draw(self, surface: pygame.Surface) -> None:
        # Outer base ring
        base_surf = pygame.Surface((int(self.base_radius * 2 + 10), int(self.base_radius * 2 + 10)), pygame.SRCALPHA)
        center_in_surf = (int(self.base_radius + 5), int(self.base_radius + 5))

        # Base disc and outline
        pygame.draw.circle(base_surf, (15, 30, 50, 90), center_in_surf, int(self.base_radius))
        pygame.draw.circle(base_surf, (70, 160, 230, 160), center_in_surf, int(self.base_radius), 2)
        pygame.draw.circle(base_surf, (50, 110, 170, 70), center_in_surf, int(self.base_radius * 0.5), 1)

        dest_x = int(self.center_x - self.base_radius - 5)
        dest_y = int(self.center_y - self.base_radius - 5)
        surface.blit(base_surf, (dest_x, dest_y))

        # Inner thumb knob
        knob_color = (90, 220, 255, 210) if self.touch_id is not None else (60, 150, 210, 140)
        knob_surf = pygame.Surface((int(self.knob_radius * 2 + 6), int(self.knob_radius * 2 + 6)), pygame.SRCALPHA)
        k_center = (int(self.knob_radius + 3), int(self.knob_radius + 3))

        pygame.draw.circle(knob_surf, (20, 45, 75, 200), k_center, int(self.knob_radius))
        pygame.draw.circle(knob_surf, knob_color, k_center, int(self.knob_radius), 2)
        pygame.draw.circle(knob_surf, (150, 240, 255, 180), k_center, int(self.knob_radius * 0.35))

        k_dest_x = int(self.knob_x - self.knob_radius - 3)
        k_dest_y = int(self.knob_y - self.knob_radius - 3)
        surface.blit(knob_surf, (k_dest_x, k_dest_y))


class TouchActionButton:
    """Circular touch action button (e.g. Fire, Boost, Pause)."""

    def __init__(
        self,
        center: Tuple[float, float],
        radius: float,
        label: str,
        color: Tuple[int, int, int],
        on_press: Optional[Callable[[], None]] = None,
    ):
        self.center_x, self.center_y = center
        self.radius = radius
        self.label = label
        self.color = color
        self.on_press = on_press

        self.touch_id: Optional[int] = None
        self.is_pressed: bool = False

    def handle_touch_down(self, touch_id: int, pos: Tuple[float, float]) -> bool:
        if self.touch_id is not None:
            return False
        x, y = pos
        if math.hypot(x - self.center_x, y - self.center_y) <= self.radius * 1.2:
            self.touch_id = touch_id
            self.is_pressed = True
            if self.on_press:
                self.on_press()
            return True
        return False

    def handle_touch_up(self, touch_id: int) -> bool:
        if self.touch_id == touch_id:
            self.touch_id = None
            self.is_pressed = False
            return True
        return False

    def draw(self, surface: pygame.Surface, font: Optional[pygame.font.Font] = None) -> None:
        btn_surf = pygame.Surface((int(self.radius * 2 + 8), int(self.radius * 2 + 8)), pygame.SRCALPHA)
        center = (int(self.radius + 4), int(self.radius + 4))

        r, g, b = self.color
        bg_alpha = 180 if self.is_pressed else 85
        border_alpha = 255 if self.is_pressed else 160

        pygame.draw.circle(btn_surf, (r // 3, g // 3, b // 3, bg_alpha), center, int(self.radius))
        pygame.draw.circle(btn_surf, (r, g, b, border_alpha), center, int(self.radius), 2 if not self.is_pressed else 3)

        if self.is_pressed:
            pygame.draw.circle(btn_surf, (r, g, b, 70), center, int(self.radius * 0.7))

        dest_x = int(self.center_x - self.radius - 4)
        dest_y = int(self.center_y - self.radius - 4)
        surface.blit(btn_surf, (dest_x, dest_y))

        # Render button label
        if font and self.label:
            txt_col = (255, 255, 255) if self.is_pressed else (r, g, b)
            txt_s = font.render(self.label, True, txt_col)
            surface.blit(txt_s, txt_s.get_rect(center=(int(self.center_x), int(self.center_y))))


class TouchController:
    """Complete multi-touch on-screen controller management."""

    def __init__(self, on_pause_callback: Optional[Callable[[], None]] = None):
        self.joystick = VirtualJoystick(center=(135.0, 665.0), base_radius=65.0, knob_radius=28.0)
        self.btn_fire = TouchActionButton(center=(885.0, 665.0), radius=42.0, label="FIRE", color=(255, 80, 80))
        self.btn_boost = TouchActionButton(center=(785.0, 715.0), radius=32.0, label="BOOST", color=(255, 180, 50))
        self.btn_pause = TouchActionButton(
            center=(955.0, 42.0),
            radius=22.0,
            label="II",
            color=(140, 200, 255),
            on_press=on_pause_callback,
        )

        self.is_active: bool = False  # Activates automatically on first touch or can be toggled
        self.font: Optional[pygame.font.Font] = None

    def _ensure_font(self) -> None:
        if self.font is None and pygame.font.get_init():
            try:
                self.font = pygame.font.SysFont("consolas", 14, bold=True)
            except Exception:
                self.font = None

    def handle_event(self, event: pygame.event.Event, logical_pos: Tuple[float, float]) -> bool:
        """Process touch or mouse events in logical coordinate space."""
        # Detect touch events (finger down, finger up, finger motion)
        if event.type == pygame.FINGERDOWN:
            self.is_active = True
            touch_id = getattr(event, "finger_id", 0)
            if self.btn_pause.handle_touch_down(touch_id, logical_pos):
                return True
            if self.joystick.handle_touch_down(touch_id, logical_pos):
                return True
            if self.btn_fire.handle_touch_down(touch_id, logical_pos):
                return True
            if self.btn_boost.handle_touch_down(touch_id, logical_pos):
                return True
            return False

        elif event.type == pygame.FINGERMOTION:
            touch_id = getattr(event, "finger_id", 0)
            self.joystick.handle_touch_motion(touch_id, logical_pos)
            return False

        elif event.type == pygame.FINGERUP:
            touch_id = getattr(event, "finger_id", 0)
            self.joystick.handle_touch_up(touch_id)
            self.btn_fire.handle_touch_up(touch_id)
            self.btn_boost.handle_touch_up(touch_id)
            self.btn_pause.handle_touch_up(touch_id)
            return False

        # Fallback mouse emulation for virtual buttons (useful in desktop browser test mode)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            touch_id = -1
            if self.btn_pause.handle_touch_down(touch_id, logical_pos):
                self.is_active = True
                return True
            if self.joystick.handle_touch_down(touch_id, logical_pos):
                self.is_active = True
                return True
            if self.btn_fire.handle_touch_down(touch_id, logical_pos):
                self.is_active = True
                return True
            if self.btn_boost.handle_touch_down(touch_id, logical_pos):
                self.is_active = True
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self.joystick.touch_id == -1:
                self.joystick.handle_touch_motion(-1, logical_pos)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.joystick.touch_id == -1:
                self.joystick.handle_touch_up(-1)
            if self.btn_fire.touch_id == -1:
                self.btn_fire.handle_touch_up(-1)
            if self.btn_boost.touch_id == -1:
                self.btn_boost.handle_touch_up(-1)
            if self.btn_pause.touch_id == -1:
                self.btn_pause.handle_touch_up(-1)

        return False

    def get_inputs(self) -> Tuple[float, float, bool, bool]:
        """Returns (thrust_normalized, turn_normalized, fire_active, boost_active)."""
        return (
            self.joystick.thrust,
            self.joystick.turn,
            self.btn_fire.is_pressed,
            self.btn_boost.is_pressed,
        )

    def draw(self, surface: pygame.Surface, show_flight_controls: bool = True) -> None:
        """Render virtual controls on top of gameplay surface."""
        self._ensure_font()

        # Pause button is always available in gameplay
        self.btn_pause.draw(surface, self.font)

        # Flight controls (joystick, fire, boost) only shown in manual player mode
        if show_flight_controls:
            self.joystick.draw(surface)
            self.btn_fire.draw(surface, self.font)
            self.btn_boost.draw(surface, self.font)
