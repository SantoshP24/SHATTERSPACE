"""Game interface, HUD, menus, and AI debug overlay for Kessler AI survival game.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pygame

from .kessler_ai import AIBehaviorState, AIDebugTelemetry
from .safe_zone import SafeZone
from .settings import (
    DIFFICULTY_DESCRIPTIONS,
    GAME_SUBTITLE,
    GAME_TAGLINE,
    GAME_TITLE,
    DifficultyLevel,
    GameSettings,
)
from .spaceship import Spaceship


class UIButton:
    """Interactive menu button with hover and click states."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        callback: Optional[Callable[[], None]] = None,
        font: Optional[pygame.font.Font] = None,
    ):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.font = font
        self.is_hovered = False

    def check_hover(self, mouse_pos: Tuple[int, int]) -> bool:
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered

    def handle_click(self, mouse_pos: Tuple[int, int]) -> bool:
        if self.rect.collidepoint(mouse_pos):
            if self.callback:
                self.callback()
            return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_hovered:
            bg_color = (22, 38, 60)
            border_color = (90, 170, 220)
            text_color = (230, 245, 255)
        else:
            bg_color = (12, 20, 32)
            border_color = (40, 75, 110)
            text_color = (160, 200, 230)

        pygame.draw.rect(surface, bg_color, self.rect, border_radius=4)
        pygame.draw.rect(surface, border_color, self.rect, 1, border_radius=4)

        if self.font:
            txt_surf = self.font.render(self.text, True, text_color)
            txt_rect = txt_surf.get_rect(center=self.rect.center)
            surface.blit(txt_surf, txt_rect)


class UISlider:
    """Interactive draggable slider for numeric parameter configuration."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        min_val: float,
        max_val: float,
        current_val: float,
        step: float = 1.0,
        is_integer: bool = False,
        suffix: str = "",
    ):
        self.rect = rect
        self.label = label
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.val = float(current_val)
        self.step = float(step)
        self.is_integer = is_integer
        self.suffix = suffix
        self.is_dragging = False

    def get_value(self) -> float:
        return int(round(self.val)) if self.is_integer else round(self.val, 2)

    def set_value(self, val: float) -> None:
        self.val = max(self.min_val, min(self.max_val, float(val)))

    def handle_event(self, event: pygame.event.Event, mouse_pos: Tuple[int, int]) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit_rect = self.rect.inflate(0, 16)
            if hit_rect.collidepoint(mouse_pos):
                self.is_dragging = True
                self._update_from_mouse(mouse_pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_dragging:
                self.is_dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            self._update_from_mouse(mouse_pos[0])
            return True
        return False

    def _update_from_mouse(self, mouse_x: int) -> None:
        frac = float(np.clip((mouse_x - self.rect.x) / max(1, self.rect.width), 0.0, 1.0))
        raw = self.min_val + frac * (self.max_val - self.min_val)
        if self.step > 0:
            stepped = round((raw - self.min_val) / self.step) * self.step + self.min_val
        else:
            stepped = raw
        self.set_value(stepped)

    def draw(self, surface: pygame.Surface, font_label: pygame.font.Font, font_val: pygame.font.Font) -> None:
        lbl_surf = font_label.render(self.label, True, (210, 230, 255))
        surface.blit(lbl_surf, (self.rect.x, self.rect.y - 18))

        v = self.get_value()
        val_str = f"{int(v)}" if self.is_integer else f"{v:.1f}"
        if self.suffix:
            val_str += f" {self.suffix}"
        v_surf = font_val.render(val_str, True, (255, 205, 70))
        v_rect = v_surf.get_rect(right=self.rect.right, bottom=self.rect.y - 2)
        surface.blit(v_surf, v_rect)

        # Track background
        pygame.draw.rect(surface, (18, 28, 44), self.rect, border_radius=4)
        pygame.draw.rect(surface, (45, 75, 110), self.rect, 1, border_radius=4)

        # Active filled track
        frac = (self.val - self.min_val) / max(1e-6, self.max_val - self.min_val)
        fill_w = int(self.rect.width * frac)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.height)
            pygame.draw.rect(surface, (40, 160, 240), fill_rect, border_radius=4)

        # Knob handle
        knob_x = int(self.rect.x + fill_w)
        knob_y = self.rect.centery
        knob_r = 7
        knob_col = (255, 255, 255) if self.is_dragging else (200, 235, 255)
        pygame.draw.circle(surface, knob_col, (knob_x, knob_y), knob_r)
        pygame.draw.circle(surface, (0, 120, 220), (knob_x, knob_y), knob_r, 2)


class UICheckbox:
    """Toggle checkbox with custom stylized box and checkmark."""

    def __init__(self, rect: pygame.Rect, label: str, checked: bool = True):
        self.rect = rect
        self.label = label
        self.checked = checked

    def handle_click(self, mouse_pos: Tuple[int, int]) -> bool:
        click_rect = pygame.Rect(self.rect.x, self.rect.y - 4, self.rect.width + 240, self.rect.height + 8)
        if click_rect.collidepoint(mouse_pos):
            self.checked = not self.checked
            return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        box_col = (20, 30, 48)
        border_col = (80, 210, 255) if self.checked else (60, 90, 130)
        pygame.draw.rect(surface, box_col, self.rect, border_radius=4)
        pygame.draw.rect(surface, border_col, self.rect, 2, border_radius=4)

        if self.checked:
            inner = self.rect.inflate(-6, -6)
            pygame.draw.rect(surface, (50, 230, 120), inner, border_radius=2)

        lbl_surf = font.render(self.label, True, (220, 235, 250))
        surface.blit(lbl_surf, (self.rect.right + 12, self.rect.y + 1))


class UICycleButton:
    """Option cycler button (dropdown alternative) for discreet settings."""

    def __init__(self, rect: pygame.Rect, label: str, options: List[str], current: str):
        self.rect = rect
        self.label = label
        self.options = options
        self.current = current if current in options else options[0]
        self.is_hovered = False

    def cycle(self) -> str:
        idx = (self.options.index(self.current) + 1) % len(self.options)
        self.current = self.options[idx]
        return self.current

    def handle_click(self, mouse_pos: Tuple[int, int]) -> bool:
        if self.rect.collidepoint(mouse_pos):
            self.cycle()
            return True
        return False

    def check_hover(self, mouse_pos: Tuple[int, int]) -> bool:
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        bg_col = (25, 45, 75) if self.is_hovered else (15, 25, 40)
        border_col = (80, 220, 255) if self.is_hovered else (40, 90, 140)
        pygame.draw.rect(surface, bg_col, self.rect, border_radius=4)
        pygame.draw.rect(surface, border_col, self.rect, 2, border_radius=4)

        txt = f"{self.label}: {self.current.upper()}"
        txt_col = (255, 255, 255) if self.is_hovered else (180, 215, 245)
        surf = font.render(txt, True, txt_col)
        surface.blit(surf, surf.get_rect(center=self.rect.center))


class UIManager:
    """Renders in-game HUD, countdown timer, AI status indicators, and modal screens."""

    def __init__(self, screen_width: int = 1000, screen_height: int = 800):
        self.width = screen_width
        self.height = screen_height
        self.notifications: List[dict] = []

        pygame.font.init()
        self.font_title = self._get_font(["consolas", "lucidaconsole", "arial"], 42, bold=True)
        self.font_timer = self._get_font(["consolas", "lucidaconsole", "arial"], 30, bold=True)
        self.font_header = self._get_font(["consolas", "lucidaconsole", "arial"], 24, bold=True)
        self.font_body = self._get_font(["consolas", "lucidaconsole", "arial"], 17, bold=False)
        self.font_small = self._get_font(["consolas", "lucidaconsole", "arial"], 13, bold=False)

    def add_notification(
        self,
        text: str,
        color: Tuple[int, int, int] = (255, 60, 60),
        duration: float = 3.0,
    ) -> None:
        """Post a prominent HUD banner notification (e.g., for Red Asteroid penalties)."""
        self.notifications.append(
            {"text": text, "color": color, "time_left": duration, "duration": duration}
        )

    def update(self, dt: float) -> None:
        """Update notification lifetimes."""
        for n in list(self.notifications):
            n["time_left"] -= dt
            if n["time_left"] <= 0:
                self.notifications.remove(n)

    def _get_font(self, names: List[str], size: int, bold: bool = False) -> pygame.font.Font:
        for name in names:
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f:
                    return f
            except Exception:
                pass
        return pygame.font.Font(None, size)

    # =========================================================================
    # IN-GAME HUD
    # =========================================================================
    def draw_hud(
        self,
        surface: pygame.Surface,
        ship: Spaceship,
        countdown_remaining: float,
        elapsed_time: float,
        asteroids_count: int,
        ai_telemetry: Optional[AIDebugTelemetry] = None,
        high_score: int = 0,
        safe_zone: Optional[SafeZone] = None,
        red_count: int = 0,
    ) -> None:
        """Render minimal HUD: timer, health, lives, score, kill count, AI state."""
        mx = 18
        my = 16
        bar_w, bar_h = 160, 10

        # ── Top-left panel: health bar + lives ──────────────────────────────
        panel_w, panel_h = 220, 72
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((6, 10, 18, 190))
        pygame.draw.rect(panel_surf, (35, 60, 90), (0, 0, panel_w, panel_h), 1, border_radius=4)
        surface.blit(panel_surf, (mx, my))

        px, py = mx + 10, my + 8

        # Hull bar
        hull_pct = max(0.0, ship.health / max(1.0, ship.max_health))
        hull_color = (
            (55, 210, 75) if hull_pct > 0.5 else ((225, 185, 35) if hull_pct > 0.25 else (235, 40, 40))
        )
        pygame.draw.rect(surface, (18, 26, 38), (px, py, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(surface, hull_color, (px, py, max(0, int(bar_w * hull_pct)), bar_h), border_radius=3)
        pygame.draw.rect(surface, (55, 85, 115), (px, py, bar_w, bar_h), 1, border_radius=3)
        lbl = self.font_small.render(f"HULL  {int(ship.health)}", True, (200, 215, 230))
        surface.blit(lbl, (px + bar_w + 8, py - 1))

        # Shield bar
        shield_pct = max(0.0, ship.shields / ship.max_shields)
        pygame.draw.rect(surface, (18, 26, 38), (px, py + 18, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(surface, (45, 180, 240), (px, py + 18, max(0, int(bar_w * shield_pct)), bar_h), border_radius=3)
        pygame.draw.rect(surface, (55, 85, 115), (px, py + 18, bar_w, bar_h), 1, border_radius=3)
        lbl2 = self.font_small.render(f"SHIELD  {int(ship.shields)}%", True, (155, 210, 245))
        surface.blit(lbl2, (px + bar_w + 8, py + 17))

        # Lives — small dot icons
        dot_col = (225, 75, 75)
        lives_x = px
        for i in range(max(0, ship.lives)):
            pygame.draw.circle(surface, dot_col, (lives_x + i * 16 + 6, py + 48), 5)
            pygame.draw.circle(surface, (255, 110, 110), (lives_x + i * 16 + 6, py + 48), 5, 1)
        lbl_lives = self.font_small.render(f"{'  ' * max(0, ship.lives)}  {ship.lives} {'life' if ship.lives == 1 else 'lives'}", True, (180, 140, 140))
        surface.blit(lbl_lives, (px + max(0, ship.lives) * 16 + 2, py + 42))

        # ── Top-centre: timer ────────────────────────────────────────────────
        cx = self.width // 2
        rem = max(0.0, countdown_remaining)
        minutes = int(rem // 60)
        seconds = rem % 60
        timer_str = f"{minutes:02d}:{seconds:04.1f}"

        tw, th = 200, 40
        tx = cx - tw // 2
        t_bg = pygame.Surface((tw, th), pygame.SRCALPHA)
        t_bg.fill((6, 10, 18, 200))
        surface.blit(t_bg, (tx, my))
        t_border = (220, 50, 50) if rem < 30.0 else (38, 110, 180)
        pygame.draw.rect(surface, t_border, (tx, my, tw, th), 1, border_radius=4)
        t_color = (240, 70, 70) if rem < 30.0 else (70, 210, 240)
        txt_timer = self.font_timer.render(timer_str, True, t_color)
        surface.blit(txt_timer, txt_timer.get_rect(center=(cx, my + th // 2)))

        # AI state badge just below the timer
        if ai_telemetry:
            self._draw_ai_status_badge(surface, cx, my + th + 8, ai_telemetry)

        # Penalty / alert banners
        if self.notifications:
            notif = self.notifications[-1]
            alpha = min(255, max(0, int(255 * (notif["time_left"] / max(0.1, notif["duration"])))))
            banner_surf = self.font_body.render(notif["text"], True, notif["color"])
            bw, bh = banner_surf.get_width() + 24, banner_surf.get_height() + 10
            b_bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
            b_bg.fill((28, 6, 10, min(210, alpha)))
            pygame.draw.rect(b_bg, (*notif["color"], alpha), (0, 0, bw, bh), 1, border_radius=3)
            b_bg.blit(banner_surf, (12, 5))
            surface.blit(b_bg, (cx - bw // 2, my + th + 54))

        # ── Top-right: score + kills ─────────────────────────────────────────
        rp_w, rp_h = 170, 56
        rp_x = self.width - rp_w - mx
        rp_surf = pygame.Surface((rp_w, rp_h), pygame.SRCALPHA)
        rp_surf.fill((6, 10, 18, 190))
        pygame.draw.rect(rp_surf, (35, 60, 90), (0, 0, rp_w, rp_h), 1, border_radius=4)
        surface.blit(rp_surf, (rp_x, my))

        score_txt = self.font_header.render(f"{ship.score}", True, (230, 240, 255))
        surface.blit(score_txt, score_txt.get_rect(right=rp_x + rp_w - 10, top=my + 6))
        score_lbl = self.font_small.render("SCORE", True, (100, 140, 175))
        surface.blit(score_lbl, (rp_x + 10, my + 7))

        kills_txt = self.font_body.render(f"{ship.kills} kills", True, (200, 180, 100))
        surface.blit(kills_txt, kills_txt.get_rect(right=rp_x + rp_w - 10, top=my + 34))

        if red_count > 0:
            red_txt = self.font_small.render(f"⚠ {red_count} red", True, (220, 65, 65))
            surface.blit(red_txt, (rp_x + 10, my + 35))

        # ── Bottom-left: minimap ─────────────────────────────────────────────
        self._draw_minimap(surface, ship, safe_zone)

    def _draw_ai_status_badge(self, surface: pygame.Surface, cx: int, y: int, tel: AIDebugTelemetry) -> None:
        """Small compact status pill for ATTACK, EVADE, REPOSITION, and SURVIVE."""
        state = tel.state

        if state == AIBehaviorState.ATTACK:
            text = "ATTACK"
            color = (55, 215, 110)
            border = (35, 170, 80)
            bg = (8, 28, 16, 210)
        elif state == AIBehaviorState.EVADE:
            text = "EVADE"
            color = (240, 55, 55)
            border = (200, 40, 40)
            bg = (35, 8, 10, 215)
        elif state == AIBehaviorState.SURVIVE:
            text = "SURVIVE"
            color = (240, 165, 35)
            border = (200, 135, 25)
            bg = (30, 20, 8, 210)
        else:  # REPOSITION
            text = "REPOSITION"
            color = (55, 200, 240)
            border = (38, 160, 210)
            bg = (8, 20, 32, 210)

        badge_w, badge_h = 130, 22
        bx = cx - badge_w // 2
        bg_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        bg_surf.fill(bg)
        surface.blit(bg_surf, (bx, y))
        pygame.draw.rect(surface, border, (bx, y, badge_w, badge_h), 1, border_radius=3)
        txt = self.font_small.render(text, True, color)
        surface.blit(txt, txt.get_rect(center=(cx, y + badge_h // 2)))

    def _draw_minimap(self, surface: pygame.Surface, ship: Spaceship, safe_zone: Optional[SafeZone]) -> None:
        """Minimap radar display in bottom right corner."""
        mm_w, mm_h = 130, 104
        mx = self.width - mm_w - 18
        my = self.height - mm_h - 18

        mm_surf = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA)
        mm_surf.fill((12, 18, 28, 210))
        pygame.draw.rect(mm_surf, (50, 100, 140), (0, 0, mm_w, mm_h), 1)

        scale_x = mm_w / self.width
        scale_y = mm_h / self.height

        if safe_zone:
            sz_x = int(safe_zone.x * scale_x)
            sz_y = int(safe_zone.y * scale_y)
            pygame.draw.circle(mm_surf, (40, 220, 140), (sz_x, sz_y), max(3, int(safe_zone.radius * scale_x)), 1)

        sx = int(ship.x * scale_x)
        sy = int(ship.y * scale_y)
        ship_color = (255, 80, 80) if ship.is_ai else (60, 210, 255)
        pygame.draw.circle(mm_surf, ship_color, (sx, sy), 3)

        surface.blit(mm_surf, (mx, my))

    # =========================================================================
    # AI DEBUG OVERLAY (TOGGLEABLE)
    # =========================================================================
    def draw_debug_overlay(
        self,
        surface: pygame.Surface,
        ship: Spaceship,
        telemetry: AIDebugTelemetry,
        settings: GameSettings,
    ) -> None:
        """Render in-world A* path, MPC trajectory arcs, and threat vectors."""
        if not settings.show_debug_overlay or not telemetry:
            return

        # 1. Detection Radius Circle
        if settings.show_detection_radius:
            det_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.circle(det_surf, (40, 140, 220, 35), (int(ship.x), int(ship.y)), 240, 1)
            surface.blit(det_surf, (0, 0))

        # 2. A* Global Path
        if settings.show_astar_path and telemetry.astar_path:
            pts = [(int(p[0]), int(p[1])) for p in telemetry.astar_path]
            if len(pts) >= 2:
                pygame.draw.lines(surface, (40, 220, 160), False, pts, 2)
                for pt in pts:
                    pygame.draw.circle(surface, (80, 255, 200), pt, 3)

        # 3. MPC Trajectory Arcs
        if settings.show_mpc_trajectories and telemetry.mpc_trajectory:
            traj_pts = [(int(p[0]), int(p[1])) for p in telemetry.mpc_trajectory]
            if len(traj_pts) >= 2:
                arc_color = (255, 60, 60) if telemetry.mpc_best_collides else (60, 220, 255)
                pygame.draw.lines(surface, arc_color, False, traj_pts, 3)
                pygame.draw.circle(surface, (255, 255, 255), traj_pts[-1], 4)

        # 4. Threat Vectors to Debris
        if settings.show_threat_vectors and telemetry.top_threats:
            for t in telemetry.top_threats:
                t_score = t.overall_threat_score
                if t_score > 0.15:
                    if t.is_critical:
                        line_color = (255, 40, 40)
                    elif t.is_urgent:
                        line_color = (255, 180, 30)
                    else:
                        line_color = (120, 160, 220)

                    ax, ay = int(t.position[0]), int(t.position[1])
                    pygame.draw.line(surface, line_color, (int(ship.x), int(ship.y)), (ax, ay), 1)
                    txt = self.font_small.render(f"T:{t_score:.2f} | TTC:{t.time_to_collision:.1f}s", True, line_color)
                    surface.blit(txt, (ax + 12, ay - 8))

        # 5. Target Reticle
        if settings.show_target_reticle and telemetry.target_pos is not None:
            tx, ty = int(telemetry.target_pos[0]), int(telemetry.target_pos[1])
            ret_r = 28
            pygame.draw.circle(surface, (255, 50, 50), (tx, ty), ret_r, 1)
            pygame.draw.line(surface, (255, 50, 50), (tx - ret_r - 6, ty), (tx - ret_r + 4, ty), 2)
            pygame.draw.line(surface, (255, 50, 50), (tx + ret_r - 4, ty), (tx + ret_r + 6, ty), 2)
            pygame.draw.line(surface, (255, 50, 50), (tx, ty - ret_r - 6), (tx, ty - ret_r + 4), 2)
            pygame.draw.line(surface, (255, 50, 50), (tx, ty + ret_r - 4), (tx, ty + ret_r + 6), 2)

    # =========================================================================
    # MENUS & MODAL SCREENS
    # =========================================================================
    def draw_main_menu(
        self,
        surface: pygame.Surface,
        buttons: List[UIButton],
        current_difficulty: DifficultyLevel,
    ) -> None:
        """Render main title screen."""
        cx = self.width // 2

        # Title: SHATTERSPACE
        title_surf = self.font_title.render(GAME_TITLE, True, (215, 235, 255))
        surface.blit(title_surf, title_surf.get_rect(center=(cx, 132)))

        # Subtitle: DEEP SPACE SURVIVAL
        sub_surf = self.font_header.render(GAME_SUBTITLE, True, (130, 185, 235))
        surface.blit(sub_surf, sub_surf.get_rect(center=(cx, 168)))

        # Thin separator line
        pygame.draw.line(surface, (38, 65, 100), (cx - 180, 188), (cx + 180, 188), 1)

        # Short tagline: "One ship. Endless asteroids. Stay alive."
        tag_surf = self.font_body.render(f'"{GAME_TAGLINE}"', True, (170, 200, 230))
        surface.blit(tag_surf, tag_surf.get_rect(center=(cx, 206)))

        for btn in buttons:
            btn.draw(surface)

        # Selected difficulty description
        desc = DIFFICULTY_DESCRIPTIONS.get(current_difficulty, "")
        if desc:
            desc_surf = self.font_small.render(f"{current_difficulty.value.upper()}  —  {desc}", True, (130, 175, 215))
            surface.blit(desc_surf, desc_surf.get_rect(center=(cx, self.height - 42)))

        ftr = self.font_small.render(f"{GAME_TITLE}  ·  Press ESC to quit", True, (65, 95, 130))
        surface.blit(ftr, ftr.get_rect(center=(cx, self.height - 22)))

    def draw_pause_menu(self, surface: pygame.Surface, buttons: List[UIButton]) -> None:
        """Render semi-transparent pause overlay."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((4, 8, 16, 185))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt = self.font_header.render(f"{GAME_TITLE}  ·  PAUSED", True, (175, 215, 245))
        surface.blit(txt, txt.get_rect(center=(cx, 220)))
        pygame.draw.line(surface, (38, 65, 100), (cx - 140, 242), (cx + 140, 242), 1)

        for btn in buttons:
            btn.draw(surface)

    def draw_defeat(
        self,
        surface: pygame.Surface,
        buttons: List[UIButton],
        score: int,
        survived_time: float,
        kills: int,
        accuracy: float,
        damage_taken: float,
    ) -> None:
        """Render defeat screen."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((22, 4, 6, 220))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt_title = self.font_title.render("DEFEATED", True, (230, 50, 50))
        surface.blit(txt_title, txt_title.get_rect(center=(cx, 145)))
        txt_sub = self.font_header.render(f"{GAME_TITLE}  ·  Hull Destroyed", True, (200, 140, 140))
        surface.blit(txt_sub, txt_sub.get_rect(center=(cx, 188)))

        # Stats card
        card_w, card_h = 400, 165
        card_x, card_y = cx - card_w // 2, 230
        pygame.draw.rect(surface, (18, 8, 12), (card_x, card_y, card_w, card_h), border_radius=5)
        pygame.draw.rect(surface, (120, 35, 42), (card_x, card_y, card_w, card_h), 1, border_radius=5)

        rows = [
            ("Score",    f"{score}"),
            ("Kills",    f"{kills}"),
            ("Accuracy", f"{int(accuracy * 100)}%"),
            ("Survived", f"{survived_time:.0f}s"),
            ("Damage",   f"{int(damage_taken)} HP"),
        ]
        for i, (k, v) in enumerate(rows):
            k_s = self.font_body.render(k, True, (145, 160, 180))
            v_s = self.font_body.render(v, True, (220, 230, 240))
            surface.blit(k_s, (card_x + 24, card_y + 14 + i * 28))
            surface.blit(v_s, v_s.get_rect(right=card_x + card_w - 24, top=card_y + 14 + i * 28))

        for btn in buttons:
            btn.draw(surface)

    def draw_victory(
        self,
        surface: pygame.Surface,
        buttons: List[UIButton],
        score: int,
        kills: int,
        accuracy: float,
        damage_taken: float,
        final_health: float,
        survived_time: float,
    ) -> None:
        """Render victory screen."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((4, 18, 12, 220))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt_title = self.font_title.render("MISSION COMPLETE", True, (50, 220, 120))
        surface.blit(txt_title, txt_title.get_rect(center=(cx, 145)))
        txt_sub = self.font_header.render(f"{GAME_TITLE}  ·  Mission Survived", True, (155, 230, 195))
        surface.blit(txt_sub, txt_sub.get_rect(center=(cx, 188)))

        card_w, card_h = 400, 180
        card_x, card_y = cx - card_w // 2, 230
        pygame.draw.rect(surface, (8, 20, 15), (card_x, card_y, card_w, card_h), border_radius=5)
        pygame.draw.rect(surface, (35, 140, 80), (card_x, card_y, card_w, card_h), 1, border_radius=5)

        rows = [
            ("Score",    f"{score}"),
            ("Kills",    f"{kills}"),
            ("Accuracy", f"{int(accuracy * 100)}%"),
            ("Health",   f"{int(final_health)} HP"),
            ("Damage",   f"{int(damage_taken)} HP"),
            ("Time",     f"{survived_time:.0f}s"),
        ]
        for i, (k, v) in enumerate(rows):
            k_s = self.font_body.render(k, True, (130, 175, 150))
            v_s = self.font_body.render(v, True, (210, 245, 225))
            surface.blit(k_s, (card_x + 24, card_y + 12 + i * 25))
            surface.blit(v_s, v_s.get_rect(right=card_x + card_w - 24, top=card_y + 12 + i * 25))

        for btn in buttons:
            btn.draw(surface)

    def draw_controls(self, surface: pygame.Surface, back_button: UIButton) -> None:
        """Render controls and AI mode reference."""
        cx = self.width // 2
        txt = self.font_header.render(f"{GAME_TITLE}  ·  Controls & Systems", True, (175, 210, 240))
        surface.blit(txt, txt.get_rect(center=(cx, 110)))
        pygame.draw.line(surface, (38, 65, 100), (cx - 240, 132), (cx + 240, 132), 1)

        instructions = [
            ("F1 / TAB",   "Toggle AI telemetry overlay"),
            ("ESC / P",    "Pause / open menu"),
            ("R",          "Restart run"),
            ("M",          "Toggle audio"),
            ("ATTACK",     "Tracking target, firing to destroy"),
            ("EVADE",      "Collision imminent — emergency dodge"),
            ("SURVIVE",    "Low hull — defensive standoff"),
            ("REPOSITION", "Navigating to open space via A*"),
        ]

        card_w, card_h = 640, 360
        card_x, card_y = cx - card_w // 2, 150
        pygame.draw.rect(surface, (10, 16, 26), (card_x, card_y, card_w, card_h), border_radius=5)
        pygame.draw.rect(surface, (35, 60, 95), (card_x, card_y, card_w, card_h), 1, border_radius=5)

        for i, (key, desc) in enumerate(instructions):
            k_surf = self.font_body.render(key, True, (200, 185, 100))
            d_surf = self.font_body.render(desc, True, (185, 210, 235))
            surface.blit(k_surf, (card_x + 28, card_y + 20 + i * 38))
            surface.blit(d_surf, (card_x + 220, card_y + 20 + i * 38))
            if i < len(instructions) - 1:
                pygame.draw.line(surface, (22, 36, 54),
                                 (card_x + 14, card_y + 54 + i * 38),
                                 (card_x + card_w - 14, card_y + 54 + i * 38), 1)

        back_button.draw(surface)


class SettingsMenuUI:
    """Comprehensive settings management screen with sliders, checkboxes, cyclers, and persistence."""

    def __init__(
        self,
        settings: GameSettings,
        screen_width: int = 1000,
        screen_height: int = 800,
        on_back: Optional[Callable[[], None]] = None,
        on_apply: Optional[Callable[[], None]] = None,
    ):
        self.settings = settings
        self.width = screen_width
        self.height = screen_height
        self.on_back = on_back
        self.on_apply = on_apply

        self.current_tab = "SPACESHIP"  # "SPACESHIP", "ASTEROIDS", "RED HAZARDS", "GAME & AI"
        self.status_msg = ""
        self.status_timer = 0.0

        # Initialize fonts
        pygame.font.init()
        self.font_title = pygame.font.SysFont("consolas", 30, bold=True)
        self.font_tab = pygame.font.SysFont("consolas", 17, bold=True)
        self.font_label = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_val = pygame.font.SysFont("consolas", 15, bold=False)
        self.font_btn = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_status = pygame.font.SysFont("consolas", 15, bold=True)

        self._init_widgets()
        self.sync_from_settings()

    def _init_widgets(self) -> None:
        cx = self.width // 2

        # 1. Navigation Tabs (4 distinct categories)
        tab_w, tab_h = 175, 36
        tab_y = 88
        self.tab_buttons = [
            UIButton(pygame.Rect(cx - 370, tab_y, tab_w, tab_h), "SPACESHIP", lambda: self.set_tab("SPACESHIP"), self.font_tab),
            UIButton(pygame.Rect(cx - 185, tab_y, tab_w, tab_h), "ASTEROIDS", lambda: self.set_tab("ASTEROIDS"), self.font_tab),
            UIButton(pygame.Rect(cx, tab_y, tab_w, tab_h), "RED HAZARDS", lambda: self.set_tab("RED HAZARDS"), self.font_tab),
            UIButton(pygame.Rect(cx + 185, tab_y, tab_w, tab_h), "GAME & AI", lambda: self.set_tab("GAME & AI"), self.font_tab),
        ]

        # Two Columns layout
        c1_x, c2_x = cx - 360, cx + 20
        track_w = 340
        y_start = 168
        dy = 54

        # 2. Spaceship Tab Widgets
        self.ship_sliders = {
            "max_speed": UISlider(pygame.Rect(c1_x, y_start, track_w, 8), "Maximum Speed", 80.0, 500.0, self.settings.ship_max_speed, 10.0, True, "px/s"),
            "thrust_max": UISlider(pygame.Rect(c1_x, y_start + dy, track_w, 8), "Thrust Acceleration", 150.0, 1000.0, self.settings.ship_thrust_max, 25.0, True, "px/s²"),
            "turn_max": UISlider(pygame.Rect(c1_x, y_start + dy * 2, track_w, 8), "Turning Speed", 60.0, 360.0, self.settings.ship_turn_max, 10.0, True, "deg/s"),
            "max_health": UISlider(pygame.Rect(c1_x, y_start + dy * 3, track_w, 8), "Ship Health", 25.0, 400.0, self.settings.ship_max_health, 25.0, True, "HP"),
            "starting_lives": UISlider(pygame.Rect(c1_x, y_start + dy * 4, track_w, 8), "Starting Lives", 1.0, 10.0, float(self.settings.ship_starting_lives), 1.0, True, "Lives"),
            "damage_large": UISlider(pygame.Rect(c1_x, y_start + dy * 5, track_w, 8), "Large Asteroid Damage", 10.0, 100.0, self.settings.damage_large_asteroid, 5.0, True, "HP"),

            "fire_rate": UISlider(pygame.Rect(c2_x, y_start, track_w, 8), "Firing Rate", 1.0, 20.0, self.settings.ship_fire_rate, 0.5, False, "shots/s"),
            "bullet_speed": UISlider(pygame.Rect(c2_x, y_start + dy, track_w, 8), "Projectile Speed", 300.0, 1500.0, self.settings.bullet_speed, 50.0, True, "px/s"),
            "damage_medium": UISlider(pygame.Rect(c2_x, y_start + dy * 2, track_w, 8), "Medium Asteroid Damage", 5.0, 80.0, self.settings.damage_medium_asteroid, 5.0, True, "HP"),
            "damage_small": UISlider(pygame.Rect(c2_x, y_start + dy * 3, track_w, 8), "Small Asteroid Damage", 2.0, 40.0, self.settings.damage_small_asteroid, 2.0, True, "HP"),
            "collision_invuln": UISlider(pygame.Rect(c2_x, y_start + dy * 4, track_w, 8), "Collision Invulnerability", 0.5, 4.0, self.settings.normal_collision_invulnerability, 0.25, False, "s"),
            "respawn_invuln": UISlider(pygame.Rect(c2_x, y_start + dy * 5, track_w, 8), "Respawn Invulnerability", 1.0, 6.0, self.settings.respawn_invulnerability, 0.5, False, "s"),
        }
        self.ship_cycle_design = UICycleButton(
            pygame.Rect(c1_x, y_start + dy * 6 + 4, track_w, 36),
            "Chassis Design",
            ["Interceptor", "Scout", "Vanguard"],
            self.settings.ship_design,
        )
        self.ship_checkboxes = {
            "health_regen": UICheckbox(pygame.Rect(c2_x, y_start + dy * 6 + 10, 22, 22), "Enable Health Regeneration", self.settings.health_regen_enabled),
        }

        # 3. Asteroids Tab Widgets
        self.asteroid_sliders = {
            "initial_asteroids": UISlider(pygame.Rect(c1_x, y_start, track_w, 8), "Initial Asteroids", 1.0, 20.0, float(self.settings.initial_asteroids), 1.0, True, "Rocks"),
            "max_asteroids": UISlider(pygame.Rect(c1_x, y_start + dy, track_w, 8), "Max Active Asteroids", 5.0, 50.0, float(self.settings.max_asteroids), 1.0, True, "Rocks"),
            "speed_mult": UISlider(pygame.Rect(c1_x, y_start + dy * 2, track_w, 8), "Speed Multiplier", 0.3, 3.0, self.settings.asteroid_speed_mult, 0.1, False, "x"),
            "spawn_interval": UISlider(pygame.Rect(c1_x, y_start + dy * 3, track_w, 8), "Spawn Interval", 2.0, 30.0, self.settings.spawn_interval_s, 1.0, True, "s"),
            "bullet_damage": UISlider(pygame.Rect(c1_x, y_start + dy * 4, track_w, 8), "Bullet Damage", 0.5, 10.0, self.settings.bullet_damage, 0.5, False, "HP"),
            "frag_speed": UISlider(pygame.Rect(c1_x, y_start + dy * 5, track_w, 8), "Fragment Speed Mult", 0.5, 2.5, self.settings.fragment_speed_mult, 0.1, False, "x"),

            "hp_small": UISlider(pygame.Rect(c2_x, y_start, track_w, 8), "Small Asteroid Health", 1.0, 10.0, self.settings.asteroid_health_small, 1.0, True, "HP"),
            "hp_medium": UISlider(pygame.Rect(c2_x, y_start + dy, track_w, 8), "Medium Asteroid Health", 1.0, 20.0, self.settings.asteroid_health_medium, 1.0, True, "HP"),
            "hp_large": UISlider(pygame.Rect(c2_x, y_start + dy * 2, track_w, 8), "Large Asteroid Health", 1.0, 30.0, self.settings.asteroid_health_large, 1.0, True, "HP"),
            "frag_min": UISlider(pygame.Rect(c2_x, y_start + dy * 3, track_w, 8), "Min Fragment Count", 1.0, 4.0, float(self.settings.fragment_count_min), 1.0, True, "Rocks"),
            "frag_max": UISlider(pygame.Rect(c2_x, y_start + dy * 4, track_w, 8), "Max Fragment Count", 2.0, 5.0, float(self.settings.fragment_count_max), 1.0, True, "Rocks"),
            "score_size_1": UISlider(pygame.Rect(c2_x, y_start + dy * 5, track_w, 8), "Small Asteroid Score", 20.0, 500.0, float(self.settings.score_size_1), 10.0, True, "pts"),
        }
        self.asteroid_checkboxes = {
            "split_enabled": UICheckbox(pygame.Rect(c1_x, y_start + dy * 6 + 12, 22, 22), "Split Into Fragments on Destruction", self.settings.asteroid_split_enabled),
            "wall_bounce": UICheckbox(pygame.Rect(c2_x, y_start + dy * 6 + 12, 22, 22), "Elastic Arena Wall Bounce (No Wrap)", self.settings.asteroid_wall_bounce),
        }

        # 4. Red Penalty Asteroid Tab Widgets
        self.red_sliders = {
            "max_active": UISlider(pygame.Rect(c1_x, y_start + dy, track_w, 8), "Max Active Red Asteroids", 1.0, 5.0, float(self.settings.red_asteroid_max_active), 1.0, True, "Max"),
            "spawn_prob": UISlider(pygame.Rect(c1_x, y_start + dy * 2, track_w, 8), "Spawn Probability per Check", 0.01, 0.30, self.settings.red_asteroid_spawn_probability, 0.01, False, "%"),
            "spawn_interval": UISlider(pygame.Rect(c1_x, y_start + dy * 3, track_w, 8), "Spawn Check Interval", 1.0, 20.0, self.settings.red_asteroid_spawn_interval, 1.0, True, "s"),

            "speed": UISlider(pygame.Rect(c2_x, y_start, track_w, 8), "Red Asteroid Speed", 40.0, 250.0, self.settings.red_asteroid_speed, 10.0, True, "px/s"),
            "radius": UISlider(pygame.Rect(c2_x, y_start + dy, track_w, 8), "Red Asteroid Radius", 15.0, 50.0, self.settings.red_asteroid_radius, 2.0, True, "px"),
            "shooting_penalty": UISlider(pygame.Rect(c2_x, y_start + dy * 2, track_w, 8), "Shooting Score Penalty", 1.0, 30.0, float(self.settings.red_asteroid_shooting_penalty), 1.0, True, "pts"),
            "collision_penalty": UISlider(pygame.Rect(c2_x, y_start + dy * 3, track_w, 8), "Collision Score Penalty", 2.0, 50.0, float(self.settings.red_asteroid_collision_penalty), 1.0, True, "pts"),
        }
        self.red_checkboxes = {
            "enabled": UICheckbox(pygame.Rect(c1_x, y_start + 10, 22, 22), "Enable Red Penalty Asteroid Hazards", self.settings.red_asteroid_enabled),
        }

        # 5. Game & AI Tab Widgets
        diff_names = [d.value for d in DifficultyLevel]
        self.game_cycle_difficulty = UICycleButton(pygame.Rect(c1_x, y_start, track_w, 38), "Difficulty Preset", diff_names, self.settings.difficulty.value)
        self.game_cycle_aggression = UICycleButton(pygame.Rect(c1_x, y_start + 50, track_w, 38), "AI Aggression", ["Defensive", "Balanced", "Aggressive"], self.settings.ai_aggression)

        self.game_sliders = {
            "countdown_duration": UISlider(pygame.Rect(c1_x, y_start + 115, track_w, 8), "Countdown Duration", 20.0, 360.0, self.settings.countdown_duration, 15.0, True, "s"),
            "collision_margin": UISlider(pygame.Rect(c1_x, y_start + 175, track_w, 8), "AI Collision Safety Margin", 15.0, 100.0, self.settings.collision_safety_margin, 5.0, True, "px"),
            "target_switch": UISlider(pygame.Rect(c1_x, y_start + 235, track_w, 8), "Target Switch Threshold", 0.05, 0.50, self.settings.target_switch_threshold, 0.05, False, ""),

            "w_risk": UISlider(pygame.Rect(c2_x, y_start, track_w, 8), "Threat Risk Weight (wR)", 0.1, 1.0, self.settings.threat_weight_risk, 0.05, False, ""),
            "w_urgency": UISlider(pygame.Rect(c2_x, y_start + dy, track_w, 8), "Threat Urgency Weight (wU)", 0.1, 1.0, self.settings.threat_weight_urgency, 0.05, False, ""),
            "w_geometry": UISlider(pygame.Rect(c2_x, y_start + dy * 2, track_w, 8), "Threat Closing Speed Weight", 0.05, 0.5, self.settings.threat_weight_geometry, 0.05, False, ""),
        }

        self.game_checkboxes = {
            "difficulty_progression": UICheckbox(pygame.Rect(c2_x, y_start + dy * 3 + 10, 22, 22), "Difficulty Escalation Over Countdown", self.settings.difficulty_progression),
            "adaptive_risk": UICheckbox(pygame.Rect(c2_x, y_start + dy * 4 + 10, 22, 22), "Adaptive Life-Dependent Risk Model", self.settings.adaptive_risk),
            "screen_shake": UICheckbox(pygame.Rect(c2_x, y_start + dy * 5 + 10, 22, 22), "Impact Camera Shake Effects", self.settings.screen_shake_enabled),
            "debug_overlay": UICheckbox(pygame.Rect(c2_x, y_start + dy * 6 + 10, 22, 22), "Show AI Telemetry Overlay (F1/Tab)", self.settings.show_debug_overlay),
        }

        # 6. Bottom Action Buttons
        btn_y = 665
        btn_h = 36
        self.apply_btn = UIButton(pygame.Rect(cx - 240, btn_y, 140, btn_h), "Apply", self.apply_to_settings, self.font_btn)
        self.reset_btn = UIButton(pygame.Rect(cx - 70, btn_y, 140, btn_h), "Reset", self.reset_defaults, self.font_btn)
        self.back_btn = UIButton(pygame.Rect(cx + 100, btn_y, 140, btn_h), "Back", self.on_back_clicked, self.font_btn)

    def set_tab(self, tab: str) -> None:
        self.current_tab = tab

    def on_back_clicked(self) -> None:
        if self.on_back:
            self.on_back()

    def sync_from_settings(self) -> None:
        """Update slider and checkbox states from current settings object."""
        s = self.settings

        # Spaceship
        self.ship_cycle_design.current = s.ship_design
        self.ship_sliders["max_speed"].set_value(s.ship_max_speed)
        self.ship_sliders["thrust_max"].set_value(s.ship_thrust_max)
        self.ship_sliders["turn_max"].set_value(s.ship_turn_max)
        self.ship_sliders["max_health"].set_value(s.ship_max_health)
        self.ship_sliders["starting_lives"].set_value(float(s.ship_starting_lives))
        self.ship_sliders["damage_large"].set_value(s.damage_large_asteroid)
        self.ship_sliders["fire_rate"].set_value(s.ship_fire_rate)
        self.ship_sliders["bullet_speed"].set_value(s.bullet_speed)
        self.ship_sliders["damage_medium"].set_value(s.damage_medium_asteroid)
        self.ship_sliders["damage_small"].set_value(s.damage_small_asteroid)
        self.ship_sliders["collision_invuln"].set_value(s.normal_collision_invulnerability)
        self.ship_sliders["respawn_invuln"].set_value(s.respawn_invulnerability)
        self.ship_checkboxes["health_regen"].checked = s.health_regen_enabled

        # Asteroids
        self.asteroid_sliders["initial_asteroids"].set_value(float(s.initial_asteroids))
        self.asteroid_sliders["max_asteroids"].set_value(float(s.max_asteroids))
        self.asteroid_sliders["speed_mult"].set_value(s.asteroid_speed_mult)
        self.asteroid_sliders["spawn_interval"].set_value(s.spawn_interval_s)
        self.asteroid_sliders["bullet_damage"].set_value(s.bullet_damage)
        self.asteroid_sliders["frag_speed"].set_value(s.fragment_speed_mult)
        self.asteroid_sliders["hp_small"].set_value(s.asteroid_health_small)
        self.asteroid_sliders["hp_medium"].set_value(s.asteroid_health_medium)
        self.asteroid_sliders["hp_large"].set_value(s.asteroid_health_large)
        self.asteroid_sliders["frag_min"].set_value(float(s.fragment_count_min))
        self.asteroid_sliders["frag_max"].set_value(float(s.fragment_count_max))
        self.asteroid_sliders["score_size_1"].set_value(float(s.score_size_1))
        self.asteroid_checkboxes["split_enabled"].checked = s.asteroid_split_enabled
        self.asteroid_checkboxes["wall_bounce"].checked = s.asteroid_wall_bounce

        # Red Penalty Asteroids
        self.red_checkboxes["enabled"].checked = s.red_asteroid_enabled
        self.red_sliders["max_active"].set_value(float(s.red_asteroid_max_active))
        self.red_sliders["spawn_prob"].set_value(s.red_asteroid_spawn_probability)
        self.red_sliders["spawn_interval"].set_value(s.red_asteroid_spawn_interval)
        self.red_sliders["speed"].set_value(s.red_asteroid_speed)
        self.red_sliders["radius"].set_value(s.red_asteroid_radius)
        self.red_sliders["shooting_penalty"].set_value(float(s.red_asteroid_shooting_penalty))
        self.red_sliders["collision_penalty"].set_value(float(s.red_asteroid_collision_penalty))

        # Game & AI
        self.game_cycle_difficulty.current = s.difficulty.value
        self.game_cycle_aggression.current = s.ai_aggression
        self.game_sliders["countdown_duration"].set_value(s.countdown_duration)
        self.game_sliders["collision_margin"].set_value(s.collision_safety_margin)
        self.game_sliders["target_switch"].set_value(s.target_switch_threshold)
        self.game_sliders["w_risk"].set_value(s.threat_weight_risk)
        self.game_sliders["w_urgency"].set_value(s.threat_weight_urgency)
        self.game_sliders["w_geometry"].set_value(s.threat_weight_geometry)
        self.game_checkboxes["difficulty_progression"].checked = s.difficulty_progression
        self.game_checkboxes["adaptive_risk"].checked = s.adaptive_risk
        self.game_checkboxes["screen_shake"].checked = s.screen_shake_enabled
        self.game_checkboxes["debug_overlay"].checked = s.show_debug_overlay

    def apply_to_settings(self) -> None:
        """Save all UI widget values back into GameSettings and persist to file."""
        s = self.settings

        # Spaceship
        s.ship_design = self.ship_cycle_design.current
        s.ship_max_speed = float(self.ship_sliders["max_speed"].get_value())
        s.ship_thrust_max = float(self.ship_sliders["thrust_max"].get_value())
        s.ship_turn_max = float(self.ship_sliders["turn_max"].get_value())
        s.ship_max_health = float(self.ship_sliders["max_health"].get_value())
        s.ship_starting_lives = int(self.ship_sliders["starting_lives"].get_value())
        s.damage_large_asteroid = float(self.ship_sliders["damage_large"].get_value())
        s.ship_fire_rate = float(self.ship_sliders["fire_rate"].get_value())
        s.bullet_speed = float(self.ship_sliders["bullet_speed"].get_value())
        s.damage_medium_asteroid = float(self.ship_sliders["damage_medium"].get_value())
        s.damage_small_asteroid = float(self.ship_sliders["damage_small"].get_value())
        s.normal_collision_invulnerability = float(self.ship_sliders["collision_invuln"].get_value())
        s.respawn_invulnerability = float(self.ship_sliders["respawn_invuln"].get_value())
        s.health_regen_enabled = self.ship_checkboxes["health_regen"].checked

        # Asteroids
        s.initial_asteroids = int(self.asteroid_sliders["initial_asteroids"].get_value())
        s.max_asteroids = int(self.asteroid_sliders["max_asteroids"].get_value())
        s.asteroid_speed_mult = float(self.asteroid_sliders["speed_mult"].get_value())
        s.spawn_interval_s = float(self.asteroid_sliders["spawn_interval"].get_value())
        s.bullet_damage = float(self.asteroid_sliders["bullet_damage"].get_value())
        s.fragment_speed_mult = float(self.asteroid_sliders["frag_speed"].get_value())
        s.asteroid_health_small = float(self.asteroid_sliders["hp_small"].get_value())
        s.asteroid_health_medium = float(self.asteroid_sliders["hp_medium"].get_value())
        s.asteroid_health_large = float(self.asteroid_sliders["hp_large"].get_value())
        s.fragment_count_min = int(self.asteroid_sliders["frag_min"].get_value())
        s.fragment_count_max = int(self.asteroid_sliders["frag_max"].get_value())
        s.score_size_1 = int(self.asteroid_sliders["score_size_1"].get_value())
        s.asteroid_split_enabled = self.asteroid_checkboxes["split_enabled"].checked
        s.asteroid_wall_bounce = self.asteroid_checkboxes["wall_bounce"].checked

        # Red Penalty Asteroids
        s.red_asteroid_enabled = self.red_checkboxes["enabled"].checked
        s.red_asteroid_max_active = int(self.red_sliders["max_active"].get_value())
        s.red_asteroid_spawn_probability = float(self.red_sliders["spawn_prob"].get_value())
        s.red_asteroid_spawn_interval = float(self.red_sliders["spawn_interval"].get_value())
        s.red_asteroid_speed = float(self.red_sliders["speed"].get_value())
        s.red_asteroid_radius = float(self.red_sliders["radius"].get_value())
        s.red_asteroid_shooting_penalty = int(self.red_sliders["shooting_penalty"].get_value())
        s.red_asteroid_collision_penalty = int(self.red_sliders["collision_penalty"].get_value())

        # Game & AI
        diff_str = self.game_cycle_difficulty.current
        for d in DifficultyLevel:
            if d.value == diff_str:
                s.difficulty = d
                break
        s.ai_aggression = self.game_cycle_aggression.current
        s.countdown_duration = float(self.game_sliders["countdown_duration"].get_value())
        s.collision_safety_margin = float(self.game_sliders["collision_margin"].get_value())
        s.target_switch_threshold = float(self.game_sliders["target_switch"].get_value())
        s.threat_weight_risk = float(self.game_sliders["w_risk"].get_value())
        s.threat_weight_urgency = float(self.game_sliders["w_urgency"].get_value())
        s.threat_weight_geometry = float(self.game_sliders["w_geometry"].get_value())
        s.difficulty_progression = self.game_checkboxes["difficulty_progression"].checked
        s.adaptive_risk = self.game_checkboxes["adaptive_risk"].checked
        s.screen_shake_enabled = self.game_checkboxes["screen_shake"].checked
        s.show_debug_overlay = self.game_checkboxes["debug_overlay"].checked

        s.validate_and_clamp()
        s.save_to_file()

        self.status_msg = "Settings saved. (CONFIG SAVED)"
        self.status_timer = 2.5

        if self.on_apply:
            self.on_apply()

    def reset_defaults(self) -> None:
        """Reset settings to default and refresh UI."""
        self.settings.reset_to_defaults()
        self.settings.save_to_file()
        self.sync_from_settings()
        self.status_msg = "Settings reset to defaults."
        self.status_timer = 2.5

    def handle_event(self, event: pygame.event.Event, mouse_pos: Tuple[int, int]) -> bool:
        """Process mouse and key interactions for active settings widgets."""
        # Check tab clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for btn in self.tab_buttons:
                if btn.handle_click(mouse_pos):
                    return True
            if self.apply_btn.handle_click(mouse_pos):
                return True
            if self.reset_btn.handle_click(mouse_pos):
                return True
            if self.back_btn.handle_click(mouse_pos):
                return True

        # Process current tab widgets
        if self.current_tab == "SPACESHIP":
            for sl in self.ship_sliders.values():
                if sl.handle_event(event, mouse_pos):
                    return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.ship_cycle_design.handle_click(mouse_pos):
                    return True
                for cb in self.ship_checkboxes.values():
                    if cb.handle_click(mouse_pos):
                        return True

        elif self.current_tab == "ASTEROIDS":
            for sl in self.asteroid_sliders.values():
                if sl.handle_event(event, mouse_pos):
                    return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for cb in self.asteroid_checkboxes.values():
                    if cb.handle_click(mouse_pos):
                        return True

        elif self.current_tab == "RED HAZARDS":
            for sl in self.red_sliders.values():
                if sl.handle_event(event, mouse_pos):
                    return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for cb in self.red_checkboxes.values():
                    if cb.handle_click(mouse_pos):
                        return True

        elif self.current_tab == "GAME & AI":
            for sl in self.game_sliders.values():
                if sl.handle_event(event, mouse_pos):
                    return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.game_cycle_difficulty.handle_click(mouse_pos):
                    return True
                if self.game_cycle_aggression.handle_click(mouse_pos):
                    return True
                for cb in self.game_checkboxes.values():
                    if cb.handle_click(mouse_pos):
                        return True

        return False

    def update(self, dt: float) -> None:
        if self.status_timer > 0:
            self.status_timer -= dt
            if self.status_timer <= 0:
                self.status_msg = ""

    def draw(self, surface: pygame.Surface) -> None:
        """Render the complete settings panel overlay."""
        cx = self.width // 2

        # 1. Background dark translucent card
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((8, 14, 24, 235))
        surface.blit(overlay, (0, 0))

        # Title
        txt_title = self.font_title.render(f"{GAME_TITLE}  ·  GAME SETTINGS", True, (210, 230, 255))
        surface.blit(txt_title, txt_title.get_rect(center=(cx, 45)))

        # Subtitle notice
        txt_sub = self.font_val.render("Changes apply to the next game session", True, (120, 155, 185))
        surface.blit(txt_sub, txt_sub.get_rect(center=(cx, 72)))

        # 2. Tab Bar
        for btn in self.tab_buttons:
            is_active = (btn.text == self.current_tab)
            if is_active:
                pygame.draw.rect(surface, (30, 65, 110), btn.rect, border_radius=6)
                pygame.draw.rect(surface, (80, 230, 255), btn.rect, 2, border_radius=6)
                txt_col = (255, 255, 255)
            else:
                pygame.draw.rect(surface, (15, 25, 40), btn.rect, border_radius=6)
                pygame.draw.rect(surface, (40, 80, 120), btn.rect, 1, border_radius=6)
                txt_col = (160, 190, 220)
            txt_s = self.font_tab.render(btn.text, True, txt_col)
            surface.blit(txt_s, txt_s.get_rect(center=btn.rect.center))

        # 3. Main Content Container
        panel_rect = pygame.Rect(cx - 390, 135, 780, 510)
        pygame.draw.rect(surface, (12, 20, 32), panel_rect, border_radius=8)
        pygame.draw.rect(surface, (40, 85, 135), panel_rect, 2, border_radius=8)

        # 4. Render Active Tab Content
        if self.current_tab == "SPACESHIP":
            for sl in self.ship_sliders.values():
                sl.draw(surface, self.font_label, self.font_val)
            self.ship_cycle_design.draw(surface, self.font_tab)
            for cb in self.ship_checkboxes.values():
                cb.draw(surface, self.font_label)

        elif self.current_tab == "ASTEROIDS":
            for sl in self.asteroid_sliders.values():
                sl.draw(surface, self.font_label, self.font_val)
            for cb in self.asteroid_checkboxes.values():
                cb.draw(surface, self.font_label)

        elif self.current_tab == "RED HAZARDS":
            for sl in self.red_sliders.values():
                sl.draw(surface, self.font_label, self.font_val)
            for cb in self.red_checkboxes.values():
                cb.draw(surface, self.font_label)

        elif self.current_tab == "GAME & AI":
            self.game_cycle_difficulty.draw(surface, self.font_tab)
            self.game_cycle_aggression.draw(surface, self.font_tab)
            for sl in self.game_sliders.values():
                sl.draw(surface, self.font_label, self.font_val)
            for cb in self.game_checkboxes.values():
                cb.draw(surface, self.font_label)

            # Active difficulty description card
            curr_diff = DifficultyLevel.from_str(self.game_cycle_difficulty.current)
            desc = DIFFICULTY_DESCRIPTIONS.get(curr_diff, "")
            card_x, card_y = cx - 360, 168 + 295
            card_w, card_h = 340, 75
            pygame.draw.rect(surface, (16, 26, 42), (card_x, card_y, card_w, card_h), border_radius=6)
            pygame.draw.rect(surface, (45, 90, 140), (card_x, card_y, card_w, card_h), 1, border_radius=6)
            txt_diff_head = self.font_label.render(f"PRESET: {curr_diff.value.upper()}", True, (80, 220, 255))
            surface.blit(txt_diff_head, (card_x + 12, card_y + 12))
            txt_desc = self.font_val.render(desc, True, (180, 205, 230))
            surface.blit(txt_desc, (card_x + 12, card_y + 38))

        # 5. Action Buttons
        self.apply_btn.draw(surface)
        self.reset_btn.draw(surface)
        self.back_btn.draw(surface)

        # 6. Status Message Banner
        if self.status_msg:
            stat_surf = self.font_status.render(self.status_msg, True, (50, 240, 140))
            surface.blit(stat_surf, stat_surf.get_rect(center=(cx, 730)))

