"""Game interface, HUD, menus, and AI debug overlay for Kessler AI survival game.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pygame

from .kessler_ai import AIDebugTelemetry, AIBehaviorState
from .safe_zone import SafeZone
from .settings import DifficultyLevel, GameSettings
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
        # Sci-fi beveled button style
        if self.is_hovered:
            bg_color = (25, 45, 75)
            border_color = (80, 220, 255)
            text_color = (255, 255, 255)
            glow_rect = self.rect.inflate(4, 4)
            pygame.draw.rect(surface, (40, 100, 160), glow_rect, 1, border_radius=6)
        else:
            bg_color = (15, 25, 40)
            border_color = (40, 90, 140)
            text_color = (180, 210, 240)

        pygame.draw.rect(surface, bg_color, self.rect, border_radius=4)
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=4)

        if self.font:
            txt_surf = self.font.render(self.text, True, text_color)
            txt_rect = txt_surf.get_rect(center=self.rect.center)
            surface.blit(txt_surf, txt_rect)


class UIManager:
    """Renders in-game HUD, menus, radar, and toggleable AI telemetry overlays."""

    def __init__(self, screen_width: int = 1000, screen_height: int = 800):
        self.width = screen_width
        self.height = screen_height

        pygame.font.init()
        # Load high-readability monospace / sci-fi system fonts with fallbacks
        self.font_title = self._get_font(["consolas", "lucidaconsole", "arial"], 44, bold=True)
        self.font_header = self._get_font(["consolas", "lucidaconsole", "arial"], 26, bold=True)
        self.font_body = self._get_font(["consolas", "lucidaconsole", "arial"], 17, bold=False)
        self.font_small = self._get_font(["consolas", "lucidaconsole", "arial"], 13, bold=False)

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
        safe_zone: SafeZone,
        elapsed_time: float,
        asteroids_count: int,
        is_ai_mode: bool = False,
        ai_telemetry: Optional[AIDebugTelemetry] = None,
        high_score: int = 0,
    ) -> None:
        """Render comprehensive in-game survival HUD."""
        # --- Top Left: Health, Shield, Boost, Lives ---
        margin_x = 20
        margin_y = 18

        # Hull Health Bar
        bar_w, bar_h = 180, 14
        hull_pct = max(0.0, ship.health / ship.max_health)
        hull_color = (
            (60, 220, 80) if hull_pct > 0.5 else ((240, 200, 40) if hull_pct > 0.25 else (240, 50, 50))
        )
        pygame.draw.rect(surface, (20, 30, 40), (margin_x, margin_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(surface, hull_color, (margin_x, margin_y, int(bar_w * hull_pct), bar_h), border_radius=3)
        pygame.draw.rect(surface, (100, 140, 180), (margin_x, margin_y, bar_w, bar_h), 1, border_radius=3)
        txt_hull = self.font_small.render(f"HULL: {int(ship.health)}%", True, (240, 240, 240))
        surface.blit(txt_hull, (margin_x + bar_w + 10, margin_y - 1))

        # Shield Bar
        shield_pct = max(0.0, ship.shields / ship.max_shields)
        pygame.draw.rect(surface, (20, 30, 40), (margin_x, margin_y + 20, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(surface, (50, 190, 255), (margin_x, margin_y + 20, int(bar_w * shield_pct), bar_h), border_radius=3)
        pygame.draw.rect(surface, (100, 140, 180), (margin_x, margin_y + 20, bar_w, bar_h), 1, border_radius=3)
        txt_shield = self.font_small.render(f"SHIELD: {int(ship.shields)}%", True, (180, 230, 255))
        surface.blit(txt_shield, (margin_x + bar_w + 10, margin_y + 19))

        # Boost Bar
        boost_pct = max(0.0, ship.boost_energy / 100.0)
        boost_color = (255, 170, 40) if not ship.is_boosting else (80, 240, 255)
        pygame.draw.rect(surface, (20, 30, 40), (margin_x, margin_y + 40, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(surface, boost_color, (margin_x, margin_y + 40, int(bar_w * boost_pct), bar_h), border_radius=3)
        pygame.draw.rect(surface, (100, 140, 180), (margin_x, margin_y + 40, bar_w, bar_h), 1, border_radius=3)
        txt_boost = self.font_small.render(f"BOOST: {int(ship.boost_energy)}%", True, (255, 200, 120))
        surface.blit(txt_boost, (margin_x + bar_w + 10, margin_y + 39))

        # Lives Icons
        for i in range(3):
            lx = margin_x + i * 22
            ly = margin_y + 64
            if i < ship.lives:
                # Small ship triangle
                pts = [(lx + 8, ly), (lx, ly + 14), (lx + 16, ly + 14)]
                pygame.draw.polygon(surface, (80, 220, 255), pts)
                pygame.draw.polygon(surface, (255, 255, 255), pts, 1)
            else:
                pts = [(lx + 8, ly), (lx, ly + 14), (lx + 16, ly + 14)]
                pygame.draw.polygon(surface, (50, 60, 75), pts, 1)

        # --- Top Right: Score, Time, Debris Count ---
        right_x = self.width - 220
        txt_score = self.font_header.render(f"SCORE: {ship.score}", True, (255, 255, 255))
        surface.blit(txt_score, (right_x, margin_y))

        txt_time = self.font_body.render(f"TIME: {elapsed_time:.1f}s", True, (180, 220, 240))
        surface.blit(txt_time, (right_x, margin_y + 30))

        txt_debris = self.font_body.render(f"HAZARDS: {asteroids_count}", True, (240, 160, 80))
        surface.blit(txt_debris, (right_x, margin_y + 52))

        # --- Bottom Left: Speedometer & Heading ---
        spd_val = abs(ship.speed)
        txt_spd = self.font_body.render(f"SPEED: {int(spd_val)} m/s", True, (160, 210, 240))
        surface.blit(txt_spd, (margin_x, self.height - 45))
        txt_hdg = self.font_small.render(f"HEADING: {int(ship.heading)}°", True, (120, 160, 190))
        surface.blit(txt_hdg, (margin_x, self.height - 24))

        # --- Top Center: Objective & Docking Indicator ---
        dist_to_sz = safe_zone.distance_to_perimeter((ship.x, ship.y))
        dock_prog = safe_zone.get_dock_progress(ship.ship_id)
        cx = self.width // 2

        if safe_zone.contains_point((ship.x, ship.y)):
            # Inside safe zone: show docking meter
            dock_w = 260
            pygame.draw.rect(surface, (15, 30, 45), (cx - dock_w // 2, margin_y, dock_w, 24), border_radius=4)
            pygame.draw.rect(surface, (40, 220, 140), (cx - dock_w // 2, margin_y, int(dock_w * dock_prog), 24), border_radius=4)
            pygame.draw.rect(surface, (100, 255, 180), (cx - dock_w // 2, margin_y, dock_w, 24), 2, border_radius=4)
            txt_dock = self.font_body.render(f"DOCKING: {int(dock_prog * 100)}%", True, (255, 255, 255))
            surface.blit(txt_dock, txt_dock.get_rect(center=(cx, margin_y + 12)))
        else:
            # Outside: show arrow pointer toward Safe Zone
            dx = safe_zone.x - ship.x
            dy = safe_zone.y - ship.y
            ang = math.atan2(dy, dx)
            # Draw pointer compass ring
            compass_r = 16
            c_center = (cx - 70, margin_y + 16)
            pygame.draw.circle(surface, (30, 60, 90), c_center, compass_r, 1)
            tip_x = c_center[0] + math.cos(ang) * (compass_r + 4)
            tip_y = c_center[1] + math.sin(ang) * (compass_r + 4)
            pygame.draw.line(surface, (50, 220, 255), c_center, (int(tip_x), int(tip_y)), 3)

            txt_nav = self.font_body.render(f"SAFE ZONE: {int(dist_to_sz)}m", True, (80, 230, 255))
            surface.blit(txt_nav, (cx - 45, margin_y + 6))

        # --- Bottom Center: AI Telemetry Banner (if in AI mode) ---
        if is_ai_mode and ai_telemetry:
            self._draw_ai_telemetry_banner(surface, ai_telemetry)

        # --- Bottom Right: Radar Minimap ---
        self._draw_minimap(surface, ship, safe_zone, is_ai_mode)

    def _draw_ai_telemetry_banner(self, surface: pygame.Surface, tel: AIDebugTelemetry) -> None:
        """Draw active state badge and threat stats for autonomous AI."""
        panel_w = 420
        panel_h = 56
        px = (self.width - panel_w) // 2
        py = self.height - panel_h - 15

        # Background card
        bg_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg_surf.fill((10, 18, 30, 210))
        surface.blit(bg_surf, (px, py))

        border_color = (255, 60, 60) if tel.emergency_override else (40, 180, 240)
        pygame.draw.rect(surface, border_color, (px, py, panel_w, panel_h), 2, border_radius=4)

        # State text
        state_txt = tel.state.value
        state_color = (255, 70, 70) if tel.emergency_override else ((50, 240, 160) if "Dock" in state_txt else (80, 220, 255))
        txt = self.font_body.render(f"AI: {state_txt}", True, state_color)
        surface.blit(txt, (px + 12, py + 8))

        # Telemetry metrics row
        crit_str = f"CRIT: {tel.critical_threats_count}"
        mult_str = f"RISK: {tel.risk_multiplier:.1f}x"
        cost_str = f"MPC: {int(tel.mpc_cost)}"
        txt_stats = self.font_small.render(f"{crit_str} | {mult_str} | {cost_str} | [F1] OVERLAY", True, (160, 200, 230))
        surface.blit(txt_stats, (px + 12, py + 32))

    def _draw_minimap(self, surface: pygame.Surface, ship: Spaceship, safe_zone: SafeZone, is_ai_mode: bool) -> None:
        """Minimap radar display in bottom right corner."""
        mm_w, mm_h = 130, 104
        mx = self.width - mm_w - 18
        my = self.height - mm_h - 18

        # Minimap background
        mm_surf = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA)
        mm_surf.fill((12, 18, 28, 210))
        pygame.draw.rect(mm_surf, (50, 100, 140), (0, 0, mm_w, mm_h), 1)

        scale_x = mm_w / self.width
        scale_y = mm_h / self.height

        # Draw safe zone square
        sz_x = int(safe_zone.x * scale_x)
        sz_y = int(safe_zone.y * scale_y)
        pygame.draw.circle(mm_surf, (40, 220, 140), (sz_x, sz_y), max(3, int(safe_zone.radius * scale_x)), 1)

        # Draw ship
        sx = int(ship.x * scale_x)
        sy = int(ship.y * scale_y)
        ship_color = (255, 80, 80) if is_ai_mode else (60, 210, 255)
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
                # Dashed or glowing green/cyan route
                pygame.draw.lines(surface, (40, 220, 160), False, pts, 2)
                for pt in pts:
                    pygame.draw.circle(surface, (80, 255, 200), pt, 3)

        # 3. MPC Trajectory Arcs
        if settings.show_mpc_trajectories and telemetry.mpc_trajectory:
            traj_pts = [(int(p[0]), int(p[1])) for p in telemetry.mpc_trajectory]
            if len(traj_pts) >= 2:
                arc_color = (255, 60, 60) if telemetry.mpc_best_collides else (60, 220, 255)
                pygame.draw.lines(surface, arc_color, False, traj_pts, 3)
                # Head marker
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

                    # Info label near asteroid
                    txt = self.font_small.render(f"T:{t_score:.2f} | TTC:{t.time_to_collision:.1f}s", True, line_color)
                    surface.blit(txt, (ax + 12, ay - 8))

        # 5. Target Reticle
        if settings.show_target_reticle and telemetry.target_pos is not None:
            tx, ty = int(telemetry.target_pos[0]), int(telemetry.target_pos[1])
            ret_r = 28
            # Crosshair circle + notches
            pygame.draw.circle(surface, (255, 50, 50), (tx, ty), ret_r, 1)
            pygame.draw.line(surface, (255, 50, 50), (tx - ret_r - 6, ty), (tx - ret_r + 4, ty), 2)
            pygame.draw.line(surface, (255, 50, 50), (tx + ret_r - 4, ty), (tx + ret_r + 6, ty), 2)
            pygame.draw.line(surface, (255, 50, 50), (tx, ty - ret_r - 6), (tx, ty - ret_r + 4), 2)
            pygame.draw.line(surface, (255, 50, 50), (tx, ty + ret_r - 4), (tx, ty + ret_r + 6), 2)

    # =========================================================================
    # MENUS & OVERLAY SCREENS
    # =========================================================================
    def draw_main_menu(
        self,
        surface: pygame.Surface,
        buttons: List[UIButton],
        current_difficulty: DifficultyLevel,
    ) -> None:
        """Render main title screen."""
        cx = self.width // 2

        # Title
        title_surf = self.font_title.render("KESSLER AI", True, (80, 220, 255))
        surface.blit(title_surf, title_surf.get_rect(center=(cx, 160)))

        # Subtitle
        sub_surf = self.font_body.render("DEEP SPACE ORBITAL SURVIVAL SIMULATION", True, (160, 200, 230))
        surface.blit(sub_surf, sub_surf.get_rect(center=(cx, 210)))

        # Difficulty tag
        diff_surf = self.font_small.render(f"DIFFICULTY: {current_difficulty.value.upper()}", True, (255, 200, 60))
        surface.blit(diff_surf, diff_surf.get_rect(center=(cx, 245)))

        # Draw all menu buttons
        for btn in buttons:
            btn.draw(surface)

        # Version & Credits footer
        ftr = self.font_small.render("V2.0 | THALES KESSLER MPC ENGINE + PYGAME INTERACTIVE", True, (90, 120, 150))
        surface.blit(ftr, ftr.get_rect(center=(cx, self.height - 30)))

    def draw_pause_menu(self, surface: pygame.Surface, buttons: List[UIButton]) -> None:
        """Render semi-transparent pause overlay."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((5, 10, 18, 190))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt = self.font_header.render("SYSTEM PAUSED", True, (80, 220, 255))
        surface.blit(txt, txt.get_rect(center=(cx, 240)))

        for btn in buttons:
            btn.draw(surface)

    def draw_game_over(
        self,
        surface: pygame.Surface,
        buttons: List[UIButton],
        score: int,
        survived_time: float,
        asteroids_hit: int,
        accuracy: float,
    ) -> None:
        """Render game over failure screen."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((25, 5, 8, 215))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt_title = self.font_title.render("CRITICAL HULL FAILURE", True, (255, 60, 60))
        surface.blit(txt_title, txt_title.get_rect(center=(cx, 180)))

        txt_sub = self.font_body.render("SHIP DESTROYED - MISSION TERMINATED", True, (240, 180, 180))
        surface.blit(txt_sub, txt_sub.get_rect(center=(cx, 225)))

        # Stats summary box
        card_w, card_h = 420, 150
        card_x = cx - card_w // 2
        card_y = 265
        pygame.draw.rect(surface, (20, 10, 15), (card_x, card_y, card_w, card_h), border_radius=6)
        pygame.draw.rect(surface, (140, 40, 50), (card_x, card_y, card_w, card_h), 2, border_radius=6)

        stats = [
            f"FINAL SCORE: {score}",
            f"SURVIVED TIME: {survived_time:.1f}s",
            f"HAZARDS DESTROYED: {asteroids_hit}",
            f"WEAPON ACCURACY: {int(accuracy * 100)}%",
        ]
        for i, s in enumerate(stats):
            t = self.font_body.render(s, True, (220, 230, 240))
            surface.blit(t, (card_x + 30, card_y + 20 + i * 28))

        for btn in buttons:
            btn.draw(surface)

    def draw_victory(
        self,
        surface: pygame.Surface,
        buttons: List[UIButton],
        score: int,
        elapsed_time: float,
        lives_remaining: int,
        asteroids_hit: int,
    ) -> None:
        """Render victory docking screen."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((5, 20, 15, 215))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt_title = self.font_title.render("ORBITAL SANCTUARY REACHED", True, (60, 240, 160))
        surface.blit(txt_title, txt_title.get_rect(center=(cx, 180)))

        txt_sub = self.font_body.render("DOCKING SEQUENCE CONFIRMED - OBJECTIVE COMPLETE!", True, (180, 255, 220))
        surface.blit(txt_sub, txt_sub.get_rect(center=(cx, 225)))

        card_w, card_h = 420, 150
        card_x = cx - card_w // 2
        card_y = 265
        pygame.draw.rect(surface, (10, 25, 20), (card_x, card_y, card_w, card_h), border_radius=6)
        pygame.draw.rect(surface, (40, 160, 100), (card_x, card_y, card_w, card_h), 2, border_radius=6)

        stats = [
            f"FINAL SCORE: {score}",
            f"MISSION DURATION: {elapsed_time:.1f}s",
            f"LIVES REMAINING: {lives_remaining}",
            f"HAZARDS DESTROYED: {asteroids_hit}",
        ]
        for i, s in enumerate(stats):
            t = self.font_body.render(s, True, (220, 255, 240))
            surface.blit(t, (card_x + 30, card_y + 20 + i * 28))

        for btn in buttons:
            btn.draw(surface)

    def draw_controls(self, surface: pygame.Surface, back_button: UIButton) -> None:
        """Render keyboard and mouse control instructions."""
        cx = self.width // 2
        txt = self.font_header.render("FLIGHT CONTROLS & INSTRUCTIONS", True, (80, 220, 255))
        surface.blit(txt, txt.get_rect(center=(cx, 120)))

        instructions = [
            ("W / UP ARROW", "Forward Thrusters (Main Engine)"),
            ("S / DOWN ARROW", "Reverse Thrusters / Kinetic Brakes"),
            ("A / D or LEFT / RIGHT", "Rotational Attitude Control (Turn Left / Right)"),
            ("MOUSE AIM", "Ship Heading Aligns with Crosshair"),
            ("SPACEBAR or LEFT CLICK", "Fire High-Velocity Plasma Cannon"),
            ("LEFT SHIFT / RIGHT SHIFT", "Activate Afterburner Boost (Speed Multiplier)"),
            ("F1 / TAB", "Toggle AI Telemetry & Trajectory Overlay"),
            ("ESC / P", "Pause Game / Open Menu"),
            ("OBJECTIVE", "Navigate through asteroid belt & dock in Safe Zone"),
        ]

        card_w, card_h = 680, 420
        card_x = cx - card_w // 2
        card_y = 170
        pygame.draw.rect(surface, (12, 20, 32), (card_x, card_y, card_w, card_h), border_radius=6)
        pygame.draw.rect(surface, (40, 90, 140), (card_x, card_y, card_w, card_h), 2, border_radius=6)

        for i, (key, desc) in enumerate(instructions):
            k_surf = self.font_body.render(key, True, (255, 200, 60))
            d_surf = self.font_body.render(desc, True, (200, 225, 245))
            surface.blit(k_surf, (card_x + 35, card_y + 24 + i * 40))
            surface.blit(d_surf, (card_x + 270, card_y + 24 + i * 40))

        back_button.draw(surface)
