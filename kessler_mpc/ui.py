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
    """Renders in-game HUD, countdown timer, AI status indicators, and modal screens."""

    def __init__(self, screen_width: int = 1000, screen_height: int = 800):
        self.width = screen_width
        self.height = screen_height

        pygame.font.init()
        self.font_title = self._get_font(["consolas", "lucidaconsole", "arial"], 42, bold=True)
        self.font_timer = self._get_font(["consolas", "lucidaconsole", "arial"], 30, bold=True)
        self.font_header = self._get_font(["consolas", "lucidaconsole", "arial"], 24, bold=True)
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
        countdown_remaining: float,
        elapsed_time: float,
        asteroids_count: int,
        ai_telemetry: Optional[AIDebugTelemetry] = None,
        high_score: int = 0,
        safe_zone: Optional[SafeZone] = None,
    ) -> None:
        """Render HUD featuring countdown timer, health, kill counter, and AI status indicators."""
        margin_x = 22
        margin_y = 18

        # --- Top Left: Health, Shield, and Boost Gauges ---
        bar_w, bar_h = 180, 14
        hull_pct = max(0.0, ship.health / ship.max_health)
        hull_color = (
            (60, 225, 80) if hull_pct > 0.5 else ((240, 200, 40) if hull_pct > 0.25 else (250, 40, 40))
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

        # Damage readout
        txt_dmg = self.font_small.render(f"DAMAGE TAKEN: {int(ship.damage_taken)}", True, (220, 130, 130))
        surface.blit(txt_dmg, (margin_x, margin_y + 60))

        # --- Top Center: Countdown Timer & AI Status Indicator ---
        cx = self.width // 2

        # Format countdown timer as MM:SS.S
        rem = max(0.0, countdown_remaining)
        minutes = int(rem // 60)
        seconds = rem % 60
        timer_str = f"{minutes:02d}:{seconds:04.1f}"

        # Timer card
        timer_box_w, timer_box_h = 240, 44
        timer_x = cx - timer_box_w // 2
        timer_bg = pygame.Surface((timer_box_w, timer_box_h), pygame.SRCALPHA)
        timer_bg.fill((10, 18, 30, 220))
        surface.blit(timer_bg, (timer_x, margin_y))

        # Border changes color when time is low (< 30s)
        timer_border = (255, 60, 60) if rem < 30.0 else (50, 150, 220)
        pygame.draw.rect(surface, timer_border, (timer_x, margin_y, timer_box_w, timer_box_h), 2, border_radius=5)

        t_color = (255, 90, 90) if rem < 30.0 else (80, 230, 255)
        txt_timer = self.font_timer.render(timer_str, True, t_color)
        surface.blit(txt_timer, txt_timer.get_rect(center=(cx, margin_y + 22)))

        # Subtitle label for timer
        txt_rem_label = self.font_small.render("MISSION COUNTDOWN", True, (140, 180, 210))
        surface.blit(txt_rem_label, txt_rem_label.get_rect(center=(cx, margin_y + timer_box_h + 10)))

        # --- AI Status Indicator Badge: ATTACK, EVADE, REPOSITION, SURVIVE ---
        if ai_telemetry:
            self._draw_ai_status_badge(surface, cx, margin_y + timer_box_h + 24, ai_telemetry)

        # --- Top Right: Score, Kills, Accuracy ---
        right_x = self.width - 230
        txt_score = self.font_header.render(f"SCORE: {ship.score}", True, (255, 255, 255))
        surface.blit(txt_score, (right_x, margin_y))

        txt_kills = self.font_body.render(f"KILLS: {ship.kills}", True, (255, 200, 60))
        surface.blit(txt_kills, (right_x, margin_y + 28))

        acc_pct = int(ship.accuracy * 100)
        txt_acc = self.font_body.render(f"ACCURACY: {acc_pct}%", True, (160, 220, 255))
        surface.blit(txt_acc, (right_x, margin_y + 50))

        txt_haz = self.font_small.render(f"ACTIVE HAZARDS: {asteroids_count}", True, (240, 160, 80))
        surface.blit(txt_haz, (right_x, margin_y + 72))

        # --- Bottom Left: Speedometer & Heading ---
        spd_val = abs(ship.speed)
        txt_spd = self.font_body.render(f"SPEED: {int(spd_val)} m/s", True, (160, 210, 240))
        surface.blit(txt_spd, (margin_x, self.height - 45))
        txt_hdg = self.font_small.render(f"HEADING: {int(ship.heading)}°", True, (120, 160, 190))
        surface.blit(txt_hdg, (margin_x, self.height - 24))

        # --- Bottom Right: Radar Minimap ---
        self._draw_minimap(surface, ship, safe_zone)

    def _draw_ai_status_badge(self, surface: pygame.Surface, cx: int, y: int, tel: AIDebugTelemetry) -> None:
        """Draw prominent status pill for ATTACK, EVADE, REPOSITION, and SURVIVE."""
        state = tel.state

        # Status badge styling
        if state == AIBehaviorState.ATTACK:
            text = "AI STATUS: [ATTACK]"
            color = (50, 240, 120)       # Emerald Green
            border = (30, 200, 90)
            bg = (10, 35, 20, 220)
        elif state == AIBehaviorState.EVADE:
            text = "AI STATUS: [EVADE]"
            color = (255, 50, 50)        # Urgent Crimson
            border = (255, 40, 40)
            bg = (40, 10, 15, 230)
        elif state == AIBehaviorState.SURVIVE:
            text = "AI STATUS: [SURVIVE]"
            color = (255, 175, 40)       # Amber Warning
            border = (230, 150, 30)
            bg = (35, 25, 10, 220)
        else:  # REPOSITION
            text = "AI STATUS: [REPOSITION]"
            color = (60, 220, 255)       # Electric Cyan
            border = (40, 180, 240)
            bg = (10, 25, 38, 220)

        badge_w, badge_h = 250, 30
        bx = cx - badge_w // 2

        bg_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        bg_surf.fill(bg)
        surface.blit(bg_surf, (bx, y))

        pygame.draw.rect(surface, border, (bx, y, badge_w, badge_h), 2, border_radius=4)
        txt = self.font_body.render(text, True, color)
        surface.blit(txt, txt.get_rect(center=(cx, y + badge_h // 2)))

        # Sub-stats row
        crit_str = f"CRIT: {tel.critical_threats_count}"
        mult_str = f"RISK: {tel.risk_multiplier:.1f}x"
        cost_str = f"MPC: {int(tel.mpc_cost)}"
        txt_stats = self.font_small.render(f"{crit_str} | {mult_str} | {cost_str} | [F1] OVERLAY", True, (150, 190, 220))
        surface.blit(txt_stats, txt_stats.get_rect(center=(cx, y + badge_h + 12)))

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

        title_surf = self.font_title.render("KESSLER AI", True, (80, 220, 255))
        surface.blit(title_surf, title_surf.get_rect(center=(cx, 160)))

        sub_surf = self.font_body.render("AUTONOMOUS DEEP-SPACE ASTEROID SURVIVAL", True, (160, 200, 230))
        surface.blit(sub_surf, sub_surf.get_rect(center=(cx, 205)))

        desc_surf = self.font_small.render("Kessler MPC + A* Global Route Planning + Collision Override", True, (140, 175, 210))
        surface.blit(desc_surf, desc_surf.get_rect(center=(cx, 232)))

        diff_surf = self.font_small.render(f"DIFFICULTY: {current_difficulty.value.upper()}", True, (255, 200, 60))
        surface.blit(diff_surf, diff_surf.get_rect(center=(cx, 260)))

        for btn in buttons:
            btn.draw(surface)

        ftr = self.font_small.render("V2.5 | 180s AUTONOMOUS COUNTDOWN SURVIVAL", True, (90, 120, 150))
        surface.blit(ftr, ftr.get_rect(center=(cx, self.height - 30)))

    def draw_pause_menu(self, surface: pygame.Surface, buttons: List[UIButton]) -> None:
        """Render semi-transparent pause overlay."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((5, 10, 18, 190))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt = self.font_header.render("SIMULATION PAUSED", True, (80, 220, 255))
        surface.blit(txt, txt.get_rect(center=(cx, 240)))

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
        """Render DEFEAT screen when ship health reaches zero."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((25, 5, 8, 225))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt_title = self.font_title.render("DEFEAT", True, (255, 50, 50))
        surface.blit(txt_title, txt_title.get_rect(center=(cx, 160)))

        txt_sub = self.font_header.render("CRITICAL HULL FAILURE - HEALTH REACHED 0", True, (255, 180, 180))
        surface.blit(txt_sub, txt_sub.get_rect(center=(cx, 210)))

        # Stats summary card
        card_w, card_h = 460, 180
        card_x = cx - card_w // 2
        card_y = 245
        pygame.draw.rect(surface, (20, 10, 15), (card_x, card_y, card_w, card_h), border_radius=6)
        pygame.draw.rect(surface, (160, 40, 50), (card_x, card_y, card_w, card_h), 2, border_radius=6)

        stats = [
            f"FINAL SCORE: {score}",
            f"ASTEROIDS DESTROYED (KILLS): {kills}",
            f"WEAPON ACCURACY: {int(accuracy * 100)}%",
            f"TOTAL DAMAGE TAKEN: {int(damage_taken)}",
            f"SURVIVAL TIME: {survived_time:.1f}s",
        ]
        for i, s in enumerate(stats):
            t = self.font_body.render(s, True, (230, 235, 245))
            surface.blit(t, (card_x + 30, card_y + 18 + i * 30))

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
        """Render VICTORY screen when AI survives the full countdown timer."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((5, 22, 15, 225))
        surface.blit(overlay, (0, 0))

        cx = self.width // 2
        txt_title = self.font_title.render("VICTORY", True, (50, 245, 130))
        surface.blit(txt_title, txt_title.get_rect(center=(cx, 160)))

        txt_sub = self.font_header.render("COUNTDOWN COMPLETE - MISSION SURVIVED!", True, (180, 255, 220))
        surface.blit(txt_sub, txt_sub.get_rect(center=(cx, 210)))

        card_w, card_h = 460, 180
        card_x = cx - card_w // 2
        card_y = 245
        pygame.draw.rect(surface, (10, 28, 20), (card_x, card_y, card_w, card_h), border_radius=6)
        pygame.draw.rect(surface, (40, 180, 100), (card_x, card_y, card_w, card_h), 2, border_radius=6)

        stats = [
            f"FINAL SCORE: {score}",
            f"ASTEROIDS DESTROYED (KILLS): {kills}",
            f"WEAPON ACCURACY: {int(accuracy * 100)}%",
            f"FINAL HEALTH REMAINING: {int(final_health)}%",
            f"TOTAL DAMAGE TAKEN: {int(damage_taken)}",
            f"SURVIVAL TIME: {survived_time:.1f}s",
        ]
        for i, s in enumerate(stats):
            t = self.font_body.render(s, True, (220, 255, 240))
            surface.blit(t, (card_x + 30, card_y + 14 + i * 26))

        for btn in buttons:
            btn.draw(surface)

    def draw_controls(self, surface: pygame.Surface, back_button: UIButton) -> None:
        """Render control instructions and AI subsystem manual."""
        cx = self.width // 2
        txt = self.font_header.render("KESSLER AI SYSTEM SPECIFICATION", True, (80, 220, 255))
        surface.blit(txt, txt.get_rect(center=(cx, 115)))

        instructions = [
            ("AUTONOMOUS CONTROL", "Ship is steered, aimed, and fired 100% by AI"),
            ("COUNTDOWN OBJECTIVE", "Survive the full 180-second orbital hazard timer"),
            ("ATTACK MODE", "AI aligns lead intercept angle & destroys targets for points"),
            ("EVADE MODE", "Emergency safety override dodges imminent collisions"),
            ("SURVIVE MODE", "Low health triggers extreme defensive standoff margins"),
            ("REPOSITION MODE", "A* global planner navigates toward open, clear space"),
            ("F1 / TAB", "Toggle AI Telemetry, Threat Vectors & Trajectory Overlay"),
            ("ESC / P", "Pause Simulation / Open Mission Menu"),
        ]

        card_w, card_h = 720, 420
        card_x = cx - card_w // 2
        card_y = 160
        pygame.draw.rect(surface, (12, 20, 32), (card_x, card_y, card_w, card_h), border_radius=6)
        pygame.draw.rect(surface, (40, 90, 140), (card_x, card_y, card_w, card_h), 2, border_radius=6)

        for i, (key, desc) in enumerate(instructions):
            k_surf = self.font_body.render(key, True, (255, 200, 60))
            d_surf = self.font_body.render(desc, True, (200, 225, 245))
            surface.blit(k_surf, (card_x + 30, card_y + 22 + i * 44))
            surface.blit(d_surf, (card_x + 240, card_y + 22 + i * 44))

        back_button.draw(surface)
