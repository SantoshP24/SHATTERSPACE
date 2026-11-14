"""Complete game loop, state management, and mode coordinator for Kessler AI.

Implements the fully autonomous 180-second countdown asteroid survival game
with escalating difficulty, accurate collision mechanics, and VICTORY / DEFEAT screens.
"""
from __future__ import annotations

import asyncio
import math
import random
import sys
import time
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pygame

from .audio import SoundManager
from .debris import DebrisManager
from .kessler_ai import KesslerAIController
from .physics import PhysicsEngine
from .player import PlayerShip
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
from .touch import TouchController
from .ui import SettingsMenuUI, UIButton, UIManager
from .visuals import ParticleSystem, ScreenShake, Starfield


class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    DEFEAT = "defeat"
    GAME_OVER = "defeat"  # Backward compatibility alias
    VICTORY = "victory"
    CONTROLS = "controls"
    SETTINGS = "settings"


class GameMode(Enum):
    AUTONOMOUS = "Autonomous AI Survival"
    PLAYER = "Manual Player Survival"
    COOP = "Player + AI Co-op / Race"


class KesslerGameApp:
    """Main application orchestrating the playable Kessler AI autonomous survival game."""

    def __init__(self, settings: Optional[GameSettings] = None):
        pygame.init()
        self.settings = settings or GameSettings.load_from_file()

        # Logical canvas setup (Physics, AI, and geometry strictly operate at 1000x800)
        self.width = self.settings.screen_width
        self.height = self.settings.screen_height
        self.virtual_surface = pygame.Surface((self.width, self.height))

        # Physical window & responsive viewport scaling
        self.window_width = self.width
        self.window_height = self.height
        self.viewport_scale: float = 1.0
        self.viewport_offset_x: int = 0
        self.viewport_offset_y: int = 0
        self.scaled_w: int = self.width
        self.scaled_h: int = self.height

        self.flags = pygame.FULLSCREEN if self.settings.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), self.flags)
        pygame.display.set_caption(f"{GAME_TITLE} - {GAME_SUBTITLE}")

        self.clock = pygame.time.Clock()
        self.is_running = True
        self.state = GameState.MENU
        self.mode = GameMode.AUTONOMOUS

        # Subsystems
        self.audio = SoundManager(enabled=self.settings.sfx_enabled, master_volume=self.settings.master_volume)
        self.visuals = ParticleSystem(max_particles=1500)
        self.starfield = Starfield(width=self.width, height=self.height)
        self.shake = ScreenShake()
        self.ui = UIManager(screen_width=self.width, screen_height=self.height)
        self.physics = PhysicsEngine(map_size=(self.width, self.height))
        self.touch = TouchController(on_pause_callback=self._on_touch_pause)

        # Settings Menu Interface
        self.settings_menu = SettingsMenuUI(
            settings=self.settings,
            screen_width=self.width,
            screen_height=self.height,
            on_back=lambda: self.set_state(GameState.MENU),
            on_apply=self._on_settings_applied,
        )

        # Gameplay Entities
        self.safe_zone: Optional[SafeZone] = SafeZone(center=(self.width - 140.0, 140.0), radius=75.0, required_dock_time=3.0)
        self.debris = DebrisManager(map_size=(self.width, self.height), settings=self.settings)
        self.player_ship: Optional[PlayerShip] = None
        self.ai_ship: Optional[Spaceship] = None
        self.ai_controller: Optional[KesslerAIController] = None

        # Countdown & Game Session Variables
        self.countdown_total: float = getattr(self.settings, "countdown_duration", 180.0)
        self.countdown_remaining: float = self.countdown_total
        self.elapsed_time: float = 0.0
        self.spawn_timer: float = 0.0
        self.last_difficulty_alert: float = 0.0
        self.high_score: int = 0

        # Build UI Menus
        self._init_ui_buttons()

    def _on_settings_applied(self) -> None:
        self.audio.play("ui_click")
        self.main_menu_buttons[3].text = f"DIFFICULTY: {self.settings.difficulty.value.upper()}"
        self.countdown_total = self.settings.countdown_duration

    def _init_ui_buttons(self) -> None:
        """Create buttons for menus."""
        cx = self.width // 2
        btn_w, btn_h = 280, 38
        font = self.ui.font_body

        # Main Menu Buttons — exact requested labels
        self.main_menu_buttons = [
            UIButton(
                pygame.Rect(cx - btn_w // 2, 235, btn_w, btn_h),
                "START GAME",
                lambda: self.start_game(GameMode.AUTONOMOUS),
                font,
            ),
            UIButton(
                pygame.Rect(cx - btn_w // 2, 282, btn_w, btn_h),
                "MANUAL PILOT",
                lambda: self.start_game(GameMode.PLAYER),
                font,
            ),
            UIButton(
                pygame.Rect(cx - btn_w // 2, 329, btn_w, btn_h),
                "PLAYER + AI",
                lambda: self.start_game(GameMode.COOP),
                font,
            ),
            UIButton(
                pygame.Rect(cx - btn_w // 2, 376, btn_w, btn_h),
                f"DIFFICULTY: {self.settings.difficulty.value.upper()}",
                self._cycle_difficulty_action,
                font,
            ),
            UIButton(
                pygame.Rect(cx - btn_w // 2, 423, btn_w, btn_h),
                "GAME SETTINGS",
                lambda: self.set_state(GameState.SETTINGS),
                font,
            ),
            UIButton(
                pygame.Rect(cx - btn_w // 2, 470, btn_w, btn_h),
                "CONTROLS",
                lambda: self.set_state(GameState.CONTROLS),
                font,
            ),
            UIButton(
                pygame.Rect(cx - btn_w // 2, 517, btn_w, btn_h),
                "QUIT GAME",
                self.quit,
                font,
            ),
        ]

        # Pause Menu Buttons
        self.pause_buttons = [
            UIButton(pygame.Rect(cx - btn_w // 2, 270, btn_w, btn_h), "RESUME", lambda: self.set_state(GameState.PLAYING), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 318, btn_w, btn_h), "RESTART", lambda: self.start_game(self.mode), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 366, btn_w, btn_h), "GAME SETTINGS", lambda: self.set_state(GameState.SETTINGS), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 414, btn_w, btn_h), "TOGGLE OVERLAY [F1]", self._toggle_overlay, font),
            UIButton(pygame.Rect(cx - btn_w // 2, 462, btn_w, btn_h), "MAIN MENU", lambda: self.set_state(GameState.MENU), font),
        ]

        # Defeat Menu Buttons
        self.defeat_buttons = [
            UIButton(pygame.Rect(cx - btn_w // 2, 420, btn_w, btn_h), "RETRY", lambda: self.start_game(self.mode), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 470, btn_w, btn_h), "MAIN MENU", lambda: self.set_state(GameState.MENU), font),
        ]
        # Keep game_over_buttons alias for tests
        self.game_over_buttons = self.defeat_buttons

        # Victory Menu Buttons
        self.victory_buttons = [
            UIButton(pygame.Rect(cx - btn_w // 2, 430, btn_w, btn_h), "PLAY AGAIN", lambda: self.start_game(self.mode), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 480, btn_w, btn_h), "MAIN MENU", lambda: self.set_state(GameState.MENU), font),
        ]

        # Controls Back Button
        self.back_button = UIButton(pygame.Rect(cx - 100, 530, 200, btn_h), "BACK", lambda: self.set_state(GameState.MENU), font)

    def _cycle_difficulty_action(self) -> None:
        self.audio.play("ui_click")
        new_diff = self.settings.cycle_difficulty()
        self.main_menu_buttons[3].text = f"DIFFICULTY: {new_diff.value.upper()}"

    def _toggle_overlay(self) -> None:
        self.audio.play("ui_click")
        self.settings.show_debug_overlay = not self.settings.show_debug_overlay

    def set_state(self, new_state: GameState) -> None:
        self.audio.play("ui_click")
        self.state = new_state

    def start_game(self, mode: Optional[GameMode] = None) -> None:
        """Initialize and launch an active countdown survival session."""
        self.audio.play("ui_click")
        if mode is not None:
            self.mode = mode
        self.state = GameState.PLAYING

        self.countdown_total = getattr(self.settings, "countdown_duration", 180.0)
        self.countdown_remaining = self.countdown_total
        self.elapsed_time = 0.0
        self.spawn_timer = 0.0
        self.last_difficulty_alert = 0.0
        self.visuals.clear()

        # Ship spawn in central sector
        spawn_pos = (self.width * 0.5, self.height * 0.5)

        # Debris scenario setup
        self.debris = DebrisManager(
            map_size=(self.width, self.height),
            score_table=self.settings.get_score_table(),
            split_enabled=self.settings.asteroid_split_enabled,
            wall_bounce=self.settings.asteroid_wall_bounce,
            base_health=self.settings.asteroid_health,
            settings=self.settings,
        )
        preset = self.settings.get_difficulty_preset()
        self.debris.populate_scenario(count=self.settings.initial_asteroids, avoid_pos=spawn_pos)
        var = self.settings.asteroid_speed_variation
        for a in self.debris.asteroids:
            jitter = (1.0 + random.uniform(-var, var)) if var > 0 else 1.0
            a.vx *= preset.asteroid_speed_mult * jitter
            a.vy *= preset.asteroid_speed_mult * jitter

        # Ship configuration based on mode
        if self.mode == GameMode.AUTONOMOUS:
            self.player_ship = None
            self.ai_ship = Spaceship(
                ship_id=0,
                position=spawn_pos,
                heading=90.0,
                map_size=(self.width, self.height),
                is_ai=True,
                settings=self.settings,
            )
            self.ai_controller = KesslerAIController(
                map_size=(self.width, self.height),
                safe_zone=self.safe_zone,
                ai_aggression=self.settings.ai_aggression,
                adaptive_risk=self.settings.adaptive_risk,
            )

        elif self.mode == GameMode.PLAYER:
            self.player_ship = PlayerShip(
                ship_id=1,
                position=spawn_pos,
                heading=90.0,
                map_size=(self.width, self.height),
                settings=self.settings,
            )
            self.ai_ship = None
            self.ai_controller = None

        elif self.mode == GameMode.COOP:
            self.player_ship = PlayerShip(
                ship_id=1,
                position=(spawn_pos[0] - 50, spawn_pos[1]),
                heading=90.0,
                map_size=(self.width, self.height),
                settings=self.settings,
            )
            self.ai_ship = Spaceship(
                ship_id=0,
                position=(spawn_pos[0] + 50, spawn_pos[1]),
                heading=90.0,
                map_size=(self.width, self.height),
                is_ai=True,
                settings=self.settings,
            )
            self.ai_controller = KesslerAIController(
                map_size=(self.width, self.height),
                safe_zone=self.safe_zone,
                ai_aggression=self.settings.ai_aggression,
                adaptive_risk=self.settings.adaptive_risk,
            )

    def quit(self) -> None:
        self.is_running = False

    def _update_viewport(self, new_w: int, new_h: int) -> None:
        """Recalculate letterbox scaling and centering offsets for arbitrary window dimensions."""
        self.window_width = max(320, int(new_w))
        self.window_height = max(240, int(new_h))

        scale_x = self.window_width / self.width
        scale_y = self.window_height / self.height
        self.viewport_scale = min(scale_x, scale_y)

        self.scaled_w = int(self.width * self.viewport_scale)
        self.scaled_h = int(self.height * self.viewport_scale)

        self.viewport_offset_x = (self.window_width - self.scaled_w) // 2
        self.viewport_offset_y = (self.window_height - self.scaled_h) // 2

    def window_to_logical(self, win_x: float, win_y: float) -> Tuple[int, int]:
        """Convert physical window pixel coordinates into logical game space (1000x800)."""
        if self.viewport_scale <= 0.0:
            return int(win_x), int(win_y)
        lx = (win_x - self.viewport_offset_x) / self.viewport_scale
        ly = (win_y - self.viewport_offset_y) / self.viewport_scale
        return int(max(0, min(self.width - 1, lx))), int(max(0, min(self.height - 1, ly)))

    def _on_touch_pause(self) -> None:
        """Toggle pause state from on-screen virtual touch button."""
        if self.state == GameState.PLAYING:
            self.set_state(GameState.PAUSED)
        elif self.state == GameState.PAUSED:
            self.set_state(GameState.PLAYING)

    # =========================================================================
    # MAIN LOOP (SYNCHRONOUS & ASYNCHRONOUS WEBASSEMBLY SUPPORT)
    # =========================================================================
    def step(self, dt: Optional[float] = None) -> bool:
        """Execute a single frame of events, physics/AI update, and rendering.

        Returns:
            bool: True if the application is still running, False if terminated.
        """
        if not self.is_running:
            return False

        if dt is None:
            dt = min(self.clock.tick(self.settings.target_fps) / 1000.0, 0.05)

        self._handle_events(dt)
        self._update(dt)
        self._render()
        return self.is_running

    def run(self) -> None:
        """Synchronous desktop game loop running at stable 60 FPS."""
        while self.is_running:
            self.step()

        pygame.quit()
        sys.exit(0)

    async def run_async(self) -> None:
        """Asynchronous game loop for Pygbag / WebAssembly browser deployment.

        Yields control to the browser event loop every frame via asyncio.sleep(0)
        to prevent blocking the single-threaded web environment.
        """
        while self.is_running:
            self.step()
            await asyncio.sleep(0)

        pygame.quit()

    # =========================================================================
    # EVENT HANDLING
    # =========================================================================
    def _handle_events(self, dt: float) -> None:
        raw_mouse = pygame.mouse.get_pos()
        mouse_pos = self.window_to_logical(*raw_mouse)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()

            elif event.type == pygame.VIDEORESIZE:
                self._update_viewport(event.w, event.h)
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)

            elif event.type == pygame.KEYDOWN:
                self.audio.unlock_audio()
                if event.key in (pygame.K_ESCAPE, pygame.K_p):
                    if self.state == GameState.PLAYING:
                        self.set_state(GameState.PAUSED)
                    elif self.state == GameState.PAUSED:
                        self.set_state(GameState.PLAYING)
                    elif self.state in (GameState.CONTROLS, GameState.SETTINGS):
                        self.set_state(GameState.MENU)

                elif event.key in (pygame.K_F1, pygame.K_TAB):
                    self._toggle_overlay()

            # Handle Browser Touch Events (Finger Down, Finger Motion, Finger Up)
            elif event.type in (pygame.FINGERDOWN, pygame.FINGERMOTION, pygame.FINGERUP):
                self.audio.unlock_audio()
                win_x = event.x * self.window_width
                win_y = event.y * self.window_height
                touch_pos = self.window_to_logical(win_x, win_y)

                if self.state == GameState.PLAYING:
                    self.touch.handle_event(event, touch_pos)
                elif event.type == pygame.FINGERDOWN:
                    # Allow touch taps in modal screens and settings
                    if self.state == GameState.SETTINGS:
                        sim_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": touch_pos})
                        self.settings_menu.handle_event(sim_event, touch_pos)
                    elif self.state == GameState.MENU:
                        for btn in self.main_menu_buttons:
                            btn.handle_click(touch_pos)
                    elif self.state == GameState.PAUSED:
                        for btn in self.pause_buttons:
                            btn.handle_click(touch_pos)
                    elif self.state in (GameState.DEFEAT, GameState.GAME_OVER):
                        for btn in self.defeat_buttons:
                            btn.handle_click(touch_pos)
                    elif self.state == GameState.VICTORY:
                        for btn in self.victory_buttons:
                            btn.handle_click(touch_pos)
                    elif self.state == GameState.CONTROLS:
                        self.back_button.handle_click(touch_pos)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.audio.unlock_audio()
                # In gameplay, allow clicking virtual on-screen controls
                if self.state == GameState.PLAYING:
                    self.touch.handle_event(event, mouse_pos)

                if self.state == GameState.SETTINGS:
                    self.settings_menu.handle_event(event, mouse_pos)
                elif self.state == GameState.MENU:
                    for btn in self.main_menu_buttons:
                        btn.handle_click(mouse_pos)
                elif self.state == GameState.PAUSED:
                    for btn in self.pause_buttons:
                        btn.handle_click(mouse_pos)
                elif self.state in (GameState.DEFEAT, GameState.GAME_OVER):
                    for btn in self.defeat_buttons:
                        btn.handle_click(mouse_pos)
                elif self.state == GameState.VICTORY:
                    for btn in self.victory_buttons:
                        btn.handle_click(mouse_pos)
                elif self.state == GameState.CONTROLS:
                    self.back_button.handle_click(mouse_pos)

            elif event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
                if self.state == GameState.PLAYING:
                    self.touch.handle_event(event, mouse_pos)
                if self.state == GameState.SETTINGS:
                    self.settings_menu.handle_event(event, mouse_pos)

        # Update button hover states using logical coordinate space
        if self.state == GameState.MENU:
            for btn in self.main_menu_buttons:
                btn.check_hover(mouse_pos)
        elif self.state == GameState.PAUSED:
            for btn in self.pause_buttons:
                btn.check_hover(mouse_pos)
        elif self.state in (GameState.DEFEAT, GameState.GAME_OVER):
            for btn in self.defeat_buttons:
                btn.check_hover(mouse_pos)
        elif self.state == GameState.VICTORY:
            for btn in self.victory_buttons:
                btn.check_hover(mouse_pos)
        elif self.state == GameState.CONTROLS:
            self.back_button.check_hover(mouse_pos)
        elif self.state == GameState.SETTINGS:
            self.settings_menu.apply_btn.check_hover(mouse_pos)
            self.settings_menu.reset_btn.check_hover(mouse_pos)
            self.settings_menu.back_btn.check_hover(mouse_pos)
            for btn in self.settings_menu.tab_buttons:
                btn.check_hover(mouse_pos)

    # =========================================================================
    # SIMULATION UPDATE
    # =========================================================================
    def _update(self, dt: float) -> None:
        self.starfield.update(dt)
        self.visuals.update(dt)
        self.shake.update(dt)
        self.settings_menu.update(dt)
        self.ui.update(dt)

        if self.state != GameState.PLAYING:
            return

        self.elapsed_time += dt
        self.countdown_remaining = max(0.0, self.countdown_remaining - dt)

        # 1. Check VICTORY condition: Timer reaches 0
        if self.countdown_remaining <= 0.0:
            self.countdown_remaining = 0.0
            self.audio.play("victory")
            self.set_state(GameState.VICTORY)
            return

        primary_ship = self.ai_ship or self.player_ship

        # 2. Check DEFEAT condition: Health reaches 0
        if primary_ship and (primary_ship.health <= 0.0 or not primary_ship.is_alive):
            primary_ship.health = 0.0
            primary_ship.is_alive = False
            self.audio.play("game_over")
            self.set_state(GameState.DEFEAT)
            return

        # 3. Dynamic Hazard Escalation over Countdown
        preset = self.settings.get_difficulty_preset()
        if self.settings.difficulty_progression:
            prog = 1.0 - (self.countdown_remaining / max(1.0, self.countdown_total))  # 0.0 -> 1.0
        else:
            prog = 0.0

        max_asteroids = int(self.settings.initial_asteroids + prog * (self.settings.max_asteroids - self.settings.initial_asteroids))
        spawn_interval = max(2.5, self.settings.spawn_interval_s * (1.0 - prog * 0.55))

        # Alert tone every 45s of survival
        if self.elapsed_time - self.last_difficulty_alert >= 45.0:
            self.last_difficulty_alert = self.elapsed_time
            self.audio.play("alarm")

        self.spawn_timer += dt
        if self.spawn_timer >= spawn_interval and len(self.debris.asteroids) < max_asteroids:
            self.spawn_timer = 0.0
            avoid = (primary_ship.x, primary_ship.y) if primary_ship else None
            new_ast = self.debris.spawn_asteroid(avoid_pos=avoid, min_distance=240.0)
            spd_boost = preset.asteroid_speed_mult * (1.0 + 0.35 * prog)
            new_ast.vx *= spd_boost
            new_ast.vy *= spd_boost

        self.debris.update(dt)

        # 4. Autonomous AI Ship Execution (No Player Input)
        if self.ai_ship and self.ai_ship.is_alive and self.ai_controller:
            ship_dict = self.ai_ship.to_dict()
            game_dict = {
                "map_size": (self.width, self.height),
                "frame_rate": 60.0,
                "time": self.elapsed_time,
                "asteroids": self.debris.get_dicts(),
            }

            ai_thrust, ai_turn, ai_fire, ai_boost = self.ai_controller.get_actions(ship_dict, game_dict)
            is_thrusting, nozzle = self.ai_ship.update_physics(ai_thrust, ai_turn, dt, boost_active=ai_boost)

            if is_thrusting:
                self.visuals.emit_thruster(nozzle, self.ai_ship.heading, is_boost=self.ai_ship.is_boosting)
            if ai_fire:
                bullet = self.ai_ship.fire()
                if bullet:
                    self.audio.play("laser")

        # 5. Optional Player Ship Controls (when in Player or Co-op mode)
        if self.player_ship and self.player_ship.is_alive:
            keys = pygame.key.get_pressed()
            raw_mouse = pygame.mouse.get_pos()
            mouse_pos = self.window_to_logical(*raw_mouse)
            mouse_btns = pygame.mouse.get_pressed()

            p_thrust, p_turn, p_fire, p_boost = self.player_ship.process_input(keys, mouse_pos, mouse_btns, dt)

            # Blend virtual touch inputs if on touchscreen
            if self.touch.is_active:
                t_thrust, t_turn, t_fire, t_boost = self.touch.get_inputs()
                if abs(t_thrust) > 0.01:
                    p_thrust = self.player_ship.params.thrust_max if t_thrust > 0.0 else self.player_ship.params.thrust_min
                if abs(t_turn) > 0.01:
                    p_turn = t_turn * self.player_ship.params.turn_max
                if t_fire:
                    p_fire = True
                if t_boost:
                    p_boost = True

            is_thrusting, nozzle = self.player_ship.update_physics(p_thrust, p_turn, dt, boost_active=p_boost)

            if is_thrusting:
                self.visuals.emit_thruster(nozzle, self.player_ship.heading, is_boost=self.player_ship.is_boosting)
            if p_boost and self.player_ship.is_boosting:
                self.audio.play("boost")
            if p_fire:
                bullet = self.player_ship.fire()
                if bullet:
                    self.audio.play("laser")

        # 6. Physics & Collision Handling
        all_ships = []
        if self.ai_ship:
            all_ships.append(self.ai_ship)
        if self.player_ship:
            all_ships.append(self.player_ship)

        # Bullet - Asteroid collisions
        bullet_hits = self.physics.check_bullet_asteroid_collisions(all_ships, self.debris)
        for hit in bullet_hits:
            if hit.event_type == "bullet_red_asteroid":
                self.visuals.emit_explosion(hit.pos, count=28, speed_mult=1.3)
                self.audio.play("alarm")
                self.shake.add_trauma(0.3)
                self.ui.add_notification(f"RED ASTEROID HIT: -{hit.penalty} POINTS", color=(255, 60, 60))
            else:
                self.visuals.emit_explosion(hit.pos, count=int(18 * hit.intensity), speed_mult=1.0)
                self.audio.play("explosion_small")
                self.shake.add_trauma(0.12 * hit.intensity)

        # Ship - Asteroid collisions
        ship_hits = self.physics.check_ship_asteroid_collisions(all_ships, self.debris)
        for hit in ship_hits:
            if hit.event_type == "ship_red_asteroid":
                self.visuals.emit_explosion(hit.pos, count=38, speed_mult=1.6)
                self.audio.play("alarm")
                self.shake.add_trauma(0.5)
                self.ui.add_notification(f"RED ASTEROID COLLISION: -{hit.penalty} POINTS", color=(255, 60, 60))
            else:
                self.visuals.emit_explosion(hit.pos, count=int(32 * hit.intensity), speed_mult=1.4)
                self.visuals.emit_shield_ripple(hit.pos, radius=30.0)
                self.audio.play("shield_hit")
                self.audio.play("explosion_large")
                self.shake.add_trauma(0.45 * hit.intensity)

        # Ship - Ship collisions (in co-op)
        self.physics.check_ship_ship_collisions(all_ships)

        # Update high score
        if primary_ship and primary_ship.score > self.high_score:
            self.high_score = primary_ship.score

    # =========================================================================
    # RENDERING
    # =========================================================================
    def _render(self) -> None:
        cam_x = self.shake.offset_x if self.settings.screen_shake_enabled else 0.0
        cam_y = self.shake.offset_y if self.settings.screen_shake_enabled else 0.0

        render_surf = self.virtual_surface

        # 1. Background Starfield & Nebulae
        self.starfield.draw(render_surf, cam_x=cam_x, cam_y=cam_y)

        # 2. Asteroids & Space Debris
        self.debris.draw(render_surf)

        # 3. Ships & Laser Projectiles
        if self.ai_ship and self.ai_ship.is_alive:
            self.ai_ship.draw(render_surf)
        if self.player_ship and self.player_ship.is_alive:
            self.player_ship.draw(render_surf)

        # 4. Particle Systems & Explosions
        self.visuals.draw(render_surf)

        # 5. AI In-World Debug Overlay (A* route, MPC arcs, threats)
        if self.ai_ship and self.ai_controller and self.settings.show_debug_overlay:
            self.ui.draw_debug_overlay(render_surf, self.ai_ship, self.ai_controller.telemetry, self.settings)

        # 6. In-Game Survival HUD
        if self.state == GameState.PLAYING:
            primary_ship = self.ai_ship or self.player_ship
            if primary_ship:
                tel = self.ai_controller.telemetry if self.ai_controller else None
                self.ui.draw_hud(
                    surface=render_surf,
                    ship=primary_ship,
                    countdown_remaining=self.countdown_remaining,
                    elapsed_time=self.elapsed_time,
                    asteroids_count=len(self.debris.normal_asteroids),
                    ai_telemetry=tel,
                    high_score=self.high_score,
                    safe_zone=self.safe_zone,
                    red_count=len(self.debris.red_asteroids),
                )

        # 7. Virtual Touch Controls (shown during PLAYING state)
        if self.state == GameState.PLAYING:
            self.touch.draw(render_surf, show_flight_controls=(self.mode == GameMode.PLAYER))

        # 8. Modal Screens
        if self.state == GameState.MENU:
            self.ui.draw_main_menu(render_surf, self.main_menu_buttons, self.settings.difficulty)

        elif self.state == GameState.PAUSED:
            self.ui.draw_pause_menu(render_surf, self.pause_buttons)

        elif self.state in (GameState.DEFEAT, GameState.GAME_OVER):
            primary = self.ai_ship or self.player_ship
            sc = primary.score if primary else 0
            kills = primary.kills if primary else 0
            acc = primary.accuracy if primary else 0.0
            dmg = primary.damage_taken if primary else 0.0
            self.ui.draw_defeat(
                surface=render_surf,
                buttons=self.defeat_buttons,
                score=sc,
                survived_time=self.elapsed_time,
                kills=kills,
                accuracy=acc,
                damage_taken=dmg,
            )

        elif self.state == GameState.VICTORY:
            primary = self.ai_ship or self.player_ship
            sc = primary.score if primary else 0
            kills = primary.kills if primary else 0
            acc = primary.accuracy if primary else 0.0
            dmg = primary.damage_taken if primary else 0.0
            hlth = primary.health if primary else 100.0
            self.ui.draw_victory(
                surface=render_surf,
                buttons=self.victory_buttons,
                score=sc,
                kills=kills,
                accuracy=acc,
                damage_taken=dmg,
                final_health=hlth,
                survived_time=self.elapsed_time,
            )

        elif self.state == GameState.CONTROLS:
            self.ui.draw_controls(render_surf, self.back_button)

        elif self.state == GameState.SETTINGS:
            self.settings_menu.draw(render_surf)

        # 9. Letterboxed Responsive Viewport Blit
        if self.window_width == self.width and self.window_height == self.height:
            # Direct blit when window matches internal logical canvas (1000x800)
            self.screen.blit(self.virtual_surface, (0, 0))
        else:
            # Fill borders with deep-space black and smoothscale canvas centered
            self.screen.fill((4, 8, 14))
            if self.scaled_w > 0 and self.scaled_h > 0:
                scaled_view = pygame.transform.smoothscale(self.virtual_surface, (self.scaled_w, self.scaled_h))
                self.screen.blit(scaled_view, (self.viewport_offset_x, self.viewport_offset_y))

        pygame.display.flip()
