"""Complete game loop, state management, and mode coordinator for Kessler AI.
"""
from __future__ import annotations

import math
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
from .settings import DifficultyLevel, GameSettings
from .spaceship import Spaceship
from .ui import UIButton, UIManager
from .visuals import ParticleSystem, ScreenShake, Starfield


class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    VICTORY = "victory"
    CONTROLS = "controls"


class GameMode(Enum):
    PLAYER = "Player Survival"
    AI_DEMO = "Autonomous AI Demonstration"
    SURVIVAL = "Endless Survival Challenge"
    COOP = "Player + AI Co-op / Race"


class KesslerGameApp:
    """Main application orchestrating the entire playable Kessler AI survival game."""

    def __init__(self, settings: Optional[GameSettings] = None):
        pygame.init()
        self.settings = settings or GameSettings()

        # Window setup
        self.width = self.settings.screen_width
        self.height = self.settings.screen_height
        self.flags = pygame.FULLSCREEN if self.settings.fullscreen else 0
        self.screen = pygame.display.set_mode((self.width, self.height), self.flags)
        pygame.display.set_caption("Kessler AI: Orbital Survival Game")

        self.clock = pygame.time.Clock()
        self.is_running = True
        self.state = GameState.MENU
        self.mode = GameMode.PLAYER

        # Systems
        self.audio = SoundManager(enabled=self.settings.sfx_enabled, master_volume=self.settings.master_volume)
        self.visuals = ParticleSystem(max_particles=1500)
        self.starfield = Starfield(width=self.width, height=self.height)
        self.shake = ScreenShake()
        self.ui = UIManager(screen_width=self.width, screen_height=self.height)
        self.physics = PhysicsEngine(map_size=(self.width, self.height))

        # Gameplay Entities
        self.safe_zone = SafeZone(center=(self.width - 140.0, 140.0), radius=75.0, required_dock_time=3.0)
        self.debris = DebrisManager(map_size=(self.width, self.height))
        self.player_ship: Optional[PlayerShip] = None
        self.ai_ship: Optional[Spaceship] = None
        self.ai_controller: Optional[KesslerAIController] = None

        # Game Session Variables
        self.elapsed_time: float = 0.0
        self.spawn_timer: float = 0.0
        self.survival_wave_timer: float = 0.0
        self.survival_wave: int = 1
        self.high_score: int = 0

        # Build UI Menus
        self._init_ui_buttons()

    def _init_ui_buttons(self) -> None:
        """Create buttons for menus."""
        cx = self.width // 2
        btn_w, btn_h = 320, 44
        font = self.ui.font_body

        # Main Menu Buttons
        self.main_menu_buttons = [
            UIButton(pygame.Rect(cx - btn_w // 2, 290, btn_w, btn_h), "PLAY SURVIVAL (PLAYER)", lambda: self.start_game(GameMode.PLAYER), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 345, btn_w, btn_h), "AI DEMONSTRATION MODE", lambda: self.start_game(GameMode.AI_DEMO), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 400, btn_w, btn_h), "ENDLESS SURVIVAL CHALLENGE", lambda: self.start_game(GameMode.SURVIVAL), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 455, btn_w, btn_h), "PLAYER + AI CO-OP / RACE", lambda: self.start_game(GameMode.COOP), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 510, btn_w, btn_h), f"DIFFICULTY: {self.settings.difficulty.value.upper()}", self._cycle_difficulty_action, font),
            UIButton(pygame.Rect(cx - btn_w // 2, 565, btn_w, btn_h), "CONTROLS & FLIGHT MANUAL", lambda: self.set_state(GameState.CONTROLS), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 620, btn_w, btn_h), "QUIT GAME", self.quit, font),
        ]

        # Pause Menu Buttons
        self.pause_buttons = [
            UIButton(pygame.Rect(cx - btn_w // 2, 300, btn_w, btn_h), "RESUME MISSION", lambda: self.set_state(GameState.PLAYING), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 360, btn_w, btn_h), "RESTART MATCH", lambda: self.start_game(self.mode), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 420, btn_w, btn_h), "TOGGLE AI OVERLAY [F1]", self._toggle_overlay, font),
            UIButton(pygame.Rect(cx - btn_w // 2, 480, btn_w, btn_h), "RETURN TO MAIN MENU", lambda: self.set_state(GameState.MENU), font),
        ]

        # Game Over Buttons
        self.game_over_buttons = [
            UIButton(pygame.Rect(cx - btn_w // 2, 450, btn_w, btn_h), "RESTART MATCH", lambda: self.start_game(self.mode), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 510, btn_w, btn_h), "RETURN TO MAIN MENU", lambda: self.set_state(GameState.MENU), font),
        ]

        # Victory Buttons
        self.victory_buttons = [
            UIButton(pygame.Rect(cx - btn_w // 2, 450, btn_w, btn_h), "PLAY AGAIN", lambda: self.start_game(self.mode), font),
            UIButton(pygame.Rect(cx - btn_w // 2, 510, btn_w, btn_h), "RETURN TO MAIN MENU", lambda: self.set_state(GameState.MENU), font),
        ]

        # Controls Back Button
        self.back_button = UIButton(pygame.Rect(cx - 140, 620, 280, btn_h), "BACK TO MENU", lambda: self.set_state(GameState.MENU), font)

    def _cycle_difficulty_action(self) -> None:
        self.audio.play("ui_click")
        new_diff = self.settings.cycle_difficulty()
        self.main_menu_buttons[4].text = f"DIFFICULTY: {new_diff.value.upper()}"

    def _toggle_overlay(self) -> None:
        self.audio.play("ui_click")
        self.settings.show_debug_overlay = not self.settings.show_debug_overlay

    def set_state(self, new_state: GameState) -> None:
        self.audio.play("ui_click")
        self.state = new_state

    def start_game(self, mode: GameMode) -> None:
        """Initialize and launch an active gameplay session."""
        self.audio.play("ui_click")
        self.mode = mode
        self.state = GameState.PLAYING
        self.elapsed_time = 0.0
        self.spawn_timer = 0.0
        self.survival_wave_timer = 0.0
        self.survival_wave = 1

        self.visuals.clear()

        # Safe Zone setup (placed in corner away from player spawn)
        sz_pos = (self.width - 130.0, 130.0)
        self.safe_zone = SafeZone(center=sz_pos, radius=75.0, required_dock_time=3.0)

        # Debris setup using difficulty presets
        preset = self.settings.get_difficulty_preset()
        self.debris.populate_scenario(count=preset.initial_asteroids)
        # Apply speed multiplier to initial asteroids
        for a in self.debris.asteroids:
            a.vx *= preset.asteroid_speed_mult
            a.vy *= preset.asteroid_speed_mult

        # Ship setup according to game mode
        spawn_center = (self.width * 0.35, self.height * 0.65)

        if mode in (GameMode.PLAYER, GameMode.SURVIVAL):
            self.player_ship = PlayerShip(ship_id=1, position=spawn_center, heading=90.0, map_size=(self.width, self.height))
            self.ai_ship = None
            self.ai_controller = None

        elif mode == GameMode.AI_DEMO:
            self.player_ship = None
            self.ai_ship = Spaceship(ship_id=0, position=spawn_center, heading=90.0, map_size=(self.width, self.height), is_ai=True)
            self.ai_controller = KesslerAIController(map_size=(self.width, self.height), safe_zone=self.safe_zone)

        elif mode == GameMode.COOP:
            self.player_ship = PlayerShip(ship_id=1, position=(spawn_center[0] - 40, spawn_center[1]), heading=90.0, map_size=(self.width, self.height))
            self.ai_ship = Spaceship(ship_id=0, position=(spawn_center[0] + 40, spawn_center[1]), heading=90.0, map_size=(self.width, self.height), is_ai=True)
            self.ai_controller = KesslerAIController(map_size=(self.width, self.height), safe_zone=self.safe_zone)

    def quit(self) -> None:
        self.is_running = False

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    def run(self) -> None:
        """Main game loop executing at stable 60 FPS."""
        while self.is_running:
            # Frame-rate independent delta time (clamped to prevent spiral of death)
            dt = min(self.clock.tick(self.settings.target_fps) / 1000.0, 0.05)

            self._handle_events(dt)
            self._update(dt)
            self._render()

        pygame.quit()
        sys.exit(0)

    # =========================================================================
    # EVENT HANDLING
    # =========================================================================
    def _handle_events(self, dt: float) -> None:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_p:
                    if self.state == GameState.PLAYING:
                        self.set_state(GameState.PAUSED)
                    elif self.state == GameState.PAUSED:
                        self.set_state(GameState.PLAYING)
                    elif self.state == GameState.CONTROLS:
                        self.set_state(GameState.MENU)

                elif event.key in (pygame.K_F1, pygame.K_TAB):
                    self._toggle_overlay()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Handle button clicks depending on state
                if self.state == GameState.MENU:
                    for btn in self.main_menu_buttons:
                        btn.handle_click(mouse_pos)
                elif self.state == GameState.PAUSED:
                    for btn in self.pause_buttons:
                        btn.handle_click(mouse_pos)
                elif self.state == GameState.GAME_OVER:
                    for btn in self.game_over_buttons:
                        btn.handle_click(mouse_pos)
                elif self.state == GameState.VICTORY:
                    for btn in self.victory_buttons:
                        btn.handle_click(mouse_pos)
                elif self.state == GameState.CONTROLS:
                    self.back_button.handle_click(mouse_pos)

        # Update button hover states
        if self.state == GameState.MENU:
            for btn in self.main_menu_buttons:
                btn.check_hover(mouse_pos)
        elif self.state == GameState.PAUSED:
            for btn in self.pause_buttons:
                btn.check_hover(mouse_pos)
        elif self.state == GameState.GAME_OVER:
            for btn in self.game_over_buttons:
                btn.check_hover(mouse_pos)
        elif self.state == GameState.VICTORY:
            for btn in self.victory_buttons:
                btn.check_hover(mouse_pos)
        elif self.state == GameState.CONTROLS:
            self.back_button.check_hover(mouse_pos)

    # =========================================================================
    # SIMULATION UPDATE
    # =========================================================================
    def _update(self, dt: float) -> None:
        self.starfield.update(dt)
        self.visuals.update(dt)
        self.shake.update(dt)

        if self.state != GameState.PLAYING:
            return

        self.elapsed_time += dt
        preset = self.settings.get_difficulty_preset()

        # 1. Update Safe Zone & Sanctuary Barrier
        ship_positions = {}
        if self.player_ship and self.player_ship.is_alive:
            ship_positions[self.player_ship.ship_id] = (self.player_ship.x, self.player_ship.y)
        if self.ai_ship and self.ai_ship.is_alive:
            ship_positions[self.ai_ship.ship_id] = (self.ai_ship.x, self.ai_ship.y)

        self.safe_zone.update(dt, ship_positions)

        # Apply safe zone shield deflection to asteroids
        if len(self.debris.asteroids) > 0:
            ast_pos = np.array([a.position for a in self.debris.asteroids])
            ast_vel = np.array([a.velocity for a in self.debris.asteroids])
            self.safe_zone.apply_debris_deflection(ast_pos, ast_vel)
            for i, a in enumerate(self.debris.asteroids):
                a.vx, a.vy = float(ast_vel[i, 0]), float(ast_vel[i, 1])

        # 2. Debris Spawning & Difficulty Escalation
        self.spawn_timer += dt
        spawn_interval = preset.spawn_interval_s

        if self.mode == GameMode.SURVIVAL:
            # Waves escalate every 15 seconds
            self.survival_wave_timer += dt
            if self.survival_wave_timer >= 15.0:
                self.survival_wave_timer = 0.0
                self.survival_wave += 1
                spawn_interval = max(3.0, spawn_interval - 1.2 * self.survival_wave)
                self.audio.play("alarm")

        if self.spawn_timer >= spawn_interval and len(self.debris.asteroids) < preset.max_asteroids:
            self.spawn_timer = 0.0
            new_ast = self.debris.spawn_asteroid()
            new_ast.vx *= preset.asteroid_speed_mult
            new_ast.vy *= preset.asteroid_speed_mult

        self.debris.update(dt)

        # 3. Player Ship Controls & Updates
        if self.player_ship and self.player_ship.is_alive:
            keys = pygame.key.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            mouse_btns = pygame.mouse.get_pressed()

            p_thrust, p_turn, p_fire, p_boost = self.player_ship.process_input(keys, mouse_pos, mouse_btns, dt)
            is_thrusting, nozzle = self.player_ship.update_physics(p_thrust, p_turn, dt, boost_active=p_boost)

            if is_thrusting:
                self.visuals.emit_thruster(nozzle, self.player_ship.heading, is_boost=self.player_ship.is_boosting)
            if p_boost and self.player_ship.is_boosting:
                self.audio.play("boost")
            if p_fire:
                bullet = self.player_ship.fire()
                if bullet:
                    self.audio.play("laser")

        # 4. Autonomous AI Ship Decisions & Updates
        if self.ai_ship and self.ai_ship.is_alive and self.ai_controller:
            # Build current state dicts for AI
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

        # 5. Physics & Collision Handling
        all_ships = []
        if self.player_ship:
            all_ships.append(self.player_ship)
        if self.ai_ship:
            all_ships.append(self.ai_ship)

        # Bullet - Asteroid collisions
        bullet_hits = self.physics.check_bullet_asteroid_collisions(all_ships, self.debris)
        for hit in bullet_hits:
            self.visuals.emit_explosion(hit.pos, count=int(18 * hit.intensity), speed_mult=1.0)
            self.audio.play("explosion_small")
            self.shake.add_trauma(0.12 * hit.intensity)

        # Ship - Asteroid collisions
        ship_hits = self.physics.check_ship_asteroid_collisions(all_ships, self.debris)
        for hit in ship_hits:
            self.visuals.emit_explosion(hit.pos, count=int(30 * hit.intensity), speed_mult=1.4)
            self.visuals.emit_shield_ripple(hit.pos, radius=30.0)
            self.audio.play("shield_hit")
            self.audio.play("explosion_large")
            self.shake.add_trauma(0.45 * hit.intensity)

        # Ship - Ship collisions (in co-op)
        self.physics.check_ship_ship_collisions(all_ships)

        # 6. Check Victory & Game Over Conditions
        primary_ship = self.player_ship or self.ai_ship

        if primary_ship:
            if primary_ship.score > self.high_score:
                self.high_score = primary_ship.score

            # Victory check (docked in safe zone)
            if self.mode != GameMode.SURVIVAL and self.safe_zone.is_ship_docked(primary_ship.ship_id):
                self.audio.play("victory")
                self.set_state(GameState.VICTORY)

            # Game over check (no lives remaining)
            elif not primary_ship.is_alive and primary_ship.lives <= 0:
                self.audio.play("game_over")
                self.set_state(GameState.GAME_OVER)

    # =========================================================================
    # RENDERING
    # =========================================================================
    def _render(self) -> None:
        cam_x = self.shake.offset_x if self.settings.screen_shake_enabled else 0.0
        cam_y = self.shake.offset_y if self.settings.screen_shake_enabled else 0.0

        # Create rendering canvas with camera offset for screen shake
        render_surf = self.screen

        # 1. Background Starfield & Nebulae
        self.starfield.draw(render_surf, cam_x=cam_x, cam_y=cam_y)

        # 2. Safe Zone Sanctuary Sanctuary (pulsating glow)
        self._draw_safe_zone(render_surf)

        # 3. Asteroids & Space Debris
        self.debris.draw(render_surf)

        # 4. Ships & Lasers
        if self.player_ship and self.player_ship.is_alive:
            self.player_ship.draw(render_surf)
        if self.ai_ship and self.ai_ship.is_alive:
            self.ai_ship.draw(render_surf)

        # 5. Particle Effects & Explosions
        self.visuals.draw(render_surf)

        # 6. AI In-World Debug Overlay (A* path, MPC arcs, threats)
        if self.ai_ship and self.ai_controller and self.settings.show_debug_overlay:
            self.ui.draw_debug_overlay(render_surf, self.ai_ship, self.ai_controller.telemetry, self.settings)

        # 7. In-Game HUD (if playing)
        if self.state == GameState.PLAYING:
            primary_ship = self.player_ship or self.ai_ship
            if primary_ship:
                is_ai = self.mode == GameMode.AI_DEMO
                tel = self.ai_controller.telemetry if self.ai_controller else None
                self.ui.draw_hud(
                    render_surf,
                    primary_ship,
                    self.safe_zone,
                    self.elapsed_time,
                    len(self.debris.asteroids),
                    is_ai_mode=is_ai,
                    ai_telemetry=tel,
                    high_score=self.high_score,
                )

        # 8. Menus & Modal Screens
        if self.state == GameState.MENU:
            self.ui.draw_main_menu(render_surf, self.main_menu_buttons, self.settings.difficulty)
        elif self.state == GameState.PAUSED:
            self.ui.draw_pause_menu(render_surf, self.pause_buttons)
        elif self.state == GameState.GAME_OVER:
            primary = self.player_ship or self.ai_ship
            acc = (primary.shots_hit / max(1, primary.shots_fired)) if primary else 0.0
            hits = primary.asteroids_destroyed if primary else 0
            sc = primary.score if primary else 0
            self.ui.draw_game_over(render_surf, self.game_over_buttons, sc, self.elapsed_time, hits, acc)
        elif self.state == GameState.VICTORY:
            primary = self.player_ship or self.ai_ship
            lives = primary.lives if primary else 0
            hits = primary.asteroids_destroyed if primary else 0
            sc = primary.score if primary else 0
            self.ui.draw_victory(render_surf, self.victory_buttons, sc, self.elapsed_time, lives, hits)
        elif self.state == GameState.CONTROLS:
            self.ui.draw_controls(render_surf, self.back_button)

        pygame.display.flip()

    def _draw_safe_zone(self, surface: pygame.Surface) -> None:
        """Draw pulsating safe zone sanctuary shield and docking ring."""
        cx, cy = int(self.safe_zone.x), int(self.safe_zone.y)
        r = int(self.safe_zone.radius)

        # Pulsating outer field aura
        pulse_r = int(r + 8.0 * math.sin(self.safe_zone.pulse_phase))
        aura_surf = pygame.Surface((pulse_r * 2 + 10, pulse_r * 2 + 10), pygame.SRCALPHA)
        pygame.draw.circle(aura_surf, (30, 220, 160, 30), (pulse_r + 5, pulse_r + 5), pulse_r)
        surface.blit(aura_surf, (cx - pulse_r - 5, cy - pulse_r - 5))

        # Main perimeter boundary
        pygame.draw.circle(surface, (40, 220, 140), (cx, cy), r, 2)

        # Rotating docking beacon spokes
        for i in range(4):
            ang = self.safe_zone.pulse_phase * 0.5 + (i * math.pi / 2.0)
            x1 = cx + math.cos(ang) * (r * 0.3)
            y1 = cy + math.sin(ang) * (r * 0.3)
            x2 = cx + math.cos(ang) * (r * 0.85)
            y2 = cy + math.sin(ang) * (r * 0.85)
            pygame.draw.line(surface, (60, 255, 180), (int(x1), int(y1)), (int(x2), int(y2)), 2)

        # Central station hub
        pygame.draw.circle(surface, (20, 45, 60), (cx, cy), 16)
        pygame.draw.circle(surface, (80, 240, 200), (cx, cy), 16, 2)
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 4)
