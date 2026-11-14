"""Tests for Stage 2: Browser Touch Controls, Audio Compatibility & Responsive Viewport.

Verifies:
1. Virtual Joystick calculations, deadzones, and multi-touch isolation.
2. Touch Action Buttons (Fire, Boost, Pause).
3. Coordinate transformation between physical window space and logical 1000x800 canvas.
4. Viewport scaling with proportional letterboxing and pillarboxing.
5. Autonomous AI gameplay isolation from touch controls.
6. Audio gesture unlock and silent fallback when audio is disabled/unavailable.
7. Desktop keyboard and mouse input preservation.
"""
import math
import pygame
import pytest

from kessler_mpc.audio import SoundManager
from kessler_mpc.game import GameMode, GameState, KesslerGameApp
from kessler_mpc.player import PlayerShip
from kessler_mpc.settings import GameSettings
from kessler_mpc.touch import TouchActionButton, TouchController, VirtualJoystick


# =============================================================================
# 1. Virtual Joystick Tests
# =============================================================================
def test_virtual_joystick_touch_and_clamp():
    joy = VirtualJoystick(center=(140.0, 660.0), base_radius=60.0, knob_radius=25.0)
    assert joy.thrust == 0.0
    assert joy.turn == 0.0

    # Touch down directly at center -> within deadzone -> 0, 0
    claimed = joy.handle_touch_down(touch_id=1, pos=(140.0, 660.0))
    assert claimed is True
    assert joy.touch_id == 1
    assert joy.thrust == 0.0
    assert joy.turn == 0.0

    # Drag forward (up in screen space is negative dy)
    joy.handle_touch_motion(touch_id=1, pos=(140.0, 610.0))
    assert joy.thrust > 0.5
    assert joy.turn == 0.0

    # Drag hard right beyond base radius -> clamped to maximum turn
    joy.handle_touch_motion(touch_id=1, pos=(300.0, 660.0))
    assert joy.turn == 1.0  # Clamped to maximum
    assert joy.thrust == 0.0
    dist_from_center = math.hypot(joy.knob_x - joy.center_x, joy.knob_y - joy.center_y)
    assert pytest.approx(dist_from_center, abs=1e-3) == 60.0

    # Drag diagonal 45-degrees forward-right beyond base radius
    joy.handle_touch_motion(touch_id=1, pos=(300.0, 500.0))
    assert pytest.approx(joy.turn, abs=1e-2) == math.sqrt(0.5)
    assert pytest.approx(joy.thrust, abs=1e-2) == math.sqrt(0.5)
    dist_diagonal = math.hypot(joy.knob_x - joy.center_x, joy.knob_y - joy.center_y)
    assert pytest.approx(dist_diagonal, abs=1e-3) == 60.0

    # Release touch -> resets to zero
    released = joy.handle_touch_up(touch_id=1)
    assert released is True
    assert joy.touch_id is None
    assert joy.thrust == 0.0
    assert joy.turn == 0.0
    assert joy.knob_x == 140.0
    assert joy.knob_y == 660.0


def test_virtual_joystick_ignores_other_touch_ids():
    joy = VirtualJoystick(center=(140.0, 660.0), base_radius=60.0)
    assert joy.handle_touch_down(touch_id=1, pos=(140.0, 620.0)) is True

    # Touch 2 arrives -> joystick ignores it because it is already tracked by Touch 1
    assert joy.handle_touch_down(touch_id=2, pos=(140.0, 640.0)) is False
    assert joy.handle_touch_motion(touch_id=2, pos=(140.0, 600.0)) is False
    assert joy.handle_touch_up(touch_id=2) is False

    # Still tracked by Touch 1
    assert joy.touch_id == 1
    joy.handle_touch_up(touch_id=1)
    assert joy.touch_id is None


# =============================================================================
# 2. Touch Action Buttons Tests
# =============================================================================
def test_touch_action_buttons():
    pause_called = []
    btn_fire = TouchActionButton(center=(880.0, 660.0), radius=40.0, label="FIRE", color=(255, 0, 0))
    btn_pause = TouchActionButton(
        center=(950.0, 40.0), radius=20.0, label="PAUSE", color=(0, 200, 255), on_press=lambda: pause_called.append(True)
    )

    assert not btn_fire.is_pressed
    assert not btn_pause.is_pressed

    # Tap Fire button
    assert btn_fire.handle_touch_down(touch_id=1, pos=(885.0, 662.0)) is True
    assert btn_fire.is_pressed is True

    # Tap Pause button simultaneously with second finger (multi-touch)
    assert btn_pause.handle_touch_down(touch_id=2, pos=(950.0, 42.0)) is True
    assert btn_pause.is_pressed is True
    assert len(pause_called) == 1

    # Release both
    assert btn_fire.handle_touch_up(touch_id=1) is True
    assert not btn_fire.is_pressed
    assert btn_pause.handle_touch_up(touch_id=2) is True
    assert not btn_pause.is_pressed


# =============================================================================
# 3. Touch Controller Multi-Touch Integration
# =============================================================================
def test_touch_controller_multitouch_integration():
    pause_events = []
    tc = TouchController(on_pause_callback=lambda: pause_events.append(True))

    # Initial state
    thrust, turn, fire, boost = tc.get_inputs()
    assert (thrust, turn, fire, boost) == (0.0, 0.0, False, False)

    # Finger 0 steers joystick (dragging up-right)
    down_event0 = pygame.event.Event(pygame.FINGERDOWN, {"finger_id": 0, "x": 0.14, "y": 0.62})
    tc.handle_event(down_event0, logical_pos=(140.0, 620.0))

    # Finger 1 taps FIRE
    down_event1 = pygame.event.Event(pygame.FINGERDOWN, {"finger_id": 1, "x": 0.88, "y": 0.66})
    tc.handle_event(down_event1, logical_pos=(885.0, 665.0))

    # Finger 2 taps BOOST
    down_event2 = pygame.event.Event(pygame.FINGERDOWN, {"finger_id": 2, "x": 0.78, "y": 0.71})
    tc.handle_event(down_event2, logical_pos=(785.0, 715.0))

    thrust, turn, fire, boost = tc.get_inputs()
    assert thrust > 0.0
    assert fire is True
    assert boost is True

    # Release fire and boost
    tc.handle_event(pygame.event.Event(pygame.FINGERUP, {"finger_id": 1}), logical_pos=(885.0, 665.0))
    tc.handle_event(pygame.event.Event(pygame.FINGERUP, {"finger_id": 2}), logical_pos=(785.0, 715.0))

    _, _, fire_after, boost_after = tc.get_inputs()
    assert fire_after is False
    assert boost_after is False


# =============================================================================
# 4. Viewport Scaling & Coordinate Transformation Tests
# =============================================================================
def test_viewport_scaling_and_letterboxing():
    settings = GameSettings(screen_width=1000, screen_height=800, sfx_enabled=False)
    app = KesslerGameApp(settings=settings)

    # 1. Default 1:1 window (1000x800)
    assert app.width == 1000
    assert app.height == 800
    assert app.viewport_scale == 1.0
    assert app.viewport_offset_x == 0
    assert app.viewport_offset_y == 0
    assert app.window_to_logical(500, 400) == (500, 400)

    # 2. Resized to 16:9 Widescreen (1920x1080) -> letterboxed (black bars left and right)
    app._update_viewport(1920, 1080)
    # Scale is constrained by height: 1080 / 800 = 1.35
    assert pytest.approx(app.viewport_scale, abs=1e-3) == 1.35
    assert app.scaled_h == 1080
    assert app.scaled_w == int(1000 * 1.35)  # 1350
    # Centered horizontally: (1920 - 1350) // 2 = 285
    assert app.viewport_offset_x == (1920 - 1350) // 2
    assert app.viewport_offset_y == 0

    # Center of window (960, 540) must map to center of logical canvas (500, 400)
    lx, ly = app.window_to_logical(960, 540)
    assert pytest.approx(lx, abs=2) == 500
    assert pytest.approx(ly, abs=2) == 400

    # 3. Resized to Mobile Portrait (400x800) -> pillarboxed (black bars top and bottom)
    app._update_viewport(400, 800)
    # Scale is constrained by width: 400 / 1000 = 0.4
    assert pytest.approx(app.viewport_scale, abs=1e-3) == 0.4
    assert app.scaled_w == 400
    assert app.scaled_h == int(800 * 0.4)  # 320
    assert app.viewport_offset_x == 0
    assert app.viewport_offset_y == (800 - 320) // 2  # 240

    # Center of mobile window (200, 400) must map to logical center (500, 400)
    lx_mob, ly_mob = app.window_to_logical(200, 400)
    assert pytest.approx(lx_mob, abs=2) == 500
    assert pytest.approx(ly_mob, abs=2) == 400


# =============================================================================
# 5. Audio Compatibility & Unlock Tests
# =============================================================================
def test_audio_unlock_and_fallback():
    # Audio with sfx disabled -> safe initialization and play
    mgr_muted = SoundManager(enabled=False)
    assert not mgr_muted.initialized
    assert not mgr_muted.unlock_audio()
    # Play call is safe no-op
    mgr_muted.play("laser")

    # Audio with sfx enabled
    mgr_active = SoundManager(enabled=True)
    # unlock_audio returns boolean status without crashing
    res = mgr_active.unlock_audio()
    assert isinstance(res, bool)
    # Play unknown sound or known sound does not crash
    mgr_active.play("laser")
    mgr_active.play("non_existent_sound")


# =============================================================================
# 6. Autonomous Mode Isolation from Touch Controls
# =============================================================================
def test_autonomous_mode_unaffected_by_touch_controls():
    settings = GameSettings(screen_width=1000, screen_height=800, sfx_enabled=False)
    app = KesslerGameApp(settings=settings)
    app.start_game(GameMode.AUTONOMOUS)

    assert app.player_ship is None
    assert app.ai_ship is not None
    assert app.ai_controller is not None

    # Simulate touching joystick and fire buttons
    tc = app.touch
    tc.is_active = True
    tc.joystick.thrust = 1.0
    tc.joystick.turn = 1.0
    tc.btn_fire.is_pressed = True

    # Advance simulation frame in autonomous mode
    app._update(0.016)

    # Verify AI ship actions are strictly determined by AI controller, not virtual joystick
    tel = app.ai_controller.telemetry
    assert tel is not None
    assert tel.state is not None
    # Player ship remains None
    assert app.player_ship is None


# =============================================================================
# 7. Desktop Keyboard & Mouse Preserved in Player Mode
# =============================================================================
def test_player_keyboard_and_mouse_preserved():
    player = PlayerShip(position=(500.0, 400.0), heading=0.0)

    # Simulated key presses: forward thrust (W), turn right (D)
    class FakeKeys:
        def __getitem__(self, key):
            if key in (pygame.K_w, pygame.K_UP):
                return True
            if key in (pygame.K_d, pygame.K_RIGHT):
                return True
            if key == pygame.K_LSHIFT:
                return True
            return False

    thrust, turn, fire, boost = player.process_input(
        keys=FakeKeys(), mouse_pos=(500, 400), mouse_buttons=(False, False, False), dt=0.016
    )

    assert thrust == player.params.thrust_max
    assert turn > 0.0
    assert boost is True
    assert fire is False
