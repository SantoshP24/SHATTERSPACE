# Kessler AI: Autonomous Asteroid Survival Game

A fully autonomous, real-time 2D space-survival game powered by **Model Predictive Control (MPC)**, **A\* Global Route Planning**, **Threat Assessment**, **Adaptive Risk Modeling**, and an **Emergency Collision-Safety Override**.

Built with **Pygame** and integrated directly with the verified **Thales Kessler MPC** navigation and physics architecture.

---

## Table of Contents
1. [Overview & Autonomous Gameplay](#overview--autonomous-gameplay)
2. [Quick Start & Launch Instructions](#quick-start--launch-instructions)
3. [Game Modes](#game-modes)
4. [AI Controller & Behavior States](#ai-controller--behavior-states)
5. [Scoring & Difficulty Escalation](#scoring--difficulty-escalation)
6. [Software Architecture & Modules](#software-architecture--modules)
7. [Testing & Validation](#testing--validation)
8. [Standalone Windows Executable](#standalone-windows-executable)

---

## Overview & Autonomous Gameplay

- **100% Autonomous Spaceship**: The spaceship is controlled entirely by AI. No player input is required for movement, steering, aiming, firing, or dodging.
- **180-Second Countdown Survival Objective**: The AI must survive an intense, escalating 180-second countdown while actively hunting down and shattering hazardous orbital debris to maximize its score.
- **Victory & Defeat Conditions**:
  - **VICTORY**: The AI successfully navigates and survives the full 180.0-second orbital hazard countdown.
  - **DEFEAT**: Critical hull failure occurs (ship health reaches 0).
- **Four Explicit AI Behavior States**:
  - **`ATTACK`**: When conditions are safe, the AI tracks targets, computes the lead-intercept angle (`aim_point_bearing_deg`), advances toward the firing envelope, and fires plasma cannons automatically.
  - **`EVADE`**: When an imminent collision hazard is predicted ($TTC \le 0.75$s or $TTC \le 1.5$s with high collision risk), the emergency safety override takes complete control, executing high-G tangential escape maneuvers and boost surges.
  - **`SURVIVE`**: Triggered when health is low ($\le 40\%$) or hazard density is extreme. The AI maintains large standoff safety margins, backs away from asteroid centroids, and avoids hazardous child debris splits.
  - **`REPOSITION`**: Global A\* route planner maps out safe corridors through the moving debris field to open, low-density space, with the MPC smoothing trajectory waypoints.
- **Space Environment & Physics**: Multi-layer parallax starfield, colored nebulae, celestial gas giant planet, procedural jagged rock asteroids (sizes 1 to 4 with rotational spin and craters), exact engine child fragmentation ($\pm 15^\circ$ fan out upon destruction), and floating space debris.
- **HUD & Telemetry**: Large digital countdown timer (`MM:SS.S`), real-time health bar, shield gauge, score, kill counter, weapon accuracy, damage taken, and toggleable (`F1` / `TAB`) debug overlay showing predicted collision points, threat vectors, A\* path, and MPC trajectory arcs.

---

## Quick Start & Launch Instructions

### Requirements
- **OS**: Windows 10 / 11 (64-bit)
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Dependencies**: `pygame`, `numpy`, `kesslergame`, `pytest`

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in the project folder:
```powershell
pip install -r requirements.txt
```

### Step 2: Launch the Game

#### Method A: Windows One-Click Batch Script
Double-click `run_game.bat` in File Explorer, or run in Command Prompt:
```cmd
run_game.bat
```

#### Method B: Windows PowerShell Script
```powershell
.\run_game.ps1
```

#### Method C: Python CLI
```powershell
# Launch into the Autonomous Survival Game (Default: 180s Countdown)
python main.py

# Custom countdown timer (e.g. 60 seconds, 120 seconds, 300 seconds)
python main.py --timer 180.0

# Optional Modes & Difficulty:
python main.py --mode ai --difficulty kessler    # Extreme debris density
python main.py --mode player                    # Manual pilot testing mode
python main.py --mode coop                      # Dual player + AI mode

# Optional Flags:
#   --width 1000 --height 800
#   --fullscreen
#   --no-sound
#   --no-overlay
```

#### Method D: Standalone Executable (No Python Required)
You can directly run the pre-compiled Windows executable:
```cmd
.\dist\KesslerAI\KesslerAI.exe
```

#### Method E: Visual Studio Code
Open the project folder in VS Code and press `F5` to select from `.vscode/launch.json`:
- **Launch Kessler AI (AI Demo Mode)** (Default Autonomous 180s Survival)
- **Launch Kessler AI (Play Mode)**
- **Run All Tests (Pytest)**

---

## Game Modes

1. **Autonomous AI Survival (Default / Primary Mode)**:
   The Kessler AI autonomously pilots the interceptor through high-velocity asteroid swarms, aims and fires plasma cannons, evades collisions, and attempts to survive the full 180-second countdown.
2. **Manual Pilot Survival**:
   Take manual control of the spaceship with WASD / Arrow keys, mouse crosshair steering, spacebar firing, and Shift boost to test your pilot survival skills against the AI's high scores.
3. **Player + AI Dual Mode**:
   Both the Player ship and the autonomous Kessler AI ship operate together in the arena simultaneously.

---

## AI Controller & Behavior States

```
+------------------------------------------------------------------------+
|                          LIVE GAME SIMULATION                          |
|         (Ship Pose, Asteroid Tracks, Velocities, Health & Shields)     |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
|                      1. THREAT ASSESSMENT (threat.py)                  |
|   - Relative velocity, Time to Closest Approach (TCA), Min Separation  |
|   - Collision Risk (C), Urgency (U), Geometry (G), Combined Score (T)  |
|   - Identifies Critical Threats (TTC <= 0.75s) & Urgent Threats        |
+------------------------------------------------------------------------+
                                    |
        +---------------------------+---------------------------+
        |                                                       |
        v                                                       v
+-----------------------------------+   +-----------------------------------+
|   2. A* GLOBAL ROUTE PLANNER      |   |   3. THREAT TARGET SELECTOR       |
|      (pathfinding.py)             |   |      (targeting.py)               |
|   - Dynamic hazard-cost corridors |   |   - Predictive moving target lead |
|   - Finds clearest open sectors   |   |   - Fragmentation safety check    |
|   - Line-of-sight path smoothing  |   |   - Small-asteroid tie breakers   |
+-----------------------------------+   +-----------------------------------+
        \                                                       /
         \                                                     /
          v                                                   v
+------------------------------------------------------------------------+
|               4. MODEL PREDICTIVE CONTROL (planner.py)                 |
|   - Evaluates candidate action sequences over planning horizon         |
|   - Dynamic ship model (thrust, turn rate, drag, speed clamping)       |
|   - Rejects sequences with predicted collisions                        |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
|                      5. BEHAVIOR ARBITRATION                           |
|   - [EVADE]:       Critical threat detected -> tangential evasion &    |
|                    safety boost override (strict precedence).          |
|   - [SURVIVE]:     Low health (<= 40%) or dense hazard cluster ->      |
|                    defensive standoff, clearance maximization.         |
|   - [ATTACK]:      Safe conditions & target selected -> lead intercept |
|                    turn tracking and plasma cannon destruction.        |
|   - [REPOSITION]:  Relocating through field via A* waypoints to open   |
|                    sectors of space.                                   |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
|                    6. FIRE CONTROL (firecontrol.py)                    |
|   - Engine-exact raycast against target asteroid radius                |
|   - Fragment simulation: ensures child debris won't collide with ship  |
+------------------------------------------------------------------------+
```

---

## Scoring & Difficulty Escalation

### Points Table
- **Size 4 Asteroid (Giant)**: 25 points
- **Size 3 Asteroid (Large)**: 50 points
- **Size 2 Asteroid (Medium)**: 100 points
- **Size 1 Asteroid (Small)**: 200 points

### Anti-Exploitation & Safe Spawning
- **Unique Asteroid IDs**: Each asteroid and spawned child fragment possesses a distinct UID. Destroyed asteroids are recorded in a scored set, preventing duplicate point awards.
- **Unavoidable Collision Prevention**: New asteroids spawn at the perimeter at least 220 units away from the ship, with initial velocities oriented away from immediate collision vectors ($> 2.0$s grace).

### Difficulty Curve Over 180 Seconds
- **Progress Ratio**: $p = 1.0 - (\text{Countdown} / 180.0)$.
- **Asteroid Density**: Active asteroids scale from initial count (e.g. 6) up to 20+ as the countdown progresses.
- **Spawn Frequency**: Spawning interval reduces dynamically from 12.0s down to 3.5s.
- **Velocity Escalation**: Asteroid speed multiplier scales smoothly by $+35\%$ over the course of the match.
- **Tactical Siren**: An alert klaxon sounds periodically as hazard density escalates.

---

## Software Architecture & Modules

```
kessler_mpc/
├── main.py                  # Application entry point with CLI parameters
├── run_game.bat             # Windows one-click batch launcher
├── run_game.ps1             # Windows PowerShell launcher
├── requirements.txt         # Project dependencies
├── verify_gameplay_loop.py  # End-to-end autonomous gameplay verification script
├── tests/
│   ├── test_gameplay_systems.py    # 19 tests: controls, collisions, A*, AI states, timer, defeat
│   ├── test_units.py               # 9 tests: geometry, lead intercept, risk levels
│   ├── test_controller_engine.py   # 5 tests: integration against kesslergame engine
│   └── test_dynamics_vs_engine.py  # 6 tests: ship model dynamics validation
└── kessler_mpc/
    ├── game.py              # Main GameApp loop, countdown timer, VICTORY/DEFEAT state machine
    ├── kessler_ai.py        # Autonomous AI coordinator (ATTACK, EVADE, REPOSITION, SURVIVE)
    ├── pathfinding.py       # A* global route planner with dynamic hazard corridors
    ├── threat_assessment.py # Per-debris metrics (TCA, min separation, C, U, G, T)
    ├── threat.py            # Vectorized collision prediction & threat equations
    ├── planner.py           # Model Predictive Control trajectory optimizer
    ├── targeting.py         # Threat-ranked target selection & lead intercept
    ├── risk.py              # Adaptive life- and health-dependent risk multiplier
    ├── firecontrol.py       # Fragmentation-aware firing decision
    ├── dynamics.py          # Vectorized ship kinematics matching engine rules
    ├── spaceship.py         # Newtonian physics, health, shields, boost, plasma bullets
    ├── debris.py            # Procedural asteroid polygons, rotation & engine splitting
    ├── physics.py           # Collision resolution (ship-asteroid, bullet-asteroid, ship-ship)
    ├── visuals.py           # Parallax starfield, nebulae, particles, shockwaves, screen shake
    ├── audio.py             # Procedural sound effects synthesizer (NumPy + Pygame Mixer)
    ├── ui.py                # Menus, countdown HUD, AI status badge, radar, telemetry overlay
    ├── settings.py          # Difficulty presets, countdown duration, visual/audio toggles
    └── engine_compat.py     # Headless engine compatibility & inputs stubbing
```

---

## Testing & Validation

All 39 unit and system integration tests pass with 100% success rate:

```powershell
python -m pytest tests -v
```

Output:
```
tests/test_controller_engine.py::test_kills_a_crossing_asteroid_without_dying PASSED
tests/test_controller_engine.py::test_evades_head_on_big_asteroid PASSED
tests/test_controller_engine.py::test_respawn_aware_no_fire_when_crowded_and_survives PASSED
tests/test_controller_engine.py::test_runtime_per_step_is_practical PASSED
tests/test_controller_engine.py::test_old_style_dict_state_adapter PASSED
tests/test_dynamics_vs_engine.py::test_full_thrust_straight PASSED
tests/test_dynamics_vs_engine.py::test_turn_and_thrust PASSED
tests/test_dynamics_vs_engine.py::test_coast_and_reverse PASSED
tests/test_dynamics_vs_engine.py::test_weak_thrust_below_drag PASSED
tests/test_dynamics_vs_engine.py::test_wrap_around_and_random PASSED
tests/test_dynamics_vs_engine.py::test_asteroid_tracks_match_engine PASSED
tests/test_gameplay_systems.py::test_player_thrust_and_drag_physics PASSED
tests/test_gameplay_systems.py::test_player_boost_mechanic PASSED
tests/test_gameplay_systems.py::test_player_turning PASSED
tests/test_gameplay_systems.py::test_bullet_asteroid_collision_and_splitting PASSED
tests/test_gameplay_systems.py::test_ship_asteroid_collision_and_shield_damage PASSED
tests/test_gameplay_systems.py::test_ship_death_when_health_reaches_zero PASSED
tests/test_gameplay_systems.py::test_threat_evaluator_metrics PASSED
tests/test_gameplay_systems.py::test_astar_planner_finds_path PASSED
tests/test_gameplay_systems.py::test_astar_avoids_large_obstacle PASSED
tests/test_gameplay_systems.py::test_waypoint_navigator_progression PASSED
tests/test_gameplay_systems.py::test_safe_zone_docking_and_victory PASSED
tests/test_gameplay_systems.py::test_emergency_collision_avoidance_trigger PASSED
tests/test_gameplay_systems.py::test_ai_behavior_state_survive_on_low_health PASSED
tests/test_gameplay_systems.py::test_ai_behavior_state_attack_when_safe PASSED
tests/test_gameplay_systems.py::test_no_duplicate_scoring_for_same_asteroid PASSED
tests/test_gameplay_systems.py::test_safe_asteroid_spawn_respects_clearance PASSED
tests/test_gameplay_systems.py::test_countdown_timer_expiration_triggers_victory PASSED
tests/test_gameplay_systems.py::test_zero_health_triggers_immediate_defeat PASSED
tests/test_gameplay_systems.py::test_game_app_state_transitions PASSED
tests/test_units.py::test_closest_approach_head_on PASSED
tests/test_units.py::test_closest_approach_zero_relative_velocity PASSED
tests/test_units.py::test_closest_approach_clamped_to_horizon_and_diverging PASSED
tests/test_units.py::test_first_time_within PASSED
tests/test_units.py::test_lead_intercept PASSED
tests/test_units.py::test_wrap_angle PASSED
tests/test_units.py::test_adaptive_risk_levels_with_three_lives PASSED
tests/test_units.py::test_adaptive_risk_off PASSED
tests/test_units.py::test_unknown_config_key_rejected PASSED

======================== 39 passed, 1 warning in 3.09s ========================
```

---

## Standalone Windows Executable

The standalone Windows directory build is available at:
- **Executable**: `dist\KesslerAI\KesslerAI.exe`
- Double-click `KesslerAI.exe` to launch directly without requiring Python.
