# Kessler AI: Deep-Space Orbital Survival Game

An interactive, real-time top-down 2D space-survival game powered by **Model Predictive Control (MPC)**, **A\* Global Route Planning**, **Threat Assessment**, **Adaptive Risk Modeling**, and an **Emergency Collision-Safety Override**.

Built with **Pygame** and integrated directly with the verified **Thales Kessler MPC** navigation and physics architecture.

---

## Table of Contents
1. [Overview & Features](#overview--features)
2. [Quick Start & Launch Instructions](#quick-start--launch-instructions)
3. [Flight Controls](#flight-controls)
4. [Game Modes](#game-modes)
5. [AI Architecture & Integration](#ai-architecture--integration)
6. [Software Architecture & Modules](#software-architecture--modules)
7. [Testing & Validation](#testing--validation)
8. [Standalone Windows Executable](#standalone-windows-executable)

---

## Overview & Features

- **Player Spaceship**: Newtonian inertia, main thrusters, kinetic reverse brakes, attitude rotation, crosshair mouse aiming, plasma laser cannons, and afterburner boost with active cooldown.
- **Autonomous Kessler AI**: Real-time debris tracking, threat ranking ($T_i = 0.50 C + 0.35 U + 0.15 G$), A\* global pathfinding through dynamic hazard fields toward the Safe Zone, local MPC trajectory optimization over candidate sequences, adaptive life-dependent risk scaling, and emergency collision avoidance overrides.
- **Space Environment**: Multi-layer parallax starfield (twinkling stars, colored gas nebulae, celestial gas giant planet silhouette), procedural jagged rock asteroids (sizes 1 to 4 with rotational spin and impact craters), exact engine child fragmentation ($\pm 15^\circ$ fan out upon destruction), and floating space debris.
- **Safe Zone Sanctuary**: Orbital sanctuary beacon with a pulsating defensive energy shield. Ships entering the sanctuary initiate an automated docking timer (3.0s); holding position inside awards victory.
- **Visuals & Procedural Audio**: Particle engine generating rocket plumes, electric blue boost fire, shockwave rings, shield ripples, and fiery explosions. Synthesized sci-fi procedural audio built using NumPy and Pygame Mixer (no missing external asset dependencies).
- **Toggleable AI Debug Telemetry (`F1` / `TAB`)**: In-world rendering of the A\* route, MPC candidate trajectory arcs, asteroid threat vectors with $TTC$ readouts, detection radius, and target lock-on reticle.

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

#### Method A: Windows Batch Launcher (One-Click)
Double-click `run_game.bat` in File Explorer, or run in cmd:
```cmd
run_game.bat
```

#### Method B: Windows PowerShell Launcher
```powershell
.\run_game.ps1
```

#### Method C: Python CLI
```powershell
# Launch into the Main Menu
python main.py

# Launch directly into a specific mode
python main.py --mode player       # Player Survival Mode
python main.py --mode ai           # Autonomous AI Demonstration
python main.py --mode survival     # Endless Survival Challenge
python main.py --mode coop         # Player + AI Co-op / Race

# Optional CLI Arguments:
#   --difficulty [casual | standard | hardcore | kessler]
#   --width 1000 --height 800
#   --fullscreen
#   --no-sound
#   --no-overlay
```

#### Method D: Visual Studio Code
Open the project folder in VS Code and press `F5` to select one of the pre-configured launch tasks in `.vscode/launch.json`:
- **Launch Kessler AI (Play Mode)**
- **Launch Kessler AI (AI Demo Mode)**
- **Launch Kessler AI (Survival Mode)**
- **Launch Kessler AI (Co-op / Race Mode)**
- **Run All Tests (Pytest)**

---

## Flight Controls

| Control | Action |
|---|---|
| **W** or **Up Arrow** | Main Thrusters (Forward Acceleration) |
| **S** or **Down Arrow** | Reverse Thrusters / Kinetic Brakes |
| **A** / **D** or **Left** / **Right** | Rotational Attitude Control (Turn Left / Right) |
| **Mouse Cursor** | Crosshair Aiming (Ship rotates smoothly toward cursor) |
| **Spacebar** or **Left Click** | Fire High-Velocity Plasma Cannon |
| **Left Shift** or **Right Shift** | Activate Afterburner Boost ($1.85\times$ thrust multiplier) |
| **F1** or **TAB** | Toggle AI Telemetry & Trajectory Debug Overlay |
| **ESC** or **P** | Pause Game / Return to Menu |

---

## Game Modes

1. **Player Survival Mode (`MODE_PLAYER`)**:
   Pilot your interceptor through dangerous orbital debris, eliminate threatening asteroids, manage shield and boost energy, and navigate safely into the orbital sanctuary to dock.
2. **AI Demonstration Mode (`MODE_AI_DEMO`)**:
   Observe the autonomous Kessler AI pilot the spacecraft in real-time. Toggle the debug overlay (`F1`) to see its internal decision-making: live threat rankings, A\* waypoints, MPC trajectory evaluations, and emergency collision overrides.
3. **Endless Survival Challenge (`MODE_SURVIVAL`)**:
   Survive escalating waves of orbital debris. Every 15 seconds, hazard speed and asteroid density increase. Compete for the highest survival score.
4. **Player + AI Co-op / Race Mode (`MODE_COOP`)**:
   Both the Player ship and the autonomous Kessler AI ship operate together in the arena. Cooperate to shatter large hazards or race to reach and dock in the sanctuary first.

---

## AI Architecture & Integration

The Kessler AI operates on a hierarchical multi-tiered control architecture:

```
+------------------------------------------------------------------------+
|                          LIVE GAME SIMULATION                          |
|   (Ship Pose, Asteroid Positions & Velocities, Safe Zone Sanctuary)   |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
|                      1. THREAT EVALUATOR (threat.py)                   |
|   - Minimum Separation (d_min) & Time to Closest Approach (t_min)     |
|   - Collision Risk: C = clip((safe - d_eff) / band, 0, 1)              |
|   - Urgency:        U = clip(1 - TTC / H, 0, 1)                        |
|   - Geometric:      G = 0.6*(size/4)*prox + 0.4*depth                  |
|   - Overall Threat: T = 0.50 C + 0.35 U + 0.15 G                       |
|   - Critical TTC <= 0.75s, Urgent TTC <= 1.50s                         |
+------------------------------------------------------------------------+
         |                                                 |
         v                                                 v
+-----------------------------+           +-----------------------------+
|   2. A* GLOBAL ROUTE        |           |  3. TARGET SELECTOR         |
|      (pathfinding.py)       |           |     (targeting.py)          |
|   - Discretized cost grid   |           |   - Threat-weighted target  |
|   - Hazard corridor penalty |           |   - Lead intercept point    |
|   - Line-of-sight smoothing |           |   - Fragmentation safety    |
|   - Navigates to Safe Zone  |           |   - Small asteroid tie band |
+-----------------------------+           +-----------------------------+
         \                                                 /
          \                                               /
           v                                             v
+------------------------------------------------------------------------+
|             4. MODEL PREDICTIVE CONTROL (mpc_planner.py)               |
|   - Evaluates candidate action sequences over planning horizon         |
|   - Exact Newtonian ship model (thrust, turn rate, drag, speed limits)  |
|   - Cost: J = J_collision + J_margin + J_aim + J_terminal + J_switch   |
|   - Rejects sequences with predicted collisions                        |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
|            5. COLLISION-SAFETY OVERRIDE (kessler_ai.py)                |
|   - Takes strict precedence over normal navigation & targeting         |
|   - Triggers when Critical Threat (TTC <= 0.75s) is detected           |
|   - Computes tangential escape vector perpendicular to hazard velocity  |
|   - Executes evasive thrust, turn, and boost if separation is critical  |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
|                     6. FIRE CONTROL (firecontrol.py)                   |
|   - Validates ray intercept against asteroid bounds                    |
|   - Simulates fragment tracks to ensure child debris does not collide  |
+------------------------------------------------------------------------+
```

---

## Software Architecture & Modules

```
kessler_mpc/
├── main.py                  # CLI and desktop game launcher entry point
├── run_game.bat             # Windows one-click batch launcher
├── run_game.ps1             # Windows PowerShell launcher
├── requirements.txt         # Project dependencies
├── verify_gameplay_loop.py  # Automated full-gameplay integration test script
├── tests/
│   ├── test_gameplay_systems.py    # 14 tests: controls, collisions, A*, AI, Safe Zone, modes
│   ├── test_units.py               # 9 tests: closest approach, geometry, risk levels
│   ├── test_controller_engine.py   # 2 tests: integration against kesslergame engine
│   └── test_dynamics_vs_engine.py  # 9 tests: ship model dynamics validation
└── kessler_mpc/
    ├── game.py              # Main GameApp loop, state machine & mode manager
    ├── player.py            # PlayerShip with keyboard, mouse, and boost input
    ├── spaceship.py         # Newtonian physics, health, shields, lives, boost, laser bullets
    ├── kessler_ai.py        # Autonomous AI coordinator (A* + MPC + Collision Override)
    ├── pathfinding.py       # A* global route planner with dynamic hazard corridors
    ├── threat_assessment.py # Per-debris metrics (TCA, min separation, C, U, G, T)
    ├── threat.py            # Vectorized collision prediction & threat equations
    ├── planner.py           # Model Predictive Control trajectory optimizer
    ├── targeting.py         # Threat-ranked target selection & lead intercept
    ├── risk.py              # Adaptive life-dependent risk multiplier
    ├── firecontrol.py       # Fragmentation-aware firing decision
    ├── dynamics.py          # Vectorized ship kinematics matching engine rules
    ├── safe_zone.py         # Sanctuary objective, arrival detection & docking progress
    ├── debris.py            # Procedural asteroid polygons, rotation & engine splitting
    ├── physics.py           # Collision resolution (ship-asteroid, bullet-asteroid, ship-ship)
    ├── visuals.py           # Parallax starfield, nebulae, particles, shockwaves, screen shake
    ├── audio.py             # Procedural sound effects synthesizer (NumPy + Pygame Mixer)
    ├── ui.py                # Menus, HUD, radar minimap, and toggleable AI telemetry overlay
    ├── settings.py          # Difficulty presets, visual/audio settings, debug toggles
    └── engine_compat.py     # Headless engine compatibility & inputs stubbing
```

---

## Testing & Validation

All 34 tests pass with 100% success rate:

```powershell
python -m pytest tests -v
```

Output:
```
tests/test_controller_engine.py::test_kills_a_crossing_asteroid_without_dying PASSED
tests/test_controller_engine.py::test_evades_head_on_big_asteroid PASSED
tests/test_dynamics_vs_engine.py::test_stationary_ship_matches_engine PASSED
tests/test_dynamics_vs_engine.py::test_full_thrust_forward_matches_engine PASSED
tests/test_dynamics_vs_engine.py::test_full_thrust_reverse_matches_engine PASSED
tests/test_dynamics_vs_engine.py::test_alternating_thrust_matches_engine PASSED
tests/test_dynamics_vs_engine.py::test_hard_turn_matches_engine PASSED
tests/test_dynamics_vs_engine.py::test_turn_while_accelerating_matches_engine PASSED
tests/test_dynamics_vs_engine.py::test_toroidal_wrapping_matches_engine PASSED
tests/test_dynamics_vs_engine.py::test_asteroid_tracks_match_engine PASSED
tests/test_dynamics_vs_engine.py::test_asteroid_wrapping_matches_engine PASSED
tests/test_gameplay_systems.py::test_player_thrust_and_drag_physics PASSED
tests/test_gameplay_systems.py::test_player_boost_mechanic PASSED
tests/test_gameplay_systems.py::test_player_turning PASSED
tests/test_gameplay_systems.py::test_bullet_asteroid_collision_and_splitting PASSED
tests/test_gameplay_systems.py::test_ship_asteroid_collision_and_shield_damage PASSED
tests/test_gameplay_systems.py::test_ship_death_and_lives_decrement PASSED
tests/test_gameplay_systems.py::test_threat_evaluator_metrics PASSED
tests/test_gameplay_systems.py::test_astar_planner_finds_path PASSED
tests/test_gameplay_systems.py::test_astar_avoids_large_obstacle PASSED
tests/test_gameplay_systems.py::test_waypoint_navigator_progression PASSED
tests/test_gameplay_systems.py::test_safe_zone_docking_and_victory PASSED
tests/test_gameplay_systems.py::test_emergency_collision_avoidance_trigger PASSED
tests/test_gameplay_systems.py::test_ai_live_simulation_integration PASSED
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

============================== 34 passed in 3.03s ==============================
```

---

## Standalone Windows Executable

The game has been compiled into a standalone Windows directory build using PyInstaller.
- **Executable path**: `dist\KesslerAI\KesslerAI.exe`
- You can double-click `KesslerAI.exe` to run the game directly without needing a Python interpreter installed.
