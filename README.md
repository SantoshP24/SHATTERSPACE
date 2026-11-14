# SHATTERSPACE: Deep Space Survival

*"One ship. Endless asteroids. Stay alive."*

A fully autonomous, real-time 2D space-survival game powered by **Model Predictive Control (MPC)**, **A\* Risk-Aware Route Planning**, **Threat-Based Target Selection**, **Adaptive Survival Risk Modeling**, and an **Emergency Collision-Safety Override**.

Built with **Pygame** and integrated directly with the verified **Thales Kessler MPC** navigation and physics architecture.

---

## Table of Contents
1. [Overview & Autonomous Gameplay](#overview--autonomous-gameplay)
2. [Advanced Autonomous AI Architecture](#advanced-autonomous-ai-architecture)
3. [Red Penalty Asteroid Hazard System](#red-penalty-asteroid-hazard-system)
4. [Spaceship & Asteroid Health Systems](#spaceship--asteroid-health-systems)
5. [Solid Arena Boundaries & No-Wrap Physics](#solid-arena-boundaries--no-wrap-physics)
6. [Interactive 4-Tab Settings Menu & Persistence](#interactive-4-tab-settings-menu--persistence)
7. [Quick Start & Launch Instructions](#quick-start--launch-instructions)
8. [Software Architecture & Modules](#software-architecture--modules)
9. [Testing & Validation](#testing--validation)
10. [Standalone Windows Executable](#standalone-windows-executable)

---

## Overview & Autonomous Gameplay

- **100% Autonomous Spaceship**: The spaceship is controlled entirely by AI. No player input is required for movement, steering, aiming, firing, or dodging.
- **180-Second Countdown Survival Objective**: The AI must survive an intense, escalating 180-second countdown while actively hunting down and shattering hazardous orbital debris to maximize its score.
- **Victory & Defeat Conditions**:
  - **VICTORY**: The AI successfully navigates and survives the full 180.0-second orbital hazard countdown.
  - **DEFEAT**: Critical hull failure occurs (all 3 ship lives depleted).
- **Four Explicit AI Behavior States**:
  - **`ATTACK`**: When conditions are safe, the AI tracks targets, computes the lead-intercept angle (`aim_point_bearing_deg`), advances toward the firing envelope, and fires plasma cannons automatically.
  - **`EVADE`**: When an imminent collision hazard is predicted ($TTC \le 0.75$s or $TTC \le 1.5$s with high collision risk), the emergency safety override takes complete control, executing high-G tangential escape maneuvers and boost surges.
  - **`SURVIVE`**: Triggered when health is low ($\le 40\%$) or hazard density is extreme. The AI maintains large standoff safety margins, backs away from asteroid centroids, and avoids hazardous child debris splits.
  - **`REPOSITION`**: Global A\* route planner maps out safe corridors through the moving debris field to open, low-density space, with the MPC smoothing trajectory waypoints.
- **Space Environment & Physics**: Multi-layer parallax starfield, colored nebulae, celestial gas giant planet, procedural jagged rock asteroids (sizes 1 to 4 with rotational spin and craters), exact engine child fragmentation ($\pm 15^\circ$ fan out upon destruction), and floating space debris.
- **HUD & Telemetry**: Large digital countdown timer (`MM:SS.S`), real-time health bar, shield gauge, score, kill counter, weapon accuracy, damage taken, floating penalty banners, and toggleable (`F1` / `TAB`) debug overlay showing predicted collision points, threat vectors, A\* path, and MPC trajectory arcs.

---

## Advanced Autonomous AI Architecture

The autonomous AI architecture integrates predictive kinematics, multi-objective optimization, and real-time state arbitration:

```
+-----------------------------------------------------------------------------------------+
|                                  GAME STATE OBSERVATION                                 |
|                Ship Pose (x, y, θ), Velocities (vx, vy, ω), Health & Lives              |
|                Asteroid States (pos, vel, radius, HP, is_penalty), Arena Bounds         |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                          1. PREDICTIVE COLLISION DETECTION                              |
|          Trajectory projection:  P_i(t) = P_i(0) + V_i * t                              |
|          Relative velocity:      v_rel = V_ast - V_ship                                 |
|          Time to closest appr:   t* = - ((P_ast - P_ship) • v_rel) / |v_rel|^2          |
|          Min separation distance: d_min = |(P_ast(t*) - P_ship(t*))|                    |
|          Collision probability:  c_i = max(0, 1 - d_min / (r_ship + r_ast + margin))    |
+-----------------------------------------------------------------------------------------+
                                             |
                      +----------------------+----------------------+
                      |                                             |
                      v                                             v
+-------------------------------------------+ +-------------------------------------------+
|      2. A* RISK-AWARE PATHFINDING         | |       3. THREAT-BASED TARGETING           |
|  - Grid cost: C(j) = 1 + λ * R(j)         | |  - Target Score:                          |
|  - R(j): hazard proximity & wall aversion | |    w_T*THREAT + w_H*VALUE + w_P*HIT_PROB  |
|  - Dynamic clearance routing              | |    - w_F*FRAG_RISK - w_E*EFFORT           |
|  - Waypoints passed to MPC trajectory     | |  - Target switching margin: ΔS >= 0.15    |
|  - Red penalty asteroids excluded         | |  - Red penalty asteroids strictly ignored |
+-------------------------------------------+ +-------------------------------------------+
                      \                                             /
                       \                                           /
                        v                                         v
+-----------------------------------------------------------------------------------------+
|                        4. MODEL PREDICTIVE CONTROL (MPC)                                |
|  - Discrete planning horizon: N steps ahead                                             |
|  - Action evaluation: (Thrust, Turn Rate) sequences                                     |
|  - Exact Newtonian ship model with rotational momentum and drag                         |
|  - Arena boundary collision penalty (walls treated as impassable hazards)               |
|  - Trajectory safety score: minimizes risk along future path                            |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                       5. ADAPTIVE SURVIVAL BEHAVIOR                                     |
|  - Damage index: q = (Lives_rem / Lives_max) * (HP_rem / HP_max)                        |
|  - Risk scaling factor: ρ(q) = 1.0 + β * (1.0 - q)                                      |
|  - Dynamically expands safety margins and lowers risk threshold when damaged             |
|  - Arbitrates states: [EVADE], [SURVIVE], [ATTACK], [REPOSITION]                        |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                  6. PREDICTIVE AIMING & SAFETY FIRE CONTROL                             |
|  - Quadratic lead intercept solver for projectile travel time τ                         |
|  - Aim bearing: θ_target = atan2(P_target(τ)_y - P_ship_y, P_target(τ)_x - P_ship_x)    |
|  - Fire Control Safety: suppresses shot if bullet path intersects any Red Hazard        |
|  - Fragment safety check: ensures child debris won't spray into ship                    |
+-----------------------------------------------------------------------------------------+
```

---

## Red Penalty Asteroid Hazard System

Special **Red Penalty Asteroids** introduce high-stakes tactical decision-making into the simulation:
- **Visuals**: Pulsating crimson-red hue (`rgb(235, 25, 25)`), outer danger glow aura, and distinct radar signature.
- **Flight Path**: Enters from one screen boundary, travels straight across the arena at constant speed, and exits the opposite edge without bouncing off walls.
- **Spawning & Population**: Spawn probability of 5% checked every 5.0 seconds; capped at a maximum of 1 concurrent active hazard.
- **Shooting Penalty**: Destroying a red asteroid deducts **-5 points** from the score and immediately removes it from play without creating any child fragments.
- **Collision Penalty**: Ramming a red asteroid deducts **-10 points** from the score and immediately removes it without reducing the spaceship's health.
- **AI Avoidance**:
  - The targeting module strictly filters out red penalty asteroids from target selection.
  - The fire control module traces bullet trajectories and aborts firing if the projectile would intercept a red penalty asteroid.
  - The MPC trajectory planner and A\* pathfinder treat red penalty asteroids as moving hazards to evade.

---

## Spaceship & Asteroid Health Systems

### Spaceship Hull & Lives
- **Health Pool**: 100 HP max per life across 3 lives.
- **Size-Based Impact Damage**:
  - **Large Asteroid**: 40 HP damage.
  - **Medium Asteroid**: 25 HP damage.
  - **Small Asteroid**: 10 HP damage.
- **Collision Invulnerability**: 1.5 seconds of invulnerability after impact, during which the ship blinks and takes no further damage from overlapping asteroids.
- **Respawn Invulnerability**: 3.0 seconds of invulnerability upon losing a life and respawning at arena center.
- **Health Regeneration**: Configurable passive hull nanite regeneration (disabled by default; adjustable rate in HP/s).

### Asteroid Hit Points & Fragmentation
- **Asteroid HP**: Small = 1 HP, Medium = 2 HP, Large = 3 HP (configurable).
- **Bullet Damage**: Configurable projectile damage (default: 1.0 HP per bullet). Asteroids flash white on damage and shatter when HP reaches 0.
- **Fragmentation**: Large and Medium normal asteroids shatter into 2–3 child fragments with a $\pm 15^\circ$ velocity fan-out and a safe clearance offset to prevent instant collisions. Red asteroids never fragment.

---

## Solid Arena Boundaries & No-Wrap Physics

Screen-wrapping has been completely removed in favor of a solid, realistic space combat arena:
- **No Edge Teleportation**: The ship, asteroids, and plasma projectiles never wrap across screen borders.
- **Strict Ship Boundary Clamping**: The ship is physically constrained within $[R_{\text{ship}}, W - R_{\text{ship}}]$ and $[R_{\text{ship}}, H - R_{\text{ship}}]$. Impacting an arena wall cancels velocity normal to the wall, preventing sticking, tunneling, or jitter.
- **Elastic Asteroid Bouncing**: Normal asteroids bounce elastically off arena boundaries upon collision, reversing their velocity component like billiard balls in a space arena.
- **Boundary-Aware MPC Trajectory Scoring**:
  - The MPC trajectory evaluator projects candidate sequences forward and penalizes candidates that exit the arena or approach within an 80px boundary margin.
  - Candidates exiting boundaries are tagged as wall collisions (`collides = True`), ensuring the planner prioritizes safe maneuvers inside the arena.
- **Corner Escape & Wall Avoidance**:
  - When the AI is near a corner ($X < 135$ or $X > W - 135$, and $Y < 135$ or $Y > H - 135$), an automated corner-escape maneuver immediately steers toward the arena center $(W/2, H/2)$ with forward thrust.
  - In `EVADE` mode, tangential evasion vectors that point toward arena walls are deflected inward toward open space, guaranteeing the AI never dodges into a dead end.

---

## Interactive 4-Tab Settings Menu & Persistence

An accessible **Settings Menu** is integrated directly into the Main Menu (`SETTINGS & TUNING`) and Pause Menu:

### Tab 1: SPACESHIP
- **Chassis Design**: Selectable ship design — `Interceptor`, `Scout`, `Vanguard`.
- **Max Speed**: 80 – 500 px/s (Default: 240 px/s).
- **Thrust Acceleration**: 150 – 1000 px/s² (Default: 480 px/s²).
- **Turning Speed**: 60 – 360 deg/s (Default: 180 deg/s).
- **Ship Health**: 25 – 400 HP (Default: 100 HP).
- **Starting Lives**: 1 – 10 Lives (Default: 3 Lives).
- **Fire Rate**: 1.0 – 20.0 shots/s (Default: 7.0 shots/s).
- **Projectile Speed**: 300 – 1500 px/s (Default: 800 px/s).
- **Bullet Damage**: 0.5 – 10.0 HP (Default: 1.0 HP).
- **Safety Margin**: 15 – 100 px (Default: 35 px).
- **Invulnerability**: Normal collision (1.5s) & Respawn (3.0s).

### Tab 2: ASTEROIDS
- **Initial Count**: 1 – 20 asteroids (Default: 6).
- **Max Active Count**: 5 – 50 asteroids (Default: 20).
- **Speed Multiplier**: 0.3x – 3.0x (Default: 1.0x).
- **Asteroid HP**: Small (1.0 HP), Medium (2.0 HP), Large (3.0 HP).
- **Fragments per Split**: Min / Max child count (Default: 3).
- **Point Values**: Small (200 pts), Medium (100 pts), Large (50 pts).
- **Asteroid Splitting**: Toggle enabled/disabled.

### Tab 3: RED HAZARDS
- **Red Asteroids Enabled**: Toggle special red penalty hazards on/off.
- **Max Active Red**: 1 – 5 concurrent red hazards (Default: 1).
- **Spawn Probability**: 1% – 25% per interval (Default: 5%).
- **Spawn Interval**: 2.0 – 20.0s (Default: 5.0s).
- **Hazard Speed**: 50 – 300 px/s (Default: 100 px/s).
- **Shooting Penalty**: -1 to -50 pts (Default: -5 pts).
- **Collision Penalty**: -1 to -50 pts (Default: -10 pts).

### Tab 4: GAME & AI
- **Round Duration**: 20 – 360 seconds (Default: 180 seconds).
- **Difficulty Presets**:
  1. `Cadet` – "A forgiving introduction to space survival."
  2. `Pilot` – "A balanced challenge."
  3. `Veteran` – "Faster threats and less room for mistakes."
  4. `Elite` – "Intense asteroid encounters."
  5. `Nightmare` – "The ultimate survival test."
- **Difficulty Escalation**: Dynamic spawn rate and speed acceleration over countdown.
- **AI Aggression**: `Defensive`, `Balanced`, `Aggressive`.
- **Adaptive Risk Beta ($\beta$)**: 0.0 – 4.0 (Default: 1.5).
- **Threat Weights**: Collision Risk ($w_R$), Urgency ($w_U$), Geometry ($w_G$).
- **Target Switch Threshold**: 0.05 – 0.50 (Default: 0.15).

### Persistence & Factory Reset
- **Local Persistence**: Settings are saved locally to `settings.json` upon clicking **APPLY**, automatically loading whenever the game starts.
- **Factory Reset**: Click **RESET** to restore default settings.
- **Validation**: Numerical inputs are clamped to verified safe ranges to protect against simulation crashes.

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
# Launch into the Autonomous Survival Game (Default: 180s Countdown, Cadet preset)
python main.py

# Custom countdown timer (e.g. 60 seconds, 120 seconds, 300 seconds)
python main.py --timer 180.0

# Optional Modes & Difficulty Presets:
python main.py --difficulty cadet        # Cadet (forgiving introduction)
python main.py --difficulty pilot        # Pilot (balanced challenge)
python main.py --difficulty veteran      # Veteran (faster threats)
python main.py --difficulty elite        # Elite (intense encounters)
python main.py --difficulty nightmare    # Nightmare (ultimate survival test)

python main.py --mode player             # Manual Pilot mode
python main.py --mode coop               # Player + AI dual mode

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

---

## Software Architecture & Modules

```
kessler_mpc/
├── main.py                  # Application entry point with CLI parameters
├── run_game.bat             # Windows one-click batch launcher
├── run_game.ps1             # Windows PowerShell launcher
├── requirements.txt         # Project dependencies
├── settings.json            # Persistent user configuration
├── verify_gameplay_loop.py  # End-to-end autonomous gameplay verification script
├── tests/
│   ├── test_advanced_features.py   # 20 tests: red hazards, health/lives, penalties, AI avoidance
│   ├── test_gameplay_systems.py    # 29 tests: controls, collisions, boundaries, settings, defeat
│   ├── test_units.py               # 9 tests: geometry, lead intercept, risk levels
│   ├── test_controller_engine.py   # 5 tests: integration against kesslergame engine
│   └── test_dynamics_vs_engine.py  # 6 tests: ship model dynamics validation
└── kessler_mpc/
    ├── game.py              # Main GameApp loop, countdown timer, VICTORY/DEFEAT state machine
    ├── kessler_ai.py        # Autonomous AI coordinator (ATTACK, EVADE, REPOSITION, SURVIVE)
    ├── pathfinding.py       # A* risk-aware route planner with dynamic hazard corridors & corner aversion
    ├── threat_assessment.py # Per-debris metrics (TCA, min separation, C, U, G, T)
    ├── threat.py            # Vectorized collision prediction & threat equations
    ├── planner.py           # Model Predictive Control trajectory optimizer with boundary penalties
    ├── targeting.py         # 5-term threat-ranked target selection & lead intercept
    ├── risk.py              # Adaptive life- and health-dependent risk multiplier
    ├── firecontrol.py       # Red hazard avoidance & fragmentation-aware firing decision
    ├── dynamics.py          # Vectorized ship kinematics matching engine rules
    ├── spaceship.py         # Newtonian physics, health, lives, invulnerability timers, regen
    ├── debris.py            # Normal asteroids & RedPenaltyAsteroid (crimson glow, edge-to-edge)
    ├── physics.py           # Size-based damage, bullet damage, red penalty deductions
    ├── visuals.py           # Parallax starfield, nebulae, particles, shockwaves, screen shake
    ├── audio.py             # Procedural sound effects synthesizer (NumPy + Pygame Mixer)
    ├── ui.py                # Menus, 4-tab SettingsMenuUI, floating penalty banners, HUD
    ├── settings.py          # GameSettings dataclass, JSON save/load, validation clamps
    └── engine_compat.py     # Headless engine compatibility & inputs stubbing
```

---

## Testing & Validation

All **71 unit and system integration tests** pass with a 100% success rate:

```powershell
python -m pytest tests -v
```

Output:
```
============================= 71 passed in 3.11s ==============================
```

---

## Standalone Windows Executable

The standalone Windows directory build is available at:
- **Executable**: `dist\KesslerAI\KesslerAI.exe`
- Double-click `KesslerAI.exe` to launch directly without requiring Python.
