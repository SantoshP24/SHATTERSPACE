"""Single configuration file. Values are engineering defaults, NOT proven optimal.

Anything the engine exposes at runtime (thrust/turn limits, max speed, drag, ship radius, map size,
frame rate, lives, cooldown) is read from the game state each call and is NOT configured here.
Only constants that the engine does not expose are configured (bullet speed -> bullet.py).
"""
from __future__ import annotations

CONFIG = {
    # ---- Overall behavior (informational flags; the controller factory reads them) -----------------------
    "behavior": "survival",
    "target_selection": "threat",
    "adaptive_risk": True,
    "planner": "mpc",

    # ---- Threat assessment ------------------------------------------------------------------------------
    "prediction_horizon_s": 3.0,
    # Multiplier on (R_ship + R_asteroid): the "safety distance". (Spec gave 1.5 without units; we read it as a factor.)
    "collision_safety_margin": 1.5,
    "critical_ttc_s": 0.75,
    "urgent_ttc_s": 1.5,
    "threat_weight_risk": 0.50,       # T_i = 0.50 C + 0.35 U + 0.15 G
    "threat_weight_urgency": 0.35,
    "threat_weight_geometry": 0.15,
    "position_uncertainty_mps": 3.0,  # clearance is reduced by this * time-to-closest-approach

    # ---- Target selection -------------------------------------------------------------------------------
    "target_score_mode": "weighted",  # "weighted" (config weights below) or "product" (spec 5B: T*P*V - lambda*E)
    "weight_collision_risk": 0.40,
    "weight_urgency": 0.25,
    "weight_hit_feasibility": 0.20,
    "weight_neutralization_value": 0.15,
    "attack_effort_lambda": 0.10,
    "target_switch_margin": 0.15,
    "tie_band": 0.05,                 # rule 3: within this band prefer smaller / nearer targets
    "min_safe_split_distance_factor": 2.5,  # don't shoot a splittable asteroid closer than this * (R_ship + R_child)

    # ---- MPC --------------------------------------------------------------------------------------------
    "planning_horizon_steps": 15,
    "mpc_step_frames": 3,             # each MPC step = this many engine frames (15 * 3 / 30 Hz = 1.5 s)
    "candidate_action_sequences": 100,
    "replan_every_step": True,
    "max_mpc_asteroids": 16,
    "cost_collision": 1000.0,
    "cost_margin": 20.0,
    "cost_threat_weight": 2.0,        # margin penalty scaled by (1 + this * T_i): "unresolved threat" term
    "cost_aim": 3.0,                  # positioning / attack-opportunity term (small vs. safety terms)
    "aim_decay": 0.85,
    "cost_terminal": 60.0,
    "terminal_lookahead_s": 1.0,
    "cost_switch": 0.3,
    "collision_pad_m": 1.5,           # extra radius in planning (frame sampling + model error)

    # ---- Adaptive risk ----------------------------------------------------------------------------------
    "low_lives_fraction": 0.33,
    "mid_lives_fraction": 0.66,
    "lives_fraction_tolerance": 0.01, # so that 1/3 counts as "<= 0.33" and 2/3 as "<= 0.66"
    "low_lives_risk_multiplier": 2.0,
    "initial_lives": None,            # None -> max lives seen so far (engine does not expose the initial count)

    # ---- Fire control -----------------------------------------------------------------------------------
    "bullet_speed": 800.0,            # engine constant (bullet.py); not exposed in state
    "aim_radius_fraction": (0.85, 0.75, 0.60),  # shot must hit this fraction of asteroid radius (normal, mid, low lives)
    "shot_cost_tolerance": 15.0,      # allowed MPC cost increase caused by a shot's fragments (divided by risk multiplier)
    "respawn_fire_clear_radius": 200.0,  # firing cancels respawn invulnerability, so only fire if nothing is this close
    "split_angle_deg": 15.0,          # engine: fixed +/-15 deg child fan (random_ast_splits=False)

    # ---- Evaluation -------------------------------------------------------------------------------------
    "evaluation_episodes": 100,
    "random_seed_start": 1000,        # FINAL evaluation seeds
    "tuning_seed_start": 1,           # TUNING seeds (kept disjoint from the final ones)
    "rng_seed": 0,
    "debug": False,
}


def make_config(**overrides) -> dict:
    cfg = dict(CONFIG)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg
