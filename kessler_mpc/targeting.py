"""Section 5: threat-ranked target selection (A_i), with the small-asteroid tie-breaker rules (5C)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import coast_trajectory
from .geometry import aim_point_bearing_deg, closest_approach, wrap_angle_deg
from .threat import Threats
from .world import World


@dataclass
class Target:
    idx: int
    score: float
    p_hit: float
    t_engage: float
    size: int


class TargetSelector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prev = None  # (pos, vel, size) of last chosen target, for identity matching + hysteresis

    def _match_prev(self, w: World):
        if self.prev is None or w.n_ast == 0:
            return None
        pos, vel, size = self.prev
        pred = np.mod(pos + vel * w.params.frame_dt, [w.params.map_w, w.params.map_h])
        d = np.linalg.norm(w.ast_pos - pred, axis=1)
        ok = (w.ast_size == size) & (d < 3.0)
        return int(np.argmin(np.where(ok, d, np.inf))) if ok.any() else None

    def evaluate(self, w: World, th: Threats):
        """Per-asteroid engagement geometry and score components (vectorised)."""
        c, p = self.cfg, w.params
        vb, dt, A = c["bullet_speed"], p.frame_dt, w.n_ast
        H = c["prediction_horizon_s"]
        d0 = w.ast_pos - w.pos
        brg = np.degrees(np.arctan2(d0[:, 1], d0[:, 0]))
        t_a = np.abs(wrap_angle_deg(brg - w.heading)) / p.turn_max + dt
        tau = np.zeros(A)
        for _ in range(3):
            s_t, _ = coast_trajectory(w.pos, w.speed, w.heading, p, t_a)
            q = w.ast_pos + w.ast_vel * t_a[:, None]
            brg, aim = aim_point_bearing_deg(s_t, q, w.ast_vel, p.radius, vb)
            tau = np.maximum(np.linalg.norm(aim - s_t, axis=1) - p.radius, 0.0) / vb
            t_a = np.abs(wrap_angle_deg(brg - w.heading)) / p.turn_max + dt
        t_eng = t_a + tau
        ip = w.ast_pos + w.ast_vel * t_eng[:, None]                      # unwrapped impact point
        valid = ((ip[:, 0] >= 0) & (ip[:, 0] <= p.map_w) & (ip[:, 1] >= 0) & (ip[:, 1] <= p.map_h) & np.isfinite(t_eng))
        in_time = t_eng < th.ttc
        P = np.where(valid, np.clip(1.0 - t_eng / H, 0.0, 1.0) * np.where(in_time, 1.0, 0.25), 0.0)

        # fragmentation-aware neutralisation value V and "engage safely" gate
        s_imp, spd_imp = coast_trajectory(w.pos, w.speed, w.heading, p, t_eng)
        u = np.array([np.cos(np.radians(w.heading)), np.sin(np.radians(w.heading))])
        sv = spd_imp[:, None] * u[None, :]
        ang = np.radians([c["split_angle_deg"], 0.0, -c["split_angle_deg"]])
        sp = np.linalg.norm(w.ast_vel, axis=1)
        th0 = np.arctan2(w.ast_vel[:, 1], w.ast_vel[:, 0])
        vc = np.stack([sp[:, None] * np.cos(th0[:, None] + ang[None, :]),
                       sp[:, None] * np.sin(th0[:, None] + ang[None, :])], axis=-1)       # (A,3,2)
        r = ip[:, None, :] - s_imp[:, None, :]
        _, dmin = closest_approach(r, vc - sv[:, None, :], 2.0)
        rc = np.maximum(w.ast_rad / np.maximum(w.ast_size, 1) * (w.ast_size - 1), 0.0)
        f_child = (dmin < c["collision_safety_margin"] * (p.radius + rc)[:, None]).mean(axis=1)
        V = np.where(w.ast_size <= 1, 1.0, 0.75 - 0.35 * f_child)
        split_dist = np.linalg.norm(ip - s_imp, axis=1)
        unsafe = (w.ast_size > 1) & (split_dist < c["min_safe_split_distance_factor"] * (p.radius + rc))

        shots_needed = (3.0 ** w.ast_size - 1.0) / 2.0
        E = 0.6 * np.clip(t_eng / H, 0, 1) + 0.4 * shots_needed / 40.0
        if A > 1:
            s = np.sort(th.ttc)
            other_min = np.where(th.ttc == s[0], s[1], s[0])
        else:
            other_min = np.full(A, np.inf)
        expo = 0.25 * (other_min < t_a)                                      # rule 4

        # Exclude red penalty asteroids: special environmental hazard, never an attack target
        is_pen = getattr(w, "ast_is_penalty", np.zeros(A, bool))

        # Normalized 5-term target scoring per specification:
        # ATTACK_SCORE_i = wT * THREAT_i + wH * STRATEGIC_VALUE_i + wP * HIT_PROBABILITY_i - wF * FRAGMENTATION_RISK_i - wE * ATTACK_EFFORT_i
        threat_term = np.clip(th.T, 0.0, 1.0)
        strat_val = np.clip(V * (w.ast_size / 4.0), 0.0, 1.0)
        hit_prob = np.clip(P, 0.0, 1.0)
        frag_risk = np.clip(f_child, 0.0, 1.0)
        attack_eff = np.clip(E, 0.0, 1.0)

        wT = float(c.get("weight_target_threat", 0.35))
        wH = float(c.get("weight_strategic_value", 0.25))
        wP = float(c.get("weight_hit_prob", 0.25))
        wF = float(c.get("weight_frag_risk", 0.10))
        wE = float(c.get("weight_attack_effort", 0.05))

        if c.get("target_score_mode", "") == "product":
            S = th.T * P * V - c.get("attack_effort_lambda", 0.25) * E - expo
        else:
            S = wT * threat_term + wH * strat_val + wP * hit_prob - wF * frag_risk - wE * attack_eff - expo

        eligible = (P > 0) & ~unsafe & ~is_pen
        return S, P, t_eng, eligible

    def select(self, w: World, th: Threats):
        if w.n_ast == 0:
            self.prev = None
            return None
        S, P, t_eng, ok = self.evaluate(w, th)
        if not ok.any():
            self.prev = None
            return None
        c = self.cfg
        Sm = np.where(ok, S, -np.inf)
        best = int(np.argmax(Sm))
        near = np.where(Sm >= Sm[best] - c.get("tie_band", 0.05))[0]                    # rule 3: prefer small / near on ties
        if len(near) > 1:
            dist = np.linalg.norm(w.ast_pos[near] - w.pos, axis=1)
            best = int(near[np.lexsort((dist, w.ast_size[near]))[0]])
        prev = self._match_prev(w)                                            # rule 5: hysteresis / target commitment
        switch_margin = float(c.get("target_switch_threshold", c.get("target_switch_margin", 0.15)))
        if prev is not None and ok[prev] and Sm[prev] >= Sm[best] - switch_margin:
            best = prev
        self.prev = (w.ast_pos[best].copy(), w.ast_vel[best].copy(), int(w.ast_size[best]))
        return Target(best, float(S[best]), float(P[best]), float(t_eng[best]), int(w.ast_size[best]))
