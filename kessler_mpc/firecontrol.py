"""Fire decision. Fire is an independent action in the engine, but firing (a) cancels respawn invulnerability
(ship.fire_bullet) and (b) destroying a size>1 asteroid spawns 3 children (fan of +/-15 deg, same speed),
so a shot is only taken if the engine-exact ray test predicts a hit AND the fragments don't worsen safety
along the planned trajectory."""
from __future__ import annotations

import numpy as np

from . import dynamics
from .geometry import first_time_within
from .planner import Plan
from .risk import RiskState
from .world import World


class FireControl:
    def __init__(self, cfg):
        self.cfg = cfg

    def _shot(self, w: World, thrust, turn, aim_fraction):
        """Engine order: asteroids/ship move, then the bullet spawns at the *new* ship pose. Returns (j, t_hit, ap1)."""
        c, p = self.cfg, w.params
        dt = p.frame_dt
        pos1, _, h1 = dynamics.step(w.pos[None], np.array([w.speed]), np.array([w.heading]),
                                    np.array([thrust]), np.array([turn]), p, dt)
        u = np.array([np.cos(np.radians(h1[0])), np.sin(np.radians(h1[0]))])
        muzzle = pos1[0] + p.radius * u
        ap1 = np.mod(w.ast_pos + w.ast_vel * dt, [p.map_w, p.map_h])
        vb = c["bullet_speed"]
        rel_v = vb * u[None, :] - w.ast_vel
        t = first_time_within(muzzle[None, :] - ap1, rel_v, aim_fraction * w.ast_rad, 2.5)
        ok = np.isfinite(t)
        if ok.any():                                    # asteroid and bullet must stay inside the map (no wrap / cull)
            ta = np.where(ok, t, 0.0)
            apos = ap1 + w.ast_vel * ta[:, None]
            bpos = muzzle[None, :] + vb * u[None, :] * ta[:, None]
            inside = lambda q: (q[:, 0] >= 0) & (q[:, 0] <= p.map_w) & (q[:, 1] >= 0) & (q[:, 1] <= p.map_h)
            ok &= inside(apos) & inside(bpos)
        if not ok.any():
            return None
        j = int(np.argmin(np.where(ok, t, np.inf)))

        # Safety check: Avoid shooting red penalty asteroids (and accidental collateral hits on red asteroids)
        is_pen = getattr(w, "ast_is_penalty", None)
        if is_pen is not None and len(is_pen) == w.n_ast:
            if is_pen[j]:
                return None
            # Also verify no red asteroid is in line of fire before target j
            if is_pen.any():
                pen_idxs = np.where(is_pen)[0]
                t_pen = t[pen_idxs]
                if (np.isfinite(t_pen) & (t_pen <= t[j])).any():
                    return None

        return j, float(t[j]), ap1

    def decide(self, w: World, thrust, turn, plan: Plan, risk: RiskState, planner, th, target, events) -> bool:
        """Fire iff the engine-exact ray test predicts a hit AND the MPC, re-run with this shot's fragments (same
        candidate sequences), still finds a safe sequence at comparable cost."""
        c, p = self.cfg, w.params
        if not w.can_fire or w.bullets_remaining == 0 or w.n_ast == 0:
            return False
        if w.respawning:   # firing would cancel invulnerability; only fire if nothing is close
            gap = np.linalg.norm(w.ast_pos - w.pos, axis=1) - w.ast_rad
            if gap.min() < c["respawn_fire_clear_radius"]:
                return False
        shot = self._shot(w, thrust, turn, risk.aim_fraction)
        if shot is None:
            return False
        j, t_hit, _ = shot
        if any(e[0] == j for e in events):             # asteroid already doomed by a bullet in flight
            return False
        dt = p.frame_dt
        k_hit = 1 + max(1, int(np.ceil(t_hit / dt - 1e-9))) if t_hit > 0 else 1
        u = np.array([np.cos(np.radians(w.heading + turn * dt)), np.sin(np.radians(w.heading + turn * dt))])
        shot_ev = (j, k_hit, c["bullet_speed"] * u, 1.0)
        alt = planner.evaluate(w, th, target, risk, plan.seqs, list(events) + [shot_ev])
        base_best = planner.best_of(plan.ev)
        alt_best = planner.best_of(alt)
        base_safe = not plan.ev["collides"][base_best]
        alt_safe = not alt["collides"][alt_best]
        if base_safe and not alt_safe:
            return False
        if not base_safe and not alt_safe:
            return True                                  # no safe option either way: score the hit
        return alt["cost"][alt_best] <= plan.ev["cost"][base_best] + c["shot_cost_tolerance"] / risk.multiplier
