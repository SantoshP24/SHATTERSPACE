"""Comparison controllers (section 8). All share the engine action interface and the same scenario/seeds."""
from __future__ import annotations

import numpy as np

from .engine_compat import import_engine

KesslerController = import_engine().KesslerController

from .config import make_config
from .geometry import aim_point_bearing_deg, closest_approach, wrap_angle_deg
from .risk import AdaptiveRisk
from .targeting import TargetSelector
from .threat import assess
from .world import read_world


class RandomController(KesslerController):
    """Sanity check: random thrust/turn, fire at random."""
    name = "Random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self._ship_id = 0

    def actions(self, ship_state, game_state):
        return (float(self.rng.uniform(-480, 480)), float(self.rng.uniform(-180, 180)),
                bool(self.rng.random() < 0.5), False)


class SimpleGeometricController(KesslerController):
    """Nearest-asteroid lead targeting + reactive evasion (flee along the side-step direction of the most urgent
    closest approach). No planning, no threat ranking."""
    name = "SimpleGeometric"
    use_threat_targeting = False

    def __init__(self, config=None, **ov):
        self.cfg = make_config(**{**(config or {}), **ov})
        self.selector = TargetSelector(self.cfg)
        self._ship_id = 0

    def _target_index(self, w, th):
        d = np.linalg.norm(w.ast_pos - w.pos, axis=1) - w.ast_rad
        return int(np.argmin(d))

    def actions(self, ship_state, game_state):
        w = read_world(ship_state, game_state)
        if w.n_ast == 0:
            return 0.0, 0.0, False, False
        p, c = w.params, self.cfg
        th = assess(w, c)
        ti = self._target_index(w, th)
        if ti is None:
            return 0.0, 0.0, False, False
        brg, _ = aim_point_bearing_deg(w.pos, w.ast_pos[ti], w.ast_vel[ti], p.radius, c["bullet_speed"])
        err = float(wrap_angle_deg(brg - w.heading))
        turn = float(np.clip(err / p.frame_dt, -p.turn_max, p.turn_max))
        thrust = 0.0
        # reactive evasion
        j = int(np.argmin(np.where(np.isfinite(th.ttc), th.ttc, np.inf))) if np.isfinite(th.ttc).any() else None
        danger = j is not None and th.ttc[j] < c["urgent_ttc_s"] or bool((th.d_min < th.safe).any() and (th.T > 0.3).any())
        if danger:
            j = int(np.argmax(th.T))
            rel_v = w.ast_vel[j] - w.vel
            r = w.ast_pos[j] - w.pos
            n = np.array([-rel_v[1], rel_v[0]])
            n = n / (np.linalg.norm(n) + 1e-9)
            if np.dot(n, r) > 0:
                n = -n                                           # side-step away from the asteroid's path
            want = np.degrees(np.arctan2(n[1], n[0]))
            e = float(wrap_angle_deg(want - w.heading))
            back = abs(e) > 90
            e = float(wrap_angle_deg(e - 180)) if back else e
            turn = float(np.clip(e / p.frame_dt, -p.turn_max, p.turn_max))
            thrust = -p.thrust_max if back else p.thrust_max
        fire = bool(abs(err) < 3.0 and w.can_fire and not (w.respawning))
        return float(thrust), turn, fire, False


class ThreatNoMPCController(SimpleGeometricController):
    """Threat-ranked target selection (section 5) + the same reactive evasion; no MPC. Measures target selection."""
    name = "ThreatTargeting-noMPC"

    def __init__(self, config=None, **ov):
        super().__init__(config, **ov)
        self.risk = AdaptiveRisk(self.cfg)

    def _target_index(self, w, th):
        t = self.selector.select(w, th)
        return t.idx if t is not None else int(np.argmin(np.linalg.norm(w.ast_pos - w.pos, axis=1)))
