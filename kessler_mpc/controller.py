"""KesslerMPCController: survival-first, threat-ranked, MPC-planned, adaptive-risk controller.

Per frame: read state -> assess threats -> select target -> MPC plan (first action executed, replan next frame)
-> fire control -> (thrust, turn_rate, fire, drop_mine). Mines are not used.
"""
from __future__ import annotations

import time
import warnings

from .engine_compat import import_engine

KesslerController = import_engine().KesslerController

from .config import make_config
from .events import bullet_events
from .firecontrol import FireControl
from .planner import MPCPlanner
from .risk import AdaptiveRisk
from .targeting import TargetSelector
from .threat import assess
from .world import read_world


class KesslerMPCController(KesslerController):
    def __init__(self, config: dict | None = None, **overrides):
        self.cfg = make_config(**{**(config or {}), **overrides})
        self.risk = AdaptiveRisk(self.cfg)
        self.selector = TargetSelector(self.cfg)
        self.planner = MPCPlanner(self.cfg)
        self.fire = FireControl(self.cfg)
        self.error_count = 0
        self.step_times: list[float] = []
        self.last_plan = None
        self._ship_id = 0

    @property
    def name(self) -> str:
        return "KesslerMPC" + ("-adaptive" if self.cfg["adaptive_risk"] else "")

    def actions(self, ship_state, game_state):
        t0 = time.perf_counter()
        try:
            out = self._act(ship_state, game_state)
        except Exception:                      # never crash the match; fall back to a drag-only coast
            if self.cfg["debug"]:
                raise
            self.error_count += 1
            warnings.warn("KesslerMPC step failed; coasting", RuntimeWarning)
            out = (0.0, 0.0, False, False)
        self.step_times.append(time.perf_counter() - t0)
        return out

    def _act(self, ship_state, game_state):
        w = read_world(ship_state, game_state)
        if w.n_ast == 0:
            return 0.0, 0.0, False, False
        risk = self.risk.update(w.lives)
        th = assess(w, self.cfg)
        target = self.selector.select(w, th)
        events = bullet_events(w)
        plan = self.planner.plan(w, th, target, risk, events)
        thrust, turn = self.planner.resolve_action(w, plan.thrust_idx, plan.turn_mode, target)
        fire = self.fire.decide(w, thrust, turn, plan, risk, self.planner, th, target, events)
        self.last_plan = plan
        return thrust, turn, bool(fire), False
