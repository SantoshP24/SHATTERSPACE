"""Adaptive risk tolerance from remaining lives (engine exposes lives_remaining but not the initial count)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskState:
    level: int            # 0 normal, 1 cautious, 2 most conservative
    multiplier: float     # scales collision / margin / fragment-risk costs only (not every penalty)
    aim_fraction: float   # fraction of asteroid radius a shot must hit (higher confidence when lives are low)
    fraction: float


class AdaptiveRisk:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lmax = int(cfg["initial_lives"] or 0)

    def update(self, lives: int, health: Optional[float] = None) -> RiskState:
        c = self.cfg
        self.lmax = max(self.lmax, lives, 1)
        q_lives = lives / self.lmax
        if health is not None:
            q_health = max(0.0, min(1.0, health / 100.0))
            q = min(q_lives, q_health)
        else:
            q = q_lives
        fr = c["aim_radius_fraction"]
        if not c["adaptive_risk"]:
            return RiskState(0, 1.0, fr[0], q)
        tol = c["lives_fraction_tolerance"]
        mult = c["low_lives_risk_multiplier"]
        if q <= c["low_lives_fraction"] + tol:
            return RiskState(2, mult, fr[2], q)
        if q <= c["mid_lives_fraction"] + tol:
            return RiskState(1, 1.0 + 0.5 * (mult - 1.0), fr[1], q)
        return RiskState(0, 1.0, fr[0], q)
