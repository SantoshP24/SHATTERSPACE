"""Threat Assessment and Collision Prediction module.

Estimates for every debris object:
- Distance from the spaceship
- Relative velocity
- Time to closest approach (TCA / t_min)
- Minimum predicted separation (d_min)
- Collision probability and collision-risk score (C, U, G, T)
- Critical and urgent collision flags
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import CONFIG
from .dynamics import ShipParams
from .geometry import closest_approach
from .threat import Threats, assess
from .world import World


@dataclass
class DebrisThreatInfo:
    asteroid_index: int
    position: np.ndarray
    velocity: np.ndarray
    size: int
    radius: float
    distance: float
    relative_velocity: np.ndarray
    relative_speed: float
    time_to_closest_approach: float
    min_predicted_separation: float
    time_to_collision: float
    collision_probability: float  # C (0..1)
    urgency_score: float          # U (0..1)
    geometric_threat: float       # G (0..1)
    overall_threat_score: float   # T (0..1)
    is_critical: bool             # TTC <= critical_ttc_s (emergency override needed)
    is_urgent: bool               # TTC <= urgent_ttc_s


class ThreatEvaluator:
    """Evaluates collision hazards and prioritizes threats for live gameplay."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or CONFIG

    def assess_threats(self, world: World) -> Tuple[Threats, List[DebrisThreatInfo]]:
        """Run the comprehensive threat model and return structured per-debris threat metrics."""
        raw_threats = assess(world, self.cfg)
        threat_list: List[DebrisThreatInfo] = []

        n_ast = world.n_ast
        if n_ast == 0:
            return raw_threats, threat_list

        ship_pos = world.pos
        ship_vel = world.vel
        ship_rad = world.params.radius

        for i in range(n_ast):
            pos = world.ast_pos[i]
            vel = world.ast_vel[i]
            rad = float(world.ast_rad[i])
            size = int(world.ast_size[i])

            rel_pos = pos - ship_pos
            rel_vel = vel - ship_vel
            dist = float(np.linalg.norm(rel_pos))
            rel_speed = float(np.linalg.norm(rel_vel))

            t_min = float(raw_threats.t_min[i]) if i < len(raw_threats.t_min) else 0.0
            d_min = float(raw_threats.d_min[i]) if i < len(raw_threats.d_min) else dist
            ttc = float(raw_threats.ttc[i]) if i < len(raw_threats.ttc) else float("inf")

            c_score = float(raw_threats.C[i]) if i < len(raw_threats.C) else 0.0
            u_score = float(raw_threats.U[i]) if i < len(raw_threats.U) else 0.0
            g_score = float(raw_threats.G[i]) if i < len(raw_threats.G) else 0.0
            t_score = float(raw_threats.T[i]) if i < len(raw_threats.T) else 0.0

            is_crit = bool(raw_threats.critical[i]) if i < len(raw_threats.critical) else False
            is_urg = bool(raw_threats.urgent[i]) if i < len(raw_threats.urgent) else False

            threat_list.append(
                DebrisThreatInfo(
                    asteroid_index=i,
                    position=pos.copy(),
                    velocity=vel.copy(),
                    size=size,
                    radius=rad,
                    distance=dist,
                    relative_velocity=rel_vel.copy(),
                    relative_speed=rel_speed,
                    time_to_closest_approach=t_min,
                    min_predicted_separation=d_min,
                    time_to_collision=ttc,
                    collision_probability=c_score,
                    urgency_score=u_score,
                    geometric_threat=g_score,
                    overall_threat_score=t_score,
                    is_critical=is_crit,
                    is_urgent=is_urg,
                )
            )

        # Sort threats in descending order of overall threat score T
        threat_list.sort(key=lambda t: t.overall_threat_score, reverse=True)
        return raw_threats, threat_list
