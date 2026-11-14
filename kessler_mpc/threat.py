"""Section 4/5A: collision prediction and asteroid threat score T_i = 0.50 C + 0.35 U + 0.15 G."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import coast_trajectory
from .world import World, asteroid_tracks


@dataclass
class Threats:
    times: np.ndarray
    ship_traj: np.ndarray        # coasting ship trajectory (F,2)
    ship_spd: np.ndarray
    tracks: np.ndarray           # asteroid tracks (F,A,2)
    rsum: np.ndarray
    safe: np.ndarray
    d_min: np.ndarray
    t_min: np.ndarray
    ttc: np.ndarray              # first predicted collision time (inf if none)
    C: np.ndarray
    U: np.ndarray
    G: np.ndarray
    T: np.ndarray
    critical: np.ndarray
    urgent: np.ndarray


def assess(w: World, cfg) -> Threats:
    p = w.params
    dt = p.frame_dt
    H = cfg["prediction_horizon_s"]
    F = max(1, int(round(H / dt)))
    times = np.arange(1, F + 1) * dt
    traj, spd = coast_trajectory(w.pos, w.speed, w.heading, p, times)
    tracks = asteroid_tracks(w, F)
    rsum = p.radius + w.ast_rad
    safe = cfg["collision_safety_margin"] * rsum
    A = w.n_ast
    if A == 0:
        z = np.zeros(0)
        return Threats(times, traj, spd, tracks, z, z, z, z, z, z, z, z, z, z.astype(bool), z.astype(bool))
    d = np.linalg.norm(tracks - traj[:, None, :], axis=-1)            # (F,A)
    vuln = times >= (w.respawn_left - 1e-9) if w.respawning else np.ones(F, bool)
    d = np.where(vuln[:, None], d, np.inf)
    idx = np.argmin(d, axis=0)
    ar = np.arange(A)
    d_min, t_min = d[idx, ar], times[idx]
    col = d <= rsum[None]
    any_col = col.any(0)
    ttc = np.where(any_col, times[np.argmax(col, axis=0)], np.inf)

    d_eff = d_min - cfg["position_uncertainty_mps"] * t_min
    band = np.maximum(safe - rsum, 1e-6)
    C = np.clip((safe - d_eff) / band, 0.0, 1.0)
    U = np.where(any_col, np.clip(1.0 - ttc / H, 0.0, 1.0), 0.0)
    clearance = np.where(np.isfinite(d_min), d_min - rsum, np.inf)
    prox = np.clip(1.0 - np.maximum(clearance, 0.0) / (3.0 * rsum), 0.0, 1.0)
    depth = np.where(np.isfinite(d_min), np.clip((rsum - d_min) / rsum, 0.0, 1.0), 0.0)
    G = 0.6 * (w.ast_size / 4.0) * prox + 0.4 * depth
    T = (cfg["threat_weight_risk"] * C + cfg["threat_weight_urgency"] * U + cfg["threat_weight_geometry"] * G)
    return Threats(times, traj, spd, tracks, rsum, safe, d_min, t_min, ttc, C, U, G, T,
                   ttc <= cfg["critical_ttc_s"], ttc <= cfg["urgent_ttc_s"])
