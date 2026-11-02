"""Predicted asteroid-destruction events. An event = (asteroid index, frame k at which it is destroyed, impactor velocity,
impactor mass). Used for bullets already in flight and for the *candidate* shot, so the planner sees fragments coming."""
from __future__ import annotations

import numpy as np

from .geometry import first_time_within
from .world import World


def bullet_events(w: World, max_t=2.5):
    """Earliest asteroid each in-flight bullet will hit (bullet head vs asteroid circle, straight-line motion,
    both inside the map at impact). One bullet per asteroid (the earliest)."""
    if len(w.bul_pos) == 0 or w.n_ast == 0:
        return []
    p, dt = w.params, w.params.frame_dt
    r = w.bul_pos[:, None, :] - w.ast_pos[None, :, :]                       # (B,A,2)
    v = w.bul_vel[:, None, :] - w.ast_vel[None, :, :]
    t = first_time_within(r, v, w.ast_rad[None, :], max_t)                   # (B,A)
    ta = np.where(np.isfinite(t), t, 0.0)
    apos = w.ast_pos[None] + w.ast_vel[None] * ta[..., None]
    bpos = w.bul_pos[:, None, :] + w.bul_vel[:, None, :] * ta[..., None]
    inside = lambda q: (q[..., 0] >= 0) & (q[..., 0] <= p.map_w) & (q[..., 1] >= 0) & (q[..., 1] <= p.map_h)
    t = np.where(np.isfinite(t) & inside(apos) & inside(bpos), t, np.inf)
    ev, taken = [], set()
    for b in range(len(w.bul_pos)):
        j = int(np.argmin(t[b]))
        if np.isfinite(t[b, j]):
            ev.append((j, max(1, int(np.ceil(t[b, j] / dt - 1e-9))), w.bul_vel[b].copy(), float(w.bul_mass[b]) or 1.0))
    ev.sort(key=lambda e: e[1])
    out = []
    for e in ev:
        if e[0] not in taken:
            taken.add(e[0]); out.append(e)
    return out


def child_velocities(parent_vel, parent_mass, imp_vel, imp_mass, split_deg, random_splits):
    """Engine (asteroid.destruct): perfectly inelastic with the bullet, same speed for all children, fan of +/-split."""
    vf = (imp_mass * np.asarray(imp_vel) + parent_mass * np.asarray(parent_vel)) / (imp_mass + parent_mass)
    sp, th = float(np.linalg.norm(vf)), np.arctan2(vf[1], vf[0])
    sa = split_deg * (2.0 if random_splits else 1.0)          # random splits: offsets drawn in [0, 2*split]
    return [sp * np.array([np.cos(th + np.radians(a)), np.sin(th + np.radians(a))]) for a in (sa, 0.0, -sa)]
