"""Pure geometry helpers (no game dependencies)."""
from __future__ import annotations

import numpy as np


def wrap_angle_deg(a):
    """Wrap to [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def closest_approach(r, v, t_max):
    """Time/distance of closest approach under constant relative velocity.
    r = p_a - p_s, v = v_a - v_s, arrays (...,2). t* = -r.v/|v|^2 clamped to [0,t_max]; |v|=0 -> t*=0."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    vv = (v * v).sum(-1)
    rv = (r * v).sum(-1)
    moving = vv > 1e-12
    t = np.where(moving, -rv / np.where(moving, vv, 1.0), 0.0)
    t = np.clip(t, 0.0, t_max)
    d = np.linalg.norm(r + t[..., None] * v, axis=-1)
    return t, d


def first_time_within(r, v, radius, t_max):
    """Earliest t in [0,t_max] with |r+vt| <= radius, else inf."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    radius = np.asarray(radius, float)
    a = (v * v).sum(-1)
    b = 2.0 * (r * v).sum(-1)
    c = (r * r).sum(-1) - radius ** 2
    inside = c <= 0
    ok = (~inside) & (a > 1e-12)
    disc = b * b - 4 * a * c
    ok &= disc >= 0
    t1 = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2 * np.where(a > 1e-12, a, 1.0))
    ok &= (t1 >= 0) & (t1 <= t_max)
    out = np.where(ok, t1, np.inf)
    return np.where(inside, 0.0, out)


def lead_intercept(shooter, target_pos, target_vel, speed):
    """Smallest t>0 with |target_pos + target_vel t - shooter| = speed*t, else inf."""
    r = np.asarray(target_pos, float) - np.asarray(shooter, float)
    v = np.asarray(target_vel, float)
    a = (v * v).sum(-1) - speed * speed
    b = 2.0 * (r * v).sum(-1)
    c = (r * r).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lin = np.where(np.abs(b) > 1e-12, -c / b, np.inf)
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
    pos1, pos2 = t1 > 0, t2 > 0
    cand = np.where(pos1 & pos2, np.minimum(t1, t2), np.where(pos1, t1, np.where(pos2, t2, np.inf)))
    cand = np.where(np.isnan(cand), np.inf, cand)
    t_lin = np.where(t_lin > 0, t_lin, np.inf)
    return np.where(np.abs(a) < 1e-9, t_lin, cand)


def aim_point_bearing_deg(pos, tgt_pos, tgt_vel, muzzle_radius, bullet_speed):
    """Bearing (deg) and lead point for a bullet leaving the muzzle (pos + R*u) at bullet_speed.
    The muzzle lies on the centre->aim line, so the bearing is the same from centre or muzzle; the offset
    only shortens flight time. Raw (unwrapped) coordinates: engine bullets do not wrap."""
    aim = tgt_pos
    for _ in range(3):
        dist = np.linalg.norm(aim - pos, axis=-1)
        tau = np.maximum(dist - muzzle_radius, 0.0) / bullet_speed
        aim = tgt_pos + tgt_vel * tau[..., None]
    d = aim - pos
    return np.degrees(np.arctan2(d[..., 1], d[..., 0])), aim
