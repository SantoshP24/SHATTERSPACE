"""Vectorised ship motion model mirroring kesslergame/ship.py (validated against the real engine in tests).

Engine facts: thrust acts along heading; speed is a signed scalar; drag opposes motion; speed clamped to
+/-max_speed; drag cannot reverse the ship (it stops at 0 if |thrust| < drag); position wraps; heading mod 360.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShipParams:
    thrust_min: float
    thrust_max: float
    turn_max: float      # deg/s
    max_speed: float
    drag: float
    radius: float
    map_w: float
    map_h: float
    frame_dt: float


def speed_step(v0, thrust, p: ShipParams, dt):
    """Constant-action interval -> (v1, signed distance along heading)."""
    v0 = np.asarray(v0, float)
    thrust = np.asarray(thrust, float)
    moving = np.abs(v0) > 1e-12
    motion_sign = np.where(moving, np.sign(v0), np.sign(thrust))
    drag_acc = -p.drag * motion_sign
    net = thrust + drag_acc
    v_end = v0 + net * dt
    safe_net = np.where(net == 0.0, 1.0, net)

    opp = moving & (net * v0 < 0)
    t_stop = np.where(opp, -v0 / safe_net, np.inf)
    crosses = opp & (t_stop < dt)
    acc2 = np.where(np.abs(drag_acc) <= np.abs(thrust), thrust - drag_acc, 0.0)

    vcap = np.copysign(p.max_speed, v_end)
    to_cap = np.where(net != 0.0, (vcap - v0) / safe_net, np.inf)
    caps = (~opp) & (net != 0.0) & (to_cap >= 0.0) & (to_cap < dt)

    t_stop = np.where(crosses, t_stop, 0.0)
    t1 = np.where(crosses, t_stop, np.where(caps, to_cap, dt))
    v_mid = np.where(crosses, 0.0, np.where(caps, vcap, v_end))
    v1 = np.where(crosses, acc2 * (dt - t_stop), np.where(caps, vcap, v_end))
    v1 = np.clip(v1, -p.max_speed, p.max_speed)
    dist = 0.5 * (v0 + v_mid) * t1 + 0.5 * (v_mid + v1) * (dt - t1)
    return v1, dist


def step(pos, speed, heading, thrust, turn, p: ShipParams, dt, wrap=True):
    """Advance ships by dt with constant (thrust, turn)."""
    v1, dist = speed_step(speed, thrust, p, dt)
    th_mid = np.radians(heading + 0.5 * turn * dt)
    chord = np.sinc(np.radians(turn) * dt * 0.5 / np.pi)
    d = (dist * chord)[..., None] * np.stack([np.cos(th_mid), np.sin(th_mid)], axis=-1)
    new_pos = pos + d
    if wrap:
        new_pos = np.mod(new_pos, [p.map_w, p.map_h])
    return new_pos, v1, np.mod(heading + turn * dt, 360.0)


def coast_trajectory(pos, speed, heading, p: ShipParams, times):
    """Closed-form zero-thrust trajectory (drag only) -> (T,2) wrapped positions, (T,) signed speeds."""
    t = np.asarray(times, float)
    s = np.sign(speed)
    t_stop = abs(speed) / p.drag if p.drag > 0 else np.inf
    tc = np.minimum(t, t_stop)
    dist = speed * tc - 0.5 * p.drag * s * tc * tc
    spd = speed - p.drag * s * tc
    u = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))])
    traj = np.mod(np.asarray(pos, float)[None, :] + dist[:, None] * u[None, :], [p.map_w, p.map_h])
    return traj, spd
