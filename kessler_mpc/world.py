"""Read-only state adapter: maps the verified kesslergame API to the internal representation.

Verified against kesslergame 2.4.0: ShipState (own ship, incl. thrust_range/turn_rate_range/max_speed/drag/
can_fire/is_respawning/respawn_time_left/lives_remaining) and GameState (compact lists:
asteroid=[x,y,vx,vy,size,mass,radius], ship=[x,y,vx,vy,speed,heading,mass,radius,id,team,is_respawning,lives,deaths]).
Falls back to dict-style access (ship_state['position'], game_state['asteroids'][i]['radius']) for other versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import ShipParams


@dataclass
class World:
    params: ShipParams
    pos: np.ndarray
    vel: np.ndarray
    speed: float
    heading: float
    lives: int
    respawning: bool
    respawn_left: float
    can_fire: bool
    bullets_remaining: int
    time: float
    ast_pos: np.ndarray
    ast_vel: np.ndarray
    ast_size: np.ndarray
    ast_rad: np.ndarray
    ast_mass: np.ndarray
    other_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    other_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    other_rad: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bul_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    bul_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    bul_mass: np.ndarray = field(default_factory=lambda: np.zeros(0))
    random_splits: bool = False

    @property
    def n_ast(self):
        return len(self.ast_rad)


def _get(obj, key, default=None):
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return default


def read_world(ship_state, game_state) -> World:
    map_w, map_h = (float(x) for x in game_state["map_size"])
    dt = 1.0 / float(_get(game_state, "frame_rate", 30.0))
    tr = _get(ship_state, "thrust_range", (-480.0, 480.0))
    tu = _get(ship_state, "turn_rate_range", (-180.0, 180.0))
    p = ShipParams(
        thrust_min=float(tr[0]), thrust_max=float(tr[1]), turn_max=float(tu[1]),
        max_speed=float(_get(ship_state, "max_speed", 240.0)), drag=float(_get(ship_state, "drag", 80.0)),
        radius=float(_get(ship_state, "radius", 20.0)), map_w=map_w, map_h=map_h, frame_dt=dt)
    my_id = ship_state["id"]

    compact = getattr(game_state, "compact", None)
    if compact is not None:
        ast = np.asarray(compact["asteroids"], float).reshape(-1, 7)
        ast_pos, ast_vel = ast[:, 0:2], ast[:, 2:4]
        ast_size, ast_mass, ast_rad = ast[:, 4], ast[:, 5], ast[:, 6]
        others = [s for s in compact["ships"] if int(s[8]) != my_id and not s[10]]
        o_pos = np.array([[s[0], s[1]] for s in others], float).reshape(-1, 2)
        o_vel = np.array([[s[2], s[3]] for s in others], float).reshape(-1, 2)
        o_rad = np.array([s[7] for s in others], float)
        bl = np.asarray(compact["bullets"], float).reshape(-1, 9)
        b_pos, b_vel, b_mass = bl[:, 0:2], bl[:, 2:4], bl[:, 7]
    else:
        al = list(game_state["asteroids"])
        ast_pos = np.array([a["position"] for a in al], float).reshape(-1, 2)
        ast_vel = np.array([a["velocity"] for a in al], float).reshape(-1, 2)
        ast_size = np.array([a["size"] for a in al], float)
        ast_mass = np.array([a["mass"] for a in al], float)
        ast_rad = np.array([a["radius"] for a in al], float)
        others = [s for s in game_state["ships"] if s["id"] != my_id and not _get(s, "is_respawning", False)]
        o_pos = np.array([s["position"] for s in others], float).reshape(-1, 2)
        o_vel = np.array([s["velocity"] for s in others], float).reshape(-1, 2)
        o_rad = np.array([s["radius"] for s in others], float)
        bl = list(_get(game_state, "bullets", []))
        b_pos = np.array([b["position"] for b in bl], float).reshape(-1, 2)
        b_vel = np.array([b["velocity"] for b in bl], float).reshape(-1, 2)
        b_mass = np.array([_get(b, "mass", 1.0) for b in bl], float)

    resp_left = float(_get(ship_state, "respawn_time_left", 0.0))
    return World(
        params=p,
        pos=np.array(ship_state["position"], float), vel=np.array(ship_state["velocity"], float),
        speed=float(ship_state["speed"]), heading=float(ship_state["heading"]),
        lives=int(ship_state["lives_remaining"]), respawning=bool(ship_state["is_respawning"]),
        respawn_left=resp_left, can_fire=bool(_get(ship_state, "can_fire", True)),
        bullets_remaining=int(_get(ship_state, "bullets_remaining", -1)),
        time=float(_get(game_state, "time", 0.0)),
        ast_pos=ast_pos, ast_vel=ast_vel, ast_size=ast_size.astype(int), ast_rad=ast_rad, ast_mass=ast_mass,
        other_pos=o_pos, other_vel=o_vel, other_rad=o_rad, bul_pos=b_pos, bul_vel=b_vel, bul_mass=b_mass,
        random_splits=bool(_get(game_state, "random_asteroid_splits", False)),
    )


def asteroid_tracks(w: World, n_frames: int, step_frames: int = 1):
    """Wrapped asteroid positions after frames 1..n_frames (engine: x = (x + v dt) % W). Shape (F, A, 2)."""
    k = np.arange(1, n_frames + 1, dtype=float)[:, None, None] * w.params.frame_dt
    return np.mod(w.ast_pos[None] + w.ast_vel[None] * k, [w.params.map_w, w.params.map_h])


def other_ship_tracks(w: World, n_frames: int):
    k = np.arange(1, n_frames + 1, dtype=float)[:, None, None] * w.params.frame_dt
    return np.mod(w.other_pos[None] + w.other_vel[None] * k, [w.params.map_w, w.params.map_h])
