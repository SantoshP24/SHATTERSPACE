"""Section 6: Model Predictive Control over action *sequences*.

Action primitive per MPC step = (thrust level, turn mode):
    thrust level: 0 = full reverse, 1 = coast (drag only), 2 = full forward
    turn mode   : 0 = -max, 1 = 0, 2 = +max, 3 = TRACK (dead-beat turn toward the lead point of the chosen target)
Each step holds the action for `mpc_step_frames` engine frames. The ship model is validated against the engine
(tests/test_dynamics_vs_engine.py). Collision checks run at engine-frame resolution on chord-interpolated paths and use
the engine's own rules: raw (non-wrapped) distances, no collisions while respawn-invulnerable.

Cost  J = collision + margin(+unresolved-threat weighting) + aim(positioning) + terminal + switch.
Sequences with a predicted collision are rejected outright whenever any collision-free sequence exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dynamics
from .events import child_velocities
from .geometry import aim_point_bearing_deg, closest_approach, wrap_angle_deg
from .risk import RiskState
from .threat import Threats
from .world import World, asteroid_tracks, other_ship_tracks

N_THRUST, N_TURN = 3, 4


@dataclass
class Plan:
    thrust_idx: int
    turn_mode: int
    seq: np.ndarray
    ship_frames: np.ndarray       # (F,2) planned wrapped ship positions at engine frames 1..F
    ast_idx: np.ndarray           # asteroid indices that were part of the plan's obstacle set
    cost: float
    n_safe: int
    n_candidates: int
    best_collides: bool
    seqs: np.ndarray = None
    ev: dict = None


class MPCPlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["rng_seed"])
        self.prev_seq = None

    # -- candidate generation -------------------------------------------------------------------------------
    def _candidates(self, H, has_target):
        N = self.cfg["candidate_action_sequences"]
        seqs = np.zeros((N, H, 2), dtype=np.int8)
        n_turn = N_TURN if has_target else N_TURN - 1
        prims = [(a, b) for a in range(N_THRUST) for b in range(n_turn)]
        for i, (a, b) in enumerate(prims[:N]):
            seqs[i, :, 0], seqs[i, :, 1] = a, b
        n = min(len(prims), N)
        if self.prev_seq is not None and n < N:
            shifted = np.concatenate([self.prev_seq[1:], self.prev_seq[-1:]], axis=0)
            if not has_target:
                shifted = shifted.copy()
                shifted[:, 1] = np.where(shifted[:, 1] == 3, 1, shifted[:, 1])
            seqs[n] = shifted
            n += 1
            n_mut = min(N - n, N // 4)
            for _ in range(n_mut):
                m = shifted.copy()
                flip = self.rng.random(H) < 0.25
                m[flip, 0] = self.rng.integers(0, N_THRUST, flip.sum())
                m[flip, 1] = self.rng.integers(0, n_turn, flip.sum())
                seqs[n] = m
                n += 1
        R = N - n
        if R > 0:   # random piecewise-constant sequences with up to 3 segments
            bounds = np.sort(self.rng.integers(1, H, size=(R, 2)), axis=1)
            s = np.arange(H)[None, :]
            seg = (s >= bounds[:, 0:1]).astype(int) + (s >= bounds[:, 1:2]).astype(int)
            th = self.rng.integers(0, N_THRUST, (R, 3))
            tu = self.rng.integers(0, n_turn, (R, 3))
            seqs[n:, :, 0] = np.take_along_axis(th, seg, axis=1)
            seqs[n:, :, 1] = np.take_along_axis(tu, seg, axis=1)
        return seqs

    # -- obstacle set -----------------------------------------------------------------------------------------
    def _obstacles(self, w: World, th: Threats, F, events):
        c, p = self.cfg, w.params
        dt = p.frame_dt
        ast_idx = np.zeros(0, dtype=int)
        tr = np.zeros((F, 0, 2)); rs = np.zeros(0); sf = np.zeros(0); wt = np.zeros(0); vel = np.zeros((0, 2))
        if w.n_ast:
            tracks = asteroid_tracks(w, F)
            times = np.arange(1, F + 1) * dt
            diff = np.abs(tracks - w.pos[None, None, :])                       # toroidal: the ship may wrap (teleport)
            diff = np.minimum(diff, np.array([p.map_w, p.map_h]) - diff)
            d0 = np.linalg.norm(diff, axis=-1)
            reach = np.minimum(p.max_speed * times,
                               abs(w.speed) * times + 0.5 * (p.thrust_max + p.drag) * times ** 2)
            rsum = p.radius + w.ast_rad
            gap = (d0 - reach[:, None] - rsum[None]).min(axis=0)
            rel = np.where(gap < 0.5 * rsum)[0]
            keep = list(rel[np.argsort(gap[rel])][: c["max_mpc_asteroids"]])
            keep += [e[0] for e in events if e[0] in set(rel.tolist()) and e[0] not in keep]   # parents of known fragments
            ast_idx = np.array(sorted(set(keep)), dtype=int)
            tr, rs, vel = tracks[:, ast_idx], rsum[ast_idx], w.ast_vel[ast_idx]
            sf = c["collision_safety_margin"] * rs
            wt = 1.0 + c["cost_threat_weight"] * th.T[ast_idx]
        M0 = tr.shape[1]
        active = np.ones((F, M0), bool)
        add = []   # (track(F,2), rsum, safe, weight, vel, active(F,))
        pos_of = {int(g): i for i, g in enumerate(ast_idx)}
        for (j, k_hit, ivel, imass) in events:
            if j not in pos_of or k_hit > F:
                continue
            i = pos_of[j]
            active[k_hit - 1:, i] = False
            size = int(w.ast_size[j])
            if size <= 1:
                continue
            origin = tr[k_hit - 1, i]
            rc = w.ast_rad[j] / size * (size - 1)
            rsc = p.radius + rc
            for vc in child_velocities(w.ast_vel[j], w.ast_mass[j], ivel, imass, c["split_angle_deg"], w.random_splits):
                steps = np.maximum(np.arange(1, F + 1) - k_hit, 0)[:, None] * dt
                add.append((np.mod(origin[None] + vc[None] * steps, [p.map_w, p.map_h]), rsc,
                            c["collision_safety_margin"] * rsc, wt[i], vc, np.arange(1, F + 1) >= k_hit))
        if add:
            tr = np.concatenate([tr, np.stack([a[0] for a in add], axis=1)], axis=1)
            rs = np.concatenate([rs, [a[1] for a in add]])
            sf = np.concatenate([sf, [a[2] for a in add]])
            wt = np.concatenate([wt, [a[3] for a in add]])
            vel = np.concatenate([vel, np.stack([a[4] for a in add])])
            active = np.concatenate([active, np.stack([a[5] for a in add], axis=1)], axis=1)
        if len(w.other_rad):
            ot = other_ship_tracks(w, F)
            ors = p.radius + w.other_rad
            tr = np.concatenate([tr, ot], axis=1)
            rs = np.concatenate([rs, ors])
            sf = np.concatenate([sf, c["collision_safety_margin"] * ors])
            wt = np.concatenate([wt, np.full(len(ors), 1.0 + 0.5 * c["cost_threat_weight"])])
            vel = np.concatenate([vel, w.other_vel])
            active = np.concatenate([active, np.ones((F, len(ors)), bool)], axis=1)
        return ast_idx, tr, rs, sf, wt, vel, active

    # -- action resolution (shared with the executed first action) --------------------------------------------
    def resolve_action(self, w: World, thrust_idx: int, turn_mode: int, target):
        p, c = w.params, self.cfg
        thrust = (p.thrust_min, 0.0, p.thrust_max)[thrust_idx]
        if turn_mode == 3 and target is not None:
            dt = p.frame_dt
            _, dist = dynamics.speed_step(np.array([w.speed]), np.array([thrust]), p, dt)
            u = np.array([np.cos(np.radians(w.heading)), np.sin(np.radians(w.heading))])
            pos_est = w.pos + dist[0] * u
            tp = np.mod(w.ast_pos[target.idx] + w.ast_vel[target.idx] * dt, [p.map_w, p.map_h])
            brg, _ = aim_point_bearing_deg(pos_est, tp, w.ast_vel[target.idx], p.radius, c["bullet_speed"])
            turn = float(np.clip(wrap_angle_deg(brg - w.heading) / dt, -p.turn_max, p.turn_max))
        else:
            turn = (-p.turn_max, 0.0, p.turn_max, 0.0)[turn_mode]
        return float(thrust), float(turn)

    # -- planning ---------------------------------------------------------------------------------------------
    def evaluate(self, w: World, th: Threats, target, risk: RiskState, seqs, events=()):
        """Simulate + score candidate sequences. Returns dict(cost, collides, frames, ast_idx)."""
        c, p = self.cfg, w.params
        H, f = c["planning_horizon_steps"], c["mpc_step_frames"]
        F, dt = H * f, p.frame_dt
        dtp = f * dt
        has_t = target is not None
        N = len(seqs)

        ast_idx, tracks, rsum, safe, wts, ovel, active = self._obstacles(w, th, F, events)
        M = tracks.shape[1]
        if has_t:
            ks = np.arange(1, H + 1)[:, None] * dtp
            tgt_steps = np.mod(w.ast_pos[target.idx][None] + w.ast_vel[target.idx][None] * ks, [p.map_w, p.map_h])
            tgt_vel = w.ast_vel[target.idx]

        thr_tab = np.array([p.thrust_min, 0.0, p.thrust_max])
        turn_tab = np.array([-p.turn_max, 0.0, p.turn_max, 0.0])
        P_unw = np.empty((N, H + 1, 2))
        P_unw[:, 0] = w.pos
        pos = np.tile(w.pos, (N, 1))
        speed = np.full(N, w.speed)
        head = np.full(N, w.heading)
        aim_err = np.zeros((N, H))
        for s in range(H):
            thrust = thr_tab[seqs[:, s, 0]]
            tm = seqs[:, s, 1]
            turn = turn_tab[tm]
            if has_t:
                v1, dist = dynamics.speed_step(speed, thrust, p, dtp)
                pos_est = pos + dist[:, None] * np.stack([np.cos(np.radians(head)), np.sin(np.radians(head))], -1)
                brg, _ = aim_point_bearing_deg(pos_est, tgt_steps[s][None, :], tgt_vel[None, :], p.radius, c["bullet_speed"])
                err = wrap_angle_deg(brg - head)
                turn = np.where(tm == 3, np.clip(err / dtp, -p.turn_max, p.turn_max), turn)
            pos, speed, head = dynamics.step(pos, speed, head, thrust, turn, p, dtp, wrap=False)
            P_unw[:, s + 1] = pos
            if has_t:
                aim_err[:, s] = np.abs(wrap_angle_deg(brg - head)) / 180.0

        # chord-interpolated engine-frame positions (unwrapped -> wrapped or bounded)
        a, b = P_unw[:, :-1, None, :], P_unw[:, 1:, None, :]
        m = (np.arange(1, f + 1) / f)[None, None, :, None]
        unwrapped_frames = (a + (b - a) * m).reshape(N, F, 2)
        wrap_bounds = c.get("wrap_boundaries", True)

        times = np.arange(1, F + 1) * dt
        vuln = (times >= w.respawn_left - 1e-9) if w.respawning else np.ones(F, bool)
        mult = risk.multiplier
        cost = np.zeros(N)

        if wrap_bounds:
            frames = np.mod(unwrapped_frames, [p.map_w, p.map_h])
            wall_col = np.zeros(N, bool)
        else:
            frames = unwrapped_frames
            # Hard wall collision check against arena boundaries
            pad = p.radius + 2.0
            wall_hit = (
                (frames[:, :, 0] < pad) |
                (frames[:, :, 0] > p.map_w - pad) |
                (frames[:, :, 1] < pad) |
                (frames[:, :, 1] > p.map_h - pad)
            )
            wall_col = wall_hit.any(axis=1)

            # Soft boundary proximity penalty guiding ship away from walls & corners
            b_margin = float(c.get("boundary_margin", 80.0))
            dist_left = np.maximum(0.0, (b_margin - frames[:, :, 0]) / b_margin)
            dist_right = np.maximum(0.0, (frames[:, :, 0] - (p.map_w - b_margin)) / b_margin)
            dist_top = np.maximum(0.0, (b_margin - frames[:, :, 1]) / b_margin)
            dist_bottom = np.maximum(0.0, (frames[:, :, 1] - (p.map_h - b_margin)) / b_margin)
            b_pen = dist_left + dist_right + dist_top + dist_bottom
            cost += 40.0 * mult * (b_pen ** 2).mean(axis=1)
            cost += c["cost_collision"] * mult * wall_col.astype(float)
        any_col = np.zeros((N, M), bool)
        if M:
            d = np.linalg.norm(frames[:, :, None, :] - tracks[None], axis=-1)           # (N,F,M)
            d = np.where(active[None], d, np.inf)
            col = (d < (rsum + c["collision_pad_m"])) & vuln[None, :, None]
            any_col = col.any(axis=1)
            t_first = (np.argmax(col, axis=1) + 1) * dt
            cost += c["cost_collision"] * mult * (any_col * (2.0 - t_first / (F * dt))).sum(axis=1)
            band = np.maximum(safe - rsum, 1e-6)
            intr = np.clip((safe - d) / band, 0.0, 1.5) * vuln[None, :, None]
            cost += c["cost_margin"] * mult * (wts * (intr.max(axis=1) ** 2 + 0.5 * (intr ** 2).mean(axis=1))).sum(axis=1)
            if vuln[-1]:                                                                   # terminal: coast lookahead
                ang = np.radians(head)
                sv = speed[:, None] * np.stack([np.cos(ang), np.sin(ang)], -1)
                r = tracks[-1][None] - frames[:, -1][:, None, :]
                _, dmin = closest_approach(r, ovel[None] - sv[:, None, :], c["terminal_lookahead_s"])
                it = np.clip((safe[None] - dmin) / band[None], 0.0, 1.5)
                cost += c["cost_terminal"] * mult * (wts[None] * it ** 2 * active[-1][None]).sum(axis=1)
        if has_t:
            g = c["aim_decay"] ** np.arange(H)
            cost += c["cost_aim"] * ((aim_err ** 2) * g).sum(axis=1) / g.sum()
        if self.prev_seq is not None and H > 1:
            cost += c["cost_switch"] * (seqs[:, 0, :] != self.prev_seq[1][None, :]).any(axis=1)
        return dict(cost=cost, collides=(any_col.any(axis=1) | wall_col), frames=frames, ast_idx=ast_idx)

    @staticmethod
    def best_of(ev):
        cost, col = ev["cost"], ev["collides"]
        sel = np.where(~col, cost, np.inf) if (~col).any() else cost
        return int(np.argmin(sel))

    def plan(self, w: World, th: Threats, target, risk: RiskState, events=()) -> Plan:
        seqs = self._candidates(self.cfg["planning_horizon_steps"], target is not None)
        ev = self.evaluate(w, th, target, risk, seqs, events)
        best = self.best_of(ev)
        self.prev_seq = seqs[best].copy()
        return Plan(int(seqs[best, 0, 0]), int(seqs[best, 0, 1]), seqs[best].copy(), ev["frames"][best], ev["ast_idx"],
                    float(ev["cost"][best]), int((~ev["collides"]).sum()), len(seqs), bool(ev["collides"][best]),
                    seqs, ev)
