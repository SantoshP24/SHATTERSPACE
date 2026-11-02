"""Section 8 evaluation harness.

  python evaluate.py --split tune  --episodes 20 --controllers all
  python evaluate.py --split final --episodes 100 --controllers all --out results.json

Seeds: --split tune uses tuning_seed_start.., --split final uses random_seed_start.. (disjoint ranges).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np

from kessler_mpc.engine_compat import import_engine
import_engine()
from kesslergame import Scenario, TrainerEnvironment

from kessler_mpc.baselines import RandomController, SimpleGeometricController, ThreatNoMPCController
from kessler_mpc.config import CONFIG, make_config
from kessler_mpc.controller import KesslerMPCController

FACTORIES = {
    "random": lambda seed, cfg: RandomController(seed),
    "geometric": lambda seed, cfg: SimpleGeometricController(cfg),
    "threat_nompc": lambda seed, cfg: ThreatNoMPCController(cfg),
    "threat_mpc": lambda seed, cfg: KesslerMPCController({**cfg, "adaptive_risk": False}),
    "mpc_adaptive": lambda seed, cfg: KesslerMPCController({**cfg, "adaptive_risk": True}),
}


def run_episode(args):
    name, seed, n_ast, tlimit, cfg = args
    ctrl = FACTORIES[name](seed, cfg)
    sc = Scenario(name=f"{name}-{seed}", num_asteroids=n_ast, seed=seed, time_limit=tlimit, map_size=(1000, 800))
    score, _ = TrainerEnvironment().run(scenario=sc, controllers=[ctrl])
    t = score.teams[0]
    st = np.array(getattr(ctrl, "step_times", []) or [0.0]) * 1000.0
    dead = t.lives_remaining <= 0
    return dict(controller=name, seed=seed, sim_time=score.sim_time, survived=not dead, deaths=t.deaths,
                asteroids_hit=t.asteroids_hit, total_asteroids=t.total_asteroids, shots=t.shots_fired,
                accuracy=t.accuracy, stop=str(score.stop_reason.name), ms_mean=float(st.mean()),
                ms_p95=float(np.percentile(st, 95)), errors=int(getattr(ctrl, "error_count", 0)))


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def summarize(rows):
    out = {}
    for name in dict.fromkeys(r["controller"] for r in rows):
        R = [r for r in rows if r["controller"] == name]
        g = lambda k: np.array([r[k] for r in R], float)
        surv = g("sim_time")
        minutes = surv.sum() / 60.0
        out[name] = dict(
            n=len(R), survival_mean=surv.mean(), survival_std=surv.std(ddof=1) if len(R) > 1 else 0.0,
            survival_ci=ci95(surv), survival_p10=float(np.percentile(surv, 10)),
            lives_lost_mean=g("deaths").mean(), lives_lost_ci=ci95(g("deaths")),
            collisions_per_min=g("deaths").sum() / minutes if minutes else 0.0,
            hit_mean=g("asteroids_hit").mean(), hit_ci=ci95(g("asteroids_hit")),
            accuracy=g("accuracy").mean(), asteroids_destroyed_frac=(g("asteroids_hit") / g("total_asteroids")).mean(),
            ms_mean=g("ms_mean").mean(), ms_p95=g("ms_p95").mean(), errors=int(g("errors").sum()))
    return out


def print_table(S):
    hdr = f"{'controller':<14}{'n':>4}{'surv s (±CI)':>16}{'p10':>7}{'lives lost':>14}{'coll/min':>9}{'hits (±CI)':>14}{'acc':>6}{'ms/step':>9}{'p95ms':>7}"
    print(hdr)
    print("-" * len(hdr))
    for k, s in S.items():
        print(f"{k:<14}{s['n']:>4}{s['survival_mean']:>9.1f}±{s['survival_ci']:<5.1f}{s['survival_p10']:>7.1f}"
              f"{s['lives_lost_mean']:>8.2f}±{s['lives_lost_ci']:<4.2f}{s['collisions_per_min']:>9.2f}"
              f"{s['hit_mean']:>9.1f}±{s['hit_ci']:<4.1f}{s['accuracy']:>6.2f}{s['ms_mean']:>9.1f}{s['ms_p95']:>7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "final"], default="tune")
    ap.add_argument("--episodes", type=int, default=CONFIG["evaluation_episodes"])
    ap.add_argument("--controllers", default="all")
    ap.add_argument("--asteroids", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (python literals)")
    a = ap.parse_args()
    cfg = make_config(**{k: eval(v) for k, v in (kv.split("=", 1) for kv in a.set)})
    names = list(FACTORIES) if a.controllers == "all" else a.controllers.split(",")
    start = cfg["random_seed_start"] if a.split == "final" else cfg["tuning_seed_start"]
    jobs = [(n, start + i, a.asteroids, a.time_limit, cfg) for n in names for i in range(a.episodes)]
    t0 = time.time()
    if a.workers > 1:
        with Pool(a.workers) as pool:
            rows = pool.map(run_episode, jobs, chunksize=1)
    else:
        rows = [run_episode(j) for j in jobs]
    S = summarize(rows)
    print(f"\n{a.episodes} episodes/controller, split={a.split}, seeds {start}..{start + a.episodes - 1}, "
          f"{a.asteroids} size-4 asteroids, {a.time_limit:.0f}s limit, wall {time.time() - t0:.0f}s\n")
    print_table(S)
    if a.out:
        json.dump(dict(config=cfg, episodes=rows, summary=S), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
