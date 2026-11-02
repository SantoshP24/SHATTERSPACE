# Kessler MPC agent

Survival-first, threat-ranked, MPC-planned, adaptive-risk controller for ThalesGroup/kessler-game
(verified against `kesslergame` 2.4.0 source; mines unused).

## Use
```python
from kessler_mpc import KesslerMPCController
ctrl = KesslerMPCController()                 # or KesslerMPCController(adaptive_risk=False, candidate_action_sequences=200)
```
Pass `ctrl` to `KesslerGame.run(scenario=..., controllers=[ctrl])`. All parameters live in `kessler_mpc/config.py`.

## Layout
- `config.py` single config; `world.py` state adapter; `dynamics.py` ship model (validated vs engine)
- `threat.py` / `targeting.py` / `planner.py` (MPC) / `events.py` (bullets in flight, fragments) / `firecontrol.py` / `risk.py`
- `baselines.py` random, simple geometric, threat-targeting-without-MPC
- `evaluate.py` harness; `tests/` 20 tests (dynamics vs real engine, geometry edge cases, integration)

## Run
```
pip install kesslergame numpy pytest
python -m pytest tests -q
python evaluate.py --split tune  --episodes 20 --controllers all     # tuning seeds 1..
python evaluate.py --split final --episodes 100 --controllers all --out results.json   # seeds 1000..
```
`--workers N` parallelises episodes. If `tkinter` is missing, `engine_compat.py` stubs it for headless runs.

## Engine facts that shaped the design
Collisions ignore world wrapping (a wrapping ship teleports); firing cancels respawn invulnerability; bullets ignore
ship velocity and don't wrap; children inherit speed and fan out +/-15 deg.

## Results so far (10 episodes, final seeds 1000-1009, 6 asteroids, 60 s) - small sample, untuned
| Controller | Lives lost | Hits | Accuracy |
|---|---|---|---|
| Random | 3.00 | 22 | 0.36 |
| Simple geometric | 1.20 +/- 0.70 | 182 | 0.72 |
| Threat targeting, no MPC | 1.10 +/- 0.74 | 203 | 0.65 |
| Threat + MPC | 0.20 +/- 0.26 | 209 | 0.99 |
| MPC + adaptive risk | 0.20 +/- 0.26 | 206 | 0.99 |

Adaptive risk shows no measurable benefit yet. ~11-15 ms/step on 1 CPU. Run the 100-episode final evaluation before claiming anything.
