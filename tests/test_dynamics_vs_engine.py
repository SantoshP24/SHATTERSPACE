"""Validate the planner's ship model and asteroid tracks against the real engine."""
import numpy as np

from kessler_mpc.engine_compat import import_engine
kg = import_engine()
from kesslergame import KesslerGame, Scenario, KesslerController, TrainerEnvironment
from kessler_mpc import dynamics
from kessler_mpc.world import read_world


class Scripted(KesslerController):
    """Replays actions and logs the state *before* each action is applied."""
    name = "scripted"

    def __init__(self, actions):
        self.script = actions
        self.i = 0
        self.log = []

    def actions(self, ship_state, game_state):
        w = read_world(ship_state, game_state)
        self.log.append(w)
        a = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return a[0], a[1], False, False


def run_script(script, pos=(500.0, 400.0), heading=90.0, frames=None):
    ctrl = Scripted(script)
    sc = Scenario(name="t", asteroid_states=[{"position": (50.0, 50.0), "speed": 0.0, "size": 1, "angle": 0.0}],
                  ship_states=[{"position": pos, "angle": heading, "lives": 3}],
                  time_limit=(frames or len(script)) / 30.0)
    game = TrainerEnvironment()
    game.run(scenario=sc, controllers=[ctrl])
    return ctrl.log


def predict(w0, script, n):
    p = w0.params
    pos, sp, hd = w0.pos[None].copy(), np.array([w0.speed]), np.array([w0.heading])
    out = []
    for k in range(n):
        th, tu = script[min(k, len(script) - 1)]
        pos, sp, hd = dynamics.step(pos, sp, hd, np.array([th]), np.array([tu]), p, p.frame_dt)
        out.append((pos[0].copy(), sp[0], hd[0]))
    return out


def check(script, n=60, tol=1.0, **kw):
    log = run_script(script, frames=n + 1, **kw)
    pred = predict(log[0], script, n)
    errs = []
    for k in range(n):
        actual = log[k + 1]
        e = np.linalg.norm(pred[k][0] - actual.pos)
        errs.append(e)
        assert abs(pred[k][1] - actual.speed) < 1e-6 * 240 + 1e-3, (k, pred[k][1], actual.speed)
        dh = (pred[k][2] - actual.heading + 180) % 360 - 180
        assert abs(dh) < 1e-6, (k, dh)
    assert max(errs) < tol, max(errs)
    return max(errs)


def test_full_thrust_straight():
    check([(480.0, 0.0)] * 60)


def test_turn_and_thrust():
    check([(480.0, 180.0)] * 30 + [(-480.0, -90.0)] * 30)


def test_coast_and_reverse():
    check([(480.0, 0.0)] * 20 + [(0.0, 0.0)] * 40 + [(-480.0, 0.0)] * 40, n=100)


def test_weak_thrust_below_drag():
    check([(480.0, 0.0)] * 10 + [(30.0, 45.0)] * 40)


def test_wrap_around_and_random():
    rng = np.random.default_rng(3)
    script = [(float(rng.choice([-480, 0, 480])), float(rng.choice([-180, 0, 180]))) for _ in range(40)]
    script = [s for s in script for _ in range(3)]
    check(script, n=110, pos=(990.0, 790.0))


def test_asteroid_tracks_match_engine():
    from kessler_mpc.world import asteroid_tracks
    states = [{"position": (990.0, 5.0), "speed": 120.0, "angle": 20.0, "size": 4},
              {"position": (10.0, 790.0), "speed": 90.0, "angle": 250.0, "size": 2}]
    ctrl = Scripted([(0.0, 0.0)])
    sc = Scenario(name="t", asteroid_states=states, ship_states=[{"position": (500.0, 400.0)}], time_limit=2.0)
    TrainerEnvironment().run(scenario=sc, controllers=[ctrl])
    w0 = ctrl.log[0]
    tracks = asteroid_tracks(w0, 40)
    for k in range(40):
        np.testing.assert_allclose(tracks[k], ctrl.log[k + 1].ast_pos, atol=1e-6)
