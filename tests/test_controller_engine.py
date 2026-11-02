"""Integration tests against the real engine."""
import numpy as np

from kessler_mpc.engine_compat import import_engine
import_engine()
from kesslergame import Scenario, TrainerEnvironment, KesslerController
from kessler_mpc.controller import KesslerMPCController


def run(ctrl, states, ship=None, T=6.0, **kw):
    sc = Scenario(name="t", asteroid_states=states, ship_states=[ship or {"position": (500.0, 400.0)}], time_limit=T, **kw)
    score, _ = TrainerEnvironment().run(scenario=sc, controllers=[ctrl])
    return score.teams[0], score


def test_kills_a_crossing_asteroid_without_dying():
    ctrl = KesslerMPCController(debug=True)
    t, _ = run(ctrl, [{"position": (200.0, 700.0), "speed": 80.0, "angle": 300.0, "size": 1}])
    assert t.asteroids_hit == 1 and t.deaths == 0 and t.bullets_hit == 1


def test_evades_head_on_big_asteroid():
    # big asteroid aimed straight at the ship; with no ammo it cannot be shot -> must dodge
    ctrl = KesslerMPCController(debug=True)
    t, _ = run(ctrl, [{"position": (900.0, 400.0), "speed": 120.0, "angle": 180.0, "size": 4}],
               T=5.0, ammo_limit_multiplier=0.0001)
    assert t.deaths == 0


def test_respawn_aware_no_fire_when_crowded_and_survives():
    ctrl = KesslerMPCController(debug=True)
    t, s = run(ctrl, [{"position": (480.0, 400.0), "speed": 0.0, "angle": 0.0, "size": 3}] +
               [{"position": (700.0, 100.0 + 120 * i), "speed": 60.0, "angle": 180.0, "size": 2} for i in range(4)],
               ship={"position": (500.0, 400.0), "lives": 3}, T=8.0)
    assert t.deaths <= 1                      # starts overlapping a big asteroid: at most the initial hit
    assert ctrl.error_count == 0


def test_runtime_per_step_is_practical():
    ctrl = KesslerMPCController(debug=True)
    sc = Scenario(name="t", num_asteroids=8, seed=5, time_limit=6.0)
    TrainerEnvironment().run(scenario=sc, controllers=[ctrl])
    st = np.array(ctrl.step_times) * 1000
    assert st.mean() < 40.0 and np.percentile(st, 95) < 80.0


def test_old_style_dict_state_adapter():
    from kessler_mpc.world import read_world
    ship = {"position": (10.0, 20.0), "velocity": (0.0, 0.0), "speed": 0.0, "heading": 90.0, "radius": 20.0, "id": 1,
            "is_respawning": False, "lives_remaining": 3, "can_fire": True, "bullets_remaining": -1,
            "thrust_range": (-480, 480), "turn_rate_range": (-180, 180), "max_speed": 240, "drag": 80, "respawn_time_left": 0.0}
    gs = {"map_size": (1000, 800), "frame_rate": 30, "time": 0.0, "ships": [ship],
          "asteroids": [{"position": (100.0, 100.0), "velocity": (1.0, 2.0), "size": 3, "mass": 400.0, "radius": 24.0}], "bullets": []}
    w = read_world(ship, gs)
    assert w.n_ast == 1 and w.ast_rad[0] == 24.0 and w.params.map_w == 1000.0
