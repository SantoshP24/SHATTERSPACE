import numpy as np
import pytest

from kessler_mpc.config import make_config
from kessler_mpc.geometry import closest_approach, first_time_within, lead_intercept, wrap_angle_deg
from kessler_mpc.risk import AdaptiveRisk


# ---- spec section 4: collision prediction -----------------------------------------------------------------
def test_closest_approach_head_on():
    t, d = closest_approach(np.array([100.0, 0.0]), np.array([-50.0, 0.0]), 10.0)
    assert t == pytest.approx(2.0) and d == pytest.approx(0.0)


def test_closest_approach_zero_relative_velocity():
    t, d = closest_approach(np.array([30.0, 40.0]), np.array([0.0, 0.0]), 3.0)
    assert t == 0.0 and d == pytest.approx(50.0)


def test_closest_approach_clamped_to_horizon_and_diverging():
    t, d = closest_approach(np.array([100.0, 0.0]), np.array([-10.0, 0.0]), 3.0)   # would meet at t=10
    assert t == pytest.approx(3.0) and d == pytest.approx(70.0)
    t, d = closest_approach(np.array([100.0, 0.0]), np.array([10.0, 0.0]), 3.0)    # moving apart
    assert t == 0.0 and d == pytest.approx(100.0)


def test_first_time_within():
    assert first_time_within(np.array([100.0, 0]), np.array([-50.0, 0]), 20.0, 5.0) == pytest.approx(1.6)
    assert first_time_within(np.array([5.0, 0]), np.array([1.0, 0]), 20.0, 5.0) == 0.0
    assert np.isinf(first_time_within(np.array([100.0, 100]), np.array([1.0, 0]), 20.0, 5.0))


def test_lead_intercept():
    t = lead_intercept(np.zeros(2), np.array([400.0, 0]), np.array([0.0, 100.0]), 800.0)
    hit = np.array([400.0, 100.0 * t])
    assert np.linalg.norm(hit) == pytest.approx(800.0 * t, rel=1e-9)
    assert np.isinf(lead_intercept(np.zeros(2), np.array([400.0, 0]), np.array([900.0, 0.0]), 800.0))  # outruns bullet


def test_wrap_angle():
    assert wrap_angle_deg(190.0) == pytest.approx(-170.0) and wrap_angle_deg(-190.0) == pytest.approx(170.0)


# ---- section 7: adaptive risk -----------------------------------------------------------------------------
def test_adaptive_risk_levels_with_three_lives():
    r = AdaptiveRisk(make_config())
    assert [r.update(L).level for L in (3, 2, 1)] == [0, 1, 2]       # 1/3 counts as "<= 33%" despite 0.3333 > 0.33
    assert r.update(1).multiplier == 2.0 and r.update(2).multiplier == 1.5 and r.update(3).multiplier == 1.0


def test_adaptive_risk_off():
    r = AdaptiveRisk(make_config(adaptive_risk=False))
    assert all(r.update(L).multiplier == 1.0 for L in (3, 2, 1))


def test_unknown_config_key_rejected():
    with pytest.raises(KeyError):
        make_config(nonexistent=1)
