"""Checks for nasch/baseline.py (triangular FD) and nasch/bottlenecks.py (bottleneck detection)."""
import numpy as np
import pandas as pd
import pytest

from nasch.baseline import fit_triangle, predict_triangle
from nasch.bottlenecks import (QUEUE_SPEED, active_mask, episodes, measure_pair, median_interval, ramp_distances,
                               ramps_between)


def triangle_data(vf=110.0, w=20.0, kj=140.0):
    k = np.r_[np.linspace(2, 15, 150), np.linspace(25, 80, 120)]
    q = np.minimum(vf * k, w * (kj - k))
    return k, q, vf * w * kj / (vf + w)


def test_noise_free_triangle_is_recovered():
    k, q, cap = triangle_data()
    fit = fit_triangle(k, q)
    assert fit["vf"] == pytest.approx(110.0, rel=1e-6)
    assert fit["jam_density"] == pytest.approx(140.0, rel=1e-6)
    assert fit["capacity"] == pytest.approx(cap, rel=1e-6)
    np.testing.assert_allclose(predict_triangle(fit, k), q, rtol=1e-6)


def test_fixed_jam_density_is_respected():
    k, q, _ = triangle_data(kj=133.3)
    fit = fit_triangle(k, q, k_jam=133.3)
    assert fit["jam_density"] == pytest.approx(133.3)
    assert predict_triangle(fit, np.array([200.0]))[0] == 0.0


def test_no_congested_branch_gives_none():
    k = np.linspace(1, 20, 100)
    assert fit_triangle(k, 100 * k) is None or fit_triangle(k, 100 * k)["capacity"] > 0


def test_five_of_seven_rule_and_no_splitting():
    n = 60
    up = np.full(n, 100.0)
    down = np.full(n, 100.0)
    up[20:40] = 30.0                      # queue upstream for 100 minutes ...
    down[20:40] = 90.0                    # ... free flow downstream
    up[30] = 100.0                        # one missed interval in the middle
    act = active_mask(up, down)
    assert act[20:40].sum() >= 19 and not act[:15].any() and not act[45:].any()
    assert len(episodes(act)) == 1        # the single gap does not split the event


def test_short_blips_are_not_bottlenecks():
    up = np.full(60, 100.0)
    down = np.full(60, 100.0)
    up[10:13] = QUEUE_SPEED - 5           # only 3 intervals
    down[10:13] = 100.0
    assert not active_mask(up, down).any()


def test_discharge_is_the_downstream_flow_while_active():
    slots = range(288)
    up = pd.DataFrame({"station_speed_kmh": 100.0, "flow_vph": 1500.0, "lane_avg_flow_vph": 1500.0}, index=slots)
    down = up.copy()
    up.loc[100:160, "station_speed_kmh"] = 30.0
    down.loc[100:160, "flow_vph"] = 1700.0
    down.loc[94:99, "flow_vph"] = 1900.0  # pre-breakdown flow just before onset
    eps = measure_pair(up, down)
    assert len(eps) == 1
    assert eps[0]["discharge_vph"] == pytest.approx(1700.0)
    assert eps[0]["prebreakdown_vph"] == pytest.approx(1900.0)


def test_episode_reports_observed_share_and_upstream_occupancy():
    slots = range(288)
    up = pd.DataFrame({"station_speed_kmh": 100.0, "flow_vph": 1500.0, "lane_avg_flow_vph": 1500.0,
                       "pct_observed": 100.0, "station_occupancy": 0.08}, index=slots)
    down = up.copy()
    up.loc[100:160, "station_speed_kmh"] = 30.0
    up.loc[100:160, "station_occupancy"] = 0.25
    down.loc[100:160, "pct_observed"] = 50.0
    ep = measure_pair(up, down)[0]
    assert ep["observed_up"] == pytest.approx(100.0)
    assert ep["observed_down"] == pytest.approx(50.0)
    assert ep["occupancy_up"] == pytest.approx(0.25)


def ramp_meta():
    # one northbound road: mainline at 0, 1 and 3 km; an on-ramp at 0.5 km, an off-ramp at 2 km,
    # an off-ramp on the southbound side (must be ignored)
    return pd.DataFrame({"station": [1, 2, 3, 10, 11, 12], "freeway": 101, "type": ["ML", "ML", "ML", "OR", "FR", "FR"],
                         "direction": ["N", "N", "N", "N", "N", "S"], "pos_km": [0.0, 1.0, 3.0, 0.5, 2.0, 1.5]})


def test_ramps_between_counts_only_the_same_road_and_kind():
    m = ramp_meta()
    assert ramps_between(m, 101, "N", 0.0, 1.0, "OR") == 1
    assert ramps_between(m, 101, "N", 0.0, 1.0, "FR") == 0     # the southbound off-ramp does not count
    assert ramps_between(m, 101, "N", 1.0, 3.0, "FR") == 1
    assert ramps_between(m, 101, "N", 2.05, 3.0, "FR") == 1    # within the tolerance of the upstream detector


def test_ramp_distances_look_the_right_way_and_are_capped():
    d = ramp_distances(ramp_meta(), cap_km=3.0).set_index("station")
    assert d.loc[2, "on_ramp_upstream_km"] == pytest.approx(0.5)
    assert d.loc[2, "off_ramp_downstream_km"] == pytest.approx(1.0)
    assert d.loc[1, "on_ramp_upstream_km"] == pytest.approx(3.0)       # no on-ramp upstream: capped
    assert d.loc[3, "off_ramp_downstream_km"] == pytest.approx(3.0)    # off-ramp is behind it: capped


def test_prebreakdown_window_stops_at_the_previous_episode():
    slots = range(288)
    up = pd.DataFrame({"station_speed_kmh": 100.0, "flow_vph": 1500.0, "lane_avg_flow_vph": 1500.0}, index=slots)
    down = up.copy()
    up.loc[100:129, "station_speed_kmh"] = 30.0     # first episode, slots 100-129
    up.loc[133:169, "station_speed_kmh"] = 30.0     # second episode after a 3-interval gap
    down.loc[100:129, "flow_vph"] = 2000.0          # flows inside the first episode must not leak
    eps = measure_pair(up, down)
    assert [(e["start"], e["end"]) for e in eps] == [(100, 130), (133, 170)]
    assert eps[1]["prebreakdown_vph"] == pytest.approx(1500.0)


def test_median_interval_is_exact_for_small_samples():
    lo, hi = median_interval([5, 1, 7, 3, 2, 6, 4])            # n = 7: P(all on one side) = 2/128 < 5%
    assert (lo, hi) == (1.0, 7.0)
    assert np.isnan(median_interval([1, 2, 3, 4, 5])[0])        # n = 5: no 95% interval exists
    lo, hi = median_interval(np.arange(1, 101))
    assert 39 <= lo <= 41 and 60 <= hi <= 62                    # n = 100: about the 40th and 61st values
