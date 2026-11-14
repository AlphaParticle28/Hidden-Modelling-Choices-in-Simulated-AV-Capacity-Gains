"""Checks for nasch/measure.py."""
import numpy as np
import pytest

from nasch.measure import default_densities, discharge_rates, fd_peak, queue_discharge_rate
from nasch.simulator import Driver


def test_discharge_rate_is_count_over_time():
    # 100 vehicles crossing exactly every 2 steps -> 0.5 vehicles per step
    times = np.arange(1, 201, 2)[None, ::-1]          # head (last index) crosses first
    assert discharge_rates(times) == pytest.approx(0.5)


def test_vehicles_that_never_arrive_are_an_error():
    times = np.array([[5, 4, 3, -1]])
    with pytest.raises(ValueError):
        discharge_rates(times)


@pytest.mark.parametrize("vmax, h", [(3, 0), (4, 0), (5, 0), (4, 1), (5, 2)])
def test_deterministic_queue_discharges_at_vmax_over_vmax_plus_one_plus_h(vmax, h):
    """Without randomness each car starts one step after its leader and ends up (vmax + 1 + h)
    cells behind it at full speed, so the queue empties at exactly vmax / (vmax + 1 + h)."""
    est = queue_discharge_rate(vmax, Driver(p=0.0, h=h), n_queue=400)
    assert est.se == 0.0
    assert est.mean == pytest.approx(vmax / (vmax + 1 + h), abs=1e-12)


def test_all_av_traffic_matches_the_deterministic_rate():
    human = Driver(p=0.05, p0=0.5, h=0)
    est = queue_discharge_rate(4, human, penetration=1.0, n_queue=400)
    assert est.mean == pytest.approx(0.8, abs=1e-12)


def test_stochastic_estimate_reaches_target_precision():
    est = queue_discharge_rate(4, Driver(p=0.05, p0=0.5, h=0), n_queue=1000, target_rel_se=0.01)
    assert est.rel_se < 0.01
    assert 0.40 < est.mean < 0.49          # human drivers discharge well below the AV rate of 0.8


def test_detector_position_does_not_change_the_rate():
    human = Driver(p=0.05, p0=0.5, h=0)
    near = queue_discharge_rate(4, human, n_queue=1000, detector_offset=10)
    far = queue_discharge_rate(4, human, n_queue=1000, detector_offset=300)
    assert abs(near.mean - far.mean) < 4 * np.hypot(near.se, far.se)


def test_fd_peak_of_deterministic_model_is_the_triangle_top():
    peak, rho, _, _ = fd_peak(1000, 4, Driver(p=0.0), densities=np.arange(0.1, 0.31, 0.01),
                              steps=3000, seeds=range(2))
    assert peak == pytest.approx(0.8, abs=1e-9)
    assert rho == pytest.approx(0.2, abs=1e-9)


def test_fd_peak_default_range_reaches_the_peak_for_small_vmax():
    peak, rho, _, _ = fd_peak(1000, 1, Driver(p=0.0), steps=3000, seeds=range(2))
    assert peak == pytest.approx(0.5, abs=1e-9)
    assert rho == pytest.approx(0.5, abs=1e-9)


def test_fd_peak_default_range_is_unchanged_for_vmax_3_and_up():
    for vmax in (3, 4, 5, 6):
        np.testing.assert_allclose(default_densities(vmax), np.arange(0.05, 0.26, 0.005), atol=1e-12)
    assert default_densities(1)[-1] >= 0.5 and default_densities(2)[-1] >= 1 / 3


def test_fd_peak_on_the_grid_edge_raises():
    with pytest.raises(ValueError, match="largest density"):
        fd_peak(1000, 4, Driver(p=0.0), densities=np.arange(0.05, 0.16, 0.01), steps=500, seeds=range(1))
