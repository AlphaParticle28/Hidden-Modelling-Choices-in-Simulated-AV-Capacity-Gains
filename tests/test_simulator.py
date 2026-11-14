"""Checks that nasch/simulator.py is correct, using results that are known exactly.

Run with:  python -m pytest -q
"""
import numba
import numpy as np
import pytest

from nasch.simulator import AV, Driver, queue_discharge_times, ring_flow, trajectory
from tests import legacy_reference as legacy

CALIBRATED_HUMAN = Driver(p=0.05, p0=0.5, h=0)   # the single-station fit from the original project


# --- 1. exact results ---------------------------------------------------------------------------

@pytest.mark.parametrize("vmax, h, rhos", [
    (5, 0, [0.05, 0.10, 0.30, 0.50, 0.80]),
    (3, 1, [0.05, 0.12, 0.30, 0.40, 0.45]),
    (4, 2, [0.05, 0.10, 0.20, 0.25, 0.30]),
])
def test_no_randomness_gives_the_triangular_fd(vmax, h, rhos):
    """With p = 0 the model is deterministic and its fundamental diagram is exactly the triangle
    J = min(vmax*rho, 1 - (1+h)*rho), the same shape as the CTM triangle."""
    L = 1000
    ns = (np.array(rhos) * L).astype(int)
    flow = ring_flow(L, ns, vmax, Driver(p=0.0, h=h), steps=3000, transient=1000, seeds=range(2))
    rho = ns / L
    exact = np.maximum(0.0, np.minimum(vmax * rho, 1 - (1 + h) * rho))
    np.testing.assert_allclose(flow.mean(axis=1), exact, atol=1e-9)


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
def test_vmax_one_matches_the_exact_solution(p):
    """For vmax = 1 the stationary flow of NaSch is known in closed form
    (Schreckenberg, Schadschneider, Nagel & Ito, Phys. Rev. E 51, 1995):
        J = (1 - sqrt(1 - 4(1-p) rho (1-rho))) / 2"""
    L, seeds = 1000, range(8)
    ns = np.array([100, 300, 500, 700])
    flow = ring_flow(L, ns, 1, Driver(p=p), steps=4000, transient=1000, seeds=seeds)
    mean = flow.mean(axis=1)
    se = flow.std(axis=1, ddof=1) / np.sqrt(len(seeds))
    rho = ns / L
    exact = 0.5 * (1 - np.sqrt(1 - 4 * (1 - p) * rho * (1 - rho)))
    assert np.all(np.abs(mean - exact) < 4 * se + 5e-4)


@pytest.mark.parametrize("vmax, p, p0", [(5, 0.3, 0.6), (4, 0.05, 0.5), (3, 0.5, None)])
def test_free_flow_speed_is_vmax_minus_p(vmax, p, p0):
    """At very low density vehicles never interact, so the mean speed is vmax - p."""
    L, n = 5000, 50
    flow = ring_flow(L, [n], vmax, Driver(p=p, p0=p0), steps=6000, transient=1000, seeds=range(4))
    speed = flow.mean() / (n / L)
    assert abs(speed - (vmax - p)) / (vmax - p) < 0.005


# --- 2. bookkeeping: no collisions, no overtaking, nothing lost ---------------------------------

def _check_motion(P, V, vmax):
    assert np.all(np.diff(P, axis=1) > 0), "vehicles overtook or collided"
    assert np.array_equal(P[1:] - P[:-1], V[1:]), "positions and speeds disagree"
    assert V.min() >= 0 and V.max() <= vmax


def test_ring_conserves_vehicles_and_order():
    human = Driver(p=0.2, p0=0.6, h=1)
    L, n, vmax = 300, 90, 5
    P, V, cls = trajectory(n, vmax, human, penetration=0.3, L=L, steps=2000, seed=3)
    _check_motion(P, V, vmax)
    assert np.all(P[:, -1] - P[:, 0] < L), "a vehicle lapped the one ahead of it"
    assert cls.sum() == round(0.3 * n)
    # every vehicle respects its own class's buffer: v <= gap - h at the moment it moved
    gaps = np.concatenate([P[:-1, 1:] - P[:-1, :-1], (P[:-1, :1] + L - P[:-1, -1:])], axis=1) - 1
    hs = np.where(cls == 1, AV.h, human.h)
    assert np.all(V[1:] <= np.maximum(gaps - hs, 0))


def test_open_road_queue_discharges_completely_and_in_order():
    human = Driver(p=0.05, p0=0.5, h=0)
    P, V, _ = trajectory(200, 4, human, ring=False, steps=1500, seed=1)
    _check_motion(P, V, 4)
    times = queue_discharge_times(200, 4, human, detector_offset=50, seeds=range(3))
    assert np.all(times > 0), "some vehicles never reached the detector"
    assert np.all(np.diff(times[:, ::-1], axis=1) >= 0), "vehicles crossed out of order"


# --- 3. reproducibility and fair comparisons -----------------------------------------------------

def test_results_are_reproducible_and_independent_of_thread_count():
    args = dict(L=200, n_vehicles=[20, 60, 120], vmax=4, human=CALIBRATED_HUMAN,
                steps=600, transient=200, seeds=range(3))
    a = ring_flow(**args)
    b = ring_flow(**args)
    default = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        c = ring_flow(**args)
    finally:
        numba.set_num_threads(default)
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_av_parameters_are_unused_when_there_are_no_avs():
    args = dict(L=200, n_vehicles=[30, 90], vmax=4, human=CALIBRATED_HUMAN,
                steps=500, transient=100, seeds=range(2), penetration=0.0)
    a = ring_flow(av=Driver(p=0.0), **args)
    b = ring_flow(av=Driver(p=0.7, p0=0.9, h=2), **args)
    assert np.array_equal(a, b)


# --- 4. same physics as the original pure-Python code -------------------------------------------

LEGACY_CASES = {
    "calibrated human, no AVs": dict(vmax=4, human=Driver(0.05, 0.5, 0), av=AV, pen=0.0),
    "plain NaSch, buffer 1":    dict(vmax=5, human=Driver(0.25, None, 1), av=AV, pen=0.0),
    "50% AVs, human buffer 1":  dict(vmax=4, human=Driver(0.05, 0.5, 1), av=AV, pen=0.5),
}


@pytest.mark.parametrize("case", LEGACY_CASES)
def test_matches_the_original_implementation(case):
    """Same fundamental diagram as the original code, within Monte Carlo error. The original
    assigns AVs by coin flip and the new code assigns an exact share, so the mixed case also
    checks that this difference does not matter on average."""
    c = LEGACY_CASES[case]
    L, T, runs = 200, 500, 30
    ns = [10, 30, 50, 80, 120]
    new = ring_flow(L, ns, c["vmax"], c["human"], av=c["av"], penetration=c["pen"],
                    steps=T, transient=200, seeds=range(runs))
    np.random.seed(2026)
    old = np.array([[legacy.measure_steady_state_flow(
        L, n, c["vmax"], T, c["human"].p, pen=c["pen"], headway_HDV=c["human"].h,
        headway_AV=c["av"].h, thresh_AV=c["av"].p, chunk_size=25, transient_chunks=8,
        thresh0=c["human"].p0, thresh0_AV=c["av"].p0) for _ in range(runs)] for n in ns])
    diff = new.mean(axis=1) - old.mean(axis=1)
    se = np.sqrt(new.var(axis=1, ddof=1) / runs + old.var(axis=1, ddof=1) / runs)
    assert np.all(np.abs(diff) < 4 * se + 1e-9), f"differences in SE units: {diff / se}"
