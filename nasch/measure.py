"""Capacity measurements built on the simulator.

Two definitions are used in this project:

* Queue discharge rate (the main one). Release a long standing queue onto an empty road and
  count vehicles at a detector just downstream of the queue head. The slope of that cumulative
  count, N(t), is the rate at which a queue empties: the capacity of an active bottleneck. It
  is measured directly, so it does not depend on the size of an artificial ring.

* Peak of the ring fundamental diagram. This is the usual definition in CA studies, but with
  slow-to-start it depends on the ring size and run length (small rings keep a fast, metastable
  state alive). It is computed here only for comparison.
"""
from dataclasses import dataclass

import numpy as np

from .simulator import AV, queue_discharge_times, ring_flow

DISCHARGE_QUEUE = 2000           # vehicles in the standing queue (15 km of jam at 7.5 m per car)
DISCHARGE_DETECTOR = 50          # detector distance downstream of the queue head (cells)
DISCHARGE_WINDOW = (0.2, 0.8)    # count from the 20% to the 80% vehicle of the queue
TARGET_REL_SE = 0.005            # keep adding runs until the standard error is below 0.5%
SEED_BATCH, MAX_SEEDS = 16, 128


@dataclass
class Estimate:
    mean: float
    se: float
    runs: np.ndarray             # one value per independent run

    @property
    def rel_se(self):
        return self.se / self.mean if self.mean else float("nan")


def _estimate(values):
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else float("nan")
    return Estimate(float(values.mean()), float(se), values)


def discharge_rates(times, window=DISCHARGE_WINDOW):
    """Slope of the detector N-curve for each run: number of vehicles that crossed between the
    window[0] and window[1] fraction of the queue, divided by the time that took."""
    t = np.sort(np.asarray(times), axis=1)
    if (t < 0).any():
        raise ValueError("some vehicles never reached the detector; increase `steps`")
    n = t.shape[1]
    a, b = int(window[0] * n), int(window[1] * n)
    return (b - a) / (t[:, b] - t[:, a])


def queue_discharge_rate(vmax, human, av=AV, penetration=0.0, n_queue=DISCHARGE_QUEUE,
                         detector_offset=DISCHARGE_DETECTOR, target_rel_se=TARGET_REL_SE,
                         first_seed=0):
    """Queue discharge rate in vehicles per step, with its Monte Carlo standard error.

    Runs are added in batches of 16 until the relative standard error is below
    `target_rel_se` (or 128 runs). A deterministic case (no randomness) stops after one batch
    with a standard error of zero."""
    rates = np.empty(0)
    start = first_seed
    while rates.size < MAX_SEEDS:
        times = queue_discharge_times(n_queue, vmax, human, av=av, penetration=penetration,
                                      detector_offset=detector_offset,
                                      seeds=range(start, start + SEED_BATCH))
        rates = np.concatenate([rates, discharge_rates(times)])
        start += SEED_BATCH
        est = _estimate(rates)
        if not est.rel_se > target_rel_se:      # also stops when se == 0
            break
    return est


def default_densities(vmax):
    """0.05 up to at least 1/(vmax+1) in steps of 0.005; the original 0.05-0.255 grid for vmax >= 3."""
    return np.arange(0.05, max(0.255, 1.0 / (vmax + 1) + 0.005) + 1e-9, 0.005)


def fd_peak(L, vmax, human, av=AV, penetration=0.0, densities=None, steps=2000,
            transient=None, seeds=range(8)):
    """Peak of the ring fundamental diagram (the literature-style capacity), for comparison.

    Returns (peak flow, density at the peak, densities used, mean flow at each density).
    Note: the maximum of noisy averages is biased slightly upward.

    The default densities run from 0.05 to at least 1/(vmax+1), the deterministic model's critical
    density; random slowing, slow-to-start and extra gaps only move the peak to lower densities. (For
    vmax >= 3 this is the original 0.05-0.255 grid.) If the highest flow sits on the last density
    of the grid, the true peak may lie beyond it, so this raises instead of returning a truncated value.
    (Only the upper edge is guarded: with the 0.05 floor the peak cannot sit on the lowest density.)"""
    if densities is None:
        densities = default_densities(vmax)
    ns = np.unique(np.maximum(1, np.rint(np.asarray(densities) * L).astype(int)))
    if transient is None:
        transient = min(steps // 3, 2000)
    flow = ring_flow(L, ns, vmax, human, av=av, penetration=penetration, steps=steps,
                     transient=transient, seeds=seeds).mean(axis=1)
    k = int(np.argmax(flow))
    if k == ns.size - 1:
        raise ValueError(f"the highest flow is at the largest density tried ({ns[-1] / L:.3f}); "
                         "the peak may lie beyond it, so pass a wider `densities` range")
    return float(flow[k]), float(ns[k] / L), ns / L, flow
