"""Calibrating the human-driver parameters against detector data.

The idea in one line: simulate the fundamental diagram once for every candidate parameter set
(the "library"), then for each station-day find the parameter set and time step that best match the
observed 5-minute (density, flow) points.

Physical units enter only as scale factors: a ring density rho (vehicles per 7.5 m cell) is
k = rho / 0.0075 veh/km, and a flow J (vehicles per step) is q = 3600 J / dt veh/h. So one library
in cells and steps serves every station and every time step dt, and dt can be fitted without any
extra simulation.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .measure import queue_discharge_rate
from .simulator import Driver, ring_flow

CELL_KM = 0.0075                                                  # 7.5 m cells

# ---- parameter grid (fixed in the analysis plan) ----
VMAX_VALUES = (3, 4, 5, 6)
P_VALUES = np.round(np.arange(0.0, 0.5001, 0.025), 3)
P0_VALUES = np.round(np.arange(0.1, 0.9001, 0.1), 1)
H_VALUES = (0, 1, 2)
DT_GRID = np.round(np.arange(0.70, 1.4001, 0.02), 2)             # seconds per step

# ---- library settings ----
RHO_LOW = np.round(np.arange(0.005, 0.3001, 0.005), 3)           # near the peak: big rings
RHO_HIGH = np.round(np.r_[np.arange(0.32, 0.9801, 0.03), 1.0], 3)
L_LOW, L_HIGH = 2000, 1000
STEPS, TRANSIENT = 3000, 1000
LIBRARY_SEEDS = range(4)


def parameter_grid():
    """All (vmax, p, p0, h) combinations. p0 = NaN means no slow-to-start; otherwise p0 > p."""
    rows = []
    for vmax in VMAX_VALUES:
        for h in H_VALUES:
            for p in P_VALUES:
                rows.append((vmax, p, np.nan, h))
                for p0 in P0_VALUES:
                    if p0 > p + 1e-9:
                        rows.append((vmax, p, p0, h))
    return pd.DataFrame(rows, columns=["vmax", "p", "p0", "h"])


def driver(row):
    p0 = None if np.isnan(row["p0"]) else float(row["p0"])
    return Driver(p=float(row["p"]), p0=p0, h=int(row["h"]))


def library_densities():
    return np.r_[RHO_LOW, RHO_HIGH]


def simulate_fd(row, seeds=LIBRARY_SEEDS):
    """Ring flow (vehicles per step) at every library density, one column per seed."""
    vmax, human = int(row["vmax"]), driver(row)
    low = ring_flow(L_LOW, np.rint(RHO_LOW * L_LOW).astype(int), vmax, human,
                    steps=STEPS, transient=TRANSIENT, seeds=seeds)
    high = ring_flow(L_HIGH, np.rint(RHO_HIGH * L_HIGH).astype(int), vmax, human,
                     steps=STEPS, transient=TRANSIENT, seeds=seeds)
    return np.vstack([low, high])


def human_discharge(row):
    est = queue_discharge_rate(int(row["vmax"]), driver(row))
    return est.mean, est.se, est.runs.size


def av_gain(vmax, human_discharge_rate):
    """Capacity gain from 0% to 100% AVs. All-AV queues discharge at exactly vmax / (vmax + 1)."""
    return (vmax / (vmax + 1.0)) / human_discharge_rate - 1.0


# --------------------------------------------------------------------------- fitting

@dataclass
class Library:
    grid: pd.DataFrame          # one row per parameter set
    rho: np.ndarray             # library densities, vehicles per cell
    flow: np.ndarray            # mean ring flow, shape (n_params, n_rho), vehicles per step

    def subset(self, idx):
        return Library(self.grid.iloc[idx].reset_index(drop=True), self.rho, self.flow[idx])


def model_flow(lib, rho_obs):
    """Model flow (vehicles per step) at each observed density for every parameter set:
    shape (n_obs, n_params). Linear interpolation; zero flow at zero density and beyond the
    last library density."""
    grid = np.r_[0.0, lib.rho]
    flow = np.hstack([np.zeros((lib.flow.shape[0], 1)), lib.flow])
    x = np.clip(np.asarray(rho_obs, float), 0.0, None)
    i = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
    w = (x - grid[i]) / (grid[i + 1] - grid[i])
    J = flow[:, i] * (1 - w) + flow[:, i + 1] * w            # (n_params, n_obs)
    J[:, x > grid[-1]] = 0.0
    return J.T


def geh(m, c):
    """GEH = sqrt(2 (m - c)^2 / (m + c)), the standard traffic-count error score."""
    m, c = np.asarray(m, float), np.asarray(c, float)
    s = m + c
    return np.sqrt(np.where(s > 0, 2.0 * (m - c) ** 2 / np.where(s > 0, s, 1.0), 0.0))


def geh_matrix(J, q_obs, dt_grid=DT_GRID):
    """GEH of every interval for every (parameter set, dt): shape (n_obs, n_params * n_dt),
    column = param_index * n_dt + dt_index."""
    J = np.asarray(J, np.float32)
    q = np.asarray(q_obs, np.float32)[:, None]
    G = np.empty((J.shape[0], J.shape[1], len(dt_grid)), np.float32)
    for d, dt in enumerate(dt_grid):                 # one dt at a time keeps memory small
        m = J * np.float32(3600.0 / dt)
        s = m + q
        G[:, :, d] = np.sqrt(2.0 * (m - q) ** 2 / np.where(s > 0, s, 1.0)) * (s > 0)
    return G.reshape(J.shape[0], -1)


def best_columns(G, weights):
    """Index of the best (parameter set, dt) column for each row of `weights` (n_rep, n_obs):
    minimises the weighted mean GEH. Also returns the minimum objective values."""
    W = np.atleast_2d(np.asarray(weights, np.float32))
    obj = (W @ G) / W.sum(axis=1, keepdims=True)
    best = obj.argmin(axis=1)
    return best, obj[np.arange(obj.shape[0]), best]


def split_column(col, n_dt=len(DT_GRID)):
    return np.asarray(col) // n_dt, np.asarray(col) % n_dt


def holdout_masks(times):
    """Test intervals = every 4th hour (03, 07, 11, 15, 19, 23 h); the rest are for training."""
    hour = np.array([int(t[:2]) for t in times])
    test = hour % 4 == 3
    return ~test, test


def load_library(path):
    z = np.load(path)
    grid = pd.DataFrame({"vmax": z["vmax"], "p": z["p"], "p0": z["p0"], "h": z["h"]})
    return Library(grid, z["rho"], z["flow"].mean(axis=2))


def fit_station_day(lib, discharge, k, q, times, n_boot=500, block=12, seed=0,
                    dt_grid=DT_GRID):
    """Calibrate one station-day.

    lib        Library (one row per parameter set)
    discharge  human queue discharge rate of each library parameter set (vehicles per step)
    k, q       observed density (veh/km) and flow (veh/h), in time order
    times      "HH:MM" of each interval (used for the held-out hours)

    Returns (summary dict, bootstrap column indices). Columns index (parameter set, dt) pairs,
    see `split_column`.
    """
    k, q = np.asarray(k, float), np.asarray(q, float)
    dt_grid = np.asarray(dt_grid, float)               # a list or tuple works too
    n, n_dt = k.size, len(dt_grid)
    G = geh_matrix(model_flow(lib, k * CELL_KM), q, dt_grid)
    vmax = lib.grid["vmax"].to_numpy()
    gain = av_gain(vmax, np.asarray(discharge, float))

    # 1. validation: fit on training hours, score the held-out hours
    train, test = holdout_masks(times)
    col_tr, _ = best_columns(G, train)
    test_geh = G[test, col_tr[0]]

    # 2. final fit on the whole day
    ones = np.ones(n, np.float32)
    mean_geh = (ones @ G) / n                                # every (parameter set, dt)
    col = int(mean_geh.argmin())
    i, d = split_column(col, n_dt)

    # 3. h: best achievable fit when h is forced to 0, 1 or 2
    h_col = np.repeat(lib.grid["h"].to_numpy(), n_dt)
    best_by_h = {hv: float(mean_geh[h_col == hv].min()) if (h_col == hv).any() else np.nan
                 for hv in (0, 1, 2)}

    # 4. bootstrap: refit on block-resampled days
    rng = np.random.default_rng(seed)
    W = block_bootstrap_weights(n, block, n_boot, rng)
    boot = np.concatenate([best_columns(G, W[s:s + 100])[0] for s in range(0, n_boot, 100)])
    bi, bd = split_column(boot, n_dt)
    g_boot = gain[bi]
    cap_boot = 3600.0 * np.asarray(discharge, float)[bi] / np.asarray(dt_grid)[bd]

    row = lib.grid.iloc[int(i)]
    dt = float(dt_grid[d])
    out = dict(
        intervals=n, param_index=int(i), vmax=int(row["vmax"]), p=float(row["p"]), p0=float(row["p0"]),
        h=int(row["h"]), dt=dt,
        geh_all=float(mean_geh[col]), pct_geh_below_5=float((G[:, col] < 5).mean() * 100),
        geh_train=float(G[train, col_tr[0]].mean()), geh_test=float(test_geh.mean()),
        pct_test_below_5=float((test_geh < 5).mean() * 100), test_intervals=int(test.sum()),
        geh_best_h0=best_by_h[0], geh_best_h1=best_by_h[1], geh_best_h2=best_by_h[2],
        human_discharge=float(discharge[int(i)]),
        capacity_vph=float(3600.0 * discharge[int(i)] / dt),
        capacity_lo=float(np.percentile(cap_boot, 2.5)),
        capacity_hi=float(np.percentile(cap_boot, 97.5)),
        capacity_se=float(cap_boot.std(ddof=1)),
        av_headway_s=float(dt * (row["vmax"] + 1) / row["vmax"]),
        free_speed_kmh=float((row["vmax"] - row["p"]) * CELL_KM * 1000 * 3.6 / dt),
        av_gain=float(gain[int(i)]),
        av_gain_lo=float(np.percentile(g_boot, 2.5)), av_gain_hi=float(np.percentile(g_boot, 97.5)),
        av_gain_se=float(g_boot.std(ddof=1)),
        boot_share_same_params=float((bi == i).mean()),
        boot_share_h0=float((lib.grid["h"].to_numpy()[bi] == 0).mean()),
        dt_lo=float(np.percentile(dt_grid[bd], 2.5)), dt_hi=float(np.percentile(dt_grid[bd], 97.5)),
        edge_dt=bool(d in (0, n_dt - 1)), edge_vmax=bool(row["vmax"] in (VMAX_VALUES[0], VMAX_VALUES[-1])),
        edge_p=bool(np.isclose(row["p"], P_VALUES[-1])),
    )
    return out, boot


def block_bootstrap_weights(n, block, n_rep, rng):
    """Moving-block bootstrap of an ordered series of length n: each replicate stitches together
    randomly chosen runs of `block` consecutive intervals. Returns how many times each interval
    was drawn, shape (n_rep, n)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_rep, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_rep, -1)[:, :n]
    W = np.zeros((n_rep, n), dtype=np.int32)
    for r in range(n_rep):
        W[r] = np.bincount(idx[r], minlength=n)
    return W
