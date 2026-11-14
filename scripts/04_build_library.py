"""Simulate everything the calibration needs, once:

  1. the fundamental-diagram library: ring flow at 84 densities x 4 seeds for each of the
     1,980 human parameter sets                       -> results/cache/fd_library.npz   (~75 min)
  2. the queue discharge rate of each parameter set   -> results/human_discharge.csv  (~100 min)

    python scripts/04_build_library.py [library|discharge|all]

Progress is saved as it goes; re-running the script resumes where it stopped.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.calibration import (LIBRARY_SEEDS, av_gain, human_discharge, library_densities,  # noqa: E402
                               parameter_grid, simulate_fd)

paths.ensure_dirs()
part = sys.argv[1] if len(sys.argv) > 1 else "all"
grid = parameter_grid()
rho = library_densities()
LIB = paths.CACHE / "fd_library.npz"
LIB_PARTIAL = paths.CACHE / "fd_library_partial.npz"
DIS = paths.RESULTS / "human_discharge.csv"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_npz(path, compressed=False, **arrays):
    """Write to a temporary file next to `path`, then swap it in, so an interruption never leaves a
    half-written file that a re-run would trust. The temporary name ends in .npz so numpy keeps it."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def save_csv(frame, path, **kwargs):
    """Same idea for a CSV."""
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, **kwargs)
    os.replace(tmp, path)


if part in ("library", "all") and not LIB.exists():
    flow = np.full((len(grid), rho.size, len(LIBRARY_SEEDS)), np.nan)
    if LIB_PARTIAL.exists():
        flow = np.load(LIB_PARTIAL)["flow"]
    todo = np.flatnonzero(np.isnan(flow[:, 0, 0]))
    log(f"library: {len(grid) - todo.size} of {len(grid)} parameter sets already done")
    t0 = time.time()
    for k, i in enumerate(todo, 1):
        flow[i] = simulate_fd(grid.iloc[i])
        if k % 50 == 0 or k == todo.size:
            save_npz(LIB_PARTIAL, flow=flow)
            rate = (time.time() - t0) / k
            log(f"library: {len(grid) - todo.size + k}/{len(grid)}  ({rate:.1f} s each, "
                f"~{rate * (todo.size - k) / 60:.0f} min left)")
    save_npz(LIB, compressed=True, flow=flow, rho=rho, vmax=grid["vmax"].to_numpy(),
             p=grid["p"].to_numpy(), p0=grid["p0"].to_numpy(), h=grid["h"].to_numpy(),
             seeds=np.array(list(LIBRARY_SEEDS)))
    LIB_PARTIAL.unlink(missing_ok=True)
    log(f"library saved: {LIB}")

if part in ("discharge", "all"):
    done = pd.read_csv(DIS) if DIS.exists() else pd.DataFrame()
    have = len(done)
    log(f"discharge: {have} of {len(grid)} parameter sets already done")
    rows = done.to_dict("records")
    t0 = time.time()
    for k, i in enumerate(range(have, len(grid)), 1):
        r = grid.iloc[i]
        mean, se, runs = human_discharge(r)
        rows.append(dict(vmax=int(r["vmax"]), p=r["p"], p0=r["p0"], h=int(r["h"]),
                         discharge=mean, discharge_se=se, runs=runs,
                         av_gain=av_gain(int(r["vmax"]), mean)))
        if k % 25 == 0 or i == len(grid) - 1:
            save_csv(pd.DataFrame(rows), DIS, index=False, float_format="%.6f")
            rate = (time.time() - t0) / k
            log(f"discharge: {i + 1}/{len(grid)}  ({rate:.1f} s each, "
                f"~{rate * (len(grid) - i - 1) / 60:.0f} min left)")
    log(f"discharge table saved: {DIS}")
