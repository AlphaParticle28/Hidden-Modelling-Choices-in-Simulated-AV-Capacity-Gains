"""Figure: the simulator against results that are known exactly, and against the original code.

    python scripts/01_validate_simulator.py      ->  figures/01_simulator_validation.png
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.plotting import INK_2, SERIES, plt, save, use_style  # noqa: E402
from nasch.simulator import Driver, ring_flow  # noqa: E402
from tests import legacy_reference as legacy  # noqa: E402

use_style()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# (a) p = 0: the fundamental diagram is exactly a triangle
ax = axes[0]
L = 1000
for color, (vmax, h) in zip(SERIES, [(5, 0), (3, 1), (4, 2)]):
    ns = np.arange(10, int(L / (1 + h)), 15)
    flow = ring_flow(L, ns, vmax, Driver(p=0.0, h=h), steps=5000, transient=2000, seeds=range(2))
    rho = np.linspace(0, 1 / (1 + h), 400)
    ax.plot(rho, np.minimum(vmax * rho, 1 - (1 + h) * rho), color=color, lw=1.2)
    ax.plot(ns / L, flow.mean(axis=1), "o", ms=4.5, color=color, label=f"vmax={vmax}, h={h}")
ax.set_xlabel("density (vehicles per cell)")
ax.set_ylabel("flow (vehicles per step)")
ax.set_title("No randomness (p = 0): exactly the triangle\nline = min(vmax·ρ, 1−(1+h)·ρ), dots = simulation")
ax.legend(loc="upper right", fontsize=9)

# (b) vmax = 1: closed-form solution for any p
ax = axes[1]
for color, p in zip(SERIES, [0.25, 0.5, 0.75]):
    ns = np.arange(20, 1000, 40)
    flow = ring_flow(L, ns, 1, Driver(p=p), steps=4000, transient=1000, seeds=range(4))
    rho = np.linspace(0, 1, 400)
    ax.plot(rho, 0.5 * (1 - np.sqrt(1 - 4 * (1 - p) * rho * (1 - rho))), color=color, lw=1.2)
    ax.plot(ns / L, flow.mean(axis=1), "o", ms=4.5, color=color, label=f"p = {p}")
ax.set_xlabel("density (vehicles per cell)")
ax.set_ylabel("flow (vehicles per step)")
ax.set_title("vmax = 1: matches the exact solution\nline = ½[1 − √(1 − 4(1−p)ρ(1−ρ))], dots = simulation")
ax.legend(loc="upper right", fontsize=9)

# (c) same physics as the original pure-Python implementation
ax = axes[2]
human = Driver(p=0.05, p0=0.5, h=0)
L, T, runs = 200, 500, 30
ns = np.arange(10, 200, 10)
new = ring_flow(L, ns, 4, human, steps=T, transient=200, seeds=range(runs))
np.random.seed(2026)
t0 = time.perf_counter()
old = np.array([[legacy.measure_steady_state_flow(L, n, 4, T, human.p, pen=0.0, headway_HDV=0,
                 chunk_size=25, transient_chunks=8, thresh0=human.p0) for _ in range(runs)]
                for n in ns])
t_old = time.perf_counter() - t0
ax.errorbar(ns / L, old.mean(axis=1), yerr=2 * old.std(axis=1, ddof=1) / np.sqrt(runs), fmt="s",
            ms=5, color=SERIES[1], capsize=3, label="original code (mean ± 2 SE)")
ax.plot(ns / L, new.mean(axis=1), color=SERIES[0], lw=1.8, label="new code")
ax.set_xlabel("density (vehicles per cell)")
ax.set_ylabel("flow (vehicles per step)")
ax.set_title("Same results as the original code\nvmax=4, p=0.05, p0=0.5, ring of 200 cells, 30 runs")
ax.legend(loc="upper right", fontsize=9)
ax.text(0.98, 0.62, f"original code: {t_old:.0f} s for these {runs * len(ns)} runs",
        transform=ax.transAxes, ha="right", color=INK_2, fontsize=9)

out = paths.FIGURES / "01_simulator_validation.png"
save(fig, out)
print(f"saved {out}")
