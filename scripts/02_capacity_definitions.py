"""Which "capacity"? Queue discharge rate versus the peak of a ring fundamental diagram.

    python scripts/02_capacity_definitions.py
        -> figures/02_capacity_definitions.png
        -> results/capacity_definitions.csv, results/discharge_convergence.csv

Takes about 2-3 minutes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.measure import discharge_rates, fd_peak, queue_discharge_rate  # noqa: E402
from nasch.plotting import BLUE, INK, INK_2, ORANGE, SERIES, plt, save, use_style  # noqa: E402
from nasch.simulator import Driver, queue_discharge_times  # noqa: E402

paths.ensure_dirs()
use_style()

CASES = {   # vmax, human driver
    "vmax=4, p=0.05, p0=0.5 (original fit)": (4, Driver(0.05, 0.5, 0)),
    "vmax=5, p=0.25, p0=0.5":               (5, Driver(0.25, 0.5, 0)),
    "vmax=5, p=0.25, no slow-to-start":     (5, Driver(0.25, None, 0)),
}
RING_SIZES = [100, 200, 500, 1000, 2000, 5000]
RUN_LENGTHS = [500, 2000, 10000]
ORIGINAL = list(CASES)[0]

# ---------------------------------------------------------------- measurements
rows = []
discharge = {}
for name, (vmax, human) in CASES.items():
    est = queue_discharge_rate(vmax, human)
    discharge[name] = est
    print(f"{name}: discharge {est.mean:.4f} +/- {est.se:.4f} ({est.runs.size} runs)")
    for L in RING_SIZES:
        for T in (RUN_LENGTHS if name == ORIGINAL else [2000]):
            peak, rho, _, _ = fd_peak(L, vmax, human, steps=T)
            rows.append(dict(case=name, vmax=vmax, p=human.p, p0=human.p0, ring_cells=L,
                             steps=T, ring_peak=peak, rho_at_peak=rho,
                             discharge=est.mean, discharge_se=est.se,
                             peak_over_discharge=peak / est.mean))
table = pd.DataFrame(rows)
table.to_csv(paths.RESULTS / "capacity_definitions.csv", index=False, float_format="%.5f")

vmax, human = CASES[ORIGINAL]
conv = []
for n in (250, 500, 1000, 2000, 4000):
    e = queue_discharge_rate(vmax, human, n_queue=n, target_rel_se=0.0, first_seed=0)
    conv.append(dict(queue_vehicles=n, runs=e.runs.size, discharge=e.mean, se=e.se,
                     rel_se=e.rel_se))
    print(f"queue {n}: {e.mean:.4f} +/- {e.se:.4f}")
pd.DataFrame(conv).to_csv(paths.RESULTS / "discharge_convergence.csv", index=False,
                          float_format="%.5f")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))

# (a) N-curves at the detector
ax = axes[0, 0]
for color, pen, label in [(BLUE, 0.0, "all human drivers"), (ORANGE, 1.0, "all automated")]:
    t = np.sort(queue_discharge_times(2000, vmax, human, penetration=pen, seeds=[0])[0])
    ax.plot(t, np.arange(1, t.size + 1), color=color, label=label)
    rate = discharge_rates(t[None, ::-1])[0]
    k = int(0.55 * t.size)
    ax.annotate(f"slope {rate:.3f} veh/step\n(= {rate * 3600:.0f} veh/h at 1 s per step)",
                (t[k], k), xytext=(12, -28 if pen == 0 else 8), textcoords="offset points",
                color=INK, fontsize=9)
ax.set_xlabel("time since the queue is released (steps)")
ax.set_ylabel("cumulative vehicles past the detector, N(t)")
ax.set_title("(a) Capacity = slope of the N-curve below a discharging queue\n"
             "original fit, queue of 2,000 vehicles, detector 50 cells downstream")
ax.legend(loc="lower right", fontsize=9)

# (b) ring peak / discharge rate, all three cases
ax = axes[0, 1]
for color, name in zip(SERIES, CASES):
    d = table[(table.case == name) & (table.steps == 2000)]
    ax.plot(d.ring_cells, d.peak_over_discharge, "o-", color=color, ms=5, label=name)
ax.axhline(1.0, color=INK_2, lw=1.2, ls="--")
ax.text(100, 0.99, "= queue discharge rate", ha="left", va="top", color=INK_2, fontsize=9)
ax.set_xscale("log")
ax.set_xticks(RING_SIZES, [str(L) for L in RING_SIZES])
ax.set_xlabel("ring size (cells; 1 cell = 7.5 m)")
ax.set_ylabel("ring FD peak / queue discharge rate")
ax.set_title("(b) The ring peak depends on the ring size\n"
             "and approaches the discharge rate as the ring grows (2,000 steps)")
ax.legend(loc="upper right", fontsize=9)

# (c) original fit: ring size x run length
ax = axes[1, 0]
est = discharge[ORIGINAL]
ax.axhspan(est.mean - 2 * est.se, est.mean + 2 * est.se, color=BLUE, alpha=0.15, lw=0)
ax.axhline(est.mean, color=BLUE, lw=1.2)
ax.text(100, est.mean - 0.004, f"queue discharge rate {est.mean:.3f} (band = 2 SE)",
        va="top", color=INK_2, fontsize=9)
for color, T in zip(SERIES, RUN_LENGTHS):
    d = table[(table.case == ORIGINAL) & (table.steps == T)]
    ax.plot(d.ring_cells, d.ring_peak, "o-", color=color, ms=5, label=f"{T:,} steps")
old = table[(table.case == ORIGINAL) & (table.ring_cells == 200) & (table.steps == 500)].iloc[0]
ax.annotate("set-up used in the original\nversion of this project", (200, old.ring_peak),
            xytext=(30, 25), textcoords="offset points", color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
ax.set_xscale("log")
ax.set_xticks(RING_SIZES, [str(L) for L in RING_SIZES])
ax.set_xlabel("ring size (cells)")
ax.set_ylabel("ring FD peak (vehicles per step)")
ax.set_title("(c) Original fit: the ring peak ranges from 0.44 to 0.66\n"
             "depending only on ring size and run length")
ax.legend(loc="upper right", fontsize=9, title="run length", title_fontsize=9)

# (d) what this does to the AV gain
ax = axes[1, 1]
av_rate = vmax / (vmax + 1)                      # exact for deterministic AVs
big = table[(table.case == ORIGINAL) & (table.ring_cells == 5000) & (table.steps == 10000)].iloc[0]
labels = ["ring peak\n200 cells, 500 steps\n(original set-up)",
          "ring peak\n5,000 cells, 10,000 steps", "queue discharge\nrate"]
human_cap = [old.ring_peak, big.ring_peak, est.mean]
gains = [100 * (av_rate / c - 1) for c in human_cap]
bars = ax.bar(labels, gains, color=[INK_2, INK_2, BLUE], width=0.55)
for b, g in zip(bars, gains):
    ax.text(b.get_x() + b.get_width() / 2, g + 1.5, f"+{g:.0f}%", ha="center", color=INK,
            fontsize=10, fontweight="bold")
ax.set_ylabel("AV capacity gain, 0% to 100% AVs (%)")
ax.set_ylim(0, max(gains) * 1.2)
ax.grid(axis="x", visible=False)
ax.set_title("(d) Same model, same vehicles: the AV gain depends\n"
             f"on how human capacity is measured (AV rate = {av_rate:.1f} exactly)")

out = paths.FIGURES / "02_capacity_definitions.png"
save(fig, out)
print(f"saved {out}")
