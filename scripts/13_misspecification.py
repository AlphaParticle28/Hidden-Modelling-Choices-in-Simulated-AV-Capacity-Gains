"""Misspecification test: can the CA calibration recover a discharge capacity that it did not generate?

Synthetic detector-days are drawn from a *different* model: a triangular fundamental diagram with a
known queue-discharge capacity, plus a capacity drop (free-flow points just below the critical
density can run up to 8% above the discharge rate, as pre-breakdown flows do), plus Poisson noise
on the 5-minute counts. Each synthetic day reuses the density sequence of a real station-day, so the
data coverage is realistic. The CA is then calibrated exactly as for real data.

    python scripts/13_misspecification.py   -> results/misspecification.csv, figures/10_misspecification.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.baseline import fit_triangle, predict_triangle  # noqa: E402
from nasch.calibration import fit_station_day, load_library  # noqa: E402
from nasch.plotting import BLUE, INK_2, plt, save, use_style  # noqa: E402

use_style()
rng = np.random.default_rng(11)
lib = load_library(paths.CACHE / "fd_library.npz")
discharge = pd.read_csv(paths.RESULTS / "human_discharge.csv")["discharge"].to_numpy()
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
kept = cal[cal["kept_for_av_gain"]]
data = pd.read_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", dtype={"time": str})
groups = {key: g.sort_values("time") for key, g in data.groupby(["station", "date"])}

rows = []
for r in kept.sample(200, random_state=3).to_dict("records"):
    d = groups[(r["station"], r["date"])]
    k = d["density_vpk"].to_numpy()
    tri = fit_triangle(k, d["flow_vph"].to_numpy())
    if tri is None:
        continue
    q_true = predict_triangle(tri, k)
    near_apex = (k < tri["k_apex"]) & (k > 0.8 * tri["k_apex"])
    q_true = np.where(near_apex, q_true * (1 + rng.uniform(0, 0.08, k.size)), q_true)
    q_syn = 12.0 * rng.poisson(np.maximum(q_true, 1) / 12.0)
    q_syn = np.maximum(q_syn, 12.0)
    out, _ = fit_station_day(lib, discharge, k, q_syn, d["time"].tolist(), n_boot=2)
    rows.append(dict(station=r["station"], date=r["date"], true_capacity=tri["capacity"],
                     ca_capacity=out["capacity_vph"], geh=out["geh_all"]))
m = pd.DataFrame(rows)
m.to_csv(paths.RESULTS / "misspecification.csv", index=False, float_format="%.3f")
ratio = m["ca_capacity"] / m["true_capacity"]
rho = spearmanr(m["true_capacity"], m["ca_capacity"])[0]
print(f"misspecification test, {len(m)} synthetic days: median ratio CA/true {ratio.median():.3f}, "
      f"IQR {ratio.quantile(0.25):.3f}-{ratio.quantile(0.75):.3f}, Spearman {rho:.2f}, "
      f"median |error| {100 * (ratio - 1).abs().median():.1f}%")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(m["true_capacity"], m["ca_capacity"], "o", ms=4, color=BLUE, alpha=0.5)
lim = [m[["true_capacity", "ca_capacity"]].min().min() - 50, m[["true_capacity", "ca_capacity"]].max().max() + 50]
ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
ax.set_xlabel("true discharge capacity of the generating triangle (veh/h)")
ax.set_ylabel("capacity recovered by the CA calibration (veh/h)")
ax.set_title(f"Data from a different model: CA recovers the discharge\nmedian ratio {ratio.median():.2f}, "
             f"Spearman {rho:.2f}, {len(m)} synthetic days")
save(fig, paths.FIGURES / "10_misspecification.png")
print("saved figures/10_misspecification.png")
