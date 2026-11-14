"""Calibrate the human-driver parameters for every kept station-day.

For each station-day: fit on training hours and score the held-out hours (every 4th hour), refit
on the whole day, compare the best fits with h forced to 0 / 1 / 2, and repeat the fit on 500
block-bootstrap resamples of the day to get uncertainty.

    python scripts/05_calibrate.py                  (add --figures-only to redraw from saved results)
        needs   results/cache/fd_library.npz, results/human_discharge.csv   (scripts/04)
                data/processed/lane1_5min.csv.gz, results/station_days.csv    (scripts/03)
        writes  results/calibration.csv, results/cache/bootstrap_columns.npz
                figures/04_fit_quality.png, figures/04_example_fits.png

Takes about 30-40 minutes for ~1,000 station-days.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.calibration import (CELL_KM, DT_GRID, fit_station_day, load_library,  # noqa: E402
                               model_flow, split_column)
from nasch.plotting import BLUE, INK, INK_2, ORANGE, plt, save, use_style  # noqa: E402

N_BOOT, BLOCK = 500, 12
MAX_TEST_GEH = 5.0                      # exclusion rule fixed in the analysis plan

paths.ensure_dirs()
use_style()
lib = load_library(paths.CACHE / "fd_library.npz")
dis = pd.read_csv(paths.RESULTS / "human_discharge.csv")
assert len(dis) == len(lib.grid)
assert np.allclose(dis[["vmax", "p", "h"]].to_numpy(), lib.grid[["vmax", "p", "h"]].to_numpy())
assert np.allclose(dis["p0"].fillna(-1), lib.grid["p0"].fillna(-1))
discharge = dis["discharge"].to_numpy()

data = pd.read_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", dtype={"time": str})
station_days = pd.read_csv(paths.RESULTS / "station_days.csv")
days = sorted(station_days["date"].unique())
groups = {key: g.sort_values("time") for key, g in data.groupby(["station", "date"])}

if "--figures-only" in sys.argv:
    cal = pd.read_csv(paths.RESULTS / "calibration.csv")
    boots = np.load(paths.CACHE / "bootstrap_columns.npz")["columns"]
else:
    rows, boots = [], []
    t0 = time.time()
    for n, sd in enumerate(station_days.itertuples(index=False), 1):
        d = groups[(sd.station, sd.date)]
        seed = int(sd.station) * 10 + days.index(sd.date)
        out, boot = fit_station_day(lib, discharge, d["density_vpk"].to_numpy(),
                                    d["flow_vph"].to_numpy(), d["time"].tolist(),
                                    n_boot=N_BOOT, block=BLOCK, seed=seed)
        rows.append(dict(station=sd.station, date=sd.date, freeway=sd.freeway,
                         direction=sd.direction, **out))
        boots.append(boot)
        if n % 50 == 0 or n == len(station_days):
            rate = (time.time() - t0) / n
            print(f"{n}/{len(station_days)} station-days ({rate:.1f} s each, "
                  f"~{rate * (len(station_days) - n) / 60:.0f} min left)", flush=True)

    cal = pd.DataFrame(rows)
    cal["kept_for_av_gain"] = cal["geh_test"] <= MAX_TEST_GEH
    cal.to_csv(paths.RESULTS / "calibration.csv", index=False, float_format="%.4f")
    boots = np.array(boots, np.int32)
    np.savez_compressed(paths.CACHE / "bootstrap_columns.npz", columns=boots,
                        station=cal["station"].to_numpy(), date=cal["date"].to_numpy().astype(str))

kept = cal[cal["kept_for_av_gain"]]
print(f"\nstation-days fitted: {len(cal)}; held-out GEH <= {MAX_TEST_GEH}: {len(kept)} "
      f"({100 * len(kept) / len(cal):.1f}%)")
print(f"held-out mean GEH: median {cal['geh_test'].median():.2f}, "
      f"90th percentile {cal['geh_test'].quantile(0.9):.2f}; "
      f"training {cal['geh_train'].median():.2f}")
print("fitted h:", cal["h"].value_counts().sort_index().to_dict(),
      "| fitted vmax:", cal["vmax"].value_counts().sort_index().to_dict())
print(f"slow-to-start chosen in {100 * cal['p0'].notna().mean():.0f}% of station-days")
print(f"grid-edge flags: dt {cal['edge_dt'].sum()}, vmax {cal['edge_vmax'].sum()}, "
      f"p {cal['edge_p'].sum()}")
print(f"AV gain (kept): median {100 * kept['av_gain'].median():.0f}%, "
      f"middle 50% {100 * kept['av_gain'].quantile(0.25):.0f}-{100 * kept['av_gain'].quantile(0.75):.0f}%")

# ---------------------------------------------------------------- figure: fit quality
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

ax = axes[0, 0]
bins = np.arange(0, max(8.0, cal["geh_test"].quantile(0.995)) + 0.25, 0.25)
ax.hist(cal["geh_test"].clip(upper=bins[-1]), bins=bins, color=BLUE, edgecolor="#fcfcfb")
ax.axvline(MAX_TEST_GEH, color=INK_2, lw=1.2, ls="--")
ax.text(MAX_TEST_GEH + 0.1, ax.get_ylim()[1] * 0.95, f"exclusion threshold\n{len(cal) - len(kept)} "
        f"station-days above", va="top", color=INK_2, fontsize=9)
ax.set_xlabel("mean GEH on the held-out hours")
ax.set_ylabel("station-days")
ax.set_title(f"(a) Out-of-sample fit: median GEH {cal['geh_test'].median():.2f}\n"
             f"{100 * len(kept) / len(cal):.1f}% of station-days pass (GEH < 5 is the usual mark)")

ax = axes[0, 1]
ax.plot(cal["geh_train"], cal["geh_test"], "o", ms=3, color=BLUE, alpha=0.35)
lim = max(cal["geh_train"].max(), cal["geh_test"].quantile(0.995))
ax.plot([0, lim], [0, lim], color=INK_2, lw=1, ls="--")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("mean GEH, training hours")
ax.set_ylabel("mean GEH, held-out hours")
ax.set_title("(b) Held-out vs training error\n(points near the line = no overfitting)")

ax = axes[0, 2]
hc = cal["h"].value_counts().reindex([0, 1, 2], fill_value=0)
bars = ax.bar(["h = 0", "h = 1", "h = 2"], hc.values, color=BLUE, width=0.55)
for b, c in zip(bars, hc.values):
    ax.text(b.get_x() + b.get_width() / 2, c, f"{c}", ha="center", va="bottom", color=INK_2)
pen = (cal["geh_best_h1"] - cal["geh_best_h0"]).median()
pen_text = f"forcing h = 1 costs a median {pen:+.2f} GEH" if np.isfinite(pen) else "h = 1 not in library"
ax.grid(axis="x", visible=False)
ax.set_ylabel("station-days")
ax.set_title(f"(c) Fitted extra buffer h\n{pen_text}")

ax = axes[1, 0]
vc = cal["vmax"].value_counts().reindex([3, 4, 5, 6], fill_value=0)
bars = ax.bar([str(v) for v in vc.index], vc.values, color=BLUE, width=0.55)
for b, c in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, c, f"{c}", ha="center", va="bottom", color=INK_2)
ax.grid(axis="x", visible=False)
ax.set_xlabel("vmax (cells per step)")
ax.set_ylabel("station-days")
ax.set_title("(d) Fitted maximum speed vmax")

ax = axes[1, 1]
ax.hist(cal["dt"], bins=np.arange(DT_GRID[0] - 0.01, DT_GRID[-1] + 0.03, 0.02), color=BLUE,
        edgecolor="#fcfcfb")
ax.set_xlabel("fitted time step dt (seconds per step)")
ax.set_ylabel("station-days")
ax.set_title(f"(e) Fitted time step\n{cal['edge_dt'].sum()} station-days at the edge of the "
             f"{DT_GRID[0]:.2f}-{DT_GRID[-1]:.2f} s range")

ax = axes[1, 2]
has_p0 = cal["p0"].notna()
ax.plot(cal.loc[has_p0, "p"], cal.loc[has_p0, "p0"], "o", ms=4, color=BLUE, alpha=0.35,
        label=f"with slow-to-start ({has_p0.sum()})")
ax.plot(cal.loc[~has_p0, "p"], np.zeros((~has_p0).sum()), "o", ms=4, color=ORANGE, alpha=0.5,
        label=f"without ({(~has_p0).sum()}), shown at p0 = 0")
ax.set_xlabel("p (dawdling while moving)")
ax.set_ylabel("p0 (dawdling when stopped)")
ax.legend(loc="lower right", fontsize=9)
ax.set_title("(f) Fitted randomness parameters")

save(fig, paths.FIGURES / "04_fit_quality.png")

# ---------------------------------------------------------------- figure: example fits
order = cal.sort_values("geh_test").reset_index()
picks = [order.iloc[int(q * (len(order) - 1))]["index"] for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.98)]
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
k_grid = np.linspace(0.5, 100, 300)
for ax, i in zip(axes.ravel(), picks):
    r = cal.loc[i]
    d = groups[(r["station"], r["date"])]
    test = d["time"].str[:2].astype(int) % 4 == 3
    bi, bd = split_column(boots[i], len(DT_GRID))
    J = model_flow(lib, k_grid * CELL_KM)                     # (300, n_params)
    curves = 3600 * J[:, bi] / DT_GRID[bd][None, :]
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=1)
    j = int(r["param_index"])
    ax.fill_between(k_grid, lo, hi, color=BLUE, alpha=0.15, lw=0, label="95% bootstrap band")
    ax.plot(k_grid, 3600 * J[:, j] / r["dt"], color=BLUE, lw=2, label="calibrated model")
    ax.plot(d.loc[~test, "density_vpk"], d.loc[~test, "flow_vph"], "o", ms=3, color=INK_2,
            alpha=0.5, label="training intervals")
    ax.plot(d.loc[test, "density_vpk"], d.loc[test, "flow_vph"], "o", ms=3.5, color=ORANGE,
            alpha=0.8, label="held-out intervals")
    p0 = "none" if np.isnan(r["p0"]) else f"{r['p0']:.1f}"
    ax.set_title(f"{r['station']} Fwy {r['freeway']} {r['direction']}, {r['date']}\n"
                 f"held-out GEH {r['geh_test']:.2f} | vmax {r['vmax']}, p {r['p']:.3f}, "
                 f"p0 {p0}, h {r['h']}, dt {r['dt']:.2f}", fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 2600)
    ax.set_xlabel("density (veh/km)")
axes[0, 0].set_ylabel("flow (veh/h)")
axes[1, 0].set_ylabel("flow (veh/h)")
axes[0, 0].legend(loc="upper right", fontsize=8.5)
fig.suptitle("Calibrated fits at the 5th, 25th, 50th, 75th, 90th and 98th percentile of "
             "held-out error", x=0.01, ha="left", fontweight="bold", color=INK)
save(fig, paths.FIGURES / "04_example_fits.png")
print("saved figures/04_fit_quality.png, figures/04_example_fits.png")
