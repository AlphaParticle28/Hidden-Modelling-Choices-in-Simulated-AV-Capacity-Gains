"""Screen every PeMS file in data/raw and keep the lane-1 station-days that show free flow,
capacity and congestion.

    python scripts/03_prepare_data.py
        -> data/processed/lane1_5min.csv.gz      (all kept 5-minute intervals; not in git)
        -> results/station_days.csv               (one row per kept station-day)
        -> results/screening_summary.csv          (how many station-days each rule removed)
        -> figures/03_data_overview.png, figures/03_example_station_days.png

Takes about 1 minute for five files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.pems import screen_file  # noqa: E402
from nasch.plotting import BLUE, INK_2, plt, save, use_style  # noqa: E402

paths.ensure_dirs()
use_style()

raw_files = paths.raw_files(paths.ANALYSIS_DAYS)
if not raw_files:
    raise SystemExit("no PeMS files in data/raw (see data/raw/README.md)")

screens, kept = [], []
for f in raw_files:
    s, d = screen_file(f)
    screens.append(s)
    kept.append(d)
    print(f"{f.name}: {len(s)} mainline stations, {(s['reason'] == '').sum()} kept")
screening = pd.concat(screens, ignore_index=True)
data = pd.concat(kept, ignore_index=True)

# ---------------------------------------------------------------- tables
data.to_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", index=False, float_format="%.4f")

station_days = screening[screening["reason"] == ""].drop(columns="reason")
station_days.to_csv(paths.RESULTS / "station_days.csv", index=False, float_format="%.2f")

reasons = screening["reason"].replace("", "kept")
summary = pd.crosstab(screening["date"], reasons)
summary["mainline stations"] = summary.sum(axis=1)
summary.loc["all days"] = summary.sum()
summary.to_csv(paths.RESULTS / "screening_summary.csv")

days_per_station = station_days.groupby("station").size()
print(f"kept {len(station_days)} station-days at {days_per_station.size} stations "
      f"({(days_per_station >= 2).sum()} stations on 2 or more days), {len(data)} intervals")

# ---------------------------------------------------------------- overview figure
blues = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
hb = ax.hexbin(data["density_vpk"], data["flow_vph"], gridsize=70, extent=(0, 120, 0, 2600),
               cmap=blues, norm=LogNorm(), mincnt=1, linewidths=0)
fig.colorbar(hb, ax=ax, label="5-minute intervals per hexagon")
for k, label in [(133.3, "jam density, h = 0"), (66.7, "jam density, h = 1")]:
    ax.axvline(k, color=INK_2, lw=1, ls="--")
    ax.text(k - 1.5, 2520, label, rotation=90, ha="right", va="top", color=INK_2, fontsize=9)
share = 100 * (data["density_vpk"] > 66.7).mean()
ax.set_xlim(0, 140)
ax.set_xlabel("density = flow / speed (veh/km, lane 1)")
ax.set_ylabel("flow (veh/h, lane 1)")
ax.set_title(f"(a) All {len(data):,} kept 5-minute intervals\n"
             f"{share:.2f}% lie above the h = 1 jam density")

ax = axes[0, 1]
ax.hist(station_days["q_max_vph"], bins=np.arange(1700, 2525, 50), color=BLUE,
        edgecolor="#fcfcfb", linewidth=1)
ax.set_xlabel("highest 5-minute flow of the day (veh/h, lane 1)")
ax.set_ylabel("station-days")
ax.set_title("(b) Peak lane-1 flow per station-day")

ax = axes[1, 0]
ax.hist(station_days["free_speed_kmh"], bins=np.arange(100, 126, 1), color=BLUE,
        edgecolor="#fcfcfb", linewidth=1)
ax.set_xlabel("free-flow speed (km/h): median speed of the least dense 25% of the day")
ax.set_ylabel("station-days")
ax.set_title("(c) Free-flow speed per station-day")

ax = axes[1, 1]
counts = days_per_station.value_counts().sort_index()
bars = ax.bar(counts.index.astype(str), counts.values, color=BLUE, width=0.6)
for b, c in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, c + 2, str(c), ha="center", color=INK_2, fontsize=9)
ax.grid(axis="x", visible=False)
ax.set_xlabel(f"number of the {len(raw_files)} days on which the station was kept")
ax.set_ylabel("stations")
ax.set_title(f"(d) {days_per_station.size} stations; "
             f"{(days_per_station >= 2).sum()} kept on 2 or more days")

save(fig, paths.FIGURES / "03_data_overview.png")

# ---------------------------------------------------------------- example station-days
rng = np.random.default_rng(1)
pick = station_days.iloc[np.sort(rng.choice(len(station_days), size=24, replace=False))]
fig, axes = plt.subplots(4, 6, figsize=(16, 10), sharex=True, sharey=True)
for ax, (_, sd) in zip(axes.ravel(), pick.iterrows()):
    d = data[(data["station"] == sd["station"]) & (data["date"] == sd["date"])]
    ax.plot(d["density_vpk"], d["flow_vph"], "o", ms=2.5, color=BLUE, alpha=0.6)
    ax.set_title(f"{sd['station']}  Fwy {sd['freeway']} {sd['direction']}\n{sd['date']}",
                 fontsize=9, fontweight="normal")
for ax in axes[-1]:
    ax.set_xlabel("veh/km")
for ax in axes[:, 0]:
    ax.set_ylabel("veh/h")
axes[0, 0].set_xlim(0, 100)
axes[0, 0].set_ylim(0, 2500)
fig.suptitle("24 randomly chosen kept station-days (lane 1, 5-minute intervals)",
             x=0.01, ha="left", fontweight="bold")
save(fig, paths.FIGURES / "03_example_station_days.png")
print("saved figures/03_data_overview.png, figures/03_example_station_days.png")
