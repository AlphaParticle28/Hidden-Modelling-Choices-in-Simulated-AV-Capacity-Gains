"""How much does the calibrated discharge capacity, and therefore a simulated AV gain, depend on
which detector and which day the human drivers were calibrated on?

The calibrated quantity is the "detector-calibrated discharge capacity": the queue-discharge rate of
the human drivers that best reproduce one detector's (density, flow) data. Most screened detectors
sit inside recurring queues, so this number reflects the detector's position (the downstream
bottleneck, ramps, lane 1's share of traffic) as much as driver behaviour. It is exactly what a
modeller would obtain by calibrating against that detector, which is the point.

AV side, defined the same way as the human side (queue discharge): an AV that never hesitates and
keeps a constant time gap g (Newell's car-following model) discharges from a queue at its full
capacity, v / (v g + L), with L = 7.5 m (the cell size) and v the site's calibrated free-flow
speed. Gaps: 0.6 s (cooperative ACC), 1.0 s and 1.5 s (typical ACC settings).

    python scripts/07_site_analysis.py
        writes  results/site_capacity.csv, results/heterogeneity_summary.csv,
                results/definition_gap.csv, results/mixed_traffic_example.csv,
                figures/05_headline.png, figures/05_checks.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.calibration import driver, load_library  # noqa: E402
from nasch.measure import fd_peak, queue_discharge_rate  # noqa: E402
from nasch.plotting import AQUA, BLUE, INK, INK_2, ORANGE, plt, save, use_style  # noqa: E402

AV_GAPS = (0.6, 1.0, 1.5)          # seconds, bumper-to-bumper time gap of the assumed AV
L_EFF = 7.5                        # metres per vehicle in a jam (the cell size)
use_style()


def av_capacity(gap_s, free_speed_kmh):
    v = np.asarray(free_speed_kmh) / 3.6
    return 3600 * v / (v * gap_s + L_EFF)


lib = load_library(paths.CACHE / "fd_library.npz")
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
k = cal[cal["kept_for_av_gain"]].copy()
k["corridor"] = "Fwy " + k["freeway"].astype(str) + " " + k["direction"]
out = {}

# ---------------------------------------------------------------- 1. nested variance decomposition
# corridor (freeway + direction) > detector within corridor > day within detector (incl. estimation)
def decompose(df):
    """Method-of-moments variance components (corridor, detector within corridor, day)."""
    s = df.groupby("station").agg(C=("capacity_vph", "mean"), days=("capacity_vph", "size"),
                                  corridor=("corridor", "first"))
    m = df.groupby("station").filter(lambda g: len(g) >= 2)
    w = m.groupby("station")["capacity_vph"].var(ddof=1).mean()
    noise = (w / s["days"]).mean()
    c = s.groupby("corridor").agg(m=("C", "mean"), n=("C", "size"), v=("C", "var"))
    vs = max(c.loc[c["n"] >= 2, "v"].mean() - noise, 0.0)
    vc = max(c["m"].var(ddof=1) - ((vs + noise) / c["n"]).mean(), 0.0)
    return vc, vs, w


sites = k.groupby("station").agg(C=("capacity_vph", "mean"), days=("capacity_vph", "size"),
                                 se=("capacity_se", lambda s: np.sqrt((s ** 2).sum()) / len(s)),
                                 corridor=("corridor", "first"), free_speed=("free_speed_kmh", "mean"))
multi = k.groupby("station").filter(lambda g: len(g) >= 2)
corr = sites.groupby("corridor").agg(m=("C", "mean"), n=("C", "size"))
var_corr, var_site_in_corr, W = decompose(k)
total = var_corr + var_site_in_corr + W

# corridor cluster bootstrap: resample whole corridors, relabelling repeats as new corridors/stations
rng = np.random.default_rng(5)
names = k["corridor"].unique()
by_corr = {c: g for c, g in k.groupby("corridor")}
boot = []
for b in range(500):
    pick = rng.choice(names, size=names.size, replace=True)
    df = pd.concat([by_corr[c].assign(corridor=f"{c}#{j}", station=by_corr[c]["station"].astype(str) + f"#{j}")
                    for j, c in enumerate(pick)])
    vc, vs, w = decompose(df)
    t = vc + vs + w
    boot.append((vc / t, vs / t, w / t, np.sqrt(vc), np.sqrt(vs), np.sqrt(w)))
boot = np.array(boot)
for j, name in enumerate(["share_corridor", "share_detector_within_corridor", "share_day",
                          "sd_corridor", "sd_detector_within_corridor", "sd_day_within_detector"]):
    out[f"{name}_lo"], out[f"{name}_hi"] = np.percentile(boot[:, j], [2.5, 97.5])
mu = sites["C"].mean()                                                          # mean over detectors
pi = (mu - 1.96 * np.sqrt(total), mu + 1.96 * np.sqrt(total))
raw = np.sqrt((multi["capacity_se"] ** 2).mean())
out.update(station_days=len(k), detectors=len(sites), corridors=len(corr),
           detectors_multi_day=int((sites["days"] >= 2).sum()),
           mean_capacity=mu, human_discharge_headway_s=3600 / mu,
           sd_corridor=np.sqrt(var_corr), sd_detector_within_corridor=np.sqrt(var_site_in_corr),
           sd_day_within_detector=np.sqrt(W),
           share_corridor=var_corr / total, share_detector_within_corridor=var_site_in_corr / total,
           share_day=W / total, new_detector_pi_lo=pi[0], new_detector_pi_hi=pi[1],
           bootstrap_se_raw=raw, bootstrap_to_daily_ratio=raw / np.sqrt(W))
sites.reset_index().to_csv(paths.RESULTS / "site_capacity.csv", index=False, float_format="%.2f")

# ---------------------------------------------------------------- 2. AV gain, same definition both sides
for g in AV_GAPS:
    gain = av_capacity(g, sites["free_speed"]) / sites["C"] - 1
    out[f"gain_gap{g}_median"] = gain.median()
    out[f"gain_gap{g}_p10"], out[f"gain_gap{g}_p90"] = gain.quantile([0.1, 0.9])

# ---------------------------------------------------------------- 3. consistency with the data
# Not independent validation: the same detector's congested flows also shaped the fit.
data = pd.read_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", dtype={"time": str})
cong = data[data["speed_kmh"] < 60].groupby(["station", "date"])["flow_vph"]
obs = pd.DataFrame({"obs_p90": cong.quantile(0.9), "n_cong": cong.size()}).reset_index()
v = k.merge(obs, on=["station", "date"])
v = v[v["n_cong"] >= 10]
rho, _ = spearmanr(v["capacity_vph"], v["obs_p90"])
out.update(consistency_n=len(v), consistency_spearman=rho,
           consistency_ratio_median=(v["capacity_vph"] / v["obs_p90"]).median())

# ---------------------------------------------------------------- 4. capacity-definition gap (cell AV)
gap = []
for i in sorted(k["param_index"].unique()):
    row = lib.grid.iloc[int(i)]
    peak, _, _, _ = fd_peak(200, int(row["vmax"]), driver(row), steps=500, transient=200)
    gap.append(dict(param_index=i, ring_peak_200=peak))
gap = k.merge(pd.DataFrame(gap), on="param_index")
gap["gain_ring_protocol"] = (gap["vmax"] / (gap["vmax"] + 1)) / gap["ring_peak_200"] - 1
gap["capacity_ring_vph"] = 3600 * gap["ring_peak_200"] / gap["dt"]
gap[["station", "date", "av_gain", "gain_ring_protocol", "capacity_vph", "capacity_ring_vph",
     "free_speed_kmh"]].to_csv(
    paths.RESULTS / "definition_gap.csv", index=False, float_format="%.4f")
out["gain_cells_discharge_median"] = gap["av_gain"].median()
out["gain_cells_ringpeak_median"] = gap["gain_ring_protocol"].median()
out["h0_share"] = (k["h"] == 0).mean()

# ---------------------------------------------------------------- 5. mixed traffic (illustration only)
ex = k.iloc[(k["capacity_vph"] - k["capacity_vph"].median()).abs().argsort().iloc[0]]
row = lib.grid.iloc[int(ex["param_index"])]
mix = []
for pen in (0.0, 0.25, 0.5, 0.75, 1.0):
    e = queue_discharge_rate(int(row["vmax"]), driver(row), penetration=pen)
    mix.append(dict(penetration=pen, discharge=e.mean, se=e.se, capacity_vph=3600 * e.mean / ex["dt"]))
mix = pd.DataFrame(mix)
mix.to_csv(paths.RESULTS / "mixed_traffic_example.csv", index=False, float_format="%.4f")

summary = pd.Series(out)
summary.to_csv(paths.RESULTS / "heterogeneity_summary.csv", header=["value"], float_format="%.4f")
print(summary.round(3).to_string())

# ---------------------------------------------------------------- figure 1: headline
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.5, 1.15, 0.8]})
ax = axes[0]
s = sites.sort_values("C").reset_index()
x = np.arange(len(s))
ax.axhspan(*pi, color=BLUE, alpha=0.08, lw=0)
ax.vlines(x, s["C"] - 1.96 * s["se"], s["C"] + 1.96 * s["se"], color=BLUE, lw=0.6, alpha=0.6)
ax.plot(x, s["C"], "o", ms=2.5, color=BLUE)
ax.text(5, pi[1] + 15, f"95% range for another screened detector: {pi[0]:.0f}-{pi[1]:.0f} veh/h",
        color=INK_2, fontsize=9)
ax.set_xlabel(f"{len(s)} detectors on {len(corr)} freeway corridors, ranked")
ax.set_ylabel("detector-calibrated discharge capacity (veh/h, lane 1)")
ax.set_title("(a) The human baseline depends on which detector\nit is calibrated against "
             "(mean over days, 95% interval)")

ax = axes[1]
cap = np.linspace(1200, 2200, 200)
gaps = np.linspace(0.4, 1.8, 200)
CC, GG = np.meshgrid(cap, gaps)
gain = 100 * (av_capacity(GG, sites["free_speed"].median()) / CC - 1)
cs = ax.contourf(CC, GG, gain, levels=np.arange(-20, 181, 20), cmap="Blues", extend="both")
ax.contour(CC, GG, gain, levels=[0], colors=[INK], linewidths=1.2)
fig.colorbar(cs, ax=ax, label="capacity gain, 0% to 100% AVs (%)")
lo, hi = np.percentile(sites["C"], [10, 90])
ax.axvspan(lo, hi, color=ORANGE, alpha=0.12, lw=0)
ax.text((lo + hi) / 2, 1.72, "middle 80% of\ndetectors", ha="center", va="top", color=INK, fontsize=9)
for g, name in [(0.6, "cooperative ACC"), (1.5, "typical ACC")]:
    ax.axhline(g, color=INK_2, lw=1, ls="--")
    ax.text(1210, g + 0.03, name, color="#fcfcfb" if g < 1 else INK_2, fontsize=9, fontweight="bold")
ax.set_xlabel("human discharge capacity used as the baseline (veh/h)")
ax.set_ylabel("AV time gap (s)")
ax.set_title("(b) The gain is set jointly by the AV assumption\nand the calibrated baseline")

ax = axes[2]
vals = [out["sd_corridor"], out["sd_detector_within_corridor"], out["sd_day_within_detector"]]
labels = ["between\ncorridors", "between detectors,\nsame corridor", "between days,\nsame detector"]
bars = ax.bar(labels, vals, color=[BLUE, AQUA, ORANGE], width=0.6)
keys = ["sd_corridor", "sd_detector_within_corridor", "sd_day_within_detector"]
ax.errorbar(range(3), vals, yerr=[[out[c] - out[c + "_lo"] for c in keys], [out[c + "_hi"] - out[c] for c in keys]],
            fmt="none", color=INK, capsize=4, lw=1)
for b, val, c in zip(bars, vals, keys):
    ax.text(b.get_x() + b.get_width() / 2, out[c + "_hi"] + 3, f"{val:.0f}", ha="center", color=INK, fontsize=10)
ax.grid(axis="x", visible=False)
ax.tick_params(axis="x", labelsize=8.5)
ax.set_ylabel("standard deviation of capacity (veh/h)")
ax.set_title("(c) Most variation is between detectors\non the same corridor, not between days")
save(fig, paths.FIGURES / "05_headline.png")

# ---------------------------------------------------------------- figure 2: supporting checks
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
ax.plot(100 * gap["av_gain"], 100 * gap["gain_ring_protocol"], "o", ms=3, color=BLUE, alpha=0.35)
lim = [100 * gap[["av_gain", "gain_ring_protocol"]].min().min() - 5,
       100 * gap[["av_gain", "gain_ring_protocol"]].max().max() + 5]
ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
ax.set_xlabel("AV gain, queue-discharge definition (%)")
ax.set_ylabel("AV gain, 200-cell ring peak (%)")
ax.set_title("(a) Capacity definition changes the gain\n(cell-encoded AV, each calibrated detector)")

ax = axes[1]
dfile = paths.RESULTS / "density_sensitivity.csv"
if dfile.exists():
    ds = pd.read_csv(dfile)
    gcols = [c for c in ds.columns if c.startswith("capacity_g")]
    for color, c in zip([AQUA, BLUE, ORANGE, INK_2, "#e87ba4"], gcols):
        ax.plot(ds["capacity_qv"], ds[c], "o", ms=2.5, color=color, alpha=0.3, label=f"g = {c[10:]} ft")
    lim = [ds["capacity_qv"].min() - 50, ds["capacity_qv"].max() + 50]
    ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
    ax.legend(fontsize=8.5)
ax.set_xlabel("capacity, density = flow / speed (veh/h)")
ax.set_ylabel("capacity, density = occupancy / g (veh/h)")
ax.set_title("(b) Sensitivity to the density conversion")

ax = axes[2]
ax.errorbar(100 * mix["penetration"], mix["capacity_vph"], yerr=1.96 * 3600 * mix["se"] / ex["dt"],
            fmt="o-", color=BLUE, capsize=3)
ax.set_xlabel("AV share (%)")
ax.set_ylabel("queue-discharge capacity (veh/h)")
ax.set_title(f"(c) Illustration only: mixed traffic at detector {ex['station']}\n"
             "(cell-encoded AV; shape, not level, is the point)")
save(fig, paths.FIGURES / "05_checks.png")
print("saved figures/05_headline.png, figures/05_checks.png")
