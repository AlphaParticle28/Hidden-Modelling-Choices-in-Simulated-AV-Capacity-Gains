"""Measured queue discharge at active bottlenecks, used as a referee for the calibrated capacities.

1. Order detectors along each freeway by postmile (PeMS station metadata) and find active
   bottlenecks (queue upstream, free flow downstream, at the same time) on all seven days.
2. Measure lane-1 discharge just downstream while each bottleneck is active, and the pre-breakdown
   flow just before it formed (the capacity drop).
3. Compare with calibrated capacities in two ways, with equal weight:
     * the same detector: the downstream detector's own calibrated capacity against its discharge;
     * the queue detector: the upstream detector's calibrated capacity against the same discharge.
   The check is ONE-SIDED. Neither detector's section is the binding one, so its capacity should be
   at least the measured discharge, and lane 1 below a bottleneck need not be saturated. Only
   "calibrated capacity < measured discharge" is evidence against the calibration; those cases are
   counted. The pre-set agreement rule of scripts/09 (rank correlation >= 0.9 and median |relative
   difference| < 3%) is still reported, as fails / agrees / undecided from its intervals.
   Intervals: corridor cluster bootstrap when n >= 20; for smaller n, the exact order-statistic
   interval of each median and no interval for the rank correlation.
4. Everything is reported for all pairs and for "credible" bottlenecks, defined on physical grounds
   only: both detectors >= 75% observed, no off-ramp between them, the episode starting between 05:00
   and 21:00, and the pair active (with episodes meeting those three conditions) on at least 3 of
   the 7 days.
5. Repeat the site-to-site analysis using only detectors next to an active bottleneck.

    python scripts/11_bottlenecks.py
        -> results/bottleneck_episodes.csv, results/bottlenecks.csv, results/bottleneck_summary.csv,
           figures/08_bottlenecks.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.bottlenecks import (load_metadata, load_positions, measure_pair, median_interval,  # noqa: E402
                               neighbour_pairs, order_along_road, ramps_between)
from nasch.pems import add_physical_units, quality_mask, read_lane1  # noqa: E402
from nasch.plotting import AQUA, BLUE, INK_2, ORANGE, plt, save, use_style  # noqa: E402

MIN_OBSERVED = 75.0              # % of the station's samples that are real, not imputed by PeMS
DAYTIME = (5 * 12, 21 * 12)      # episode must start between 05:00 and 21:00 (5-minute slots)
MIN_DAYS_ACTIVE = 3              # recurrent bottleneck: active on at least 3 of the 7 days
PEAK_HOURS = ((6, 10), (15, 19))
N_BOOT = 2000
MIN_N_BOOTSTRAP = 20             # below this, exact order-statistic intervals for medians

use_style()
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
kept = cal[cal["kept_for_av_gain"]]
site_cap = kept.groupby("station")["capacity_vph"].mean()
tri = pd.read_csv(paths.RESULTS / "baseline_comparison.csv").groupby("station")["capacity_triangle"].mean()

files = paths.raw_files(paths.ANALYSIS_DAYS + paths.TEST_DAYS)
positions = load_positions()
meta = load_metadata()
source = positions["source"].iloc[0]
meta_lanes = positions.set_index("station")["lanes"] if "lanes" in positions else pd.Series(dtype=float)
panels = {}
for f in files:
    df = add_physical_units(read_lane1(f))
    df = df[df["station"].isin(positions["station"])]
    if "freeway" not in positions:
        positions = positions.merge(df[["station", "freeway", "direction"]].drop_duplicates("station"), on="station")
    df.loc[~quality_mask(df), ["speed_kmh", "flow_vph"]] = np.nan
    df["station_speed_kmh"] = df["avg_speed"] * 1.609344
    df["station_occupancy"] = df["avg_occupancy"]
    df["slot"] = df["time"].str[:2].astype(int) * 12 + df["time"].str[3:].astype(int) // 5
    for st, g in df.groupby("station"):
        g = g.set_index("slot")
        if st in meta_lanes.index:
            n_lanes = int(meta_lanes[st])
        else:                                   # no metadata: infer the lane count from lane 1's share
            share = (g["l1_flow"] / g["total_flow"]).median()
            n_lanes = max(1, int(round(1 / share))) if np.isfinite(share) and share > 0 else 1
        g["lane_avg_flow_vph"] = 12.0 * g["total_flow"] / n_lanes
        panels[(st, df["date"].iloc[0])] = g[["station_speed_kmh", "flow_vph", "lane_avg_flow_vph",
                                              "pct_observed", "station_occupancy"]].reindex(range(288))
if "pos_km" not in positions:
    positions["pos_km"] = np.nan
    for _, g in positions.groupby(["freeway", "direction"]):
        positions.loc[g.index, "pos_km"] = order_along_road(g)
pairs = neighbour_pairs(positions)
print(f"positions from {source}: {len(positions)} stations, {len(pairs)} neighbouring pairs")

rows = []
days = sorted({d for _, d in panels})
for up, down, gap in pairs:
    for day in days:
        if (up, day) in panels and (down, day) in panels:
            for ep in measure_pair(panels[(up, day)], panels[(down, day)]):
                rows.append(dict(upstream=up, downstream=down, spacing_km=gap, date=day, **ep))
eps = pd.DataFrame(rows)
eps["capacity_drop"] = 1 - eps["discharge_vph"] / eps["prebreakdown_vph"]
hour = eps["start"] / 12
eps["peak_hours"] = np.any([(hour >= a) & (hour < b) for a, b in PEAK_HOURS], axis=0)

# pair-level facts: ramps between the detectors (metadata only) and how many days the pair is active
pos = positions.set_index("station")
pair_info = eps[["upstream", "downstream"]].drop_duplicates().reset_index(drop=True)
for kind, col in [("OR", "on_ramps_between"), ("FR", "off_ramps_between")]:
    pair_info[col] = [ramps_between(meta, pos.loc[u, "freeway"], pos.loc[u, "direction"], pos.loc[u, "pos_km"],
                                    pos.loc[d, "pos_km"], kind) if meta is not None else np.nan
                      for u, d in zip(pair_info["upstream"], pair_info["downstream"])]
pair_info["corridor"] = [f"{pos.loc[u, 'freeway']}{pos.loc[u, 'direction']}" for u in pair_info["upstream"]]
eps = eps.merge(pair_info, on=["upstream", "downstream"])
eps["qualifies"] = ((eps["observed_up"] >= MIN_OBSERVED) & (eps["observed_down"] >= MIN_OBSERVED)
                    & (eps["off_ramps_between"] == 0) & eps["start"].between(DAYTIME[0], DAYTIME[1] - 1))
active_days = eps[eps["qualifies"]].groupby(["upstream", "downstream"])["date"].nunique().rename("days_active")
pair_info = pair_info.merge(active_days, left_on=["upstream", "downstream"], right_index=True, how="left")
pair_info["days_active"] = pair_info["days_active"].fillna(0).astype(int)
eps = eps.merge(pair_info[["upstream", "downstream", "days_active"]], on=["upstream", "downstream"])
eps["credible"] = eps["qualifies"] & (eps["days_active"] >= MIN_DAYS_ACTIVE)
eps.to_csv(paths.RESULTS / "bottleneck_episodes.csv", index=False, float_format="%.3f")


def per_pair(e):
    """Measured discharge per bottleneck (upstream/downstream pair), weighted by the number of
    usable 5-minute discharge intervals in each episode."""
    w = e.assign(wq=e["discharge_vph"] * e["intervals"])
    b = w.groupby(["upstream", "downstream"]).agg(minutes=("minutes", "sum"), intervals=("intervals", "sum"),
                                                  wq=("wq", "sum"), episodes=("minutes", "size"),
                                                  capacity_drop=("capacity_drop", "median")).reset_index()
    b["measured_discharge"] = b["wq"] / b["intervals"]
    b["model_capacity_upstream"] = b["upstream"].map(site_cap)
    b["model_capacity_downstream"] = b["downstream"].map(site_cap)
    b["triangle_capacity_upstream"] = b["upstream"].map(tri)
    b["triangle_capacity_downstream"] = b["downstream"].map(tri)
    return b.drop(columns="wq").merge(pair_info, on=["upstream", "downstream"])


def compare(b, col, rng):
    """One-sided count (calibrated capacity below measured discharge), plus Spearman, median
    model/measured ratio and median |relative difference| with 95% intervals, and the three-valued
    verdict of the pre-set agreement rule."""
    d = b.dropna(subset=[col]).reset_index(drop=True)
    out = dict(n=len(d))
    if len(d) < 3:
        return out
    m, x = d[col].to_numpy(), d["measured_discharge"].to_numpy()
    r = m / x
    out.update(n_model_below_measured=int((r < 1).sum()),
               max_shortfall_pct=float(100 * (1 - r.min())) if (r < 1).any() else 0.0)

    def stats(i):
        return spearmanr(m[i], x[i])[0], np.median(r[i]), np.median(np.abs(r[i] - 1))

    point = stats(np.arange(len(d)))
    if len(d) >= MIN_N_BOOTSTRAP:                      # resample whole corridors, as elsewhere
        groups = [g.to_numpy() for _, g in d.groupby("corridor").groups.items()]
        boots = np.array([stats(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]))
                          for _ in range(N_BOOT)])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        out["interval"] = "corridor bootstrap"
    else:                                              # exact order-statistic intervals for the medians
        (r_lo, r_hi), (a_lo, a_hi) = median_interval(r), median_interval(np.abs(r - 1))
        lo, hi = np.array([np.nan, r_lo, a_lo]), np.array([np.nan, r_hi, a_hi])
        out["interval"] = "exact order statistic (median); none for rank correlation"
    for j, name in enumerate(["spearman", "ratio", "abs_rel_diff"]):
        out.update({name: point[j], f"{name}_lo": lo[j], f"{name}_hi": hi[j]})
    if hi[0] < 0.9 or lo[2] >= 0.03:
        out["verdict"] = "fails"
    elif lo[0] >= 0.9 and hi[2] < 0.03:
        out["verdict"] = "agrees"
    else:
        out["verdict"] = "undecided"
    return out


rng = np.random.default_rng(11)
bn = per_pair(eps)
bn_cred = per_pair(eps[eps["credible"]])
bn["credible"] = bn.set_index(["upstream", "downstream"]).index.isin(
    bn_cred.set_index(["upstream", "downstream"]).index)
results = {}
for label, b in [("all", bn), ("credible", bn_cred)]:
    for target, col in [("same_detector", "model_capacity_downstream"), ("queue_detector", "model_capacity_upstream"),
                        ("triangle_same_detector", "triangle_capacity_downstream")]:
        for key, val in compare(b, col, rng).items():
            results[f"{label}_{target}_{key}"] = val

# peak hours vs the rest, credible episodes at calibrated downstream detectors (descriptive)
ce = eps[eps["credible"]].assign(model=lambda d: d["downstream"].map(site_cap)).dropna(subset=["model"])
for name, part in [("peak", ce[ce["peak_hours"]]), ("offpeak", ce[~ce["peak_hours"]])]:
    results[f"credible_{name}_episodes"] = len(part)
    results[f"credible_{name}_discharge_median"] = part["discharge_vph"].median()
    results[f"credible_{name}_model_to_measured_median"] = (part["model"] / part["discharge_vph"]).median()

# site-to-site analysis restricted to detectors next to an active bottleneck
adjacent = set(bn["upstream"]) | set(bn["downstream"])
sub = kept[kept["station"].isin(adjacent)]
multi = sub.groupby("station").filter(lambda g: len(g) >= 2)
W = multi.groupby("station")["capacity_vph"].var(ddof=1).mean()
ms = multi.groupby("station")["capacity_vph"].agg(["mean", "size"])
var_site = max(ms["mean"].var(ddof=1) - (W / ms["size"]).mean(), 0.0) if len(ms) > 2 else np.nan

cred = eps[eps["credible"]]
summary = pd.Series(dict(
    position_source=source, stations_with_positions=len(positions), neighbour_pairs=len(pairs),
    episodes=len(eps), bottlenecks=len(bn), bottleneck_minutes=eps["minutes"].sum(),
    credible_episodes=len(cred), credible_bottlenecks=len(bn_cred),
    excluded_low_observed=int(((eps["observed_up"] < MIN_OBSERVED) | (eps["observed_down"] < MIN_OBSERVED)).sum()),
    excluded_off_ramp=int((eps["off_ramps_between"] > 0).sum()),
    excluded_night=int((~eps["start"].between(DAYTIME[0], DAYTIME[1] - 1)).sum()),
    excluded_not_recurrent=int((eps["days_active"] < MIN_DAYS_ACTIVE).sum()),
    upstream_occupancy_median=eps["occupancy_up"].median(),
    measured_discharge_median=bn["measured_discharge"].median(),
    credible_measured_discharge_median=bn_cred["measured_discharge"].median(),
    discharge_per_lane_all_lanes_median=eps["discharge_per_lane_all_lanes"].median(),
    credible_discharge_per_lane_all_lanes_median=cred["discharge_per_lane_all_lanes"].median(),
    capacity_drop_median=eps["capacity_drop"].median(),
    capacity_drop_iqr_lo=eps["capacity_drop"].quantile(0.25), capacity_drop_iqr_hi=eps["capacity_drop"].quantile(0.75),
    capacity_drop_negative_share=(eps["capacity_drop"].dropna() < 0).mean(),
    credible_capacity_drop_negative_share=(cred["capacity_drop"].dropna() < 0).mean(),
    credible_capacity_drop_median=cred["capacity_drop"].median(),
    credible_capacity_drop_iqr_lo=cred["capacity_drop"].quantile(0.25),
    credible_capacity_drop_iqr_hi=cred["capacity_drop"].quantile(0.75),
    **results,
    adjacent_sites=sub["station"].nunique(), adjacent_sites_multi_day=len(ms),
    adjacent_sd_between_sites=np.sqrt(var_site) if np.isfinite(var_site) else np.nan,
    adjacent_sd_within_site=np.sqrt(W) if np.isfinite(W) else np.nan,
))
bn.to_csv(paths.RESULTS / "bottlenecks.csv", index=False, float_format="%.3f")
summary.to_csv(paths.RESULTS / "bottleneck_summary.csv", header=["value"])
print(summary.to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
for ax, (target, col, title) in zip(axes[:2], [
        ("same_detector", "model_capacity_downstream", "(a) Same detector: its calibrated capacity\nvs its own discharge"),
        ("queue_detector", "model_capacity_upstream", "(b) Queue detector upstream: its calibrated\ncapacity vs the discharge below")]):
    a = bn.dropna(subset=[col])
    c = bn_cred.dropna(subset=[col])          # discharge from credible episodes only
    ax.plot(a["measured_discharge"], a[col], "o", ms=4, color=BLUE, alpha=0.35, label=f"all bottlenecks (n={len(a)})")
    ax.plot(c["measured_discharge"], c[col], "o", ms=7, color=ORANGE, mec="#fcfcfb",
            label=f"credible (n={len(c)})")
    lim = [min(a["measured_discharge"].min(), a[col].min()) - 100, max(a["measured_discharge"].max(), a[col].max()) + 100]
    ax.plot(lim, lim, color=INK_2, lw=1, ls="--", label="model = measured")
    note = ("below the line: model < measured\n(the only evidence against it)" if target == "same_detector" else
            "below the line: model < measured\n(possible evidence against it; on-ramps\nbetween the detectors can also cause it)")
    ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xlabel("measured lane-1 discharge while the bottleneck is active (veh/h)")
    ax.set_ylabel("calibrated lane-1 capacity (veh/h)")
    below = results.get(f"all_{target}_n_model_below_measured", 0)
    r_all = results.get(f"all_{target}_spearman", np.nan)
    q_all, q_c = results.get(f"all_{target}_ratio", np.nan), results.get(f"credible_{target}_ratio", np.nan)
    ax.set_title(f"{title}\nmodel/measured {q_all:.2f} (credible {q_c:.2f}); rank corr. {r_all:.2f}; "
                 f"below the line: {below} of {len(a)}", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
ax = axes[2]
bins = np.linspace(-40, 60, 26)
ax.hist(np.clip(100 * eps["capacity_drop"].dropna(), -40, 60), bins=bins, color=BLUE, alpha=0.35,
        edgecolor="#fcfcfb", label=f"all episodes (n={len(eps)})")
ax.hist(np.clip(100 * cred["capacity_drop"].dropna(), -40, 60), bins=bins, color=AQUA, edgecolor="#fcfcfb",
        label=f"credible (n={len(cred)})")
ax.axvline(100 * summary["credible_capacity_drop_median"], color=INK_2, lw=1.2, ls="--")
ax.set_xlabel("1 - discharge / pre-breakdown flow (%)")
ax.set_ylabel("bottleneck episodes")
ax.set_title(f"(c) Capacity drop, lane 1, 5-min data (clipped at -40 and 60)\nmedian "
             f"{100 * summary['capacity_drop_median']:.0f}% (credible "
             f"{100 * summary['credible_capacity_drop_median']:.0f}%); none in "
             f"{100 * summary['capacity_drop_negative_share']:.0f}% of episodes", fontsize=10)
ax.legend(fontsize=8.5)
fig.suptitle("Calibrated capacities against discharge measured below active bottlenecks (PeMS, 7 weekdays). "
             "One-sided check: a detector below a bottleneck need not be at capacity.",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, paths.FIGURES / "08_bottlenecks.png")
print("saved figures/08_bottlenecks.png")
