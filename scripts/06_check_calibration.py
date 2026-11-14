"""Three checks that the calibration and its error bars can be trusted. Each is applied to the
calibrated human queue-discharge capacity (veh/h) and to the AV gain.

  recovery   Make synthetic station-days from the calibrated model itself (known truth) by adding
             the real residuals back with their signs flipped at random in 1-hour blocks,
             re-calibrate them, and check that the truth is recovered and that the 95% bootstrap
             interval contains it ~95% of the time.
  mc         Rebuild the library with independent random numbers for the parameter sets that matter
             and refit: does Monte Carlo noise in the library change the answers?
  blocks     Repeat the bootstrap with 30-minute and 2-hour blocks instead of 1-hour blocks.

    python scripts/06_check_calibration.py [recovery|mc|blocks|all]
        needs   the outputs of scripts/04 and scripts/05
        writes  results/check_recovery.csv, results/check_mc.csv, results/check_blocks.csv,
                results/cache/fd_library_mc.npz,
                figures/04_calibration_checks.png (when all three exist)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.calibration import (CELL_KM, DT_GRID, Library, av_gain, fit_station_day,  # noqa: E402
                               geh_matrix, load_library, model_flow, simulate_fd)
from nasch.plotting import BLUE, INK_2, ORANGE, plt, save, use_style  # noqa: E402

part = sys.argv[1] if len(sys.argv) > 1 else "all"
paths.ensure_dirs()
lib = load_library(paths.CACHE / "fd_library.npz")
discharge = pd.read_csv(paths.RESULTS / "human_discharge.csv")["discharge"].to_numpy()
gain = av_gain(lib.grid["vmax"].to_numpy(), discharge)
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
data = pd.read_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", dtype={"time": str})
groups = {key: g.sort_values("time") for key, g in data.groupby(["station", "date"])}
kept = cal[cal["kept_for_av_gain"]].reset_index(drop=True)


def day_arrays(r):
    d = groups[(r["station"], r["date"])]
    return d["density_vpk"].to_numpy(), d["flow_vph"].to_numpy(), d["time"].tolist()


# ---------------------------------------------------------------- 1. parameter recovery
if part in ("recovery", "all"):
    rng = np.random.default_rng(2026)
    sample = kept.sample(min(200, len(kept)), random_state=7).reset_index(drop=True)
    rows = []
    t0 = time.time()
    for n, r in enumerate(sample.to_dict("records"), 1):
        k, q, times = day_arrays(r)
        i_true, dt_true = int(r["param_index"]), float(r["dt"])
        m = 3600 * model_flow(lib.subset([i_true]), k * CELL_KM)[:, 0] / dt_true
        # synthetic day = model at the true parameters + the real residuals, each kept at its own
        # time (so it keeps its real size) with its sign flipped at random in 1-hour blocks
        resid = q - m
        signs = np.repeat(rng.choice([-1.0, 1.0], size=int(np.ceil(k.size / 12))), 12)[:k.size]
        q_syn = np.maximum(m + signs * resid, 12.0)
        out, _ = fit_station_day(lib, discharge, k, q_syn, times, n_boot=200, seed=n)
        g_true, c_true = gain[i_true], 3600 * discharge[i_true] / dt_true
        rows.append(dict(
            station=r["station"], date=r["date"], true_param=i_true, fit_param=out["param_index"],
            true_dt=dt_true, fit_dt=out["dt"],
            true_capacity=c_true, fit_capacity=out["capacity_vph"],
            capacity_lo=out["capacity_lo"], capacity_hi=out["capacity_hi"],
            capacity_covered=out["capacity_lo"] <= c_true <= out["capacity_hi"],
            true_gain=g_true, fit_gain=out["av_gain"], gain_lo=out["av_gain_lo"],
            gain_hi=out["av_gain_hi"], gain_covered=out["av_gain_lo"] <= g_true <= out["av_gain_hi"]))
        if n % 50 == 0:
            print(f"recovery {n}/{len(sample)} ({(time.time() - t0) / n:.1f} s each)", flush=True)
    rec = pd.DataFrame(rows)
    rec.to_csv(paths.RESULTS / "check_recovery.csv", index=False, float_format="%.5f")
    cap_err = 100 * (rec.fit_capacity / rec.true_capacity - 1)
    print(f"recovery: same parameter set {100 * (rec.fit_param == rec.true_param).mean():.0f}%; "
          f"capacity error median |{cap_err.abs().median():.1f}|%, interval coverage "
          f"{100 * rec.capacity_covered.mean():.1f}%; AV-gain error median "
          f"|{100 * (rec.fit_gain - rec.true_gain).abs().median():.1f}| points, interval coverage "
          f"{100 * rec.gain_covered.mean():.1f}%")

# ---------------------------------------------------------------- 2. Monte Carlo check
if part in ("mc", "all"):
    mc_lib = paths.CACHE / "fd_library_mc.npz"
    if mc_lib.exists():
        z = np.load(mc_lib)
        union, alt_flow = z["union"], z["flow"]
        print(f"mc: reusing {union.size} re-simulated parameter sets from {mc_lib.name}")
    else:
        top = set()
        for r in kept.to_dict("records"):
            k, q, _ = day_arrays(r)
            G = geh_matrix(model_flow(lib, k * CELL_KM), q).reshape(k.size, len(lib.grid), len(DT_GRID))
            top.update(np.argsort(G.mean(axis=0).min(axis=1))[:20].tolist())
        union = np.array(sorted(top))
        print(f"mc: {union.size} parameter sets are in some station-day's top 20; re-simulating them",
              flush=True)
        alt_flow = np.empty((union.size, lib.rho.size))
        t0 = time.time()
        for n, i in enumerate(union, 1):
            alt_flow[n - 1] = simulate_fd(lib.grid.iloc[i], seeds=range(100, 104)).mean(axis=1)
            if n % 50 == 0:
                print(f"mc library {n}/{union.size} ({(time.time() - t0) / n:.1f} s each)", flush=True)
        np.savez_compressed(mc_lib, union=union, flow=alt_flow)
    main_sub = lib.subset(union)
    alt_sub = Library(lib.grid.iloc[union].reset_index(drop=True), lib.rho, alt_flow)
    rows = []
    for r in kept.to_dict("records"):
        k, q, times = day_arrays(r)
        a, _ = fit_station_day(main_sub, discharge[union], k, q, times, n_boot=2)
        b, _ = fit_station_day(alt_sub, discharge[union], k, q, times, n_boot=2)
        rows.append(dict(station=r["station"], date=r["date"], same_params=a["param_index"] == b["param_index"],
                         capacity_main=a["capacity_vph"], capacity_alt=b["capacity_vph"],
                         gain_main=a["av_gain"], gain_alt=b["av_gain"],
                         dt_main=a["dt"], dt_alt=b["dt"], geh_main=a["geh_all"], geh_alt=b["geh_all"]))
    mc = pd.DataFrame(rows)
    mc.to_csv(paths.RESULTS / "check_mc.csv", index=False, float_format="%.5f")
    dc = 100 * (mc.capacity_alt / mc.capacity_main - 1)
    dg = 100 * (mc.gain_alt - mc.gain_main)
    print(f"mc: same best parameter set {100 * mc.same_params.mean():.0f}%; capacity change SD "
          f"{dc.std():.2f}% vs median bootstrap SE {100 * (kept.capacity_se / kept.capacity_vph).median():.2f}%; "
          f"AV-gain change SD {dg.std():.2f} vs median bootstrap SE {100 * kept.av_gain_se.median():.2f} points")

# ---------------------------------------------------------------- 3. bootstrap block length
if part in ("blocks", "all"):
    rows = []
    t0 = time.time()
    for n, r in enumerate(kept.to_dict("records"), 1):
        k, q, times = day_arrays(r)
        res = dict(station=r["station"], date=r["date"],
                   cap_se_block12=r["capacity_se"] / r["capacity_vph"], gain_se_block12=r["av_gain_se"])
        for block in (6, 24):
            out, _ = fit_station_day(lib, discharge, k, q, times, n_boot=200, block=block, seed=n)
            res[f"cap_se_block{block}"] = out["capacity_se"] / out["capacity_vph"]
            res[f"gain_se_block{block}"] = out["av_gain_se"]
        rows.append(res)
        if n % 100 == 0:
            print(f"blocks {n}/{len(kept)} ({(time.time() - t0) / n:.1f} s each)", flush=True)
    bl = pd.DataFrame(rows)
    bl.to_csv(paths.RESULTS / "check_blocks.csv", index=False, float_format="%.5f")
    print("blocks: median relative SE of capacity: "
          + ", ".join(f"{b} intervals {100 * bl[f'cap_se_block{b}'].median():.2f}%" for b in (6, 12, 24)))

# ---------------------------------------------------------------- figure
files = [paths.RESULTS / f"check_{x}.csv" for x in ("recovery", "mc", "blocks")]
if all(f.exists() for f in files):
    use_style()
    rec, mc, bl = (pd.read_csv(f) for f in files)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    def identity(ax, lo, hi):
        ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls="--")

    ax = axes[0, 0]
    ax.errorbar(rec.true_capacity, rec.fit_capacity,
                yerr=[rec.fit_capacity - rec.capacity_lo, rec.capacity_hi - rec.fit_capacity],
                fmt="o", ms=3, color=BLUE, elinewidth=0.6, alpha=0.6)
    identity(ax, rec[["true_capacity", "capacity_lo"]].min().min() - 50,
             rec[["true_capacity", "capacity_hi"]].max().max() + 50)
    ax.set_xlabel("true human discharge capacity (veh/h)")
    ax.set_ylabel("recovered, with 95% interval (veh/h)")
    ax.set_title(f"(a) Recovery: human discharge capacity, {len(rec)} synthetic days\n"
                 f"interval contains the truth {100 * rec.capacity_covered.mean():.0f}% of the time (target 95%)")

    ax = axes[0, 1]
    ax.errorbar(100 * rec.true_gain, 100 * rec.fit_gain,
                yerr=[100 * (rec.fit_gain - rec.gain_lo), 100 * (rec.gain_hi - rec.fit_gain)],
                fmt="o", ms=3, color=BLUE, elinewidth=0.6, alpha=0.6)
    identity(ax, 100 * rec[["true_gain", "gain_lo"]].min().min() - 5,
             100 * rec[["true_gain", "gain_hi"]].max().max() + 5)
    ax.set_xlabel("true AV gain, AV defined in the site's cells and steps (%)")
    ax.set_ylabel("recovered, with 95% interval (%)")
    ax.set_title(f"(b) Recovery: AV gain as pre-registered\n"
                 f"interval contains the truth {100 * rec.gain_covered.mean():.0f}% of the time (target 95%)")

    ax = axes[1, 0]
    ax.plot(mc.capacity_main, mc.capacity_alt, "o", ms=3, color=BLUE, alpha=0.4)
    identity(ax, mc[["capacity_main", "capacity_alt"]].min().min() - 50,
             mc[["capacity_main", "capacity_alt"]].max().max() + 50)
    sd = (100 * (mc.capacity_alt / mc.capacity_main - 1)).std()
    ax.set_xlabel("human discharge capacity, main library (veh/h)")
    ax.set_ylabel("same, library with new random numbers (veh/h)")
    ax.set_title(f"(c) Monte Carlo check: library noise moves capacity\nby {sd:.1f}% (SD) per station-day")

    ax = axes[1, 1]
    grey = dict(color=INK_2, lw=1.2)
    ax.boxplot([100 * bl[f"cap_se_block{b}"] for b in (6, 12, 24)],
               tick_labels=["30 min", "1 h (main)", "2 h"], widths=0.5, showfliers=False,
               medianprops=dict(color=ORANGE, lw=2), boxprops=grey, whiskerprops=grey, capprops=grey)
    ax.set_xlabel("bootstrap block length")
    ax.set_ylabel("bootstrap SE of human discharge capacity (%)")
    ax.set_title("(d) Error bars vs bootstrap block length\n(longer blocks respect more autocorrelation)")
    save(fig, paths.FIGURES / "04_calibration_checks.png")
    print("saved figures/04_calibration_checks.png")
