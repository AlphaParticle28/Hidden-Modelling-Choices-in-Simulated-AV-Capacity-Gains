"""Is the CA's weaker congested fit about the model or about the objective?

The main calibration minimises mean GEH over all intervals, and about 80% of them are free flow. Here
each training day is refitted with free-flow and congested intervals weighted 50/50, then scored on
congested held-out intervals, as in scripts/09. Capacities from both objectives are compared.

    python scripts/15_balanced_fit.py   -> results/balanced_fit.csv, results/balanced_summary.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.calibration import (CELL_KM, DT_GRID, best_columns, geh, geh_matrix, holdout_masks,  # noqa: E402
                               load_library, model_flow, split_column)

lib = load_library(paths.CACHE / "fd_library.npz")
discharge = pd.read_csv(paths.RESULTS / "human_discharge.csv")["discharge"].to_numpy()
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
kept = cal[cal["kept_for_av_gain"]]
base = pd.read_csv(paths.RESULTS / "baseline_comparison.csv")
data = pd.read_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", dtype={"time": str})
groups = {key: g.sort_values("time") for key, g in data.groupby(["station", "date"])}


def balanced_weights(mask_use, cong):
    w = np.zeros(mask_use.size, np.float32)
    for part in (cong, ~cong):
        sel = mask_use & part
        if sel.any():
            w[sel] = 0.5 / sel.sum()
    return w


rows = []
for r in kept.to_dict("records"):
    d = groups[(r["station"], r["date"])]
    k, q, v = d["density_vpk"].to_numpy(), d["flow_vph"].to_numpy(), d["speed_kmh"].to_numpy()
    train, test = holdout_masks(d["time"].tolist())
    cong = v < 60
    J = model_flow(lib, k * CELL_KM)
    G = geh_matrix(J, q)
    col_tr, _ = best_columns(G, balanced_weights(train, cong))
    col_all, _ = best_columns(G, balanced_weights(np.ones(k.size, bool), cong))
    i, dd = split_column(int(col_tr[0]))
    target = test & cong
    row = dict(station=r["station"], date=r["date"], capacity_ca=r["capacity_vph"])
    if target.sum() >= 3:
        row["geh_congested_test_balanced"] = float(geh(3600 * J[target, int(i)] / DT_GRID[dd], q[target]).mean())
    ia, da = split_column(int(col_all[0]))
    row["capacity_balanced"] = float(3600 * discharge[int(ia)] / DT_GRID[da])
    rows.append(row)

b = pd.DataFrame(rows).merge(base[["station", "date", "geh_ca", "geh_triangle"]], on=["station", "date"], how="left")
b.to_csv(paths.RESULTS / "balanced_fit.csv", index=False, float_format="%.4f")
ok = b.dropna(subset=["geh_congested_test_balanced", "geh_ca", "geh_triangle"])
multi = b.groupby("station").filter(lambda g: len(g) >= 2)
summary = pd.Series(dict(
    scored=len(ok),
    congested_test_geh_balanced=ok["geh_congested_test_balanced"].median(),
    congested_test_geh_original=ok["geh_ca"].median(), congested_test_geh_triangle=ok["geh_triangle"].median(),
    share_balanced_better_than_triangle=(ok["geh_congested_test_balanced"] < ok["geh_triangle"]).mean(),
    capacity_spearman_balanced_vs_original=spearmanr(b["capacity_ca"], b["capacity_balanced"])[0],
    capacity_ratio_balanced_to_original=(b["capacity_balanced"] / b["capacity_ca"]).median(),
    detector_sd_balanced=b.groupby("station")["capacity_balanced"].mean().std(),
    day_sd_balanced=np.sqrt(multi.groupby("station")["capacity_balanced"].var(ddof=1).mean()),
))
summary.to_csv(paths.RESULTS / "balanced_summary.csv", header=["value"])
print(summary.round(3).to_string())
