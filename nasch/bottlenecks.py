"""Finding active freeway bottlenecks in PeMS data and measuring their discharge directly.

An active bottleneck sits between two neighbouring detectors when the upstream one is in a queue
and the downstream one is flowing freely at the same time (Chen, Skabardonis & Varaiya, 2004).
While a bottleneck is active, the flow at the first free-flowing detector below it is the
bottleneck's discharge (plus or minus any ramp in between), measured rather than modelled. It is
not that detector's own capacity: a free-flowing section below a bottleneck carries at most its
capacity, and its lane 1 need not be saturated. So comparing it with a calibrated capacity is a
one-sided check. The highest flow just before the queue formed is the pre-breakdown flow; the gap
between the two is the capacity drop (Cassidy & Bertini, 1999).

Detector positions come from the PeMS station metadata file if it is in data/raw/; otherwise from
the open PEMS-BAY sensor coordinates (Li et al., 2018), which cover part of District 4. The
metadata also lists on-ramp ("OR") and off-ramp ("FR") detectors, which locate the ramps.
"""
import numpy as np
import pandas as pd
from scipy.stats import binom

from . import paths

# detection rule (speeds in km/h): upstream in a queue, downstream much faster, detectors close
QUEUE_SPEED = 40 * 1.609344          # 40 mph
SPEED_DIFFERENCE = 20 * 1.609344     # 20 mph
MAX_SPACING_KM = 2 * 1.609344        # 2 miles (Chen et al. 2004)
PERSIST = (5, 7)                     # active if the rule holds in at least 5 of 7 consecutive intervals
MIN_EPISODE = 6                      # an episode must last at least 30 minutes
SKIP_ONSET = 2                       # ignore the first 10 minutes of an episode (transient)
PRE_WINDOW = 6                       # look for the pre-breakdown flow in the 30 minutes before onset

DIRECTION_SIGN = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}   # (lat, lon) of travel
RAMP_TOLERANCE_KM = 0.1              # a ramp this close to a detector counts as "between" (conservative)
RAMP_CAP_KM = 3.0                    # distances to ramps are capped here


def load_metadata():
    """Every station in the newest PeMS station metadata file in data/raw/, or None if there is none.
    pos_km is the absolute postmile in km, signed so that it increases in the direction of travel
    (postmiles increase northbound and eastbound)."""
    meta = sorted(paths.DATA_RAW.glob("d04_text_meta_*.txt"))
    if not meta:
        return None
    m = pd.read_csv(meta[-1], sep="\t")
    sign = m["Dir"].map({"N": 1, "E": 1, "S": -1, "W": -1})
    return pd.DataFrame({"station": m["ID"], "freeway": m["Fwy"], "direction": m["Dir"], "type": m["Type"],
                         "lanes": m["Lanes"], "lat": m["Latitude"], "lon": m["Longitude"],
                         "pos_km": m["Abs_PM"] * 1.609344 * sign})


def load_positions():
    """Mainline station, freeway, direction and position along the road (km, increasing downstream)."""
    meta = load_metadata()
    if meta is not None:
        out = meta[meta["type"] == "ML"].drop(columns="type").dropna().reset_index(drop=True)
        out["source"] = "PeMS metadata"
        return out
    loc = pd.read_csv(paths.ROOT / "data" / "external" / "pems_bay_sensor_locations.csv",
                      header=None, names=["station", "lat", "lon"])
    loc["source"] = "PEMS-BAY coordinates"
    return loc


def ramps_between(meta, freeway, direction, pos_up, pos_down, kind, tol_km=RAMP_TOLERANCE_KM):
    """Number of ramp detectors of `kind` ("OR" on-ramp, "FR" off-ramp) on the same freeway and
    direction between two mainline positions. A ramp within tol_km of either detector counts as
    between, so an ambiguous ramp excludes a pair rather than slipping through."""
    r = meta[(meta["type"] == kind) & (meta["freeway"] == freeway) & (meta["direction"] == direction)]
    return int(r["pos_km"].between(pos_up - tol_km, pos_down + tol_km).sum())


def ramp_distances(meta, cap_km=RAMP_CAP_KM, tol_km=RAMP_TOLERANCE_KM):
    """For each mainline station: distance (km) to the nearest on-ramp upstream and to the nearest
    off-ramp downstream on the same freeway and direction, capped at cap_km."""
    ml = meta[meta["type"] == "ML"].dropna(subset=["pos_km"])
    rows = []
    for (fwy, d), g in ml.groupby(["freeway", "direction"]):
        r = meta[(meta["freeway"] == fwy) & (meta["direction"] == d)]
        on = r.loc[r["type"] == "OR", "pos_km"].dropna().to_numpy()
        off = r.loc[r["type"] == "FR", "pos_km"].dropna().to_numpy()
        for st, x in zip(g["station"], g["pos_km"]):
            up = x - on[on <= x + tol_km]
            down = off[off >= x - tol_km] - x
            rows.append(dict(station=st,
                             on_ramp_upstream_km=min(max(up.min(), 0.0), cap_km) if up.size else cap_km,
                             off_ramp_downstream_km=min(max(down.min(), 0.0), cap_km) if down.size else cap_km))
    return pd.DataFrame(rows)


def order_along_road(group):
    """Order one freeway-direction's stations by projecting their coordinates on the road's main
    axis, oriented in the direction of travel. Returns positions in km."""
    lat0 = group["lat"].mean()
    xy = np.c_[group["lat"].to_numpy() * 111.0,
               group["lon"].to_numpy() * 111.0 * np.cos(np.radians(lat0))]
    centred = xy - xy.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    dlat, dlon = DIRECTION_SIGN.get(group["direction"].iloc[0], (1, 0))
    if axis @ np.array([dlat, dlon]) < 0:
        axis = -axis
    return centred @ axis


def neighbour_pairs(positions):
    """(upstream, downstream) station pairs that are consecutive along the road and close enough."""
    pairs = []
    for (_, _), g in positions.groupby(["freeway", "direction"]):
        if len(g) < 2:
            continue
        g = g.sort_values("pos_km")
        st, pos = g["station"].to_numpy(), g["pos_km"].to_numpy()
        xy = np.c_[g["lat"].to_numpy() * 111.0, g["lon"].to_numpy() * 111.0 * np.cos(np.radians(g["lat"].mean()))]
        for i in range(len(g) - 1):
            gap = pos[i + 1] - pos[i]
            straight = np.hypot(*(xy[i + 1] - xy[i]))
            if 0 < gap <= MAX_SPACING_KM and straight <= MAX_SPACING_KM:
                pairs.append((st[i], st[i + 1], gap))
    return pairs


def median_interval(x, level=0.95):
    """Distribution-free interval for a median from order statistics (binomial), with coverage of
    at least `level`. For 7 values at 95% it is (smallest, largest). NaN if there are too few values."""
    x = np.sort(np.asarray(x, float))
    n, k = x.size, 0
    while k < n and binom.cdf(k, n, 0.5) <= (1 - level) / 2:
        k += 1
    return (x[k - 1], x[n - k]) if k > 0 else (np.nan, np.nan)


def active_mask(v_up, v_down):
    """Interval-by-interval bottleneck activity between two detectors, from all-lane station speeds.
    The rule must hold in at least 5 of 7 consecutive intervals. Within such a window, everything from
    the first to the last interval where the rule holds counts as active, so a single missed interval
    does not split one event, while the event does not spill past its real start or end."""
    rule = (v_up < QUEUE_SPEED) & (v_down - v_up > SPEED_DIFFERENCE)
    need, window = PERSIST
    hits = pd.Series(rule.astype(float)).rolling(window, min_periods=window).sum().to_numpy()
    active = np.zeros(len(rule), bool)
    for t in np.flatnonzero(hits >= need):
        idx = np.flatnonzero(rule[t - window + 1:t + 1]) + t - window + 1
        active[idx[0]:idx[-1] + 1] = True
    return active


def episodes(active):
    """(start, end) index ranges of contiguous active stretches, end exclusive."""
    out, start = [], None
    for t, a in enumerate(np.r_[active, False]):
        if a and start is None:
            start = t
        elif not a and start is not None:
            out.append((start, t))
            start = None
    return out


def measure_pair(day_up, day_down):
    """Discharge episodes for one upstream/downstream pair on one day. Inputs: 288-row frames indexed
    by interval number with columns station_speed_kmh (all lanes, used for detection), flow_vph
    (lane 1) and lane_avg_flow_vph (total flow / lanes); NaN where missing. If both frames also have
    pct_observed and station_occupancy, each episode reports their means over the discharge window
    (observed_up, observed_down, occupancy_up). The pre-breakdown window never reaches back into the
    previous active stretch."""
    v_up, v_dn = day_up["station_speed_kmh"].to_numpy(), day_down["station_speed_kmh"].to_numpy()
    q_dn = day_down["flow_vph"].to_numpy()
    q_all = day_down["lane_avg_flow_vph"].to_numpy()
    ok = np.isfinite(v_up) & np.isfinite(v_dn)
    act = active_mask(np.where(ok, v_up, np.inf), np.where(ok, v_dn, -np.inf))
    rows, prev_end = [], 0
    for s, e in episodes(act):
        pre_start, prev_end = max(prev_end, s - PRE_WINDOW), e
        if e - s < MIN_EPISODE:
            continue
        dis = q_dn[s + SKIP_ONSET:e]
        dis = dis[np.isfinite(dis)]
        pre = pd.Series(q_dn[pre_start:s]).rolling(3, min_periods=3).mean().max()
        if dis.size < 3:
            continue
        dis_all = q_all[s + SKIP_ONSET:e]
        dis_all = dis_all[np.isfinite(dis_all)]
        row = dict(start=s, end=e, minutes=5 * (e - s), intervals=int(dis.size), discharge_vph=float(dis.mean()),
                   prebreakdown_vph=float(pre) if np.isfinite(pre) else np.nan,
                   discharge_per_lane_all_lanes=float(dis_all.mean()) if dis_all.size else np.nan)
        extra = ("pct_observed", "station_occupancy")
        if all(c in day_up and c in day_down for c in extra):
            window = slice(s + SKIP_ONSET, e)
            row.update(observed_up=float(np.nanmean(day_up["pct_observed"].to_numpy()[window])),
                       observed_down=float(np.nanmean(day_down["pct_observed"].to_numpy()[window])),
                       occupancy_up=float(np.nanmean(day_up["station_occupancy"].to_numpy()[window])))
        rows.append(row)
    return rows
