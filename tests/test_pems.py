"""Checks for nasch/pems.py, using a small synthetic file in the real PeMS format."""
import gzip

import numpy as np
import pandas as pd
import pytest

from nasch.pems import (MPH_TO_KMH, add_physical_units, density_from_occupancy, quality_mask,
                        read_lane1, screen_file, screen_station_day)

G_FT = 20.0


def _row(time, station, k_vpk, v_kmh, lane_type="ML", observed=1, samples=10, count=None):
    """One raw line: 12 station columns, lane-1 columns, then empty columns for lanes 2-3."""
    if count is None:
        count = int(round(k_vpk * v_kmh / 12.0))           # 5-minute count
    q = 12.0 * count
    v_mph = v_kmh / MPH_TO_KMH
    k = q / v_kmh if v_kmh > 0 else 0.0
    occ = k * MPH_TO_KMH * G_FT / 5280.0
    station_cols = [f"08/27/2026 {time}:00", station, 4, 880, "N", lane_type, 0.5, 10, 100,
                    count, round(occ, 6), round(v_mph, 3)]
    lane1 = [samples, count, round(occ, 6), round(v_mph, 3), observed]
    return ",".join(str(x) for x in station_cols + lane1) + ",,,,," * 2


def _congested_day(station):
    """288 intervals tracing a full fundamental diagram: free flow, capacity, congestion."""
    rows = []
    for i in range(288):
        time = f"{i // 12:02d}:{(i % 12) * 5:02d}"
        if i < 150:                                        # free flow at 110 km/h, 3-20 veh/km
            k, v = 3 + 17 * i / 149, 110.0
        else:                                              # congested, 25-80 veh/km
            k = 25 + 55 * (i - 150) / 137
            v = (2000 - 20 * (k - 25)) / k
        rows.append(_row(time, station, k, v))
    return rows


@pytest.fixture
def raw_file(tmp_path):
    lines = _congested_day(400001)
    lines += [_row(f"{i // 12:02d}:{(i % 12) * 5:02d}", 400002, 10, 110) for i in range(100)]
    lines += [_row(f"{i // 12:02d}:{(i % 12) * 5:02d}", 400003, 10, 110, lane_type="HV")
              for i in range(288)]
    path = tmp_path / "d04_text_station_5min_2026_08_27.txt.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def test_units_density_and_vehicle_length(raw_file):
    df = add_physical_units(read_lane1(raw_file))
    r = df.iloc[0]
    assert r["flow_vph"] == 12 * r["l1_flow"]
    assert r["speed_kmh"] == pytest.approx(110.0, abs=1e-2)
    assert r["density_vpk"] == pytest.approx(r["flow_vph"] / r["speed_kmh"])
    assert r["g_ft"] == pytest.approx(G_FT, rel=1e-3)
    assert (df["date"] == "2026-08-27").all()
    assert set(df["lane_type"]) == {"ML"}                  # HOV station dropped on reading


def test_density_from_occupancy():
    # 10% occupancy with a 25 ft effective length = 21.12 veh/mile = 13.12 veh/km
    assert density_from_occupancy(0.10, 25.0) == pytest.approx(0.10 * 5280 / 25 / 1.609344)


def test_quality_rules_drop_unobserved_empty_and_impossible_intervals():
    good = dict(l1_observed=1, l1_samples=10, l1_flow=100, l1_speed_mph=60.0, l1_occupancy=0.1)
    rows = [good,
            {**good, "l1_observed": 0},                    # imputed by PeMS
            {**good, "l1_samples": 0},
            {**good, "l1_flow": 0},
            {**good, "l1_occupancy": 0.9}]                 # implies a ~400 ft vehicle
    df = add_physical_units(pd.DataFrame(rows))
    assert quality_mask(df).tolist() == [True, False, False, False, False]


def test_screening_keeps_only_full_fundamental_diagrams(raw_file):
    screening, data = screen_file(raw_file)
    verdict = dict(zip(screening["station"], screening["reason"]))
    assert verdict[400001] == ""
    assert verdict[400002] == "too few usable intervals"
    assert 400003 not in verdict                           # not a mainline station
    assert set(data["station"]) == {400001}
    assert len(data) == 288
    row = screening.set_index("station").loc[400001]
    assert row["free_speed_kmh"] == pytest.approx(110.0, abs=0.5)
    assert 1700 <= row["q_max_vph"] <= 2500
    assert row["congested_intervals"] >= 15


def _day(k, v):
    k, v = np.asarray(k, float), np.asarray(v, float)
    return pd.DataFrame({"density_vpk": k, "speed_kmh": v, "flow_vph": k * v})


def test_screening_reasons():
    free = np.linspace(3, 20, 260)
    assert screen_station_day(_day(free, np.full(260, 110.0)))["reason"] == "not enough congestion"
    low = np.linspace(1, 10, 260)                          # never above 1100 veh/h
    assert screen_station_day(_day(low, np.full(260, 110.0)))["reason"].startswith("peak flow")
