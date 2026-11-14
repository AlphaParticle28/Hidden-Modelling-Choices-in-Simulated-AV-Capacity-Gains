"""Checks for nasch/calibration.py. These use a small library of exact triangular fundamental
diagrams (the p = 0 case), so they run in a fraction of a second."""
import numpy as np
import pandas as pd
import pytest

from nasch.calibration import (CELL_KM, DT_GRID, Library, av_gain, best_columns,
                               block_bootstrap_weights, fit_station_day, geh, geh_matrix,
                               holdout_masks, library_densities, model_flow, parameter_grid,
                               split_column)


def triangle_library():
    grid = pd.DataFrame({"vmax": [3, 4, 4, 5, 4], "p": 0.0, "p0": np.nan, "h": [0, 0, 1, 0, 2]})
    rho = library_densities()
    flow = np.array([np.maximum(0, np.minimum(v * rho, 1 - (1 + h) * rho))
                     for v, h in zip(grid["vmax"], grid["h"])])
    return Library(grid, rho, flow)


def a_day(n=288):
    times = [f"{i // 12:02d}:{(i % 12) * 5:02d}" for i in range(n)]
    k = np.r_[np.linspace(2, 25, n // 2), np.linspace(30, 90, n - n // 2)]   # veh/km
    return k, times


def test_parameter_grid_matches_the_analysis_plan():
    g = parameter_grid()
    assert len(g) == 1980
    assert set(g["vmax"]) == {3, 4, 5, 6} and set(g["h"]) == {0, 1, 2}
    has_p0 = g["p0"].notna()
    assert (g.loc[has_p0, "p0"] > g.loc[has_p0, "p"]).all()     # slow-to-start means p0 > p


def test_geh():
    assert geh(100, 100) == 0
    assert geh(1100, 1000) == pytest.approx(np.sqrt(2 * 100 ** 2 / 2100))
    assert geh(0, 0) == 0


def test_model_flow_is_exact_at_library_points_and_zero_beyond():
    lib = triangle_library()
    J = model_flow(lib, np.r_[lib.rho[:3], 0.0, 1.5])
    np.testing.assert_allclose(J[:3], lib.flow[:, :3].T)
    assert np.all(J[3] == 0) and np.all(J[4] == 0)


def test_holdout_is_every_fourth_hour():
    _, times = a_day()
    train, test = holdout_masks(times)
    hours = sorted({int(t[:2]) for t, s in zip(times, test) if s})
    assert hours == [3, 7, 11, 15, 19, 23] and test.sum() == 72 and train.sum() == 216


def test_block_bootstrap_weights():
    rng = np.random.default_rng(0)
    W = block_bootstrap_weights(288, 12, 50, rng)
    assert W.shape == (50, 288) and np.all(W.sum(axis=1) == 288)
    assert np.all(block_bootstrap_weights(288, 288, 3, rng) == 1)    # one block = the whole day


def test_best_columns_minimises_weighted_mean():
    G = np.array([[1.0, 5.0], [9.0, 1.0]], np.float32)
    assert best_columns(G, [1, 0])[0][0] == 0
    assert best_columns(G, [0, 1])[0][0] == 1
    assert best_columns(G, [1, 1])[0][0] == 1                          # means 5.0 vs 3.0


@pytest.mark.parametrize("true_idx, true_dt", [(1, 1.0), (2, 0.9), (3, 1.2)])
def test_noise_free_data_are_fitted_exactly(true_idx, true_dt):
    lib = triangle_library()
    k, times = a_day()
    q = 3600 * model_flow(lib, k * CELL_KM)[:, true_idx] / true_dt
    discharge = np.array([v / (v + 1 + h) for v, h in zip(lib.grid["vmax"], lib.grid["h"])])
    out, boot = fit_station_day(lib, discharge, k, q, times, n_boot=50)
    row = lib.grid.iloc[true_idx]
    assert (out["vmax"], out["h"]) == (row["vmax"], row["h"])
    assert out["dt"] == pytest.approx(true_dt)
    assert out["geh_all"] == pytest.approx(0, abs=1e-6) and out["geh_test"] == pytest.approx(0, abs=1e-6)
    assert out["boot_share_same_params"] == 1.0
    assert out["av_gain"] == pytest.approx(av_gain(row["vmax"], discharge[true_idx]))
    cap = 3600 * discharge[true_idx] / true_dt
    assert out["capacity_vph"] == pytest.approx(cap)
    assert out["capacity_lo"] == pytest.approx(cap) and out["capacity_hi"] == pytest.approx(cap)
    i, d = split_column(boot, len(DT_GRID))
    assert np.all(i == true_idx) and np.allclose(DT_GRID[d], true_dt)


def test_library_without_every_h_value_still_fits():
    lib = triangle_library().subset([0, 1, 3])                       # h = 0 only
    k, times = a_day()
    q = 3600 * model_flow(lib, k * CELL_KM)[:, 1]
    out, _ = fit_station_day(lib, np.full(3, 0.5), k, q, times, n_boot=10)
    assert out["h"] == 0 and np.isnan(out["geh_best_h1"]) and np.isnan(out["geh_best_h2"])


def test_custom_dt_grid_as_a_list():
    lib = triangle_library()
    k, times = a_day()
    q = 3600 * model_flow(lib, k * CELL_KM)[:, 1] / 1.2
    out, _ = fit_station_day(lib, np.full(5, 0.5), k, q, times, n_boot=10, dt_grid=[0.8, 1.0, 1.2])
    assert out["dt"] == pytest.approx(1.2) and out["dt_lo"] == pytest.approx(1.2)


def test_forcing_the_wrong_buffer_costs_fit():
    lib = triangle_library()
    k, times = a_day()
    q = 3600 * model_flow(lib, k * CELL_KM)[:, 1]                     # true: vmax 4, h 0, dt 1
    out, _ = fit_station_day(lib, np.full(5, 0.5), k, q, times, n_boot=20)
    assert out["geh_best_h0"] == pytest.approx(0, abs=1e-6)
    assert out["geh_best_h1"] > 1 and out["geh_best_h2"] > out["geh_best_h1"]
