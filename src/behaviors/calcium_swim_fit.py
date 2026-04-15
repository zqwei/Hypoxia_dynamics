from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_LAG_GRID = list(range(6, 61, 2))
DEFAULT_RIDGE_LAMBDA = 0.1
DEFAULT_TAU_FRAMES = 10.0
DEFAULT_SWIM_PERCENTILE = 70


@dataclass(frozen=True)
class CalciumSwimFitConfig:
    fit_window_s: tuple[float, float]
    color: str = "tab:blue"


def preprocess_swim(
    swim_signal: np.ndarray,
    percentile: float = DEFAULT_SWIM_PERCENTILE,
) -> np.ndarray:
    threshold = np.percentile(swim_signal, percentile)
    processed = np.asarray(swim_signal, dtype=float).copy()
    processed[processed < threshold] = threshold
    return np.sqrt(processed - threshold)


def build_lag_matrix(signal: np.ndarray, n_lags: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    n_time = len(signal)
    X = np.zeros((n_time, n_lags), dtype=float)
    X[:, 0] = signal
    for lag in range(1, n_lags):
        X[lag:, lag] = signal[:-lag]
    return X


def lag_penalty_diag(
    n_lags: int,
    tau_frames: float = DEFAULT_TAU_FRAMES,
) -> np.ndarray:
    return np.exp(np.arange(n_lags, dtype=float) / tau_frames)


def fit_weighted_ridge(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
) -> dict[str, np.ndarray | float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean

    penalty = np.diag(lag_penalty_diag(X.shape[1], tau_frames=tau_frames))
    weights = np.linalg.solve(Xc.T @ Xc + ridge_lambda * penalty, Xc.T @ yc)

    intercept = float(y_mean - x_mean @ weights)
    pred = intercept + X @ weights

    ss_res = np.square(y - pred).sum()
    ss_tot = np.square(y - y_mean).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    beta = float(np.linalg.norm(weights))
    kernel = weights / beta if beta > 0 else np.zeros_like(weights)

    return {
        "intercept": intercept,
        "weights": weights,
        "beta": beta,
        "kernel": kernel,
        "pred": pred,
        "r2": float(r2),
    }


def fit_weighted_ridge_batch(
    X: np.ndarray,
    Y: np.ndarray,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - x_mean
    Yc = Y - y_mean

    penalty = np.diag(lag_penalty_diag(X.shape[1], tau_frames=tau_frames))
    weights = np.linalg.solve(Xc.T @ Xc + ridge_lambda * penalty, Xc.T @ Yc)
    intercept = y_mean.ravel() - x_mean @ weights
    pred = intercept[None, :] + X @ weights

    ss_res = np.square(Y - pred).sum(axis=0)
    ss_tot = np.square(Y - y_mean).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "intercept": intercept,
        "weights": weights,
        "pred": pred,
        "r2": r2,
    }


def fit_calcium_swim_state(
    state_name: str,
    config: CalciumSwimFitConfig,
    time_stamp: np.ndarray,
    swim_signal: np.ndarray,
    dff: np.ndarray,
    lag_grid: list[int] | tuple[int, ...] = DEFAULT_LAG_GRID,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
) -> dict[str, object]:
    lo, hi = config.fit_window_s
    mask = (time_stamp >= lo) & (time_stamp < hi)
    time_s = time_stamp[mask]
    swim_window = np.asarray(swim_signal[mask], dtype=float)
    dff_window = np.asarray(dff[mask], dtype=float)

    lag_scan = []
    best_result: dict[str, object] | None = None
    best_r2 = -np.inf

    for n_lags in lag_grid:
        X = build_lag_matrix(swim_window, n_lags)
        fit = fit_weighted_ridge(X, dff_window, ridge_lambda=ridge_lambda, tau_frames=tau_frames)
        lag_scan.append({"state": state_name, "n_lags": n_lags, "r2": fit["r2"]})
        if fit["r2"] > best_r2:
            best_r2 = fit["r2"]
            best_result = fit | {"n_lags": n_lags, "X": X}

    if best_result is None:
        raise ValueError(f"No lag-fit result produced for state {state_name}.")

    return {
        "state": state_name,
        "config": config,
        "mask": mask,
        "time_s": time_s,
        "swim": swim_window,
        "dff": dff_window,
        "lag_scan_df": pd.DataFrame(lag_scan),
        "n_lags": best_result["n_lags"],
        "intercept": best_result["intercept"],
        "weights": best_result["weights"],
        "beta": best_result["beta"],
        "kernel": best_result["kernel"],
        "pred": best_result["pred"],
        "r2": best_result["r2"],
        "X": best_result["X"],
    }


def fit_calcium_swim_states(
    windows: dict[str, CalciumSwimFitConfig],
    time_stamp: np.ndarray,
    swim_signal: np.ndarray,
    dff: np.ndarray,
    lag_grid: list[int] | tuple[int, ...] = DEFAULT_LAG_GRID,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
) -> dict[str, dict[str, object]]:
    return {
        state: fit_calcium_swim_state(
            state,
            config,
            time_stamp,
            swim_signal,
            dff,
            lag_grid=lag_grid,
            ridge_lambda=ridge_lambda,
            tau_frames=tau_frames,
        )
        for state, config in windows.items()
    }


def summarize_calcium_swim_fits(
    fit_results: dict[str, dict[str, object]],
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state": state,
                "fit_window_s": result["config"].fit_window_s,
                "n_timepoints": len(result["dff"]),
                "selected_lags": result["n_lags"],
                "lambda": ridge_lambda,
                "tau_frames": tau_frames,
                "intercept": result["intercept"],
                "beta": result["beta"],
                "r2": result["r2"],
            }
            for state, result in fit_results.items()
        ]
    )


def benchmark_single_trace_search(
    fit_results: dict[str, dict[str, object]],
    lag_grid: list[int] | tuple[int, ...] = DEFAULT_LAG_GRID,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
) -> float:
    started = time.perf_counter()
    for result in fit_results.values():
        for n_lags in lag_grid:
            X = build_lag_matrix(result["swim"], n_lags)
            fit_weighted_ridge(X, result["dff"], ridge_lambda=ridge_lambda, tau_frames=tau_frames)
    return time.perf_counter() - started


def benchmark_batch_search(
    fit_results: dict[str, dict[str, object]],
    lag_grid: list[int] | tuple[int, ...] = DEFAULT_LAG_GRID,
    batch_cells: int = 1000,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for result in fit_results.values():
        base = np.asarray(result["dff"], dtype=float)
        Y = np.tile(base[:, None], (1, batch_cells))
        Y = Y + 0.1 * rng.standard_normal((len(base), batch_cells))
        for n_lags in lag_grid:
            X = build_lag_matrix(result["swim"], n_lags)
            fit_weighted_ridge_batch(X, Y, ridge_lambda=ridge_lambda, tau_frames=tau_frames)
    return time.perf_counter() - started


def summarize_calcium_swim_runtime(
    fit_results: dict[str, dict[str, object]],
    lag_grid: list[int] | tuple[int, ...] = DEFAULT_LAG_GRID,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
    batch_cells: int = 1000,
    projected_cells: int = 100_000,
    seed: int = 0,
) -> pd.DataFrame:
    single_trace_runtime_s = benchmark_single_trace_search(
        fit_results,
        lag_grid=lag_grid,
        ridge_lambda=ridge_lambda,
        tau_frames=tau_frames,
    )
    batch_runtime_s = benchmark_batch_search(
        fit_results,
        lag_grid=lag_grid,
        batch_cells=batch_cells,
        ridge_lambda=ridge_lambda,
        tau_frames=tau_frames,
        seed=seed,
    )

    return pd.DataFrame(
        [
            {
                "single_trace_runtime_s": single_trace_runtime_s,
                "batch_runtime_s": batch_runtime_s,
                "batch_cells": batch_cells,
                "projected_naive_100k_cells_hours": single_trace_runtime_s * projected_cells / 3600,
                "projected_batch_100k_cells_hours": batch_runtime_s
                * (projected_cells / batch_cells)
                / 3600,
            }
        ]
    )
