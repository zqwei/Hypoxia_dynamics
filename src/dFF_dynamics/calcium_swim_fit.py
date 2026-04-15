from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.paths import load_datalist


DEFAULT_LAG_GRID = list(range(6, 61, 2))
DEFAULT_RIDGE_LAMBDA = 0.1
DEFAULT_TAU_FRAMES = 10.0
DEFAULT_SWIM_PERCENTILE = 70
DEFAULT_NORMOXIA_WINDOW_S = (600.0, 1200.0)
DEFAULT_HYPOXIA_WINDOW_S = (1800.0, 2400.0)
DEFAULT_SPEARMAN_THRESHOLD = 0.2
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_N_JOBS = min(8, os.cpu_count() or 1)


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
    return_pred: bool = True,
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
    fitted = intercept[None, :] + X @ weights
    pred = fitted if return_pred else None
    ss_res = np.square(Y - fitted).sum(axis=0)
    ss_tot = np.square(Y - y_mean).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot

    result = {
        "intercept": intercept,
        "weights": weights,
        "r2": r2,
    }
    if pred is not None:
        result["pred"] = pred
    return result


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


def _chunk_slices(n_items: int, chunk_size: int) -> list[slice]:
    return [
        slice(start, min(start + chunk_size, n_items))
        for start in range(0, n_items, chunk_size)
    ]


def _parallel_map_chunks(
    fn,
    slices: list[slice],
    n_jobs: int,
):
    if n_jobs <= 1 or len(slices) <= 1:
        return [fn(chunk_slice) for chunk_slice in slices]
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(fn, slices))


def load_glm_calcium_swim_inputs(save_root: str | Path) -> dict[str, np.ndarray]:
    save_root = Path(save_root)
    time_stamp = np.load(save_root / "locs_cam.npy").astype(float) / 6000.0
    swim_signal = np.load(save_root / "swim_ds.npy").astype(np.float32)

    with np.load(save_root / "baseline_clusters.npz", allow_pickle=True) as cluster_data:
        invalid_ = cluster_data["invalid_"]
        ev_thres = cluster_data["ev_thres"]
    with np.load(save_root / "O2_clusters.npz", allow_pickle=True) as o2_data:
        idx_f = o2_data["idx_F"]
    with np.load(save_root / "cell_dff.npz", allow_pickle=True) as cell_dff:
        dff = cell_dff["dFF"].astype(np.float16)

    selected_dff = dff[~invalid_][ev_thres][idx_f]
    if selected_dff.shape[1] != len(time_stamp) or len(time_stamp) != len(swim_signal):
        min_len = min(selected_dff.shape[1], len(time_stamp), len(swim_signal))
        selected_dff = selected_dff[:, :min_len]
        time_stamp = time_stamp[:min_len]
        swim_signal = swim_signal[:min_len]

    return {
        "time_stamp": time_stamp,
        "swim_signal": swim_signal,
        "invalid_": invalid_,
        "ev_thres": ev_thres,
        "idx_F": idx_f,
        "selected_dff": selected_dff,
    }


def compute_chunked_spearman_correlations(
    cell_traces: np.ndarray,
    regressor: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> np.ndarray:
    cell_traces = np.asarray(cell_traces, dtype=np.float32)
    regressor = np.asarray(regressor, dtype=np.float32)

    x_rank = rankdata(regressor).astype(np.float32)
    x_rank -= x_rank.mean()
    x_norm = np.linalg.norm(x_rank)
    if x_norm == 0:
        return np.full(cell_traces.shape[0], np.nan, dtype=np.float32)

    def process_chunk(chunk_slice: slice) -> np.ndarray:
        y_chunk = cell_traces[chunk_slice]
        if y_chunk.size == 0:
            return np.empty(0, dtype=np.float32)
        y_rank = rankdata(y_chunk, axis=1).astype(np.float32)
        y_rank -= y_rank.mean(axis=1, keepdims=True)
        y_norm = np.linalg.norm(y_rank, axis=1)
        numer = y_rank @ x_rank
        denom = y_norm * x_norm
        corr = np.full(y_chunk.shape[0], np.nan, dtype=np.float32)
        valid = denom > 0
        corr[valid] = numer[valid] / denom[valid]
        return corr

    slices = _chunk_slices(cell_traces.shape[0], chunk_size)
    return np.concatenate(_parallel_map_chunks(process_chunk, slices, n_jobs=n_jobs))


def _fit_state_chunk(
    X: np.ndarray,
    y_chunk: np.ndarray,
    ridge_lambda: float,
    tau_frames: float,
) -> dict[str, np.ndarray]:
    fit = fit_weighted_ridge_batch(
        X,
        y_chunk.T,
        ridge_lambda=ridge_lambda,
        tau_frames=tau_frames,
        return_pred=False,
    )
    weights = fit["weights"].T.astype(np.float32)
    beta = np.linalg.norm(weights, axis=1).astype(np.float32)
    intercept = fit["intercept"].astype(np.float32)
    r2 = fit["r2"].astype(np.float32)
    return {
        "weights": weights,
        "beta": beta,
        "intercept": intercept,
        "r2": r2,
    }


def fit_calcium_swim_state_cells(
    cell_traces: np.ndarray,
    swim_window: np.ndarray,
    lag_grid: list[int] | tuple[int, ...] = DEFAULT_LAG_GRID,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
    verbose_prefix: str | None = None,
) -> dict[str, np.ndarray]:
    cell_traces = np.asarray(cell_traces, dtype=np.float32)
    swim_window = np.asarray(swim_window, dtype=np.float32)

    n_cells = cell_traces.shape[0]
    max_lag = max(lag_grid)

    best_r2 = np.full(n_cells, np.nan, dtype=np.float32)
    best_intercept = np.full(n_cells, np.nan, dtype=np.float32)
    best_lags = np.full(n_cells, np.nan, dtype=np.float32)
    best_weights = np.full((n_cells, max_lag), np.nan, dtype=np.float32)

    slices = _chunk_slices(n_cells, chunk_size)

    for n_lags in lag_grid:
        if verbose_prefix is not None:
            print(f"{verbose_prefix} lag={n_lags}")
        X = build_lag_matrix(swim_window, n_lags)

        def process_chunk(chunk_slice: slice) -> tuple[slice, dict[str, np.ndarray]]:
            return chunk_slice, _fit_state_chunk(
                X,
                cell_traces[chunk_slice],
                ridge_lambda=ridge_lambda,
                tau_frames=tau_frames,
            )

        chunk_results = _parallel_map_chunks(process_chunk, slices, n_jobs=n_jobs)
        for chunk_slice, chunk_fit in chunk_results:
            row_idx = np.arange(chunk_slice.start, chunk_slice.stop)
            chunk_r2 = chunk_fit["r2"]
            better = np.isfinite(chunk_r2) & (
                np.isnan(best_r2[row_idx]) | (chunk_r2 > best_r2[row_idx])
            )
            if not np.any(better):
                continue
            chosen = row_idx[better]
            best_r2[chosen] = chunk_r2[better]
            best_intercept[chosen] = chunk_fit["intercept"][better]
            best_lags[chosen] = float(n_lags)
            best_weights[chosen] = np.nan
            best_weights[chosen, :n_lags] = chunk_fit["weights"][better]

    weights_filled = np.nan_to_num(best_weights, nan=0.0)
    beta = np.linalg.norm(weights_filled, axis=1).astype(np.float32)
    kernel = np.full_like(best_weights, np.nan, dtype=np.float32)
    nonzero = beta > 0
    kernel[nonzero] = best_weights[nonzero] / beta[nonzero, None]

    return {
        "beta": beta,
        "r2": best_r2,
        "intercept": best_intercept,
        "selected_lags": best_lags,
        "kernel": kernel,
    }


def fit_calcium_swim_fish(
    save_root: str | Path,
    normoxia_window_s: tuple[float, float] = DEFAULT_NORMOXIA_WINDOW_S,
    hypoxia_window_s: tuple[float, float] = DEFAULT_HYPOXIA_WINDOW_S,
    lag_grid: list[int] | tuple[int, ...] = DEFAULT_LAG_GRID,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
    verbose: bool = True,
) -> dict[str, np.ndarray | float | int]:
    inputs = load_glm_calcium_swim_inputs(save_root)
    time_stamp = inputs["time_stamp"]
    swim_signal = preprocess_swim(inputs["swim_signal"])
    selected_dff = inputs["selected_dff"]

    norm_mask = (time_stamp >= normoxia_window_s[0]) & (time_stamp < normoxia_window_s[1])
    hyp_mask = (time_stamp >= hypoxia_window_s[0]) & (time_stamp < hypoxia_window_s[1])

    norm_swim = swim_signal[norm_mask]
    hyp_swim = swim_signal[hyp_mask]
    norm_dff = selected_dff[:, norm_mask]
    hyp_dff = selected_dff[:, hyp_mask]

    norm_spearman = compute_chunked_spearman_correlations(
        norm_dff,
        norm_swim,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
    )
    hyp_spearman = compute_chunked_spearman_correlations(
        hyp_dff,
        hyp_swim,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
    )
    spearman_r_max = np.maximum(np.abs(norm_spearman), np.abs(hyp_spearman)).astype(np.float32)
    cell_skipped = (spearman_r_max < spearman_threshold) | ~np.isfinite(spearman_r_max)

    n_cells = selected_dff.shape[0]
    max_lag = max(lag_grid)
    result: dict[str, np.ndarray | float | int] = {
        "invalid_": inputs["invalid_"],
        "ev_thres": inputs["ev_thres"],
        "idx_F": inputs["idx_F"],
        "normoxia_spearman_r": norm_spearman.astype(np.float32),
        "hypoxia_spearman_r": hyp_spearman.astype(np.float32),
        "spearman_r_max": spearman_r_max,
        "cell_skipped": cell_skipped,
        "normoxia_beta": np.full(n_cells, np.nan, dtype=np.float32),
        "hypoxia_beta": np.full(n_cells, np.nan, dtype=np.float32),
        "normoxia_r2": np.full(n_cells, np.nan, dtype=np.float32),
        "hypoxia_r2": np.full(n_cells, np.nan, dtype=np.float32),
        "normoxia_selected_lags": np.full(n_cells, np.nan, dtype=np.float32),
        "hypoxia_selected_lags": np.full(n_cells, np.nan, dtype=np.float32),
        "normoxia_intercept": np.full(n_cells, np.nan, dtype=np.float32),
        "hypoxia_intercept": np.full(n_cells, np.nan, dtype=np.float32),
        "normoxia_kernel": np.full((n_cells, max_lag), np.nan, dtype=np.float32),
        "hypoxia_kernel": np.full((n_cells, max_lag), np.nan, dtype=np.float32),
        "lag_grid": np.asarray(lag_grid, dtype=np.int16),
        "ridge_lambda": float(ridge_lambda),
        "tau_frames": float(tau_frames),
        "spearman_threshold": float(spearman_threshold),
        "normoxia_window_s": np.asarray(normoxia_window_s, dtype=np.float32),
        "hypoxia_window_s": np.asarray(hypoxia_window_s, dtype=np.float32),
        "fit_num_cells": int(n_cells),
        "fit_num_cells_passing": int((~cell_skipped).sum()),
        "fit_num_frames_normoxia": int(norm_mask.sum()),
        "fit_num_frames_hypoxia": int(hyp_mask.sum()),
    }

    fit_mask = ~cell_skipped
    if not np.any(fit_mask):
        return result

    if verbose:
        print(
            f"{Path(save_root).name}: selected={n_cells} pass={int(fit_mask.sum())} "
            f"({fit_mask.mean():.1%})"
        )

    state_specs = [
        ("normoxia", norm_dff[fit_mask], norm_swim),
        ("hypoxia", hyp_dff[fit_mask], hyp_swim),
    ]
    for state_name, state_dff, state_swim in state_specs:
        state_result = fit_calcium_swim_state_cells(
            state_dff,
            state_swim,
            lag_grid=lag_grid,
            ridge_lambda=ridge_lambda,
            tau_frames=tau_frames,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            verbose_prefix=f"{Path(save_root).name} {state_name}" if verbose else None,
        )
        result[f"{state_name}_beta"][fit_mask] = state_result["beta"]
        result[f"{state_name}_r2"][fit_mask] = state_result["r2"]
        result[f"{state_name}_selected_lags"][fit_mask] = state_result["selected_lags"]
        result[f"{state_name}_intercept"][fit_mask] = state_result["intercept"]
        result[f"{state_name}_kernel"][fit_mask] = state_result["kernel"]

    return result


def export_glm_calcium_swim_fit(
    datalist_name: str,
    start_index: int = 0,
    max_index: int | None = None,
    normoxia_window_s: tuple[float, float] = DEFAULT_NORMOXIA_WINDOW_S,
    hypoxia_window_s: tuple[float, float] = DEFAULT_HYPOXIA_WINDOW_S,
    lag_grid: list[int] | tuple[int, ...] = DEFAULT_LAG_GRID,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    tau_frames: float = DEFAULT_TAU_FRAMES,
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
    force: bool = False,
) -> list[Path]:
    df = load_datalist(datalist_name)
    if max_index is None:
        indices = [ind for ind in df.index if ind >= start_index]
    else:
        indices = [ind for ind in df.index if start_index <= ind <= max_index]

    written_paths: list[Path] = []
    for ind in indices:
        row = df.loc[ind]
        save_root = Path(row["save_root"])
        output_path = save_root / "GLM_calcium_swim_fit.npz"
        if output_path.exists() and not force:
            print(f"skip existing {output_path}")
            written_paths.append(output_path)
            continue
        print(f"fit {ind} {save_root.name}")
        started = time.perf_counter()
        result = fit_calcium_swim_fish(
            save_root,
            normoxia_window_s=normoxia_window_s,
            hypoxia_window_s=hypoxia_window_s,
            lag_grid=lag_grid,
            ridge_lambda=ridge_lambda,
            tau_frames=tau_frames,
            spearman_threshold=spearman_threshold,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            verbose=True,
        )
        np.savez_compressed(output_path, **result)
        elapsed = time.perf_counter() - started
        print(f"saved {output_path} ({elapsed:.1f}s)")
        written_paths.append(output_path)
    return written_paths
