from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings

from h5py import File
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from tqdm import tqdm


SAMPLE_RATE_HZ = 6000


def bin_arr_1d(arr: np.ndarray, bin_size_samples: int) -> np.ndarray:
    arr_max = len(arr) // bin_size_samples * bin_size_samples
    return arr[:arr_max].reshape(-1, bin_size_samples).mean(axis=1)


def get_data_windowed(
    save_root: str | Path,
    bin_size: float = 1,
    pre_s: float = 10,
    post_s: float = 30,
    sr: int = SAMPLE_RATE_HZ,
    return_window_info: bool = False,
):
    """Load swim and oxygen traces around swim onsets with an explicit time window.

    `bin_size` is specified in seconds and converted to samples internally.
    """

    save_root = Path(save_root)
    bin_size_s = float(bin_size)
    bin_size_samples = int(bin_size_s * sr)
    pre_samples = int(sr * pre_s)
    post_samples = int(sr * post_s)
    pre_bins = pre_samples // bin_size_samples
    post_bins = post_samples // bin_size_samples

    if bin_size_samples <= 0:
        raise ValueError("bin_size must define a positive bin width in seconds.")
    if pre_samples <= 0 or post_samples <= 0:
        raise ValueError("pre_s and post_s must define positive windows.")
    if (pre_samples % bin_size_samples != 0) or (post_samples % bin_size_samples != 0):
        warnings.warn(
            "Window size is not an exact multiple of bin_size; binned traces "
            "will be truncated at the end."
        )

    swim_list = []
    bin_swim_list = []
    o2_list = []
    swim_on_list = []
    swim_dur_list = []
    pre_swim_list = []
    post_swim_list = []

    ephys_data = File(save_root / "data.mat", "r")["data"]
    flt_ch1 = ephys_data["fltCh2"][()].squeeze()
    back1 = ephys_data["back2"][()].squeeze()
    swim_ = flt_ch1 - back1
    swim_[swim_ < 0] = 0
    swim_start_ind = ephys_data["swimStartIndT"][()].squeeze().astype(int)
    swim_end_ind = ephys_data["swimEndIndT"][()].squeeze().astype(int)

    oxygen = File(save_root / "O2_real.mat", "r")["O2_real"][()].squeeze()
    oxygen = oxygen / oxygen[sr * 60 * 5 : sr * 60 * 10].mean()
    swim_ = swim_ / np.percentile(swim_[swim_ > 0], 99.99)

    binarized_swim_trace = np.zeros(len(flt_ch1))
    for start_idx, end_idx in zip(swim_start_ind, swim_end_ind):
        binarized_swim_trace[start_idx:end_idx] = 1

    swim_idx = np.where(
        ((swim_start_ind - pre_samples * 2) > 0)
        & ((swim_start_ind + post_samples * 2) < len(swim_))
    )[0]

    for n_swim in swim_idx[:-1]:
        swim_on_t = swim_start_ind[n_swim] / sr / 60
        swim_on = int(swim_start_ind[n_swim])
        swim_off = swim_end_ind[n_swim]
        swim_off_pre = swim_end_ind[n_swim - 1]
        swim_on_next = swim_start_ind[n_swim + 1]

        swim_on_list.append(swim_on_t)
        swim_dur_list.append((swim_off - swim_on) / sr)
        pre_swim_list.append((swim_on - swim_off_pre) / sr)
        post_swim_list.append((swim_on_next - swim_off) / sr)

        swim_slice = slice(swim_on - pre_samples, swim_on + post_samples)
        swim_list.append(bin_arr_1d(swim_[swim_slice], bin_size_samples))
        bin_swim_list.append(
            bin_arr_1d(binarized_swim_trace[swim_slice], bin_size_samples)
        )
        o2_list.append(bin_arr_1d(oxygen[swim_slice], bin_size_samples))

    swim_list = np.array(swim_list)
    o2_list = np.array(o2_list)
    swim_on_list = np.array(swim_on_list)
    swim_dur_list = np.array(swim_dur_list)
    pre_swim_list = np.array(pre_swim_list)
    post_swim_list = np.array(post_swim_list)
    bin_swim_list = np.array(bin_swim_list)

    out = (
        swim_list,
        o2_list,
        bin_swim_list,
        swim_on_list,
        swim_dur_list,
        pre_swim_list,
        post_swim_list,
    )
    if return_window_info:
        window_info = {
            "sr": sr,
            "bin_size_s": bin_size_s,
            "bin_size_samples": bin_size_samples,
            "pre_s": pre_s,
            "post_s": post_s,
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "pre_bins": pre_bins,
            "post_bins": post_bins,
        }
        return out + (window_info,)
    return out


def _normalize_lags(lags: dict[str, int] | None = None) -> dict[str, int]:
    if lags is None:
        return {"o2": 0, "swim": 0}
    if not isinstance(lags, dict):
        raise TypeError("lags must be a dict like {'o2': 0, 'swim': 0}")
    return lags


def _resolve_current_lag(feature_name: str, lags: dict[str, int] | None) -> int:
    lags = _normalize_lags(lags)
    if feature_name not in {"o2", "swim"}:
        raise ValueError(f"Unknown feature_name: {feature_name}")
    if feature_name not in lags:
        raise KeyError(f"Missing lag for feature_name: {feature_name}")
    return int(lags[feature_name])


def _build_model_o2_swim_predictors(
    bin_swim_row: np.ndarray,
    o2_row: np.ndarray,
    event_idx: int,
    lags: dict[str, int] | None = None,
    include_const: bool = True,
) -> tuple[np.ndarray, list[tuple[str, int, int] | tuple[str, int]]]:
    lags = _normalize_lags(lags)
    if not lags:
        raise ValueError("At least one predictor must be included.")

    x_blocks = []
    feature_info: list[tuple[str, int, int] | tuple[str, int]] = []

    if include_const:
        x_blocks.append(np.array([1.0]))
        feature_info.append(("const", 1))

    for feature_name in lags:
        current_lag = _resolve_current_lag(feature_name, lags=lags)
        if feature_name == "o2":
            if current_lag < 0:
                raise ValueError(f"o2 lag must be >= 0, got {current_lag}")
            if current_lag > event_idx:
                raise ValueError(
                    f"o2 lag {current_lag} exceeds available history at "
                    f"event_idx={event_idx}"
                )
            x_window = o2_row[event_idx - current_lag : event_idx + 1]
        else:
            if current_lag < 0:
                raise ValueError(f"swim lag must be >= 0, got {current_lag}")
            if current_lag > event_idx:
                raise ValueError(
                    f"swim lag {current_lag} exceeds available history at "
                    f"event_idx={event_idx}"
                )
            if current_lag == 0:
                x_window = np.array([], dtype=float)
            else:
                x_window = bin_swim_row[event_idx - current_lag : event_idx]

        x_window = np.asarray(x_window, dtype=float)
        x_blocks.append(x_window)
        feature_info.append((feature_name, current_lag, len(x_window)))

    total_columns = sum(len(block) for block in x_blocks)
    if total_columns == 0:
        raise ValueError(
            "Predictor design is empty. Use include_const=True or a positive lag."
        )

    return np.concatenate(x_blocks), feature_info


def _feature_info_template(
    lags: dict[str, int] | None,
    include_const: bool = True,
) -> list[tuple[str, int, int] | tuple[str, int]]:
    lags = _normalize_lags(lags)
    feature_info: list[tuple[str, int, int] | tuple[str, int]] = []
    if include_const:
        feature_info.append(("const", 1))
    for feature_name in lags:
        current_lag = _resolve_current_lag(feature_name, lags=lags)
        block_len = current_lag + 1 if feature_name == "o2" else current_lag
        feature_info.append((feature_name, current_lag, block_len))
    return feature_info


def _glm_keep_mask(
    y_list: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        random_draws = np.random.randn(len(y_list))
    else:
        random_draws = rng.standard_normal(len(y_list))
    return (y_list > 0) | (random_draws > 0.5)


def model_o2_swim_flexible(
    bin_swim_list,
    o2_list,
    pre_: int = 10,
    lags: dict[str, int] | None = None,
    include_const: bool = True,
    return_feature_info: bool = False,
    rng: np.random.Generator | None = None,
    return_warning: bool = False,
):
    """Flexible oxygen/swim GLM with per-feature lag control."""

    lags = _normalize_lags(lags)
    x_list = []
    y_list = []
    feature_info = _feature_info_template(lags, include_const=include_const)
    len_ = len(bin_swim_list)
    total_columns = sum(item[-1] if len(item) == 3 else item[1] for item in feature_info)
    empty_out = (
        np.nan,
        np.nan,
        np.full(total_columns, np.nan),
        np.full(total_columns, np.nan),
        np.nan,
    )

    if len_ == 0:
        if return_feature_info and return_warning:
            return empty_out + (feature_info, False)
        if return_feature_info:
            return empty_out + (feature_info,)
        if return_warning:
            return empty_out + (False,)
        return empty_out

    for n in range(len_):
        for n_dat in range(2, 15):
            event_idx = pre_ + n_dat
            y_list.append(bin_swim_list[n][event_idx] > 0.02)
            x_curr, feature_info = _build_model_o2_swim_predictors(
                bin_swim_list[n],
                o2_list[n],
                event_idx,
                lags=lags,
                include_const=include_const,
            )
            x_list.append(x_curr)

    y_list = np.array(y_list)
    x_list = np.array(x_list)
    idx_y_list = _glm_keep_mask(y_list, rng=rng)
    if x_list.size == 0 or not idx_y_list.any():
        if return_feature_info and return_warning:
            return empty_out + (feature_info, False)
        if return_feature_info:
            return empty_out + (feature_info,)
        if return_warning:
            return empty_out + (False,)
        return empty_out

    perfect_separation = False
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", PerfectSeparationWarning)
        glm_logit = sm.GLM(
            y_list[idx_y_list], x_list[idx_y_list], family=sm.families.Binomial()
        )
        glm_results = glm_logit.fit(max_iter=1000, tol=1e-6, tol_criterion="params")
        perfect_separation = any(
            issubclass(w.category, PerfectSeparationWarning) for w in captured
        )

    out = (
        glm_results.llf,
        glm_results.pseudo_rsquared(kind="cs"),
        glm_results.params,
        x_list[idx_y_list].mean(axis=0),
        glm_results.aic,
    )
    if return_feature_info and return_warning:
        return out + (feature_info, perfect_separation)
    if return_feature_info:
        return out + (feature_info,)
    if return_warning:
        return out + (perfect_separation,)
    return out


def preload_model_fit_data(
    files_: list[str] | list[Path],
    removed_one: list[int],
    bin_size: float = 1,
    pre_s: float = 10,
    post_s: float = 30,
) -> list[dict[str, object]]:
    preloaded_data = []
    for n_file, save_root in enumerate(files_):
        if n_file in removed_one:
            continue
        (
            swim_list,
            o2_list,
            bin_swim_list,
            swim_on_list,
            swim_dur_list,
            pre_swim_list,
            post_swim_list,
            window_info,
        ) = get_data_windowed(
            save_root,
            bin_size=bin_size,
            pre_s=pre_s,
            post_s=post_s,
            return_window_info=True,
        )
        preloaded_data.append(
            {
                "save_root": Path(save_root),
                "swim_list": swim_list,
                "o2_list": o2_list,
                "bin_swim_list": bin_swim_list,
                "swim_on_list": swim_on_list,
                "swim_dur_list": swim_dur_list,
                "pre_swim_list": pre_swim_list,
                "post_swim_list": post_swim_list,
                "window_info": window_info,
            }
        )
    return preloaded_data


def fit_one_run_preloaded(
    preloaded_data: list[dict[str, object]],
    model_lags: dict[str, int] | None = None,
    seed: int | None = None,
) -> list[list[tuple]]:
    rng = np.random.default_rng(seed)
    model_list_swim = []
    for item in preloaded_data:
        o2_list = item["o2_list"]
        bin_swim_list = item["bin_swim_list"]
        swim_on_list = item["swim_on_list"]
        pre_swim_list = item["pre_swim_list"]
        pre_bins = item["window_info"]["pre_bins"]

        models_ = []
        o2_list_norm = (
            o2_list / o2_list[:, pre_bins - 1 : pre_bins + 1].mean(axis=1, keepdims=True)
            - 1
        )
        o2_pre = o2_list_norm[:, : pre_bins - 3].mean(axis=1)

        idx_n = (
            (swim_on_list > 5)
            & (swim_on_list < 20)
            & (pre_swim_list > 3)
            & (o2_pre < 0.01)
        )
        idx_h = (
            (swim_on_list > 30)
            & (swim_on_list < 40)
            & (pre_swim_list > 3)
            & (o2_pre < 0.01)
        )

        curr_model = model_o2_swim_flexible(
            bin_swim_list[idx_n],
            o2_list[idx_n],
            pre_=pre_bins,
            lags=model_lags,
            rng=rng,
            return_warning=True,
        )
        models_.append(curr_model)

        curr_model = model_o2_swim_flexible(
            bin_swim_list[idx_h],
            o2_list[idx_h],
            pre_=pre_bins,
            lags=model_lags,
            rng=rng,
            return_warning=True,
        )
        models_.append(curr_model)

        model_list_swim.append(models_)
    return model_list_swim


def fit_many_runs_preloaded(
    preloaded_data: list[dict[str, object]],
    n_runs: int = 20,
    model_lags: dict[str, int] | None = None,
    n_jobs: int = 1,
    base_seed: int | None = 0,
    retry_on_separation: int = 0,
) -> list[list[list[tuple]]]:
    if base_seed is None:
        seeds = [None] * n_runs
    else:
        seeds = [base_seed + n_run for n_run in range(n_runs)]

    def has_perfect_separation(run_models: list[list[tuple]]) -> bool:
        for models_for_fish in run_models:
            for model_out in models_for_fish:
                if len(model_out) > 5 and model_out[5]:
                    return True
        return False

    def run_with_retry(seed: int | None) -> list[list[tuple]]:
        attempts = retry_on_separation + 1
        for attempt in range(attempts):
            if base_seed is None and attempt > 0:
                seed = int(np.random.default_rng().integers(0, 2**32 - 1))
            run_models = fit_one_run_preloaded(
                preloaded_data, model_lags=model_lags, seed=seed
            )
            if not has_perfect_separation(run_models):
                return run_models
        return run_models

    if n_jobs == 1:
        return [run_with_retry(seed) for seed in tqdm(seeds)]

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_with_retry, seed) for seed in seeds]
        return [future.result() for future in tqdm(futures)]


def fit_model_spec_preloaded(
    preloaded_data: list[dict[str, object]],
    model_lags: dict[str, int] | None = None,
    n_runs: int = 20,
    n_jobs: int = 1,
    base_seed: int | None = 0,
    retry_on_separation: int = 0,
) -> dict[str, object]:
    model_lags = _normalize_lags(model_lags).copy()
    runs = fit_many_runs_preloaded(
        preloaded_data,
        n_runs=n_runs,
        model_lags=model_lags,
        n_jobs=n_jobs,
        base_seed=base_seed,
        retry_on_separation=retry_on_separation,
    )
    return {
        "model_lags": model_lags,
        "runs": runs,
    }


def format_model_lags(model_lags: dict[str, int]) -> str:
    return ", ".join(f"{name}:{lag}" for name, lag in model_lags.items())


def _safe_nan_summary(values: np.ndarray) -> tuple[int, float, float, float]:
    finite = np.isfinite(values)
    n_valid = int(finite.sum())
    if n_valid == 0:
        return n_valid, np.nan, np.nan, np.nan
    finite_values = values[finite]
    return (
        n_valid,
        float(np.mean(finite_values)),
        float(np.std(finite_values)),
        float(np.median(finite_values)),
    )


def summarize_model_run_list(
    model_run_list: list[dict[str, object]],
    selected_spec_idx: int = 0,
    condition_labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    if not model_run_list:
        raise ValueError("model_run_list is empty.")

    selected_model_spec = model_run_list[selected_spec_idx]
    selected_model_lags = selected_model_spec["model_lags"]
    runs_ = selected_model_spec["runs"]
    model_list_swim = runs_[0]

    num_spec = len(model_run_list)
    num_trial = len(model_run_list[0]["runs"])
    num_fish = len(model_run_list[0]["runs"][0])
    num_model = len(model_run_list[0]["runs"][0][0])

    if condition_labels is None:
        condition_labels = ["normoxia", "hypoxia"][:num_model]
    else:
        condition_labels = list(condition_labels)[:num_model]

    pseudo_r2_all = np.full((num_spec, num_fish, num_model, num_trial), np.nan)
    aic_all = np.full((num_spec, num_fish, num_model, num_trial), np.nan)

    for n_spec, model_spec in enumerate(model_run_list):
        runs_spec = model_spec["runs"]
        for n_trial in range(num_trial):
            for n_fish in range(num_fish):
                for n_model in range(num_model):
                    pseudo_r2_all[n_spec, n_fish, n_model, n_trial] = runs_spec[n_trial][
                        n_fish
                    ][n_model][1]
                    aic_all[n_spec, n_fish, n_model, n_trial] = runs_spec[n_trial][
                        n_fish
                    ][n_model][4]

    r2_list_swim = pseudo_r2_all[selected_spec_idx]
    aic_list_swim = aic_all[selected_spec_idx]

    summary_rows = []
    for n_spec, model_spec in enumerate(model_run_list):
        spec_label = format_model_lags(model_spec["model_lags"])
        for n_model, condition_name in enumerate(condition_labels):
            r2_vals = pseudo_r2_all[n_spec, :, n_model, :].reshape(-1)
            aic_vals = aic_all[n_spec, :, n_model, :].reshape(-1)
            n_valid, r2_mean, r2_std, r2_median = _safe_nan_summary(r2_vals)
            _, aic_mean, aic_std, aic_median = _safe_nan_summary(aic_vals)
            summary_rows.append(
                {
                    "spec_idx": n_spec,
                    "model_lags": model_spec["model_lags"],
                    "spec_label": spec_label,
                    "condition": condition_name,
                    "n_valid": n_valid,
                    "pseudo_r2_mean": r2_mean,
                    "pseudo_r2_std": r2_std,
                    "pseudo_r2_median": r2_median,
                    "aic_mean": aic_mean,
                    "aic_std": aic_std,
                    "aic_median": aic_median,
                }
            )

    metrics_summary_df = pd.DataFrame(summary_rows)
    return {
        "selected_spec_idx": selected_spec_idx,
        "selected_model_spec": selected_model_spec,
        "selected_model_lags": selected_model_lags,
        "runs": runs_,
        "model_list_swim": model_list_swim,
        "condition_labels": condition_labels,
        "pseudo_r2_all": pseudo_r2_all,
        "aic_all": aic_all,
        "r2_list_swim": r2_list_swim,
        "aic_list_swim": aic_list_swim,
        "metrics_summary_df": metrics_summary_df,
    }
