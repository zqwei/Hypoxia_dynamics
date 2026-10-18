from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings

from h5py import File
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
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
    print("swim_list shape:", swim_list.shape)

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
    subsample: bool = True,
):
    """Flexible oxygen/swim GLM with per-feature lag control.

    ``subsample=True`` (default) fits on all swim rows plus a random share of the
    non-swim rows (``_glm_keep_mask``); ``False`` fits on every row.
    """

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
        for n_dat in range(0, 20):
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
    idx_y_list = _glm_keep_mask(y_list, rng=rng) if subsample else np.ones(len(y_list), dtype=bool)
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
        glm_results.pseudo_rsquared(kind="mcf"), # cs
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


def get_trace_binned(
    save_root: str | Path,
    bin_size: float = 1,
    t_start_min: float = 3,
    t_end_min: float = 40,
    sr: int = SAMPLE_RATE_HZ,
):
    """Load one continuous binned O2 trace and binarized swim trace.

    Preprocessing matches get_data_windowed. Returns (o2, bin_swim, time_s)
    where time_s is the start time (seconds) of each bin.
    """

    save_root = Path(save_root)
    bin_size_samples = int(float(bin_size) * sr)
    if bin_size_samples <= 0:
        raise ValueError("bin_size must define a positive bin width in seconds.")

    ephys_data = File(save_root / "data.mat", "r")["data"]
    swim_start_ind = ephys_data["swimStartIndT"][()].squeeze().astype(int)
    swim_end_ind = ephys_data["swimEndIndT"][()].squeeze().astype(int)
    n_samples = len(ephys_data["fltCh2"][()].squeeze())

    oxygen = File(save_root / "O2_real.mat", "r")["O2_real"][()].squeeze()
    oxygen = oxygen / oxygen[sr * 60 * 5 : sr * 60 * 10].mean()

    binarized_swim_trace = np.zeros(n_samples)
    for start_idx, end_idx in zip(swim_start_ind, swim_end_ind):
        binarized_swim_trace[start_idx:end_idx] = 1

    sample_slice = slice(int(t_start_min * 60 * sr), int(t_end_min * 60 * sr))
    o2 = bin_arr_1d(oxygen[sample_slice], bin_size_samples)
    bin_swim = bin_arr_1d(binarized_swim_trace[sample_slice], bin_size_samples)
    time_s = t_start_min * 60 + np.arange(len(o2)) * float(bin_size)
    return o2, bin_swim, time_s


def _build_full_trace_design(
    o2,
    bin_swim,
    lags: dict[str, int] | None = None,
    include_const: bool = True,
    first_bin: int | None = None,
    bin_mask=None,
    extra=None,
):
    """Design matrix, swim labels and bin indices for every usable bin of a trace.

    ``first_bin`` defaults to the largest lag; pass a larger common value to score
    specs with different lags on identical bins. ``bin_mask`` (bool, one entry per
    bin) restricts the bins, e.g. to one period. ``extra`` (array of shape
    (n_bins_of_trace,) or (n_bins_of_trace, k), e.g. from ``history_features``) adds
    its value at the predicted bin as extra columns after the lag blocks.
    """

    lags = _normalize_lags(lags)
    o2 = np.asarray(o2, dtype=float)
    bin_swim = np.asarray(bin_swim, dtype=float)
    if first_bin is None:
        first_bin = max(int(lag) for lag in lags.values())
    bin_idx = np.arange(first_bin, len(o2))
    if bin_mask is not None:
        bin_idx = bin_idx[np.asarray(bin_mask, dtype=bool)[bin_idx]]
    x_list = np.array(
        [
            _build_model_o2_swim_predictors(
                bin_swim, o2, t, lags=lags, include_const=include_const
            )[0]
            for t in bin_idx
        ]
    )
    if extra is not None:
        extra = np.asarray(extra, dtype=float)
        extra = extra[:, None] if extra.ndim == 1 else extra
        x_list = np.hstack([x_list, extra[bin_idx]])
    y = bin_swim[bin_idx] > 0.02
    return x_list, y, bin_idx


def history_features(bin_swim, threshold: float = 0.02, empty_bins: int = 9, rate_windows=(30, 120), gap_cap: int = 600):
    """Causal long-memory swim-history features, one value per bin (using only earlier bins).

    - ``empty``: 1 if there was no swim bin in the previous ``empty_bins`` bins (the "quiet state" indicator);
    - ``log_gap``: log(1 + seconds since the last swim bin), capped at ``gap_cap``;
    - ``rate<w>``: fraction of swim bins in the previous ``w`` bins (trailing mean, excluding the current bin).
    Returns a dict name -> array of the trace's length.
    """

    swim = np.asarray(bin_swim, dtype=float) > threshold
    n = len(swim)
    csum = np.r_[0, np.cumsum(swim)]
    out = {}
    idx = np.arange(n)
    out["empty"] = np.array([float(csum[t] - csum[max(0, t - empty_bins)] == 0) for t in idx])
    last = -10**9
    gap = np.zeros(n)
    for t in range(n):
        gap[t] = min(t - last, gap_cap) if last > -10**8 else gap_cap
        if swim[t]:
            last = t
    out["log_gap"] = np.log1p(gap)
    for w in rate_windows:
        out[f"rate{w}"] = np.array([(csum[t] - csum[max(0, t - w)]) / max(1, min(w, t)) for t in idx])
    return out


def evaluate_model_full_trace(
    params,
    o2,
    bin_swim,
    lags: dict[str, int] | None = None,
    include_const: bool = True,
    correct_subsampling: bool = True,
):
    """Apply fitted GLM params to every bin of a continuous trace.

    ``params`` come from a fit on swim-onset trials (model_o2_swim_flexible,
    index 2 of its output) with the same ``lags``/``include_const``. The
    Binomial log-likelihood is evaluated on every bin with enough history, then
    AIC = 2k - 2*llf, with k = len(params) (the fit is not re-estimated).

    The training fit keeps all swim bins but only P(randn > 0.5) ~ 0.31 of
    non-swim bins, which shifts the intercept up by -log(0.31). With
    ``correct_subsampling`` the intercept is shifted back before scoring so the
    model is calibrated to the full-trace base rate.

    Returns a dict with llf, aic, aic_per_bin (aic / n_bins), pseudo_r2_mcf
    (vs. a full-trace base-rate null), n_bins, k, and the per-bin
    log-likelihood ``ll_per_bin`` with its bin indices ``bin_idx``.
    """

    lags = _normalize_lags(lags)
    params = np.array(params, dtype=float)
    o2 = np.asarray(o2, dtype=float)
    bin_swim = np.asarray(bin_swim, dtype=float)

    if correct_subsampling:
        if not include_const:
            raise ValueError("correct_subsampling requires include_const=True.")
        params[0] += np.log(norm.sf(0.5))

    x_list, y, bin_idx = _build_full_trace_design(
        o2, bin_swim, lags, include_const=include_const
    )
    if x_list.shape[1] != len(params):
        raise ValueError(
            f"params has {len(params)} entries but lags {lags} give "
            f"{x_list.shape[1]} predictors."
        )

    eta = x_list @ params
    ll_per_bin = np.where(y, -np.logaddexp(0, -eta), -np.logaddexp(0, eta))
    llf = float(ll_per_bin.sum())
    n_bins = len(bin_idx)
    k = len(params)
    aic = 2 * k - 2 * llf

    p_base = y.mean()
    if 0 < p_base < 1:
        llf_null = n_bins * (p_base * np.log(p_base) + (1 - p_base) * np.log(1 - p_base))
        pseudo_r2 = 1.0 - llf / llf_null
    else:
        pseudo_r2 = np.nan

    return {
        "llf": llf,
        "aic": aic,
        "aic_per_bin": aic / n_bins,
        "pseudo_r2_mcf": pseudo_r2,
        "n_bins": n_bins,
        "k": k,
        "ll_per_bin": ll_per_bin,
        "bin_idx": bin_idx,
    }


def _fit_binomial_glm(x_list, y, start_params=None):
    """Binomial GLM by IRLS, falling back to Newton if IRLS gives a NaN likelihood.

    IRLS can diverge under (quasi-)separation, e.g. a binary swim history that
    almost determines the next bin; Newton stays finite. Returns
    (results, perfect_separation, fallback_optimizer); ``results`` is None if the
    Newton fallback also fails (singular Hessian).
    """

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", PerfectSeparationWarning)
        glm_results = sm.GLM(y, x_list, family=sm.families.Binomial()).fit(
            start_params=start_params, max_iter=1000, tol=1e-6, tol_criterion="params"
        )
        perfect_separation = any(
            issubclass(w.category, PerfectSeparationWarning) for w in captured
        )

    fallback_optimizer = not np.isfinite(glm_results.llf)
    if fallback_optimizer:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                glm_results = sm.GLM(y, x_list, family=sm.families.Binomial()).fit(
                    method="newton", maxiter=1000
                )
        except np.linalg.LinAlgError:
            glm_results = None
    return glm_results, perfect_separation, fallback_optimizer


def fit_model_full_trace(
    o2,
    bin_swim,
    lags: dict[str, int] | None = None,
    include_const: bool = True,
    first_bin: int | None = None,
    bin_mask=None,
    start_params=None,
    ridge: float | None = None,
    ridge_free_o2_current: bool = False,
    ridge_decay_tau: float | None = None,
    extra_features=None,
):
    """Fit the Binomial GLM by maximum likelihood on every bin of a continuous trace.

    No row subsampling is applied, so AIC = 2k - 2*llf is an ordinary in-sample AIC
    on the fitted bins. ``start_params`` (e.g. params from a swim-trial fit) only
    sets the starting point of the optimizer; the optimum is unchanged.

    With ``ridge`` (a number, or a sequence of penalties searched by an inner
    contiguous CV) the slopes get an L2 penalty instead (see _fit_ridge_logit;
    intercept free, features standardized). The likelihood and R2 are computed on the
    penalized params, ``df_eff`` is the effective number of parameters and ``aic`` is
    the effective-df AIC ``2 * df_eff - 2 * llf`` (``k`` is then also ``df_eff``).

    Returns a dict with llf, aic, aic_per_bin (aic / n_bins), pseudo_r2_mcf,
    params, n_bins, k, ll_per_bin, bin_idx, perfect_separation, fallback_optimizer
    (IRLS gave a non-finite likelihood, so Newton was used) and max_abs_param (a
    very large value signals quasi-separation: params are not identified, though
    the AIC is stable).
    """

    x_list, y, bin_idx = _build_full_trace_design(
        o2, bin_swim, lags, include_const=include_const,
        first_bin=first_bin, bin_mask=bin_mask, extra=extra_features,
    )
    if ridge is not None:
        col = _o2_current_col(_normalize_lags(lags), include_const=include_const) if ridge_free_o2_current else None
        free = () if col is None else (col,)
        weights = None if ridge_decay_tau is None else _pad_pen_weights(_ridge_age_weights(
            _normalize_lags(lags), include_const=include_const, tau=ridge_decay_tau
        ), x_list.shape[1])
        lam = float(ridge) if np.ndim(ridge) == 0 else _select_ridge_lambda(
            x_list, y, ridge, include_const=include_const, free_cols=free, pen_weights=weights)
        params, df_eff = _fit_ridge_logit(
            x_list, y, lam, include_const=include_const, free_cols=free, pen_weights=weights, return_df=True,
        )
        eta = x_list @ params
        ll_per_bin = np.where(y, -np.logaddexp(0, -eta), -np.logaddexp(0, eta))
        p_base = y.mean()
        llf_null = len(y) * (p_base * np.log(p_base) + (1 - p_base) * np.log(1 - p_base))
        return {
            "llf": float(ll_per_bin.sum()),
            "aic": 2 * df_eff - 2 * float(ll_per_bin.sum()),
            "aic_per_bin": (2 * df_eff - 2 * float(ll_per_bin.sum())) / len(bin_idx),
            "pseudo_r2_mcf": 1.0 - ll_per_bin.sum() / llf_null if 0 < p_base < 1 else np.nan,
            "params": params,
            "n_bins": len(bin_idx),
            "k": df_eff,
            "df_eff": df_eff,
            "ridge_lambda": lam,
            "ll_per_bin": ll_per_bin,
            "bin_idx": bin_idx,
            "perfect_separation": False,
            "fallback_optimizer": False,
            "max_abs_param": float(np.max(np.abs(params))),
        }

    glm_results, perfect_separation, fallback_optimizer = _fit_binomial_glm(
        x_list, y, start_params=start_params
    )
    if glm_results is None:
        return {
            "llf": np.nan,
            "aic": np.nan,
            "aic_per_bin": np.nan,
            "pseudo_r2_mcf": np.nan,
            "params": np.full(x_list.shape[1], np.nan),
            "n_bins": len(bin_idx),
            "k": x_list.shape[1],
            "ll_per_bin": np.full(len(bin_idx), np.nan),
            "bin_idx": bin_idx,
            "perfect_separation": perfect_separation,
            "fallback_optimizer": True,
            "max_abs_param": np.nan,
        }

    eta = x_list @ np.asarray(glm_results.params)
    ll_per_bin = np.where(y, -np.logaddexp(0, -eta), -np.logaddexp(0, eta))
    n_bins = len(bin_idx)
    return {
        "llf": glm_results.llf,
        "aic": glm_results.aic,
        "aic_per_bin": glm_results.aic / n_bins,
        "pseudo_r2_mcf": glm_results.pseudo_rsquared(kind="mcf"),
        "params": np.asarray(glm_results.params),
        "n_bins": n_bins,
        "k": len(glm_results.params),
        "ll_per_bin": ll_per_bin,
        "bin_idx": bin_idx,
        "perfect_separation": perfect_separation,
        "fallback_optimizer": fallback_optimizer,
        "max_abs_param": float(np.max(np.abs(glm_results.params))),
    }


def _bernoulli_ll(y, eta):
    return np.where(y, -np.logaddexp(0, -eta), -np.logaddexp(0, eta)).sum()


def _o2_current_col(lags, include_const=True):
    """Column index of the current-second O2 predictor (None if there is no O2 block)."""

    if "o2" not in lags:
        return None
    pos = 1 if include_const else 0
    for name, lag in lags.items():
        if name == "o2":
            return pos + int(lag)  # the O2 block runs oldest -> current, so current is last
        pos += int(lag)
    return None


def _ridge_age_weights(lags, include_const=True, tau=1.0):
    """Per-column penalty multipliers 1 - exp(-age / tau), age in bins (0 = current second).

    The constant gets 0, the current-second O2 gets 0, and old lags approach 1, so
    recent lags are penalized less. The O2 block runs oldest -> current (ages lag..0),
    the swim block oldest -> previous bin (ages lag..1).
    """

    if tau is None or tau <= 0:
        raise ValueError("tau must be positive")
    weights = [0.0] if include_const else []
    for name, lag in lags.items():
        ages = np.arange(int(lag), -1, -1) if name == "o2" else np.arange(int(lag), 0, -1)
        weights.extend(1.0 - np.exp(-ages / float(tau)))
    return np.array(weights)


def _pad_pen_weights(weights, n_cols):
    """Extra feature columns after the lag blocks (e.g. history features) get the full penalty weight 1."""

    weights = np.asarray(weights, dtype=float)
    return weights if len(weights) >= n_cols else np.concatenate([weights, np.ones(n_cols - len(weights))])


def _bernoulli_ll_bins(y, eta):
    return np.where(y, -np.logaddexp(0, -eta), -np.logaddexp(0, eta))


def _fit_ridge_logit(x_list, y, lam, include_const=True, max_iter=100, tol=1e-8, free_cols=(), pen_weights=None, offset=None, return_df=False):
    """L2-penalized logistic regression by Newton with step halving.

    Minimizes -loglik + 0.5 * lam * sum(beta_j**2) over the slope coefficients; the
    intercept (column 0 when ``include_const``) is not penalized. Features are
    standardized with the training mean/sd before penalizing, so ``lam`` means the
    same thing for O2 and swim columns; the returned params are on the original
    scale, so ``x_list @ params`` is the linear predictor. Columns in ``free_cols``
    are not penalized either (e.g. the current-O2 weight). ``pen_weights`` (one per
    column) scales the penalty per coefficient, e.g. from _ridge_age_weights.
    ``offset`` (one value per row) is added to the linear predictor and not fitted,
    so the fit estimates a deviation from an existing model. With ``return_df`` the
    function returns ``(params, df_eff)``, where ``df_eff = tr[(Z'WZ + P)^-1 Z'WZ]`` is
    the effective number of parameters of the penalized fit (each coefficient counts
    1 if unpenalized and less as the penalty shrinks it).
    """

    x = np.asarray(x_list, dtype=float)
    yv = np.asarray(y, dtype=float)
    start = 1 if include_const else 0
    n_cols = x.shape[1]
    center = np.zeros(n_cols)
    scale = np.ones(n_cols)
    if include_const:  # without an intercept column there is nothing to absorb the centering, so only scale
        center[start:] = x[:, start:].mean(axis=0)
    sd = x[:, start:].std(axis=0)
    scale[start:] = np.where(sd > 0, sd, 1.0)
    z = (x - center) / scale
    off = np.zeros(len(yv)) if offset is None else np.asarray(offset, dtype=float)

    pen = float(lam) * (np.ones(n_cols) if pen_weights is None else np.asarray(pen_weights, dtype=float))
    if include_const:
        pen[0] = 0.0
    for col in free_cols:
        pen[col] = 0.0
    b = np.zeros(n_cols)
    if include_const:
        p0 = np.clip(yv.mean(), 1e-6, 1 - 1e-6)
        b[0] = np.log(p0 / (1 - p0)) - off.mean()

    def objective(b_):
        eta_ = z @ b_ + off
        return -(yv * eta_ - np.logaddexp(0, eta_)).sum() + 0.5 * (pen * b_ * b_).sum()

    f = objective(b)
    for _ in range(max_iter):
        mu = expit(z @ b + off)
        w = mu * (1 - mu)
        grad = z.T @ (yv - mu) - pen * b
        hess = (z * w[:, None]).T @ z + np.diag(pen)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        t = 1.0
        while t > 1e-10:
            b_new = b + t * step
            f_new = objective(b_new)
            if f_new <= f + 1e-12:
                break
            t /= 2
        else:
            break
        converged = np.max(np.abs(b_new - b)) < tol
        b, f = b_new, f_new
        if converged:
            break

    beta = b / scale
    if include_const:
        beta[0] = b[0] - np.sum(b[start:] * center[start:] / scale[start:])
    if return_df:
        mu = expit(z @ b + off)
        w = mu * (1 - mu)
        zwz = (z * w[:, None]).T @ z
        df_eff = float(np.trace(np.linalg.solve(zwz + np.diag(pen), zwz)))
        return beta, df_eff
    return beta


def _select_ridge_lambda(x_list, y, lams, include_const=True, n_inner=4, free_cols=(), pen_weights=None, offset=None):
    """Pick the penalty with the best pooled held-out log-likelihood (contiguous inner CV)."""

    n = len(y)
    scores = np.zeros(len(lams))
    for test_pos in np.array_split(np.arange(n), n_inner):
        is_test = np.zeros(n, dtype=bool)
        is_test[test_pos] = True
        y_tr = y[~is_test]
        if not 0 < y_tr.mean() < 1:
            continue
        for i, lam in enumerate(lams):
            off = None if offset is None else np.asarray(offset)
            beta = _fit_ridge_logit(
                x_list[~is_test], y_tr, lam, include_const=include_const, free_cols=free_cols,
                pen_weights=pen_weights, offset=None if off is None else off[~is_test],
            )
            scores[i] += _bernoulli_ll(
                y[is_test], x_list[is_test] @ beta + (0.0 if off is None else off[is_test])
            )
    best = np.flatnonzero(scores == scores.max())
    return float(np.asarray(lams, dtype=float)[best[-1]])  # ties -> larger penalty


def cv_model_full_trace(
    o2,
    bin_swim,
    lags: dict[str, int] | None = None,
    include_const: bool = True,
    n_folds: int = 5,
    first_bin: int | None = None,
    bin_mask=None,
    block_len: int | None = None,
    gap: int = 0,
    rng: np.random.Generator | None = None,
    ridge=None,
    ridge_free_o2_current: bool = False,
    ridge_decay_tau: float | None = None,
    extra_features=None,
    subsample_nonswim: float | None = None,
    correct_subsampling: bool = True,
):
    """Cross-validation of the full-trace GLM over blocks of time bins.

    By default (``block_len=None``) the usable bins are cut into ``n_folds``
    contiguous time blocks, each held out in turn. With ``block_len`` the bins are
    cut into blocks of that many bins and the blocks are randomly assigned to
    folds; ``block_len=1`` is a random-row split, which leaks heavily because
    adjacent bins share lagged features (useful to measure how much overlap
    matters). ``gap`` drops training bins within that many bins of any test bin
    (purging), so no lagged feature crosses a train/test boundary.

    ``ridge`` adds an L2 penalty on the slopes (intercept free, features
    standardized; see _fit_ridge_logit). ``None`` fits by plain maximum
    likelihood, a number is a fixed penalty, and a sequence of penalties is
    searched per fold by an inner contiguous CV on that fold's training bins, so
    the test fold never influences the choice. With ``ridge_free_o2_current`` the
    current-second O2 coefficient is exempt from the penalty (the older O2 lags and
    the swim lags are still penalized). ``ridge_decay_tau`` (in bins) grades the
    penalty by age instead: each coefficient is penalized by
    lambda * (1 - exp(-age / tau)), so recent lags are penalized less and the
    current-second O2 is free (see _ridge_age_weights).

    Test McFadden R2 is 1 - llf_test(model) / llf_test(null), where the null is an
    intercept-only model at the training base rate.

    Returns a dict with r2_test (pooled over all held-out bins; NaN if any fold
    could not be fit), r2_test_folds, r2_train (mean over folds; likelihood part
    only for ridge fits), test_ll_per_bin (mean held-out log-likelihood per bin,
    comparable across specs scored on the same bins), n_bins, n_fallback (folds
    that needed the Newton fallback), n_failed (folds where the fit failed),
    ridge_selected (penalty used in each fold, or None), and test_ll_bins /
    null_ll_bins / bin_idx (held-out log-likelihood of the model and of the null
    for every scored bin, NaN where the fold failed, with the trace index of each
    bin, e.g. to split the score by period).

    ``subsample_nonswim`` (a keep probability, e.g. 0.3085) trains each fold on all swim
    bins plus a random fraction of the non-swim bins (``rng`` sets the draw) while the
    held-out bins are always scored in full; with ``correct_subsampling`` the intercept
    shift this causes (log of the keep probability) is undone before scoring. The null
    model for R2 uses the swim rate of all training bins, not of the subsample.
    """

    x_list, y, bin_idx = _build_full_trace_design(
        o2, bin_swim, lags, include_const=include_const,
        first_bin=first_bin, bin_mask=bin_mask, extra=extra_features,
    )
    n_bins = len(y)
    pos = np.arange(n_bins)
    rs_sub = (rng if rng is not None else np.random.default_rng(0)) if subsample_nonswim is not None else None
    if block_len is None:
        fold_of_bin = np.zeros(n_bins, dtype=int)
        for f, test_pos in enumerate(np.array_split(pos, n_folds)):
            fold_of_bin[test_pos] = f
    else:
        rng = rng if rng is not None else np.random.default_rng()
        block_id = pos // int(block_len)
        n_blocks = block_id.max() + 1
        fold_of_block = rng.permutation(np.arange(n_blocks) % n_folds)
        fold_of_bin = fold_of_block[block_id]

    ll_model_total = 0.0
    ll_null_total = 0.0
    r2_test_folds = []
    r2_train_folds = []
    n_fallback = 0
    n_failed = 0
    ridge_selected = []
    ll_bins = np.full(n_bins, np.nan)
    null_bins = np.full(n_bins, np.nan)
    free_cols = ()
    if ridge_free_o2_current:
        col = _o2_current_col(_normalize_lags(lags), include_const=include_const)
        free_cols = () if col is None else (col,)
    pen_weights = None
    if ridge_decay_tau is not None:
        pen_weights = _pad_pen_weights(_ridge_age_weights(_normalize_lags(lags), include_const=include_const, tau=ridge_decay_tau), x_list.shape[1])

    for f in range(n_folds):
        is_test = fold_of_bin == f
        if not is_test.any():
            continue
        is_train = ~is_test
        if gap > 0:
            test_on_timeline = np.zeros(int(bin_idx.max()) + 1, dtype=bool)
            test_on_timeline[bin_idx[is_test]] = True
            near = np.convolve(test_on_timeline, np.ones(2 * gap + 1), mode="same") > 0
            is_train &= ~near[bin_idx]
        p_null_rate = y[is_train].mean() if is_train.any() else np.nan     # swim rate of ALL training bins (the null's rate)
        if subsample_nonswim is not None:
            is_train = is_train & (y | (rs_sub.random(n_bins) < subsample_nonswim))
        x_tr, y_tr, x_te, y_te = x_list[is_train], y[is_train], x_list[is_test], y[is_test]

        p_train = y_tr.mean() if len(y_tr) else np.nan
        params = None
        r2_train_fold = np.nan
        if len(y_tr) and 0 < p_train < 1:
            eta_null_tr = np.full(len(y_tr), np.log(p_train / (1 - p_train)))
            if ridge is None:
                glm_results, _, fallback = _fit_binomial_glm(x_tr, y_tr)
                n_fallback += int(fallback)
                ridge_selected.append(None)
                if glm_results is not None:
                    params = np.asarray(glm_results.params)
                    r2_train_fold = glm_results.pseudo_rsquared(kind="mcf")
            else:
                lam = (
                    float(ridge)
                    if np.ndim(ridge) == 0
                    else _select_ridge_lambda(
                        x_tr, y_tr, ridge, include_const=include_const, free_cols=free_cols,
                        pen_weights=pen_weights,
                    )
                )
                ridge_selected.append(lam)
                params = _fit_ridge_logit(
                    x_tr, y_tr, lam, include_const=include_const, free_cols=free_cols,
                    pen_weights=pen_weights,
                )
                r2_train_fold = 1.0 - _bernoulli_ll(y_tr, x_tr @ params) / _bernoulli_ll(y_tr, eta_null_tr)
        if params is None:
            n_failed += 1
            ll_model_total = np.nan
            r2_test_folds.append(np.nan)
            r2_train_folds.append(np.nan)
            continue

        p_null = p_null_rate if subsample_nonswim is not None else p_train
        eta_null = np.full(len(y_te), np.log(p_null / (1 - p_null)))
        shift = np.log(subsample_nonswim) if (subsample_nonswim is not None and correct_subsampling) else 0.0
        ll_vec = _bernoulli_ll_bins(y_te, x_te @ params + shift)
        null_vec = _bernoulli_ll_bins(y_te, eta_null)
        ll_bins[is_test] = ll_vec
        null_bins[is_test] = null_vec
        ll_model, ll_null = ll_vec.sum(), null_vec.sum()
        ll_model_total += ll_model
        ll_null_total += ll_null
        r2_test_folds.append(1.0 - ll_model / ll_null)
        r2_train_folds.append(r2_train_fold)

    return {
        "r2_test": 1.0 - ll_model_total / ll_null_total if ll_null_total != 0 else np.nan,
        "r2_test_folds": np.array(r2_test_folds),
        "r2_train": float(np.nanmean(r2_train_folds)) if np.isfinite(r2_train_folds).any() else np.nan,
        "test_ll_per_bin": ll_model_total / n_bins,
        "n_bins": n_bins,
        "n_fallback": n_fallback,
        "n_failed": n_failed,
        "ridge_selected": ridge_selected if ridge is not None else None,
        "test_ll_bins": ll_bins,
        "null_ll_bins": null_bins,
        "bin_idx": bin_idx,
    }


def _fit_condition_adjusted(
    x_tr, y_tr, cid_tr, n_cond, adjust_slopes, adjust_intercept, ridge_base, ridge_adj, has_const=True
):
    """Stage 1 (pooled ridge model) and stage 2 (per-condition ridge-shrunk deviations).

    Returns (beta_base, betas, lam_base, lam_adj, df_parts) with one beta vector per
    condition; ``lam_adj`` has one entry per condition (None if that condition was not
    adjusted). ``df_parts = (df_base, [df_adj per condition])`` holds the effective
    number of parameters of the base fit and of each adjustment; the whole model has
    ``df_base + sum(df_adj)`` (approximate because stage 2 is conditional on stage 1).
    """

    def as_grid(v):
        return [float(v)] if np.ndim(v) == 0 else list(v)

    if not has_const and adjust_intercept:
        raise ValueError("adjust_intercept needs a design with a constant column")
    grid_b = as_grid(ridge_base)
    lam_b = grid_b[0] if len(grid_b) == 1 else _select_ridge_lambda(x_tr, y_tr, grid_b, include_const=has_const)
    beta_base, df_base = _fit_ridge_logit(x_tr, y_tr, lam_b, include_const=has_const, return_df=True)
    betas = [beta_base.copy() for _ in range(n_cond)]
    lam_adj = [None] * n_cond
    df_adj = [0.0] * n_cond
    if adjust_slopes or adjust_intercept:
        for c in range(n_cond):
            sel = cid_tr == c
            yc = y_tr[sel]
            if len(yc) < 50 or not 0 < yc.mean() < 1:
                continue
            xc = x_tr[sel]
            off = xc @ beta_base
            grid_a = as_grid(ridge_adj)
            if adjust_slopes:
                if not has_const:      # no intercept anywhere: every column is adjusted, none is a constant
                    xa, inc = xc, False
                else:
                    xa, inc = (xc, True) if adjust_intercept else (xc[:, 1:], False)
                lam_a = grid_a[0] if len(grid_a) == 1 else _select_ridge_lambda(
                    xa, yc, grid_a, include_const=inc, offset=off)
                delta, df_c = _fit_ridge_logit(xa, yc, lam_a, include_const=inc, offset=off, return_df=True)
                df_adj[c] = df_c
                lam_adj[c] = lam_a
                betas[c] = beta_base + (delta if (inc or not has_const) else np.r_[0.0, delta])
            else:  # intercept-only adjustment
                delta0 = _fit_ridge_logit(xc[:, :1], yc, 0.0, include_const=True, offset=off)
                df_adj[c] = 1.0
                betas[c] = beta_base + np.r_[delta0, np.zeros(xc.shape[1] - 1)]
    return beta_base, betas, lam_b, lam_adj, (df_base, df_adj)


_DEFAULT_RIDGE_GRID = (0.3, 1, 3, 10, 30, 100, 300, 1000)


def cv_model_condition_adjusted(
    o2,
    bin_swim,
    cond_masks,
    lags: dict[str, int] | None = None,
    n_folds: int = 10,
    first_bin: int | None = None,
    adjust_slopes: bool = True,
    adjust_intercept: bool = False,
    ridge_base=_DEFAULT_RIDGE_GRID,
    ridge_adj=_DEFAULT_RIDGE_GRID,
    include_const: bool = True,
):
    """Cross-validate a base model plus per-condition adjustments.

    Stage 1 fits one ridge model on the training bins of all conditions together
    (shared intercept and slopes). Stage 2 fits, for each condition, a ridge-shrunk
    deviation from that base model on that condition's training bins:
    ``adjust_slopes`` lets the swim/O2 coefficients differ by condition (shrunk toward
    the base by ``ridge_adj``), ``adjust_intercept`` lets the baseline differ
    (unpenalized). With both False this is the pooled model; with only
    ``adjust_slopes`` the intercept stays fixed at the shared value. Contiguous
    ``n_folds`` blocks over the bins of all conditions; penalties are chosen by an
    inner CV inside each training set (a number or a sequence for each).

    ``cond_masks`` is a list of boolean arrays over the trace, one per condition
    (disjoint). Returns r2_test (pooled, null = intercept-only at the pooled
    training rate), test_ll_bins / null_ll_bins / test_p_bins (held-out log-likelihood,
    null log-likelihood and predicted swim probability of every scored bin), bin_idx,
    cond_id (condition index of every scored bin) and the penalties chosen per fold.
    ``include_const=False`` drops the intercept completely (no constant column; the O2
    or swim coefficients must then carry any baseline); the null model used for R2 is
    still an intercept-only model at the pooled training rate.
    """

    lags = _normalize_lags(lags)
    masks = [np.asarray(m, dtype=bool) for m in cond_masks]
    union = np.any(masks, axis=0)
    x_list, y, bin_idx = _build_full_trace_design(o2, bin_swim, lags, include_const=include_const,
                                                  first_bin=first_bin, bin_mask=union)
    cond_id = np.argmax(np.array([m[bin_idx] for m in masks]), axis=0)
    n_bins = len(y)
    fold_of_bin = np.zeros(n_bins, dtype=int)
    for f, test_pos in enumerate(np.array_split(np.arange(n_bins), n_folds)):
        fold_of_bin[test_pos] = f

    ll_bins = np.full(n_bins, np.nan)
    null_bins = np.full(n_bins, np.nan)
    p_bins = np.full(n_bins, np.nan)
    lam_base_chosen, lam_adj_chosen = [], []
    for f in range(n_folds):
        is_test = fold_of_bin == f
        is_train = ~is_test
        y_tr = y[is_train]
        if not is_test.any() or not 0 < y_tr.mean() < 1:
            continue
        beta_base, betas, lam_b, lam_a, _ = _fit_condition_adjusted(
            x_list[is_train], y_tr, cond_id[is_train], len(masks),
            adjust_slopes, adjust_intercept, ridge_base, ridge_adj, has_const=include_const)
        lam_base_chosen.append(lam_b)
        lam_adj_chosen.extend(v for v in lam_a if v is not None)
        eta_null = np.log(y_tr.mean() / (1 - y_tr.mean()))
        for c in range(len(masks)):
            te_c = is_test & (cond_id == c)
            if not te_c.any():
                continue
            eta = x_list[te_c] @ betas[c]
            ll_bins[te_c] = _bernoulli_ll_bins(y[te_c], eta)
            null_bins[te_c] = _bernoulli_ll_bins(y[te_c], np.full(te_c.sum(), eta_null))
            p_bins[te_c] = expit(eta)
    return {
        "r2_test": 1.0 - np.nansum(ll_bins) / np.nansum(null_bins),
        "test_ll_bins": ll_bins,
        "null_ll_bins": null_bins,
        "test_p_bins": p_bins,
        "bin_idx": bin_idx,
        "cond_id": cond_id,
        "lam_base": lam_base_chosen,
        "lam_adj": lam_adj_chosen,
    }


def fit_model_condition_adjusted(
    o2,
    bin_swim,
    cond_masks,
    lags: dict[str, int] | None = None,
    first_bin: int | None = None,
    adjust_slopes: bool = True,
    adjust_intercept: bool = False,
    ridge_base=_DEFAULT_RIDGE_GRID,
    ridge_adj=_DEFAULT_RIDGE_GRID,
    include_const: bool = True,
):
    """Final (all-bins) fit of the base model plus per-condition adjustments.

    Same two stages as cv_model_condition_adjusted but fitted once on every bin of
    all conditions (penalties chosen by inner CV over those bins). Returns beta_base,
    betas (one vector per condition, base + adjustment), the chosen penalties, the
    design matrix x, labels y, bin_idx, cond_id and in-sample p_hat per bin. Column
    order of the betas: constant, then the O2 block (oldest -> current), then the swim
    block (oldest -> previous bin), following ``lags`` (no constant column when
    ``include_const=False``, which drops the intercept completely). Also returns the in-sample
    log-likelihood ``llf``, the effective number of parameters ``df_eff`` (split into
    ``df_base`` and ``df_adj`` per condition) and the effective-df AIC
    ``aic = 2 * df_eff - 2 * llf`` (per bin: ``aic_per_bin``).
    """

    lags = _normalize_lags(lags)
    masks = [np.asarray(m, dtype=bool) for m in cond_masks]
    union = np.any(masks, axis=0)
    x_list, y, bin_idx = _build_full_trace_design(o2, bin_swim, lags, include_const=include_const,
                                                  first_bin=first_bin, bin_mask=union)
    cond_id = np.argmax(np.array([m[bin_idx] for m in masks]), axis=0)
    beta_base, betas, lam_b, lam_a, (df_base, df_adj) = _fit_condition_adjusted(
        x_list, y, cond_id, len(masks), adjust_slopes, adjust_intercept, ridge_base, ridge_adj, has_const=include_const)
    df_eff = df_base + float(sum(df_adj))
    eta = np.array([x_list[i] @ betas[cond_id[i]] for i in range(len(y))])
    ll_bins = _bernoulli_ll_bins(y, eta)
    llf = float(ll_bins.sum())
    return {
        "beta_base": beta_base, "betas": betas, "lam_base": lam_b, "lam_adj": lam_a,
        "x": x_list, "y": y, "bin_idx": bin_idx, "cond_id": cond_id, "p_hat": expit(eta),
        "ll_bins": ll_bins, "llf": llf, "df_eff": df_eff, "df_base": df_base, "df_adj": df_adj,
        "aic": 2 * df_eff - 2 * llf,
        "aic_per_bin": (2 * df_eff - 2 * llf) / len(y), "n_bins": len(y),
    }


def fit_model_shared_intercept(
    o2,
    bin_swim,
    lags: dict[str, int] | None = None,
    intercept: float = 0.0,
    first_bin: int | None = None,
    bin_mask=None,
    ridge=_DEFAULT_RIDGE_GRID,
    count_intercept: bool = True,
    ridge_free_o2_current: bool = False,
    ridge_decay_tau: float | None = None,
):
    """Ridge fit of a model whose intercept is fixed to a given value.

    ``intercept`` (e.g. the intercept of another model's fit) enters as a fixed offset;
    only the O2/swim coefficients are fitted (no constant column). ``ridge`` is a number or
    a sequence searched by an inner contiguous CV. ``count_intercept`` adds 1 to the
    effective df because the borrowed intercept was estimated from the same data.
    ``ridge_free_o2_current`` leaves the current-second O2 weight unpenalized; ``ridge_decay_tau``
    uses the age-decaying penalty weights 1 - exp(-age / tau) (see _ridge_age_weights).
    Returns llf, aic (= 2 * df_eff - 2 * llf), aic_per_bin, pseudo_r2_mcf (null: intercept-only
    at the swim rate of the fitted bins), params, intercept, df_eff, ll_per_bin, p_hat, bin_idx, ridge_lambda.
    """

    x_list, y, bin_idx = _build_full_trace_design(o2, bin_swim, lags, include_const=False,
                                                  first_bin=first_bin, bin_mask=bin_mask)
    off = np.full(len(y), float(intercept))
    grid = [float(ridge)] if np.ndim(ridge) == 0 else list(ridge)
    col = _o2_current_col(_normalize_lags(lags), include_const=False) if ridge_free_o2_current else None
    free = () if col is None else (col,)
    weights = None if ridge_decay_tau is None else _ridge_age_weights(_normalize_lags(lags), include_const=False, tau=ridge_decay_tau)
    lam = grid[0] if len(grid) == 1 else _select_ridge_lambda(x_list, y, grid, include_const=False, free_cols=free, pen_weights=weights, offset=off)
    params, df = _fit_ridge_logit(x_list, y, lam, include_const=False, free_cols=free, pen_weights=weights, offset=off, return_df=True)
    eta = x_list @ params + off
    ll = _bernoulli_ll_bins(y, eta)
    llf = float(ll.sum())
    df_eff = df + (1.0 if count_intercept else 0.0)
    p_base = y.mean()
    llf_null = len(y) * (p_base * np.log(p_base) + (1 - p_base) * np.log(1 - p_base))
    return {
        "llf": llf, "aic": 2 * df_eff - 2 * llf, "aic_per_bin": (2 * df_eff - 2 * llf) / len(y),
        "pseudo_r2_mcf": 1.0 - llf / llf_null, "params": params, "intercept": float(intercept),
        "df_eff": df_eff, "ll_per_bin": ll, "p_hat": expit(eta), "bin_idx": bin_idx, "ridge_lambda": lam,
    }


def cv_models_shared_intercept(
    o2,
    bin_swim,
    specs,
    ref_index: int = 0,
    n_folds: int = 10,
    first_bin: int | None = None,
    bin_mask=None,
    ridge=_DEFAULT_RIDGE_GRID,
):
    """Contiguous-block CV of several models that all use the reference model's intercept.

    In every fold the reference model (``specs[ref_index]``, with its own intercept) is fitted
    on the training bins; its intercept is then fixed as an offset in all the other models,
    which are fitted on the same training bins (ridge penalties chosen by an inner CV) and
    scored on the held-out bins. The null for R2 is an intercept-only model at the training
    swim rate. Returns a list (one dict per spec) with r2_test, test_ll_bins, null_ll_bins,
    test_p_bins and bin_idx; the reference entry uses its own fitted intercept.
    """

    designs = []
    for k, lags in enumerate(specs):
        x_k, y, bin_idx = _build_full_trace_design(o2, bin_swim, lags, include_const=(k == ref_index),
                                                   first_bin=first_bin, bin_mask=bin_mask)
        designs.append(x_k)
    n_bins = len(y)
    fold_of_bin = np.zeros(n_bins, dtype=int)
    for f, test_pos in enumerate(np.array_split(np.arange(n_bins), n_folds)):
        fold_of_bin[test_pos] = f
    grid = [float(ridge)] if np.ndim(ridge) == 0 else list(ridge)
    out = [{"test_ll_bins": np.full(n_bins, np.nan), "null_ll_bins": np.full(n_bins, np.nan),
            "test_p_bins": np.full(n_bins, np.nan), "bin_idx": bin_idx} for _ in specs]
    for f in range(n_folds):
        te = fold_of_bin == f
        tr = ~te
        y_tr = y[tr]
        if not te.any() or not 0 < y_tr.mean() < 1:
            continue
        x_ref = designs[ref_index]
        lam_ref = grid[0] if len(grid) == 1 else _select_ridge_lambda(x_ref[tr], y_tr, grid)
        beta_ref = _fit_ridge_logit(x_ref[tr], y_tr, lam_ref)
        b0 = float(beta_ref[0])
        eta_null = np.log(y_tr.mean() / (1 - y_tr.mean()))
        for k in range(len(specs)):
            if k == ref_index:
                eta = x_ref[te] @ beta_ref
            else:
                off = np.full(tr.sum(), b0)
                lam = grid[0] if len(grid) == 1 else _select_ridge_lambda(
                    designs[k][tr], y_tr, grid, include_const=False, offset=off)
                beta = _fit_ridge_logit(designs[k][tr], y_tr, lam, include_const=False, offset=off)
                eta = designs[k][te] @ beta + b0
            out[k]["test_ll_bins"][te] = _bernoulli_ll_bins(y[te], eta)
            out[k]["null_ll_bins"][te] = _bernoulli_ll_bins(y[te], np.full(te.sum(), eta_null))
            out[k]["test_p_bins"][te] = expit(eta)
    for o in out:
        o["r2_test"] = 1.0 - np.nansum(o["test_ll_bins"]) / np.nansum(o["null_ll_bins"])
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
