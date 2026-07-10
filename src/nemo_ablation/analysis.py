from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import time
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from numpy.lib import format as npy_format
from scipy.stats import mannwhitneyu, rankdata, t as student_t


COHORT_DIRS = {
    "neuron": "neuron_fish",
    "glia": "glia_fish",
}


@dataclass(frozen=True)
class AblationAnalysisConfig:
    pre_normoxia_window_s: tuple[float, float] = (10 * 60, 20 * 60)
    pre_hypoxia_window_s: tuple[float, float] = (25 * 60, 35 * 60)
    post_normoxia_window_s: tuple[float, float] = (70 * 60, 80 * 60)
    post_hypoxia_window_s: tuple[float, float] = (85 * 60, 95 * 60)
    pre_r_window_s: tuple[float, float] = (10 * 60, 60 * 60)
    post_r_window_s: tuple[float, float] = (70 * 60, 120 * 60)
    post_time_shift_s: float = 60 * 60
    use_idx_f: bool = True


def iter_ablation_recordings(
    data_root: str | Path,
    dataset: str = "all",
) -> list[tuple[str, Path]]:
    root = Path(data_root)
    if dataset not in {*COHORT_DIRS, "all"}:
        raise ValueError(f"Unknown dataset {dataset!r}; use neuron, glia, or all.")

    cohorts = COHORT_DIRS if dataset == "all" else {dataset: COHORT_DIRS[dataset]}
    recordings: list[tuple[str, Path]] = []
    for cohort, dirname in cohorts.items():
        cohort_dir = root / dirname
        if not cohort_dir.is_dir():
            raise FileNotFoundError(f"Missing cohort directory: {cohort_dir}")
        recordings.extend(
            (cohort, path) for path in sorted(cohort_dir.iterdir()) if path.is_dir()
        )
    return recordings


def _read_npy_header(handle) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = npy_format.read_magic(handle)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(handle)
    return npy_format.read_array_header_2_0(handle)


def _open_stored_npz_array(npz_path: Path, key: str) -> np.memmap:
    """Memory-map an uncompressed NPY member without loading the full NPZ."""
    member_name = f"{key}.npy"
    with ZipFile(npz_path) as archive:
        try:
            member = archive.getinfo(member_name)
        except KeyError as exc:
            raise KeyError(f"{member_name!r} is missing from {npz_path}") from exc
        if member.compress_type != ZIP_STORED:
            raise ValueError(
                f"{member_name} in {npz_path} is compressed and cannot be memory-mapped."
            )

    with npz_path.open("rb") as handle:
        handle.seek(member.header_offset)
        local_header = handle.read(30)
        if len(local_header) != 30:
            raise ValueError(f"Invalid ZIP local header in {npz_path}")
        fields = struct.unpack("<4s5H3I2H", local_header)
        if fields[0] != b"PK\x03\x04":
            raise ValueError(f"Invalid ZIP signature in {npz_path}")
        filename_length, extra_length = fields[9], fields[10]
        npy_start = member.header_offset + 30 + filename_length + extra_length
        handle.seek(npy_start)
        shape, fortran_order, dtype = _read_npy_header(handle)
        data_offset = handle.tell()

    order = "F" if fortran_order else "C"
    return np.memmap(
        npz_path,
        mode="r",
        dtype=dtype,
        offset=data_offset,
        shape=shape,
        order=order,
    )


def _window_mask(time_stamp: np.ndarray, window_s: tuple[float, float]) -> np.ndarray:
    start, stop = window_s
    return (time_stamp > start) & (time_stamp < stop)


def _pooled_d_prime(
    traces: np.ndarray,
    normoxia_mask: np.ndarray,
    hypoxia_mask: np.ndarray,
) -> np.ndarray:
    normoxia = traces[:, normoxia_mask].astype(float, copy=False)
    hypoxia = traces[:, hypoxia_mask].astype(float, copy=False)
    pooled_std = np.sqrt(
        (np.nanvar(normoxia, axis=1) + np.nanvar(hypoxia, axis=1)) / 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            np.nanmean(normoxia, axis=1) - np.nanmean(hypoxia, axis=1)
        ) / pooled_std


def _mean_diff_and_rank_sum(
    traces: np.ndarray,
    normoxia_mask: np.ndarray,
    hypoxia_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, mean_difference, p_value = _window_mean_comparison_and_rank_sum(
        traces,
        normoxia_mask,
        hypoxia_mask,
    )
    return mean_difference, p_value


def _window_mean_comparison_and_rank_sum(
    traces: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = traces[:, first_mask].astype(float, copy=False)
    second = traces[:, second_mask].astype(float, copy=False)
    first_mean = np.nanmean(first, axis=1)
    second_mean = np.nanmean(second, axis=1)
    mean_difference = second_mean - first_mean
    test_result = mannwhitneyu(
        second,
        first,
        axis=1,
        alternative="two-sided",
        method="asymptotic",
        nan_policy="omit",
    )
    return first_mean, second_mean, mean_difference, np.asarray(test_result.pvalue)


def _spearman_against_reference(
    traces: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(traces, dtype=float)
    reference = np.asarray(reference, dtype=float)
    num_cells, num_samples = traces.shape
    r_values = np.full(num_cells, np.nan, dtype=float)
    p_values = np.full(num_cells, np.nan, dtype=float)
    if num_samples < 3:
        return r_values, p_values

    finite_rows = np.isfinite(traces).all(axis=1)
    if not np.isfinite(reference).all() or not finite_rows.any():
        return r_values, p_values

    reference_rank = rankdata(reference)
    reference_rank -= reference_rank.mean()
    reference_norm = np.linalg.norm(reference_rank)
    if reference_norm == 0:
        return r_values, p_values

    trace_rank = rankdata(traces[finite_rows], axis=1)
    trace_rank -= trace_rank.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(trace_rank, axis=1) * reference_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (trace_rank @ reference_rank) / denominator
    corr = np.clip(corr, -1, 1)

    dof = num_samples - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = corr * np.sqrt(dof / ((corr + 1) * (1 - corr)))
    p_value = 2 * student_t.sf(np.abs(statistic), dof)
    r_values[finite_rows] = corr
    p_values[finite_rows] = p_value
    return r_values, p_values


def _atomic_savez(path: Path, **arrays) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez(temporary_path, **arrays)
    temporary_path.replace(path)


def _required_paths(recording_dir: Path, use_idx_f: bool) -> list[Path]:
    paths = [
        recording_dir / "cell_dff.npz",
        recording_dir / "time_stamp.npy",
        recording_dir / "invalid_.npy",
    ]
    if use_idx_f:
        paths.append(recording_dir / "idx_F.npy")
    return paths


def analyze_ablation_recording(
    recording_dir: str | Path,
    oxygen_mean: np.ndarray,
    *,
    config: AblationAnalysisConfig | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, int | float | str]:
    config = config or AblationAnalysisConfig()
    recording_dir = Path(recording_dir)
    d_prime_path = recording_dir / "d_prime_ablation_comparison.npz"
    r_path = recording_dir / "r_ablation_comparison.npz"
    dff_diff_path = recording_dir / "dff_diff_ablation_comparison.npz"
    baseline_diff_path = recording_dir / "baseline_diff_ablation_comparison.npz"
    dff_normoxia_path = recording_dir / "dff_normoxia_ablation_comparison.npz"

    missing = [path for path in _required_paths(recording_dir, config.use_idx_f) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(map(str, missing)))
    write_d_prime = force or not d_prime_path.exists()
    write_r = force or not r_path.exists()
    write_dff_diff = force or not dff_diff_path.exists()
    write_baseline_diff = force or not baseline_diff_path.exists()
    write_dff_normoxia = force or not dff_normoxia_path.exists()
    if not any(
        (
            write_d_prime,
            write_r,
            write_dff_diff,
            write_baseline_diff,
            write_dff_normoxia,
        )
    ):
        return {"status": "skipped_existing", "recording": recording_dir.name}

    dff = _open_stored_npz_array(recording_dir / "cell_dff.npz", "dFF")
    baseline = _open_stored_npz_array(recording_dir / "cell_dff.npz", "baseline")
    time_stamp = np.load(recording_dir / "time_stamp.npy", mmap_mode="r")
    invalid = np.asarray(np.load(recording_dir / "invalid_.npy"), dtype=bool)

    if dff.ndim != 2 or baseline.shape != dff.shape:
        raise ValueError(f"Expected matching 2D dFF/baseline arrays at {recording_dir}")
    num_cells = dff.shape[0]
    if invalid.shape != (num_cells,):
        raise ValueError(f"invalid_.npy does not align with cell_dff.npz at {recording_dir}")

    valid_mask = ~invalid
    if config.use_idx_f:
        idx_f = np.asarray(np.load(recording_dir / "idx_F.npy"), dtype=bool)
        if idx_f.shape != (num_cells,):
            raise ValueError(f"idx_F.npy does not align with cell_dff.npz at {recording_dir}")
        valid_mask &= idx_f

    num_frames = min(dff.shape[1], baseline.shape[1], time_stamp.shape[0])
    time_values = np.asarray(time_stamp[:num_frames], dtype=float)
    if np.any(np.diff(time_values) <= 0):
        raise ValueError(f"time_stamp.npy is not strictly increasing at {recording_dir}")

    masks = {
        "pre_normoxia": _window_mask(time_values, config.pre_normoxia_window_s),
        "pre_hypoxia": _window_mask(time_values, config.pre_hypoxia_window_s),
        "post_normoxia": _window_mask(time_values, config.post_normoxia_window_s),
        "post_hypoxia": _window_mask(time_values, config.post_hypoxia_window_s),
        "pre_r": _window_mask(time_values, config.pre_r_window_s),
        "post_r": _window_mask(time_values, config.post_r_window_s),
    }
    empty_windows = [name for name, mask in masks.items() if not mask.any()]
    if empty_windows:
        raise ValueError(f"Empty analysis windows at {recording_dir}: {empty_windows}")

    oxygen_mean = np.asarray(oxygen_mean, dtype=float).squeeze()
    if oxygen_mean.ndim != 1:
        raise ValueError("oxygen_mean must be one-dimensional.")
    oxygen_time_s = np.arange(oxygen_mean.size, dtype=float) + 1
    pre_oxygen = np.interp(time_values[masks["pre_r"]], oxygen_time_s, oxygen_mean)
    post_local_time = time_values[masks["post_r"]] - config.post_time_shift_s
    post_oxygen = np.interp(post_local_time, oxygen_time_s, oxygen_mean)

    summary: dict[str, int | float | str] = {
        "status": "validated" if dry_run else "completed",
        "recording": recording_dir.name,
        "num_cells": num_cells,
        "num_valid_cells": int(valid_mask.sum()),
        "num_frames": num_frames,
    }
    if dry_run:
        summary.update({f"frames_{name}": int(mask.sum()) for name, mask in masks.items()})
        return summary

    started = time.perf_counter()
    output_index = np.flatnonzero(valid_mask)
    common_metadata = {
        "valid_mask": valid_mask,
        "num_frames_used": np.asarray(num_frames),
        "post_time_shift_s": np.asarray(config.post_time_shift_s),
    }

    if write_d_prime or write_dff_diff or write_dff_normoxia:
        print(f"  loading dFF for {output_index.size:,} selected cells", flush=True)
        dff_selected = np.asarray(dff[valid_mask, :num_frames])

        if write_d_prime:
            d_prime1 = np.full(num_cells, np.nan, dtype=float)
            d_prime2 = np.full(num_cells, np.nan, dtype=float)
            d_prime1[output_index] = _pooled_d_prime(
                dff_selected,
                masks["pre_normoxia"],
                masks["pre_hypoxia"],
            )
            d_prime2[output_index] = _pooled_d_prime(
                dff_selected,
                masks["post_normoxia"],
                masks["post_hypoxia"],
            )
            _atomic_savez(
                d_prime_path,
                d_prime1=d_prime1,
                d_prime2=d_prime2,
                pre_normoxia_window_s=np.asarray(config.pre_normoxia_window_s),
                pre_hypoxia_window_s=np.asarray(config.pre_hypoxia_window_s),
                post_normoxia_window_s=np.asarray(config.post_normoxia_window_s),
                post_hypoxia_window_s=np.asarray(config.post_hypoxia_window_s),
                **common_metadata,
            )

        if write_dff_diff:
            diff_1 = np.full(num_cells, np.nan, dtype=float)
            p_1 = np.full(num_cells, np.nan, dtype=float)
            diff_2 = np.full(num_cells, np.nan, dtype=float)
            p_2 = np.full(num_cells, np.nan, dtype=float)
            diff_1_selected, p_1_selected = _mean_diff_and_rank_sum(
                dff_selected,
                masks["pre_normoxia"],
                masks["pre_hypoxia"],
            )
            diff_2_selected, p_2_selected = _mean_diff_and_rank_sum(
                dff_selected,
                masks["post_normoxia"],
                masks["post_hypoxia"],
            )
            diff_1[output_index], p_1[output_index] = diff_1_selected, p_1_selected
            diff_2[output_index], p_2[output_index] = diff_2_selected, p_2_selected
            _atomic_savez(
                dff_diff_path,
                p_1=p_1,
                diff_1=diff_1,
                p_2=p_2,
                diff_2=diff_2,
                difference_definition=np.asarray("mean_hypoxia_minus_mean_normoxia"),
                rank_sum_test=np.asarray("mannwhitneyu_two_sided_asymptotic"),
                pre_normoxia_window_s=np.asarray(config.pre_normoxia_window_s),
                pre_hypoxia_window_s=np.asarray(config.pre_hypoxia_window_s),
                post_normoxia_window_s=np.asarray(config.post_normoxia_window_s),
                post_hypoxia_window_s=np.asarray(config.post_hypoxia_window_s),
                **common_metadata,
            )

        if write_dff_normoxia:
            normoxia_mean_1 = np.full(num_cells, np.nan, dtype=float)
            normoxia_mean_2 = np.full(num_cells, np.nan, dtype=float)
            normoxia_diff = np.full(num_cells, np.nan, dtype=float)
            p = np.full(num_cells, np.nan, dtype=float)
            mean_1_selected, mean_2_selected, diff_selected, p_selected = (
                _window_mean_comparison_and_rank_sum(
                    dff_selected,
                    masks["pre_normoxia"],
                    masks["post_normoxia"],
                )
            )
            normoxia_mean_1[output_index] = mean_1_selected
            normoxia_mean_2[output_index] = mean_2_selected
            normoxia_diff[output_index] = diff_selected
            p[output_index] = p_selected
            _atomic_savez(
                dff_normoxia_path,
                normoxia_mean_1=normoxia_mean_1,
                normoxia_mean_2=normoxia_mean_2,
                normoxia_diff=normoxia_diff,
                p=p,
                difference_definition=np.asarray("post_normoxia_minus_pre_normoxia"),
                source_signal=np.asarray("dFF"),
                rank_sum_test=np.asarray("mannwhitneyu_two_sided_asymptotic"),
                pre_normoxia_window_s=np.asarray(config.pre_normoxia_window_s),
                post_normoxia_window_s=np.asarray(config.post_normoxia_window_s),
                **common_metadata,
            )
        del dff_selected

    if write_r or write_baseline_diff:
        print(f"  loading baseline for {output_index.size:,} selected cells", flush=True)
        baseline_selected = np.asarray(baseline[valid_mask, :num_frames])

        if write_r:
            r1 = np.full(num_cells, np.nan, dtype=float)
            p1 = np.full(num_cells, np.nan, dtype=float)
            r2 = np.full(num_cells, np.nan, dtype=float)
            p2 = np.full(num_cells, np.nan, dtype=float)
            r1_selected, p1_selected = _spearman_against_reference(
                baseline_selected[:, masks["pre_r"]],
                pre_oxygen,
            )
            r2_selected, p2_selected = _spearman_against_reference(
                baseline_selected[:, masks["post_r"]],
                post_oxygen,
            )
            r1[output_index], p1[output_index] = r1_selected, p1_selected
            r2[output_index], p2[output_index] = r2_selected, p2_selected
            _atomic_savez(
                r_path,
                r1=r1,
                p1=p1,
                r2=r2,
                p2=p2,
                pre_r_window_s=np.asarray(config.pre_r_window_s),
                post_r_window_s=np.asarray(config.post_r_window_s),
                oxygen_reference=oxygen_mean,
                **common_metadata,
            )

        if write_baseline_diff:
            diff_1 = np.full(num_cells, np.nan, dtype=float)
            p_1 = np.full(num_cells, np.nan, dtype=float)
            diff_2 = np.full(num_cells, np.nan, dtype=float)
            p_2 = np.full(num_cells, np.nan, dtype=float)
            diff_1_selected, p_1_selected = _mean_diff_and_rank_sum(
                baseline_selected,
                masks["pre_normoxia"],
                masks["pre_hypoxia"],
            )
            diff_2_selected, p_2_selected = _mean_diff_and_rank_sum(
                baseline_selected,
                masks["post_normoxia"],
                masks["post_hypoxia"],
            )
            diff_1[output_index], p_1[output_index] = diff_1_selected, p_1_selected
            diff_2[output_index], p_2[output_index] = diff_2_selected, p_2_selected
            _atomic_savez(
                baseline_diff_path,
                p_1=p_1,
                diff_1=diff_1,
                p_2=p_2,
                diff_2=diff_2,
                difference_definition=np.asarray("mean_hypoxia_minus_mean_normoxia"),
                source_signal=np.asarray("baseline"),
                rank_sum_test=np.asarray("mannwhitneyu_two_sided_asymptotic"),
                pre_normoxia_window_s=np.asarray(config.pre_normoxia_window_s),
                pre_hypoxia_window_s=np.asarray(config.pre_hypoxia_window_s),
                post_normoxia_window_s=np.asarray(config.post_normoxia_window_s),
                post_hypoxia_window_s=np.asarray(config.post_hypoxia_window_s),
                **common_metadata,
            )
        del baseline_selected

    summary["outputs_written"] = ",".join(
        name
        for name, write_output in (
            (d_prime_path.name, write_d_prime),
            (r_path.name, write_r),
            (dff_diff_path.name, write_dff_diff),
            (
                baseline_diff_path.name,
                write_baseline_diff,
            ),
            (
                dff_normoxia_path.name,
                write_dff_normoxia,
            ),
        )
        if write_output
    )
    summary["elapsed_s"] = time.perf_counter() - started
    return summary


def analyze_ablation_dataset(
    data_root: str | Path,
    oxygen_mean: np.ndarray,
    *,
    dataset: str = "all",
    config: AblationAnalysisConfig | None = None,
    force: bool = False,
    dry_run: bool = False,
    max_fish: int | None = None,
) -> list[dict[str, int | float | str]]:
    recordings = iter_ablation_recordings(data_root, dataset)
    if max_fish is not None:
        recordings = recordings[:max_fish]

    results = []
    for fish_index, (cohort, recording_dir) in enumerate(recordings, start=1):
        print(f"[{fish_index}/{len(recordings)}] {cohort}: {recording_dir.name}")
        result = analyze_ablation_recording(
            recording_dir,
            oxygen_mean,
            config=config,
            force=force,
            dry_run=dry_run,
        )
        results.append(result)
        print("  " + ", ".join(f"{key}={value}" for key, value in result.items()))
    return results
