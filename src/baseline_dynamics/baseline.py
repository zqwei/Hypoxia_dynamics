from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import rankdata, spearmanr, zscore
from sklearn.decomposition import FactorAnalysis, PCA
from tqdm import tqdm

from ..paths import data_file, ensure_directory, load_datalist


def load_baseline_datalist(name: str = "datalist.csv") -> pd.DataFrame:
    return load_datalist(name)


def load_oxygen_mean(name: str = "O2_internal.npz") -> np.ndarray:
    with np.load(data_file(name), allow_pickle=True) as oxygen_data:
        return oxygen_data["oxy_mean"]


def ecdf(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from statsmodels.distributions.empirical_distribution import ECDF

    ecdf_ = ECDF(sample)
    x_values = np.linspace(np.min(sample), np.max(sample))
    y_values = ecdf_(x_values)
    return x_values, y_values


def _iter_selected_rows(
    df: pd.DataFrame,
    *,
    start_index: int | None = None,
    max_index: int | None = None,
):
    for ind, row in df.iterrows():
        if start_index is not None and ind < start_index:
            continue
        if max_index is not None and ind > max_index:
            continue
        yield ind, row


def _split_indices(num_items: int, chunk_size: int = 1000) -> list[np.ndarray]:
    if num_items <= 0:
        return []
    num_splits = max(1, num_items // chunk_size)
    return list(np.array_split(np.arange(num_items, dtype=int), num_splits))


def _save_root(row: pd.Series) -> Path:
    return ensure_directory(str(row["save_root"]))


def _align_time_series(
    traces: np.ndarray,
    time_stamp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(time_stamp) < traces.shape[1]:
        traces = traces[:, : len(time_stamp)]
    elif len(time_stamp) > traces.shape[1]:
        time_stamp = time_stamp[: traces.shape[1]]
    return traces, time_stamp


def _spearman_corr_matrix(
    traces: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    references = np.atleast_2d(np.asarray(references, dtype=float))

    if traces.shape[1] != references.shape[1]:
        raise ValueError("Trace and reference matrices must have the same length.")

    traces_ranked = np.apply_along_axis(rankdata, 1, traces)
    references_ranked = np.apply_along_axis(rankdata, 1, references)

    traces_ranked -= traces_ranked.mean(axis=1, keepdims=True)
    references_ranked -= references_ranked.mean(axis=1, keepdims=True)

    denominator = (
        np.linalg.norm(traces_ranked, axis=1, keepdims=True)
        @ np.linalg.norm(references_ranked, axis=1, keepdims=True).T
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = traces_ranked @ references_ranked.T / denominator
    return np.nan_to_num(corr)


def _configure_plotting() -> None:
    sns.set(style="ticks", font_scale=1.0)
    plt.rcParams["figure.figsize"] = (4, 3)


def export_baseline_correlations(
    datalist_name: str = "datalist.csv",
    *,
    max_index: int | None = None,
) -> None:
    df = load_baseline_datalist(datalist_name)

    for ind, row in _iter_selected_rows(df, max_index=max_index):
        save_root = _save_root(row)
        output_path = save_root / "baseline_oxy.npz"
        cell_dff_path = save_root / "cell_dff.npz"

        if output_path.exists() or not cell_dff_path.exists():
            continue

        with np.load(cell_dff_path, allow_pickle=True) as cell_dff:
            dff = cell_dff["dFF"].astype("float16")
            baseline = cell_dff["baseline"].astype("float16")

        valid_f = np.ones(dff.shape[0], dtype=bool)
        for cell_index, trace in enumerate(dff):
            if np.isnan(trace).any() or trace.max() > 10:
                valid_f[cell_index] = False

        baseline = baseline[valid_f]
        mean_baseline = baseline.mean(axis=0)
        p_values = np.zeros(baseline.shape[0])
        r_values = np.zeros(baseline.shape[0])

        for cell_index in tqdm(
            range(baseline.shape[0]),
            desc=f"baseline_corr {ind}",
            leave=False,
        ):
            corr, p_value = spearmanr(mean_baseline, baseline[cell_index])
            p_values[cell_index] = p_value
            r_values[cell_index] = corr

        np.savez(
            output_path,
            valid_F=valid_f,
            mean_baseline_=mean_baseline,
            p_=p_values,
            r_=r_values,
        )


def export_baseline_averages(
    datalist_name: str = "datalist.csv",
    *,
    max_index: int | None = None,
    negative_threshold: float = -0.1,
    positive_threshold: float = 0.8,
) -> None:
    df = load_baseline_datalist(datalist_name)

    for _, row in _iter_selected_rows(df, max_index=max_index):
        save_root = _save_root(row)
        baseline_oxy_path = save_root / "baseline_oxy.npz"
        cell_dff_path = save_root / "cell_dff.npz"

        if not baseline_oxy_path.exists() or not cell_dff_path.exists():
            continue

        with np.load(baseline_oxy_path, allow_pickle=True) as baseline_oxy:
            valid_f = baseline_oxy["valid_F"]
            r_values = baseline_oxy["r_"]

        with np.load(cell_dff_path, allow_pickle=True) as cell_dff:
            baseline = cell_dff["baseline"].astype("float16")[valid_f]

        neg_baseline = baseline[r_values < negative_threshold].mean(axis=0)
        pos_baseline = baseline[r_values > positive_threshold].mean(axis=0)

        np.savez(
            save_root / "baseline_oxy_ave.npz",
            neg_baseline_=neg_baseline,
            pos_baseline_=pos_baseline,
        )


def export_baseline_clusters(
    datalist_name: str = "datalist.csv",
    *,
    max_index: int | None = None,
    n_components: int = 20,
    explained_variance_threshold: float = 0.9,
    seed_threshold: float = 0.4,
    min_seed_size: int = 100,
    assignment_threshold: float = 0.4,
) -> None:
    df = load_baseline_datalist(datalist_name)

    for ind, row in _iter_selected_rows(df, max_index=max_index):
        save_root = _save_root(row)
        output_path = save_root / "baseline_clusters.npz"
        cell_dff_path = save_root / "cell_dff.npz"

        if output_path.exists() or not cell_dff_path.exists():
            continue

        print(f"Processing {ind} at {save_root}")
        with np.load(cell_dff_path, allow_pickle=True) as cell_dff:
            dff = cell_dff["dFF"].astype("float16")
            baseline = cell_dff["baseline"].astype("float16")

        invalid = np.isnan(dff).sum(axis=-1) > 0
        baseline = baseline[~invalid]
        zbaseline = zscore(baseline.astype(float), axis=-1)

        component_count = min(n_components, zbaseline.shape[0], zbaseline.shape[1])
        if component_count == 0:
            np.savez(
                output_path,
                invalid_=invalid,
                ev_thres=np.zeros(zbaseline.shape[0], dtype=bool),
                cell_cluster=np.array([], dtype=int),
                cluster_act_mat=np.empty((0, 0), dtype="float16"),
            )
            continue

        pca = PCA(n_components=component_count)
        pca.fit(zbaseline)
        zbaseline_pca = pca.transform(zbaseline)
        zbaseline_res = pca.inverse_transform(zbaseline_pca)
        denominator = np.maximum((zbaseline**2).sum(axis=-1), np.finfo(float).eps)
        exp_var = 1 - ((zbaseline_res - zbaseline) ** 2).sum(axis=-1) / denominator
        ev_thres = exp_var > explained_variance_threshold
        zbaseline_res = zbaseline_res[ev_thres].astype("float16")
        zbaseline_pca = zbaseline_pca[ev_thres].astype("float16")

        num_cells = zbaseline_res.shape[0]
        cell_cluster = np.full(num_cells, -1, dtype=np.int16)
        if num_cells == 0:
            np.savez(
                output_path,
                invalid_=invalid,
                ev_thres=ev_thres,
                cell_cluster=cell_cluster,
                cluster_act_mat=np.empty((0, 0), dtype="float16"),
            )
            continue

        z_pca_max = np.abs(zbaseline_pca).max(axis=-1, keepdims=True)
        ind_pca_max = np.argmax(np.abs(zbaseline_pca), axis=-1)
        sign_pca_max = zbaseline_pca[np.arange(num_cells), ind_pca_max] > 0
        sparse_pca = (np.abs(zbaseline_pca) > z_pca_max * seed_threshold).sum(axis=-1)

        cluster_act_mat = []
        for component_index in range(zbaseline_pca.shape[1]):
            neg_idx = (
                (ind_pca_max == component_index)
                & (sparse_pca == 1)
                & ~sign_pca_max
            )
            if neg_idx.sum() > min_seed_size:
                cluster_act_mat.append(zbaseline_res[neg_idx].mean(axis=0))

            pos_idx = (
                (ind_pca_max == component_index)
                & (sparse_pca == 1)
                & sign_pca_max
            )
            if pos_idx.sum() > min_seed_size:
                cluster_act_mat.append(zbaseline_res[pos_idx].mean(axis=0))

        cluster_act_mat = np.asarray(cluster_act_mat, dtype="float16")
        if cluster_act_mat.size == 0:
            np.savez(
                output_path,
                invalid_=invalid,
                ev_thres=ev_thres,
                cell_cluster=cell_cluster,
                cluster_act_mat=np.empty((0, 0), dtype="float16"),
            )
            continue

        max_corr = np.full(num_cells, -1.0)
        for split in tqdm(
            _split_indices(num_cells),
            desc=f"baseline_clusters {ind}",
            leave=False,
        ):
            corr = _spearman_corr_matrix(zbaseline_res[split], cluster_act_mat)
            best_match = np.argmax(corr, axis=1)
            max_corr[split] = corr[np.arange(len(split)), best_match]
            cell_cluster[split] = best_match

        cell_cluster[max_corr < assignment_threshold] = -1
        cluster_act_mat_refined = []
        for cluster_index in range(cluster_act_mat.shape[0]):
            member_mask = cell_cluster == cluster_index
            if member_mask.any():
                cluster_act_mat_refined.append(zbaseline_res[member_mask].mean(axis=0))
            else:
                cluster_act_mat_refined.append(
                    np.full(zbaseline_res.shape[1], np.nan, dtype=float)
                )

        np.savez(
            output_path,
            invalid_=invalid,
            ev_thres=ev_thres,
            cell_cluster=cell_cluster,
            cluster_act_mat=np.asarray(cluster_act_mat_refined, dtype="float16"),
        )


def export_baseline_stats(
    datalist_name: str = "datalist.csv",
    *,
    oxygen_file: str = "O2_internal.npz",
    max_index: int | None = None,
) -> None:
    df = load_baseline_datalist(datalist_name)
    oxy_mean = load_oxygen_mean(oxygen_file)

    for ind, row in _iter_selected_rows(df, max_index=max_index):
        save_root = _save_root(row)
        output_path = save_root / "baseline_stats.npz"
        cell_dff_path = save_root / "cell_dff.npz"
        clusters_path = save_root / "baseline_clusters.npz"
        locs_cam_path = save_root / "locs_cam.npy"

        if output_path.exists() or not cell_dff_path.exists():
            continue
        if not clusters_path.exists() or not locs_cam_path.exists():
            continue

        print(ind)
        with np.load(cell_dff_path, allow_pickle=True) as cell_dff:
            baseline = cell_dff["baseline"].astype("float16")
        with np.load(clusters_path, allow_pickle=True) as cluster_data:
            invalid = cluster_data["invalid_"]
            ev_thres = cluster_data["ev_thres"]

        baseline = baseline[~invalid][ev_thres]
        time_stamp = np.load(locs_cam_path) / 6000
        baseline, time_stamp = _align_time_series(baseline, time_stamp)

        baseline_float = baseline.astype(float)
        baseline_std = baseline_float.std(axis=1)
        baseline_mean = baseline_float.mean(axis=1)

        idx_time = (time_stamp > 10 * 60) & (time_stamp < 60 * 60)
        oxy_interp = np.interp(time_stamp[idx_time], np.arange(oxy_mean.shape[0]) + 1, oxy_mean)

        r_cell = []
        p_cell = []
        for split in tqdm(
            _split_indices(baseline.shape[0]),
            desc=f"baseline_stats {ind}",
            leave=False,
        ):
            for trace in baseline_float[split][:, idx_time]:
                corr, p_value = spearmanr(trace, oxy_interp)
                r_cell.append(corr)
                p_cell.append(p_value)

        idx_hypo = (time_stamp > 35 * 60) & (time_stamp < 40 * 60)
        idx_norm = (time_stamp > 15 * 60) & (time_stamp < 20 * 60)
        hypo_baseline_mean = baseline_float[:, idx_hypo].mean(axis=1)
        norm_baseline_mean = baseline_float[:, idx_norm].mean(axis=1)
        hypo_baseline_std = baseline_float[:, idx_hypo].std(axis=1)
        norm_baseline_std = baseline_float[:, idx_norm].std(axis=1)

        np.savez(
            output_path,
            baseline_std=baseline_std,
            baseline_mean=baseline_mean,
            r_cell=np.asarray(r_cell),
            p_cell=np.asarray(p_cell),
            hypo_baseline_mean=hypo_baseline_mean,
            norm_baseline_mean=norm_baseline_mean,
            hypo_baseline_std=hypo_baseline_std,
            norm_baseline_std=norm_baseline_std,
        )


def export_baseline_subclusters(
    datalist_name: str = "datalist.csv",
    *,
    oxygen_file: str = "O2_internal.npz",
    start_index: int | None = 17,
    max_index: int | None = None,
    n_components: int = 10,
    loading_threshold: float = 0.1,
    min_seed_size: int = 100,
    assignment_threshold: float = 0.4,
) -> None:
    df = load_baseline_datalist(datalist_name)
    oxy_mean = load_oxygen_mean(oxygen_file)

    for _, row in _iter_selected_rows(
        df,
        start_index=start_index,
        max_index=max_index,
    ):
        save_root = _save_root(row)
        cell_dff_path = save_root / "cell_dff.npz"
        clusters_path = save_root / "baseline_clusters.npz"
        locs_cam_path = save_root / "locs_cam.npy"

        if not cell_dff_path.exists() or not clusters_path.exists() or not locs_cam_path.exists():
            continue

        with np.load(cell_dff_path, allow_pickle=True) as cell_dff:
            baseline = cell_dff["baseline"].astype("float16")
        with np.load(clusters_path, allow_pickle=True) as cluster_data:
            invalid = cluster_data["invalid_"]
            ev_thres = cluster_data["ev_thres"]
            cell_cluster = cluster_data["cell_cluster"]

        baseline = baseline[~invalid][ev_thres]
        zbaseline = zscore(baseline.astype(float), axis=-1)

        num_cluster = int(cell_cluster.max()) + 1
        cell_subcluster = np.full(zbaseline.shape[0], -1, dtype=int)
        num_subcluster = 0

        for cluster_index in tqdm(
            range(num_cluster),
            desc=f"baseline_subclusters {save_root.name}",
            leave=False,
        ):
            cluster_mask = cell_cluster == cluster_index
            num_cells = int(cluster_mask.sum())
            if num_cells == 0:
                continue

            tmp_dat = zbaseline[cluster_mask]
            component_count = min(n_components, tmp_dat.shape[0], tmp_dat.shape[1])
            if component_count == 0:
                continue

            fa = FactorAnalysis(n_components=component_count, rotation="varimax")
            fa.fit(tmp_dat.T)

            cluster_mat = []
            for component_index in range(component_count):
                neg_idx = fa.components_[component_index] < -loading_threshold
                if neg_idx.sum() > min_seed_size:
                    cluster_mat.append(tmp_dat[neg_idx].mean(axis=0))

                pos_idx = fa.components_[component_index] > loading_threshold
                if pos_idx.sum() > min_seed_size:
                    cluster_mat.append(tmp_dat[pos_idx].mean(axis=0))

            if not cluster_mat:
                continue

            cluster_mat = np.asarray(cluster_mat)
            max_corr = np.full(num_cells, -1.0)
            subcluster = np.full(num_cells, -1, dtype=int)
            for split in _split_indices(num_cells):
                corr = _spearman_corr_matrix(tmp_dat[split], cluster_mat)
                best_match = np.argmax(corr, axis=1)
                max_corr[split] = corr[np.arange(len(split)), best_match]
                subcluster[split] = best_match

            cell_cluster_local = subcluster.copy()
            if np.any(max_corr < assignment_threshold):
                cell_cluster_local[max_corr < assignment_threshold] = -1
                cell_cluster_local = cell_cluster_local + 1

            cell_subcluster[cluster_mask] = cell_cluster_local + num_subcluster
            num_subcluster = int(cell_cluster_local.max()) + 1 + num_subcluster

        if cell_subcluster.max() < 0:
            np.savez(
                save_root / "baseline_subclusters.npz",
                cell_subcluster=cell_subcluster,
                cell_subcluster_mat=np.empty((0, 0), dtype="float16"),
                p_oxy=np.array([]),
                r_oxy=np.array([]),
            )
            continue

        cell_subcluster_mat = []
        num_subcluster = int(cell_subcluster.max()) + 1
        for subcluster_index in range(num_subcluster):
            cell_subcluster_mat.append(
                zbaseline[cell_subcluster == subcluster_index].mean(axis=0)
            )
        cell_subcluster_mat = np.asarray(cell_subcluster_mat)

        locs_cam = np.load(locs_cam_path)
        time_values = locs_cam / 6000
        time_values = time_values - time_values[0]
        valid_time = time_values < (oxy_mean.shape[0] - 1)
        time_values = time_values[valid_time]
        oxy_interp = interp1d(np.arange(oxy_mean.shape[0]), oxy_mean)
        oxy_new = oxy_interp(time_values)

        max_length = min(cell_subcluster_mat.shape[1], oxy_new.shape[0], time_values.shape[0])
        time_values = time_values[:max_length]
        oxy_new = oxy_new[:max_length]
        time_mask = time_values > 300

        p_values = np.zeros(cell_subcluster_mat.shape[0])
        r_values = np.zeros(cell_subcluster_mat.shape[0])
        for subcluster_index in tqdm(
            range(cell_subcluster_mat.shape[0]),
            desc=f"baseline_subclusters_oxy {save_root.name}",
            leave=False,
        ):
            corr, p_value = spearmanr(
                oxy_new[time_mask],
                cell_subcluster_mat[subcluster_index, :max_length][time_mask],
            )
            p_values[subcluster_index] = p_value
            r_values[subcluster_index] = corr

        np.savez(
            save_root / "baseline_subclusters.npz",
            cell_subcluster=cell_subcluster,
            cell_subcluster_mat=cell_subcluster_mat,
            p_oxy=p_values,
            r_oxy=r_values,
        )


def export_oxygen_clusters(
    datalist_name: str = "datalist_gfap_gc6f.csv",
    *,
    row_index: int = 6,
    oxygen_file: str = "O2_internal.npz",
    baseline_mean_threshold: float = 20,
    assignment_threshold: float = 0.7,
    show: bool = True,
) -> None:
    _configure_plotting()
    df = load_baseline_datalist(datalist_name)
    row = df.iloc[row_index]
    save_root = _save_root(row)

    with np.load(save_root / "cell_dff.npz", allow_pickle=True) as cell_dff:
        baseline = cell_dff["baseline"].astype("float16")
    with np.load(save_root / "baseline_clusters.npz", allow_pickle=True) as cluster_data:
        invalid = cluster_data["invalid_"]
        ev_thres = cluster_data["ev_thres"]
    with np.load(save_root / "baseline_stats.npz", allow_pickle=True) as baseline_stats:
        baseline_std = baseline_stats["baseline_std"]
        baseline_mean = baseline_stats["baseline_mean"]
        r_cell = baseline_stats["r_cell"]
        p_cell = baseline_stats["p_cell"]

    baseline = baseline[~invalid][ev_thres]
    time_stamp = np.load(save_root / "locs_cam.npy") / 6000
    baseline, time_stamp = _align_time_series(baseline, time_stamp)

    oxy_mean = load_oxygen_mean(oxygen_file)
    baseline_std_mean = baseline_std / baseline_mean
    a_center = np.load(save_root / "cell_center.npy")[~invalid][ev_thres]
    brain_map = np.load(save_root / "Y_ave.npy")

    low_mean_mask = baseline_mean < baseline_mean_threshold
    z_loc = np.arange(brain_map.shape[0])
    fig, axes = plt.subplots(2, 5, figsize=(60, 15))
    axes = axes.flatten()
    for panel_index in range(10):
        z_idx = (z_loc >= panel_index * 3) & (z_loc < (panel_index + 1) * 3)
        axes[panel_index].imshow(brain_map[z_idx].max(0), cmap=plt.cm.gray)
        scatter_idx = (
            (a_center[~low_mean_mask, 0] >= panel_index * 3)
            & (a_center[~low_mean_mask, 0] < (panel_index + 1) * 3)
        )
        axes[panel_index].scatter(
            a_center[~low_mean_mask, 1][scatter_idx],
            a_center[~low_mean_mask, 2][scatter_idx],
            s=1,
            c=a_center[~low_mean_mask, 0][scatter_idx],
        )
        axes[panel_index].set_axis_off()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    idx_f = (
        (baseline_mean > baseline_mean_threshold)
        & (baseline_std > 2)
        & (baseline_std_mean > 0.05)
        & (p_cell < 0.001)
    )
    idx_time = (time_stamp > 10 * 60) & (time_stamp < 60 * 60)
    oxy_interp = np.interp(time_stamp[idx_time], np.arange(oxy_mean.shape[0]) + 1, oxy_mean)
    zbaseline = zscore(baseline[idx_f][:, idx_time].astype(float), axis=-1)

    r_idx = np.r_[-1, np.arange(-0.8, 0.9, 0.1), 1]
    ref = []
    for ref_index in range(len(r_idx) - 1):
        ref_mask = (r_cell[idx_f] > r_idx[ref_index]) & (r_cell[idx_f] < r_idx[ref_index + 1])
        if ref_mask.any():
            ref.append(zbaseline[ref_mask].mean(axis=0))
    ref = np.asarray(ref)

    if ref.size == 0 or zbaseline.size == 0:
        np.savez(save_root / "O2_clusters.npz", idx_F=idx_f, ref=ref, r_cell_=np.empty((0, 0)))
        return

    r_cell_matrix = []
    for split in tqdm(
        _split_indices(zbaseline.shape[0]),
        desc=f"O2_clusters {save_root.name}",
        leave=False,
    ):
        r_cell_matrix.append(_spearman_corr_matrix(zbaseline[split], ref))
    r_cell_matrix = np.concatenate(r_cell_matrix, axis=0)
    r_cell_max = r_cell_matrix.max(axis=1)
    r_cell_max_idx = np.argmax(r_cell_matrix, axis=1)

    np.savez(save_root / "O2_clusters.npz", idx_F=idx_f, ref=ref, r_cell_=r_cell_matrix)

    colors = plt.cm.rainbow(np.linspace(0, 1, ref.shape[0]))
    ref_fig, ref_ax = plt.subplots()
    for ref_index in range(ref.shape[0]):
        ref_ax.plot(ref[ref_index], color=colors[ref_index])
    ref_fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(ref_fig)

    a_center_filtered = a_center[idx_f]
    color_scale = r_cell_max_idx / max(1, r_cell_max_idx.max())

    fig, axes = plt.subplots(2, 5, figsize=(60, 15))
    axes = axes.flatten()
    for panel_index in range(10):
        z_idx = (z_loc >= panel_index * 3) & (z_loc < (panel_index + 1) * 3)
        axes[panel_index].imshow(brain_map[z_idx].max(0), cmap=plt.cm.gray)
        scatter_idx = (
            (a_center_filtered[:, 0] >= panel_index * 3)
            & (a_center_filtered[:, 0] < (panel_index + 1) * 3)
            & (r_cell_max > assignment_threshold)
        )
        axes[panel_index].scatter(
            a_center_filtered[scatter_idx, 1],
            a_center_filtered[scatter_idx, 2],
            s=1,
            c=color_scale[scatter_idx],
            vmax=1,
            vmin=0,
            cmap=plt.cm.rainbow,
        )
        axes[panel_index].set_axis_off()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)
