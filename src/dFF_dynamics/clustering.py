from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.stats import rankdata, spearmanr, zscore
from sklearn.decomposition import FactorAnalysis

from ..paths import load_datalist


DEFAULT_CLUSTER_DATALIST = "datalist.csv"
DEFAULT_CLUSTER_STAGES = ("r1", "r2", "r3", "r3_dynamics", "activity")


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float = 1, order: int = 5) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b_coeff, a_coeff = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b_coeff, a_coeff, data)


def factor_scores(y_values: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    from scipy.linalg import lstsq

    return lstsq(loadings, y_values)[0]


def factor_loadings(y_values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    from scipy.linalg import lstsq

    return lstsq(scores.T, y_values.T)[0].T


def smooth(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(values, kernel, "full")[kernel.shape[0] // 2 : -(kernel.shape[0] // 2)]


def gauss_kernel(sigma: int = 20) -> np.ndarray:
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(np.arange(-sigma * 3, sigma * 3 + 1) ** 2) / (2 * sigma**2)
    )
    return kernel / kernel.sum()


def loadings_to_labels(
    loadings: np.ndarray,
    *,
    min_cluster: int = 100,
    min_weight: float = 0.2,
    thres_large_cluster: float = 0.9,
) -> np.ndarray:
    if loadings.size == 0:
        return np.zeros(loadings.shape[0], dtype=int)

    loadings_pos = loadings.copy()
    loadings_neg = loadings.copy()
    loadings_pos[loadings < 0] = 0
    loadings_neg[loadings > 0] = 0
    working = np.concatenate([loadings_pos, loadings_neg], axis=1)
    working[np.abs(working) < min_weight] = 0
    working = working[:, (np.abs(working) > 0).sum(axis=0) > min_cluster]

    if working.shape[1] == 0:
        return np.zeros(loadings.shape[0], dtype=int)

    for _ in range(3):
        labels = np.argmax(np.abs(working), axis=1)
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid = np.zeros(working.shape[1], dtype=bool)
        for label, count in zip(unique_labels, counts):
            if count > min_cluster:
                valid[label] = True
        working = working[:, valid]
        if working.shape[1] == 0:
            return np.zeros(loadings.shape[0], dtype=int)

    labels = np.argmax(np.abs(working), axis=1)
    _, counts = np.unique(labels, return_counts=True)
    working = working[:, np.argsort(-counts)]
    labels = np.argmax(np.abs(working), axis=1)

    if labels.size and (labels == 0).mean() > thres_large_cluster and working.shape[1] > 1:
        label_max = labels.max()
        valid_secondary = ((np.abs(working[:, 1:]) > 0).sum(axis=1) > 0) & (labels == 0)
        if valid_secondary.mean() > 0.5:
            labels[valid_secondary] = (
                np.argmax(np.abs(working[valid_secondary, 1:]), axis=1) + label_max
            )
    return labels


def _iter_selected_rows(df, *, start_index: int | None = None, max_index: int | None = None):
    for ind, row in df.iterrows():
        if start_index is not None and ind < start_index:
            continue
        if max_index is not None and ind > max_index:
            continue
        yield ind, row


def _load_raw_dff(save_root: Path) -> np.ndarray:
    with np.load(save_root / "cell_dff.npz", allow_pickle=True) as cell_dff:
        return cell_dff["dFF"].astype("float16")


def _load_cell_centers(save_root: Path) -> np.ndarray:
    return np.load(save_root / "cell_center.npy")


def _compute_invalid_mask(dff: np.ndarray) -> np.ndarray:
    return (np.isnan(dff).sum(axis=1) > 0) | (np.isinf(dff).sum(axis=1) > 0)


def _load_clean_zdff(save_root: Path) -> tuple[np.ndarray, np.ndarray]:
    dff = _load_raw_dff(save_root)
    invalid = _compute_invalid_mask(dff)
    zdff = zscore(dff[~invalid].astype(float), axis=-1)
    return zdff, invalid


def _load_factor_analysis(path: Path) -> FactorAnalysis:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_factor_analysis(path: Path, fa: FactorAnalysis) -> None:
    with path.open("wb") as handle:
        pickle.dump(fa, handle)


def _fit_factor_analysis(zdff: np.ndarray, *, n_components: int = 100) -> FactorAnalysis:
    component_count = min(n_components, zdff.shape[0], zdff.shape[1])
    if component_count <= 0:
        raise ValueError("Cannot fit FactorAnalysis on an empty matrix.")
    fa = FactorAnalysis(n_components=component_count, rotation="varimax")
    fa.fit(zdff.T)
    return fa


def process_cluster_r1(save_root: str | Path, *, force: bool = False) -> bool:
    save_root = Path(save_root)
    output_path = save_root / "FA_R1_ind.npz"
    fa_path = save_root / "FA_R1.pkl"
    cell_dff_path = save_root / "cell_dff.npz"

    if not cell_dff_path.exists():
        return False
    if output_path.exists() and not force:
        return False

    zdff, _invalid = _load_clean_zdff(save_root)
    if zdff.shape[0] == 0:
        np.savez(output_path, ind__=np.array([], dtype=np.int16), noise_ratio_=np.array([]))
        return True

    if force or not fa_path.exists():
        fa = _fit_factor_analysis(zdff)
        _save_factor_analysis(fa_path, fa)
    else:
        fa = _load_factor_analysis(fa_path)

    loadings = fa.components_.T
    scores = factor_scores(zdff, loadings)
    filtered_scores = np.array(
        [butter_lowpass_filter(trace, cutoff=0.03, fs=1, order=5) for trace in scores]
    )
    denominator = np.maximum((scores**2).sum(axis=1), np.finfo(float).eps)
    noise_ratio = ((scores - filtered_scores) ** 2).sum(axis=1) / denominator

    retained = loadings[:, noise_ratio < 0.3]
    if retained.size == 0:
        labels = np.zeros(loadings.shape[0], dtype=np.int16)
        np.savez(output_path, ind__=labels, noise_ratio_=noise_ratio.astype(np.float32))
        return True

    retained_masked = retained.copy()
    retained_masked[np.abs(retained_masked) < 0.2] = 0
    retained = retained[:, (np.abs(retained_masked) > 0).sum(axis=0) > 0]
    if retained.size == 0:
        labels = np.zeros(loadings.shape[0], dtype=np.int16)
        np.savez(output_path, ind__=labels, noise_ratio_=noise_ratio.astype(np.float32))
        return True

    component_index = np.argmax(np.abs(retained), axis=1)
    sign = retained[np.arange(len(component_index)), component_index] > 0
    labels = component_index * (-1 + 2 * sign.astype(int))
    labels[~sign] = labels[~sign] - 1
    np.savez(output_path, ind__=labels.astype(np.int16), noise_ratio_=noise_ratio.astype(np.float32))
    return True


def process_cluster_r2(save_root: str | Path, *, force: bool = False) -> bool:
    save_root = Path(save_root)
    output_path = save_root / "FA_R2_ind.npz"
    fa_path = save_root / "FA_R1.pkl"
    r1_path = save_root / "FA_R1_ind.npz"
    cell_dff_path = save_root / "cell_dff.npz"

    if not cell_dff_path.exists() or not fa_path.exists() or not r1_path.exists():
        return False
    if output_path.exists() and not force:
        return False

    zdff, _invalid = _load_clean_zdff(save_root)
    num_cells = zdff.shape[0]
    fa = _load_factor_analysis(fa_path)
    with np.load(r1_path, allow_pickle=True) as r1_data:
        noise_ratio = r1_data["noise_ratio_"]

    loadings = fa.components_.T[:, noise_ratio < 0.3]
    cluster_id = loadings_to_labels(loadings, min_cluster=100, min_weight=0.07, thres_large_cluster=0.9)

    max_cluster_size = 0.05
    lind, counts = np.unique(cluster_id, return_counts=True)
    max_cluster = len(counts) // 2 + 5
    valid_cluster = counts / max(num_cells, 1) < max_cluster_size
    run_loop = (~valid_cluster).sum() > 0

    res_fa: list[FactorAnalysis] = []
    res_noise_ratio: list[np.ndarray] = []
    res_layer: list[int] = []
    res_fa_ind: list[int] = []

    counts_previous = counts.copy()
    layer = 0

    while run_loop:
        layer += 1
        for cluster_index in lind[~valid_cluster]:
            idx = cluster_id == cluster_index
            num_cluster = min(max_cluster, max(1, idx.sum() // 1000))
            fa_sub = FactorAnalysis(n_components=num_cluster, rotation="varimax")
            zdff_sub = zdff[idx]
            fa_sub.fit(zdff_sub.T)
            sub_loadings = fa_sub.components_.T
            sub_scores = factor_scores(zdff_sub, sub_loadings)
            filtered_scores = np.array(
                [butter_lowpass_filter(trace, cutoff=0.03, fs=1, order=5) for trace in sub_scores]
            )
            denominator = np.maximum((sub_scores**2).sum(axis=1), np.finfo(float).eps)
            sub_noise_ratio = ((sub_scores - filtered_scores) ** 2).sum(axis=1) / denominator
            res_fa.append(fa_sub)
            res_noise_ratio.append(sub_noise_ratio)
            res_layer.append(layer)
            res_fa_ind.append(cluster_index)

        current_indices = np.where(np.asarray(res_layer) == layer)[0]
        cluster_max = cluster_id.max()
        cluster_max_layer = cluster_id.max()

        for result_index in current_indices:
            idx = cluster_id == res_fa_ind[result_index]
            sub_loadings = res_fa[result_index].components_.T
            sub_noise_ratio = res_noise_ratio[result_index]
            sub_loadings = sub_loadings[:, sub_noise_ratio < 0.3]
            sub_labels = loadings_to_labels(
                sub_loadings,
                min_cluster=100,
                min_weight=0.07,
                thres_large_cluster=0.9,
            )
            if (sub_labels > 0).sum() > 0:
                original_indices = np.where(idx)[0]
                cluster_id[original_indices[sub_labels > 0]] = sub_labels[sub_labels > 0] + cluster_max
                cluster_max = cluster_id.max()

        lind, counts = np.unique(cluster_id, return_counts=True)
        max_cluster = max((cluster_max - cluster_max_layer) // max(len(current_indices), 1) + 5, 20)
        valid_cluster = counts / max(num_cells, 1) < max_cluster_size
        non_factored = np.where(counts[: len(counts_previous)] / counts_previous > 0.9)[0]
        valid_cluster[non_factored] = True
        run_loop = (~valid_cluster).sum() > 0
        counts_previous = counts.copy()

    np.savez(
        output_path,
        cluster_id=cluster_id.astype(np.int16),
        res_nFA=np.asarray(res_fa, dtype=object),
        res_n_noise_ratio_=np.asarray(res_noise_ratio, dtype=object),
        res_nLayer=np.asarray(res_layer, dtype=np.int16),
        res_nFA_ind=np.asarray(res_fa_ind, dtype=np.int16),
    )
    return True


def _spearman_to_templates(traces: np.ndarray, templates: np.ndarray) -> np.ndarray:
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    templates = np.atleast_2d(np.asarray(templates, dtype=float))
    trace_ranks = np.apply_along_axis(rankdata, 1, traces)
    template_ranks = np.apply_along_axis(rankdata, 1, templates)
    trace_ranks -= trace_ranks.mean(axis=1, keepdims=True)
    template_ranks -= template_ranks.mean(axis=1, keepdims=True)
    denominator = (
        np.linalg.norm(trace_ranks, axis=1, keepdims=True)
        @ np.linalg.norm(template_ranks, axis=1, keepdims=True).T
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = trace_ranks @ template_ranks.T / denominator
    return np.nan_to_num(corr)


def process_cluster_r3(save_root: str | Path, *, force: bool = False) -> bool:
    save_root = Path(save_root)
    output_path = save_root / "FA_R3_ind.npz"
    r2_path = save_root / "FA_R2_ind.npz"
    cell_dff_path = save_root / "cell_dff.npz"

    if not cell_dff_path.exists() or not r2_path.exists():
        return False
    if output_path.exists() and not force:
        return False

    zdff, invalid = _load_clean_zdff(save_root)
    with np.load(r2_path, allow_pickle=True) as r2_data:
        cluster_id = r2_data["cluster_id"]

    lind, counts = np.unique(cluster_id, return_counts=True)
    lind = lind[np.argsort(-counts)]
    sub_cluster = []

    for cluster_label in lind:
        idx = cluster_id == cluster_label
        zdff_sub = zdff[idx]
        num_cells = zdff_sub.shape[0]
        if num_cells < 200:
            sub_cluster.append(np.zeros(num_cells, dtype=np.int16))
            continue

        step = num_cells // 9000 + 1
        corr_matrix = spearmanr(zdff_sub[::step], axis=1)[0]
        linkage = sch.linkage(corr_matrix, method="ward")
        k_clusters = min(10, ((zdff_sub[::step].shape[0] // 1000) + 1) * 2)
        labels = sch.fcluster(linkage, k_clusters, criterion="maxclust")

        if step > 1:
            templates = np.array(
                [zdff_sub[::step][labels == label].mean(axis=0) for label in range(1, k_clusters + 1)]
            )
            assigned = np.zeros(num_cells, dtype=np.int16)
            for split in np.array_split(np.arange(num_cells, dtype=int), num_cells // 1000 + 1):
                corr = _spearman_to_templates(zdff_sub[split], templates)
                assigned[split] = np.argmax(corr, axis=1).astype(np.int16)
            sub_cluster.append(assigned)
        else:
            sub_cluster.append((labels - 1).astype(np.int16))

    num_cells = zdff.shape[0]
    clusters_ind = np.zeros(num_cells, dtype=np.int16) - 1
    subclusters_ind = np.zeros(num_cells, dtype=np.int16) - 1

    subcluster_max = 0
    for cluster_index, cluster_label in enumerate(lind):
        idx = np.where(cluster_id == cluster_label)[0]
        clusters_ind[idx] = cluster_index
        subclusters_ind[idx] = sub_cluster[cluster_index] + subcluster_max
        subcluster_max = subclusters_ind.max() + 1

    np.savez(
        output_path,
        invalid_=invalid,
        cluster_id_R2=cluster_id.astype(np.int16),
        lind_=lind.astype(np.int16),
        sub_cluster=np.asarray(sub_cluster, dtype=object),
        clusters_ind=clusters_ind.astype(np.int16),
        subclusters_ind=subclusters_ind.astype(np.int16),
    )
    return True


def _cluster_activity_from_labels(
    dff: np.ndarray,
    labels: np.ndarray,
    *,
    min_cells: int = 50,
    cell_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    cluster_act = []
    cluster_size = []
    cell_locs = []
    num_cluster = int(labels.max()) + 1 if labels.size else 0
    for cluster_index in range(num_cluster):
        idx = labels == cluster_index
        if idx.sum() < min_cells:
            continue
        zdff_sub = zscore(dff[idx].astype(float), axis=1)
        cluster_act.append(zdff_sub.mean(axis=0))
        cluster_size.append(idx.sum())
        if cell_centers is not None:
            cell_locs.append(cell_centers[idx])

    cluster_act_arr = np.asarray(cluster_act)
    cluster_size_arr = np.asarray(cluster_size)
    if cell_centers is None:
        return cluster_act_arr, cluster_size_arr, None
    return cluster_act_arr, cluster_size_arr, np.asarray(cell_locs, dtype=object)


def process_cluster_r3_dynamics(save_root: str | Path, *, force: bool = False) -> bool:
    save_root = Path(save_root)
    output_path = save_root / "FA_R3_dynamics.npz"
    r3_path = save_root / "FA_R3_ind.npz"
    cell_dff_path = save_root / "cell_dff.npz"

    if not cell_dff_path.exists() or not r3_path.exists():
        return False
    if output_path.exists() and not force:
        return False

    dff = _load_raw_dff(save_root)
    centers = _load_cell_centers(save_root)
    with np.load(r3_path, allow_pickle=True) as r3_data:
        invalid = r3_data["invalid_"]
        subclusters_ind = r3_data["subclusters_ind"]

    dff = dff[~invalid]
    centers = centers[~invalid]
    cluster_act, num_cells, cell_locs = _cluster_activity_from_labels(
        dff,
        subclusters_ind,
        min_cells=50,
        cell_centers=centers,
    )
    np.savez(output_path, cluster_act=cluster_act, num_cells=num_cells, cell_locs=cell_locs)
    return True


def process_cluster_activity(save_root: str | Path, *, force: bool = False) -> bool:
    save_root = Path(save_root)
    cluster_output = save_root / "FA_R3_ind_cluster_dat.npz"
    subcluster_output = save_root / "FA_R3_ind_subcluster_dat.npz"
    r3_path = save_root / "FA_R3_ind.npz"
    cell_dff_path = save_root / "cell_dff.npz"

    if not cell_dff_path.exists() or not r3_path.exists():
        return False
    if cluster_output.exists() and subcluster_output.exists() and not force:
        return False

    dff = _load_raw_dff(save_root)
    with np.load(r3_path, allow_pickle=True) as r3_data:
        invalid = r3_data["invalid_"]
        clusters_ind = r3_data["clusters_ind"]
        subclusters_ind = r3_data["subclusters_ind"]

    dff = dff[~invalid]
    cluster_act, cluster_size, _ = _cluster_activity_from_labels(dff, clusters_ind, min_cells=50)
    np.savez(cluster_output, cluster_act=cluster_act, cluster_size=cluster_size)

    subcluster_act, subcluster_size, _ = _cluster_activity_from_labels(
        dff,
        subclusters_ind,
        min_cells=50,
    )
    np.savez(subcluster_output, cluster_act=subcluster_act, cluster_size=subcluster_size)
    return True


def process_cluster_pipeline(
    save_root: str | Path,
    *,
    stages: tuple[str, ...] = DEFAULT_CLUSTER_STAGES,
    force: bool = False,
) -> dict[str, bool]:
    results = {}
    if "r1" in stages:
        results["r1"] = process_cluster_r1(save_root, force=force)
    if "r2" in stages:
        results["r2"] = process_cluster_r2(save_root, force=force)
    if "r3" in stages:
        results["r3"] = process_cluster_r3(save_root, force=force)
    if "r3_dynamics" in stages:
        results["r3_dynamics"] = process_cluster_r3_dynamics(save_root, force=force)
    if "activity" in stages:
        results["activity"] = process_cluster_activity(save_root, force=force)
    return results


def export_dff_cluster_pipeline(
    datalist_name: str = DEFAULT_CLUSTER_DATALIST,
    *,
    start_index: int | None = None,
    max_index: int | None = None,
    stages: tuple[str, ...] = DEFAULT_CLUSTER_STAGES,
    force: bool = False,
) -> None:
    df = load_datalist(datalist_name)
    for ind, row in _iter_selected_rows(df, start_index=start_index, max_index=max_index):
        save_root = Path(row["save_root"])
        print(f"Processing {ind} at {save_root}")
        process_cluster_pipeline(save_root, stages=stages, force=force)


__all__ = [
    "DEFAULT_CLUSTER_DATALIST",
    "DEFAULT_CLUSTER_STAGES",
    "butter_lowpass_filter",
    "export_dff_cluster_pipeline",
    "factor_loadings",
    "factor_scores",
    "gauss_kernel",
    "loadings_to_labels",
    "process_cluster_activity",
    "process_cluster_pipeline",
    "process_cluster_r1",
    "process_cluster_r2",
    "process_cluster_r3",
    "process_cluster_r3_dynamics",
    "smooth",
]
