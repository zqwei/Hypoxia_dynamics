from __future__ import annotations

from pathlib import Path

import numpy as np

from src.paths import ensure_directory, load_datalist


DEFAULT_ATLAS_PATH = Path("/nrs/ahrens/Ziqiang/Atlas/atlas.npy")
DEFAULT_BRAIN_MAP_FOLDER = Path("/nrs/ahrens/Ziqiang/Motor_clamp/Brain_maps/dFF_dynamics")


def _is_valid_registration(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return False
    return str(value) != "None"


def collect_beta_ratio_cells(
    datalist_name: str,
    start_index: int = 0,
    max_index: int | None = None,
    ratio_mode: str = "raw",
    min_r2_threshold: float | None = None,
    sign_filter: str = "all",
) -> tuple[np.ndarray, int]:
    df = load_datalist(datalist_name)
    if max_index is None:
        indices = [ind for ind in df.index if ind >= start_index]
    else:
        indices = [ind for ind in df.index if start_index <= ind <= max_index]

    rows = []
    num_animal = 0
    for ind in indices:
        row = df.loc[ind]
        if not _is_valid_registration(row["registration_root"]):
            continue

        save_root = Path(row["save_root"])
        center_path = save_root / "cell_center_affine_registered.npy"
        fit_path = save_root / "GLM_calcium_swim_fit.npz"
        if not center_path.exists() or not fit_path.exists():
            continue

        centers = np.load(center_path)
        with np.load(fit_path, allow_pickle=True) as fit_data:
            invalid_ = fit_data["invalid_"]
            ev_thres = fit_data["ev_thres"]
            idx_f = fit_data["idx_F"]
            norm_beta = fit_data["normoxia_beta"].astype(np.float32)
            hyp_beta = fit_data["hypoxia_beta"].astype(np.float32)
            norm_r2 = fit_data["normoxia_r2"].astype(np.float32)
            hyp_r2 = fit_data["hypoxia_r2"].astype(np.float32)
            cell_skipped = fit_data["cell_skipped"]

        selected_centers = centers[~invalid_][ev_thres][idx_f]
        if selected_centers.shape[0] != len(norm_beta):
            raise ValueError(
                f"Selected cell centers do not align with GLM output at {save_root}"
            )

        valid = (
            ~cell_skipped
            & np.isfinite(norm_beta)
            & np.isfinite(hyp_beta)
            & (norm_beta > 0)
        )
        if min_r2_threshold is not None:
            valid = (
                valid
                & np.isfinite(norm_r2)
                & np.isfinite(hyp_r2)
                & (np.minimum(norm_r2, hyp_r2) >= min_r2_threshold)
            )
        if not np.any(valid):
            continue

        ratio = hyp_beta[valid] / norm_beta[valid]
        if ratio_mode == "log2":
            ratio = np.log2(ratio)
        elif ratio_mode != "raw":
            raise ValueError(f"Unsupported ratio_mode={ratio_mode!r}")

        if sign_filter == "positive":
            sign_mask = ratio > 0
        elif sign_filter == "negative":
            sign_mask = ratio < 0
        elif sign_filter == "all":
            sign_mask = np.ones_like(ratio, dtype=bool)
        else:
            raise ValueError(f"Unsupported sign_filter={sign_filter!r}")

        if not np.any(sign_mask):
            continue

        centers_valid = selected_centers[valid][sign_mask]
        ratio = ratio[sign_mask]

        rows.append(
            np.hstack(
                [
                    centers_valid.astype(np.float32),
                    np.full((len(ratio), 1), num_animal, dtype=np.float32),
                    ratio[:, None].astype(np.float32),
                ]
            )
        )
        num_animal += 1

    if not rows:
        return np.empty((0, 5), dtype=np.float32), 0
    return np.vstack(rows), num_animal


def build_beta_ratio_brain_map(
    cell_rows: np.ndarray,
    atlas: np.ndarray,
    radius_z: int = 5,
    radius_y: int = 5,
    radius_x: int = 5,
    min_support: int = 10,
) -> dict[str, np.ndarray]:
    atlas_shape = atlas.shape
    value_sum = np.zeros(atlas_shape, dtype=np.float32)
    value_count = np.zeros(atlas_shape, dtype=np.uint16)

    if cell_rows.size == 0:
        value_map = np.full(atlas_shape, np.nan, dtype=np.float32)
        return {"result_": value_map, "count_": value_count}

    zyx = np.round(cell_rows[:, :3]).astype(int)
    values = cell_rows[:, 4].astype(np.float32)

    for (z, x, y), value in zip(zyx, values):
        if (
            z <= 0
            or z >= atlas_shape[0] - 1
            or y <= 0
            or y >= atlas_shape[1] - 1
            or x <= 0
            or x >= atlas_shape[2] - 1
        ):
            continue

        z0 = max(0, z - radius_z)
        z1 = min(atlas_shape[0], z + radius_z)
        y0 = max(0, y - radius_y)
        y1 = min(atlas_shape[1], y + radius_y)
        x0 = max(0, x - radius_x)
        x1 = min(atlas_shape[2], x + radius_x)
        value_sum[z0:z1, y0:y1, x0:x1] += value
        value_count[z0:z1, y0:y1, x0:x1] += 1

    value_map = np.full(atlas_shape, np.nan, dtype=np.float32)
    valid = value_count >= min_support
    value_map[valid] = value_sum[valid] / value_count[valid]
    return {"result_": value_map, "count_": value_count}


def export_beta_ratio_brain_map(
    datalist_name: str,
    output_path: str | Path,
    atlas_path: str | Path = DEFAULT_ATLAS_PATH,
    start_index: int = 0,
    max_index: int | None = None,
    ratio_mode: str = "raw",
    min_r2_threshold: float | None = None,
    sign_filter: str = "all",
    radius_z: int = 5,
    radius_y: int = 5,
    radius_x: int = 5,
    min_support: int = 10,
) -> Path:
    atlas = np.load(atlas_path)
    cell_rows, num_animal = collect_beta_ratio_cells(
        datalist_name,
        start_index=start_index,
        max_index=max_index,
        ratio_mode=ratio_mode,
        min_r2_threshold=min_r2_threshold,
        sign_filter=sign_filter,
    )
    result = build_beta_ratio_brain_map(
        cell_rows,
        atlas=atlas,
        radius_z=radius_z,
        radius_y=radius_y,
        radius_x=radius_x,
        min_support=min_support,
    )

    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    np.savez_compressed(
        output_path,
        result_=result["result_"],
        count_=result["count_"],
        cell_rows=cell_rows,
        num_animal=np.asarray(num_animal, dtype=np.int16),
        ratio_mode=np.asarray(ratio_mode),
        sign_filter=np.asarray(sign_filter),
        min_r2_threshold=np.asarray(
            np.nan if min_r2_threshold is None else min_r2_threshold,
            dtype=np.float32,
        ),
        min_support=np.asarray(min_support, dtype=np.int16),
        radius=np.asarray([radius_z, radius_y, radius_x], dtype=np.int16),
    )
    return output_path
