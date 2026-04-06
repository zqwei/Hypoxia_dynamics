from __future__ import annotations

from pathlib import Path
import shutil

import dask.array as da
from h5py import File
import numpy as np

from ..paths import data_file, ensure_directory, load_datalist
from .preprocess import baseline, cell_loc


def copy_registered_cell_centers(
    datalist_name: str = "datalist_huc_ablation.csv",
) -> None:
    df = load_datalist(datalist_name)

    for _, row in df.iterrows():
        ephys_dir = Path(str(row["dir_"])) / "proc_data"
        save_root = ensure_directory(row["save_root"])
        source = ephys_dir / "cell_center_affine_registered.npy"
        target = save_root / "cell_center_affine_registered.npy"

        if not source.exists():
            print(f"Missing cell center file at {source}")
            continue

        shutil.copyfile(source, target)


def extract_locs_cam(
    datalist_name: str = "datalist_gfap_gc6f_v2.csv",
    max_index: int | None = 10,
) -> None:
    df = load_datalist(datalist_name)

    for ind, row in df.iterrows():
        if max_index is not None and ind > max_index:
            continue

        raw_dir = Path(str(row["dir_"]))
        analysis_dir = raw_dir / "ephys" / "analysis"
        save_root = ensure_directory(row["save_root"])
        locs_cam_path = save_root / "locs_cam.npy"
        data_mat_path = save_root / "data.mat"

        if locs_cam_path.exists():
            continue

        source_data_mat = analysis_dir / "data.mat"
        if source_data_mat.exists():
            shutil.copyfile(source_data_mat, data_mat_path)
        else:
            print(f"Missing data file at {save_root}")

        source_locs_cam = analysis_dir / "locs_cam.mat"
        if source_locs_cam.exists():
            with File(source_locs_cam, "r") as handle:
                locs_cam = handle["locs_cam"][()].squeeze()
        else:
            source_x3 = analysis_dir / "x3.mat"
            if not source_x3.exists():
                print(f"Missing locs_cam.mat and x3.mat at {analysis_dir}")
                continue

            with File(source_x3, "r") as handle:
                x3 = handle["x3"][()].squeeze()
            locs_cam = np.where((x3[:-1] < 3.8) & (x3[1:] > 3.8))[0] + 1

            cell_dff_path = raw_dir / "proc_data" / "cell_dff.npz"
            if cell_dff_path.exists():
                shape_ = np.load(cell_dff_path, allow_pickle=True)["dFF"].shape[1]
                print(locs_cam.shape, shape_)

        np.save(locs_cam_path, locs_cam.astype(int))


def export_swim_ds(
    datalist_name: str = "datalist_gfap_gc6f_v2.csv",
    max_index: int | None = 10,
) -> None:
    df = load_datalist(datalist_name)

    for ind, row in df.iterrows():
        if max_index is not None and ind > max_index:
            continue

        save_root = ensure_directory(row["save_root"])
        swim_ds_path = save_root / "swim_ds.npy"
        data_mat_path = save_root / "data.mat"
        locs_cam_path = save_root / "locs_cam.npy"

        if swim_ds_path.exists():
            print(swim_ds_path)
            continue

        if not data_mat_path.exists():
            print(f"Missing data file at {save_root}")
            continue

        if not locs_cam_path.exists():
            print(f"Missing locs_cam file at {save_root}")
            continue

        locs_cam = np.load(locs_cam_path)
        len_cam = int(np.unique(np.diff(locs_cam)).min())

        with File(data_mat_path, "r") as handle:
            ephys_data = handle["data"]
            flt_ch1 = ephys_data["fltCh2"][()].squeeze()
            back1 = ephys_data["back2"][()].squeeze()

        swim_ = flt_ch1 - back1
        swim_[swim_ < 0] = 0

        swim_ds = np.zeros(len(locs_cam))
        for n_ in range(len_cam):
            swim_ds = swim_ds + swim_[locs_cam - n_]

        np.save(swim_ds_path, swim_ds)


def export_segmented_data(
    datalist_name: str = "datalist.csv",
    start_index: int | None = 38,
) -> None:
    df = load_datalist(datalist_name)

    for ind, row in df.iterrows():
        if start_index is not None and ind < start_index:
            continue

        seg_dir = Path(str(row["dir_"])) / "seg_mika"
        if not seg_dir.exists():
            continue

        save_root = ensure_directory(row["save_root"])
        cell_center_path = save_root / "cell_center.npy"
        if cell_center_path.exists():
            continue

        print(f"Processing {ind} at {save_root}")

        with File(seg_dir / "cells0_clean.hdf5", "r") as cell_file, File(
            seg_dir / "volume0.hdf5", "r"
        ) as volume_file:
            x_coord = cell_file["cell_x"]
            y_coord = cell_file["cell_y"]
            z_coord = cell_file["cell_z"]
            weights = cell_file["cell_weights"][()]
            volume_weight = cell_file["volume_weight"]

            fluorescence = cell_file["cell_timeseries_raw"][()]
            background = cell_file["background"][()]
            brain_map = volume_file["volume_mean"][()]

            fluorescence = fluorescence - (background - 10)
            fluorescence[fluorescence < 0] = 0
            fluorescence_dask = da.from_array(fluorescence, chunks=("auto", -1))
            baseline_ = da.map_blocks(
                baseline,
                fluorescence_dask,
                dtype="float",
                window=400,
                percentile=20,
                downsample=10,
            ).compute()
            dff = fluorescence / baseline_ - 1

            np.savez(
                save_root / "cell_dff.npz",
                dFF=dff.astype("float16"),
                baseline=baseline_.astype("float16"),
                brain_shape=volume_weight.shape,
                X=x_coord,
                Y=y_coord,
                Z=z_coord,
                W=weights,
            )
            np.save(save_root / "Y_ave.npy", brain_map)

            num_cells = fluorescence.shape[0]
            centers = np.zeros((num_cells, 3))
            for cell_id in range(num_cells):
                centers[cell_id] = cell_loc(
                    x_coord, y_coord, z_coord, weights, cell_id
                )

        np.save(cell_center_path, centers)
