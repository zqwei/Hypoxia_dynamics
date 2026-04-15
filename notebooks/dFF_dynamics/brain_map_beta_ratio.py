import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


DEFAULT_MAX_INDEX = {
    "neuron": 8,
    "glia": 5,
}
DEFAULT_DATALIST = {
    "neuron": "datalist_huc_h2b_gc7f.csv",
    "glia": "datalist_gfap_gc6f.csv",
}
DEFAULT_ATLAS = {
    "neuron": "/nrs/ahrens/Ziqiang/Atlas/atlas.npy",
    "glia": "/nrs/ahrens/Ziqiang/Motor_clamp/Brain_maps/Glia_Brain_Fused_sliced_DS.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build across-fish brain maps for hypoxia/normoxia beta ratio."
    )
    parser.add_argument(
        "--dataset",
        choices=("neuron", "glia", "all"),
        default="all",
        help="Dataset cohort to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Smallest dataframe index to process within each selected cohort.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        help="Largest dataframe index to process within each selected cohort.",
    )
    parser.add_argument(
        "--atlas-path",
        type=str,
        default=None,
        help="Override atlas path.",
    )
    parser.add_argument(
        "--brain-map-folder",
        type=str,
        default=None,
        help="Folder for output npz maps.",
    )
    parser.add_argument(
        "--ratio-mode",
        choices=("raw", "log2"),
        default="raw",
        help="Map raw beta ratio or log2 ratio.",
    )
    parser.add_argument(
        "--min-r2-threshold",
        type=float,
        default=None,
        help="Optional minimum min(normoxia_r2, hypoxia_r2) filter per cell.",
    )
    parser.add_argument(
        "--radius-z",
        type=int,
        default=5,
        help="Half-width of smoothing box in z.",
    )
    parser.add_argument(
        "--radius-y",
        type=int,
        default=5,
        help="Half-width of smoothing box in y.",
    )
    parser.add_argument(
        "--radius-x",
        type=int,
        default=5,
        help="Half-width of smoothing box in x.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=10,
        help="Minimum number of overlapping cells required per voxel.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from src.dFF_dynamics import (
        DEFAULT_ATLAS_PATH,
        DEFAULT_BRAIN_MAP_FOLDER,
        export_beta_ratio_brain_map,
    )

    datasets = ["neuron", "glia"] if args.dataset == "all" else [args.dataset]
    brain_map_folder = (
        DEFAULT_BRAIN_MAP_FOLDER
        if args.brain_map_folder is None
        else Path(args.brain_map_folder)
    )

    for dataset in datasets:
        atlas_path = (
            Path(DEFAULT_ATLAS[dataset])
            if args.atlas_path is None
            else Path(args.atlas_path)
        )
        max_index = DEFAULT_MAX_INDEX[dataset] if args.max_index is None else args.max_index
        output_name = f"brain_map_beta_ratio_{dataset}.npz"
        output_path = brain_map_folder / output_name
        print(f"build {dataset} -> {output_path}")
        export_beta_ratio_brain_map(
            DEFAULT_DATALIST[dataset],
            output_path=output_path,
            atlas_path=atlas_path,
            start_index=args.start_index,
            max_index=max_index,
            ratio_mode=args.ratio_mode,
            min_r2_threshold=args.min_r2_threshold,
            radius_z=args.radius_z,
            radius_y=args.radius_y,
            radius_x=args.radius_x,
            min_support=args.min_support,
        )
        print(f"saved {output_path}")
