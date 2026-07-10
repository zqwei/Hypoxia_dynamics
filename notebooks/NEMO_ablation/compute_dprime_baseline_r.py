import argparse
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


DEFAULT_DATA_ROOT = Path("/nrs/ahrens/Ziqiang/Motor_clamp/NEMO_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pre/post-ablation d-prime, dFF and baseline differences, "
            "normoxia dFF changes, and Spearman correlation against oxygen."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Ablation dataset root (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--dataset",
        choices=("neuron", "glia", "all"),
        default="all",
        help="Cohort to process (default: all).",
    )
    parser.add_argument(
        "--oxygen-file",
        default="O2_internal.npz",
        help="Oxygen reference under notebooks/data or data.",
    )
    parser.add_argument(
        "--all-valid-cells",
        action="store_true",
        help="Use all non-invalid cells instead of also requiring idx_F.",
    )
    parser.add_argument(
        "--max-fish",
        type=int,
        default=None,
        help="Process only the first N recordings, useful for smoke tests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing comparison output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files, shapes, and windows without computing or writing outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from src.nemo_ablation import AblationAnalysisConfig, analyze_ablation_dataset
    from src.paths import data_file

    with np.load(data_file(args.oxygen_file), allow_pickle=False) as oxygen_data:
        oxygen_mean = np.asarray(oxygen_data["oxy_mean"], dtype=float)

    config = AblationAnalysisConfig(
        use_idx_f=not args.all_valid_cells,
    )
    analyze_ablation_dataset(
        args.data_root,
        oxygen_mean,
        dataset=args.dataset,
        config=config,
        force=args.force,
        dry_run=args.dry_run,
        max_fish=args.max_fish,
    )
