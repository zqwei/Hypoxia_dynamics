import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dF/F clustering pipeline stages (R1, R2, R3, dynamics, activity)."
    )
    parser.add_argument(
        "datalist",
        nargs="?",
        default="datalist.csv",
        help="CSV file name under notebooks/data or data.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Smallest dataframe index to process.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        help="Largest dataframe index to process.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", "r1", "r2", "r3", "r3_dynamics", "activity"],
        default=["all"],
        help="Pipeline stages to run. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if target files already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from src.dFF_dynamics import DEFAULT_CLUSTER_STAGES, export_dff_cluster_pipeline

    stages = DEFAULT_CLUSTER_STAGES if "all" in args.stages else tuple(args.stages)
    export_dff_cluster_pipeline(
        args.datalist,
        start_index=args.start_index,
        max_index=args.max_index,
        stages=stages,
        force=args.force,
    )
