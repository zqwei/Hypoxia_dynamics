import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build oxygen-correlated reference traces and plot baseline brain maps."
    )
    parser.add_argument(
        "datalist",
        nargs="?",
        default="datalist_gfap_gc6f.csv",
        help="CSV file name under notebooks/data or data.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=6,
        help="Row index within the datalist to visualize.",
    )
    parser.add_argument(
        "--oxygen-file",
        default="O2_internal.npz",
        help="Oxygen reference file under notebooks/data or data.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Compute outputs without displaying matplotlib figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from src.neural_dynamics_baseline import export_oxygen_clusters

    export_oxygen_clusters(
        args.datalist,
        row_index=args.row_index,
        oxygen_file=args.oxygen_file,
        show=not args.no_show,
    )
