import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy registered cell center arrays into the configured save roots."
    )
    parser.add_argument(
        "datalist",
        nargs="?",
        default="datalist_huc_ablation.csv",
        help="CSV file name under notebooks/data or data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from src.data import copy_registered_cell_centers

    copy_registered_cell_centers(args.datalist)
