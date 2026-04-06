import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export swim downsampled traces from saved analysis outputs."
    )
    parser.add_argument(
        "datalist",
        nargs="?",
        default="datalist_gfap_gc6f_v2.csv",
        help="CSV file name under notebooks/data or data.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=10,
        help="Largest dataframe index to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all rows in the datalist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from src.data import export_swim_ds

    max_index = None if args.all else args.max_index
    export_swim_ds(args.datalist, max_index=max_index)
