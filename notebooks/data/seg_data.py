import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export segmented fluorescence and cell center data."
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
        default=38,
        help="Smallest dataframe index to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all rows in the datalist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from src.data import export_segmented_data

    start_index = None if args.all else args.start_index
    export_segmented_data(args.datalist, start_index=start_index)
