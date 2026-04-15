import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subcluster baseline populations and correlate them with oxygen."
    )
    parser.add_argument(
        "datalist",
        nargs="?",
        default="datalist.csv",
        help="CSV file name under notebooks/data or data.",
    )
    parser.add_argument(
        "--oxygen-file",
        default="O2_internal.npz",
        help="Oxygen reference file under notebooks/data or data.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=17,
        help="Smallest dataframe index to process.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
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
    from src.baseline_dynamics import export_baseline_subclusters

    start_index = None if args.all else args.start_index
    max_index = None if args.all else args.max_index
    export_baseline_subclusters(
        args.datalist,
        oxygen_file=args.oxygen_file,
        start_index=start_index,
        max_index=max_index,
    )
