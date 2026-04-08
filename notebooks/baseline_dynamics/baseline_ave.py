import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average baseline traces for negatively and positively correlated cells."
    )
    parser.add_argument(
        "datalist",
        nargs="?",
        default="datalist.csv",
        help="CSV file name under notebooks/data or data.",
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
    from src.neural_dynamics_baseline import export_baseline_averages

    max_index = None if args.all else args.max_index
    export_baseline_averages(args.datalist, max_index=max_index)
