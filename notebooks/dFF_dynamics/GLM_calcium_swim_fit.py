import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


DEFAULT_MAX_INDEX = {
    "neuron": 8,
    "glia": 5,
}
DEFAULT_DATALIST = {
    "neuron": "datalist_huc_h2b_gc7f.csv",
    "glia": "datalist_gfap_gc6f.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit calcium activity to swim drive for selected neuron and glia datasets."
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
        "--chunk-size",
        type=int,
        default=None,
        help="Number of cells to fit per chunk.",
    )
    parser.add_argument(
        "--lag-grid",
        type=int,
        nargs="+",
        default=None,
        help="Explicit lag grid in frames, for example: --lag-grid 6 8 10 12",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=None,
        help="Weighted ridge lambda.",
    )
    parser.add_argument(
        "--tau-frames",
        type=float,
        default=None,
        help="Exponential lag penalty timescale in frames.",
    )
    parser.add_argument(
        "--normoxia-window",
        type=float,
        nargs=2,
        metavar=("START_S", "END_S"),
        default=None,
        help="Normoxia fit window in seconds.",
    )
    parser.add_argument(
        "--hypoxia-window",
        type=float,
        nargs=2,
        metavar=("START_S", "END_S"),
        default=None,
        help="Hypoxia fit window in seconds.",
    )
    parser.add_argument(
        "--spearman-threshold",
        type=float,
        default=None,
        help="Minimum max absolute Spearman correlation across the two periods.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Thread count for chunked Spearman and fit operations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing GLM_calcium_swim_fit.npz outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from src.dFF_dynamics import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_HYPOXIA_WINDOW_S,
        DEFAULT_LAG_GRID,
        DEFAULT_NORMOXIA_WINDOW_S,
        DEFAULT_N_JOBS,
        DEFAULT_RIDGE_LAMBDA,
        DEFAULT_SPEARMAN_THRESHOLD,
        DEFAULT_TAU_FRAMES,
        export_glm_calcium_swim_fit,
    )

    datasets = ["neuron", "glia"] if args.dataset == "all" else [args.dataset]
    lag_grid = DEFAULT_LAG_GRID if args.lag_grid is None else args.lag_grid
    ridge_lambda = DEFAULT_RIDGE_LAMBDA if args.ridge_lambda is None else args.ridge_lambda
    tau_frames = DEFAULT_TAU_FRAMES if args.tau_frames is None else args.tau_frames
    normoxia_window = (
        DEFAULT_NORMOXIA_WINDOW_S if args.normoxia_window is None else tuple(args.normoxia_window)
    )
    hypoxia_window = (
        DEFAULT_HYPOXIA_WINDOW_S if args.hypoxia_window is None else tuple(args.hypoxia_window)
    )
    spearman_threshold = (
        DEFAULT_SPEARMAN_THRESHOLD
        if args.spearman_threshold is None
        else args.spearman_threshold
    )
    chunk_size = DEFAULT_CHUNK_SIZE if args.chunk_size is None else args.chunk_size
    n_jobs = DEFAULT_N_JOBS if args.n_jobs is None else args.n_jobs

    for dataset in datasets:
        max_index = DEFAULT_MAX_INDEX[dataset] if args.max_index is None else args.max_index
        export_glm_calcium_swim_fit(
            DEFAULT_DATALIST[dataset],
            start_index=args.start_index,
            max_index=max_index,
            normoxia_window_s=normoxia_window,
            hypoxia_window_s=hypoxia_window,
            lag_grid=lag_grid,
            ridge_lambda=ridge_lambda,
            tau_frames=tau_frames,
            spearman_threshold=spearman_threshold,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            force=args.force,
        )
