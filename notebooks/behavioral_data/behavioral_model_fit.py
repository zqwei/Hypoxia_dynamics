from __future__ import annotations

from glob import glob
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "src").exists():
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        if (parent / "src").exists():
            REPO_ROOT = parent
            break
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.behaviors import (  # noqa: E402
    DEFAULT_CONDITION_ORDER,
    DEFAULT_SPEC_ORDER,
    fit_model_spec_preloaded,
    format_model_lags,
    load_runs_json,
    plot_models_no_lines,
    plot_models_with_lines,
    preload_model_fit_data,
    run_two_way_anova,
    summarize_model_run_list,
)


DATA_GLOB = (
    "/nrs/ahrens/Ziqiang/Motor_clamp/oxygen_glia/20250812_data_to_ziqiang/"
    "paralyzed/*"
)
REMOVED_ONE = [2, 3, 4, 6, 7]

BIN_SIZE_S = 1
PRE_S = 10
POST_S = 30

N_RUNS = 20
N_JOBS = 4
BASE_SEED = None
RETRY_ON_SEPARATION = 2

MODEL_SPECS = [
    {"o2": 8, "swim": 9},  # full model with both o2 and swim history
    {"o2": 8},  # o2 only
    {"swim": 9},  # swim only
    {"o2": 8, "swim": 1},  # single swim state + o2 model
]

RUNS_JSON = Path(__file__).with_name("behavioral_model_fit_runs.json")
REPORT_PLOTS = True
REPORT_STATS = True
PLOT_WITH_LINES = True
PLOT_NO_LINES = True

PLOT_EV_PATH = Path(__file__).with_name("EV_models.pdf")
PLOT_AIC_PATH = Path(__file__).with_name("AIC_models.pdf")
PLOT_EV_NOLINE_PATH = Path(__file__).with_name("EV_models_no_line.pdf")
PLOT_AIC_NOLINE_PATH = Path(__file__).with_name("AIC_models_no_line.pdf")


def model_run_list_to_json(
    model_run_list: list[dict[str, object]],
    condition_labels: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    if not model_run_list:
        return []

    num_model = len(model_run_list[0]["runs"][0][0])
    if condition_labels is None:
        condition_labels = ["normoxia", "hypoxia"][:num_model]
    else:
        condition_labels = list(condition_labels)[:num_model]

    rows = []
    for spec_idx, model_spec in enumerate(model_run_list):
        spec_label = format_model_lags(model_spec["model_lags"])
        runs_spec = model_spec["runs"]
        for run_idx, run in enumerate(runs_spec):
            for fish_idx, models_for_fish in enumerate(run):
                for cond_idx, model_out in enumerate(models_for_fish):
                    params = model_out[2]
                    if hasattr(params, "tolist"):
                        params = params.tolist()
                    rows.append(
                        {
                            "spec_idx": spec_idx,
                            "spec_label": spec_label,
                            "run_idx": run_idx,
                            "fish_idx": fish_idx,
                            "condition": condition_labels[cond_idx],
                            "r2": model_out[1],
                            "aic": model_out[4],
                            "params": params,
                            "perfect_separation": bool(
                                model_out[5] if len(model_out) > 5 else False
                            ),
                        }
                    )
    return rows


def run_behavioral_model_fit() -> pd.DataFrame:
    files_ = glob(DATA_GLOB)

    preloaded_data = preload_model_fit_data(
        files_,
        REMOVED_ONE,
        bin_size=BIN_SIZE_S,
        pre_s=PRE_S,
        post_s=POST_S,
    )

    model_run_list = [
        fit_model_spec_preloaded(
            preloaded_data,
            model_lags=lags,
            n_runs=N_RUNS,
            n_jobs=N_JOBS,
            base_seed=BASE_SEED,
            retry_on_separation=RETRY_ON_SEPARATION,
        )
        for lags in MODEL_SPECS
    ]

    metrics = summarize_model_run_list(model_run_list, selected_spec_idx=0)
    metrics_summary_df = metrics["metrics_summary_df"]
    runs_json = model_run_list_to_json(
        model_run_list, condition_labels=metrics["condition_labels"]
    )
    RUNS_JSON.write_text(json.dumps(runs_json, indent=2))

    if REPORT_PLOTS or REPORT_STATS:
        runs_df = load_runs_json(RUNS_JSON)

    if REPORT_PLOTS:
        if PLOT_WITH_LINES:
            plot_models_with_lines(
                runs_df,
                value_col="r2",
                output_path=PLOT_EV_PATH,
                spec_order=DEFAULT_SPEC_ORDER,
                condition_order=DEFAULT_CONDITION_ORDER,
                ylim=(0, 0.75),
            )
            plot_models_with_lines(
                runs_df,
                value_col="aic",
                output_path=PLOT_AIC_PATH,
                spec_order=DEFAULT_SPEC_ORDER,
                condition_order=DEFAULT_CONDITION_ORDER,
                ylim=(0, 200),
            )
        if PLOT_NO_LINES:
            plot_models_no_lines(
                runs_df,
                value_col="r2",
                output_path=PLOT_EV_NOLINE_PATH,
                spec_order=DEFAULT_SPEC_ORDER,
                condition_order=DEFAULT_CONDITION_ORDER,
                ylim=(0, 0.75),
            )
            plot_models_no_lines(
                runs_df,
                value_col="aic",
                output_path=PLOT_AIC_NOLINE_PATH,
                spec_order=DEFAULT_SPEC_ORDER,
                condition_order=DEFAULT_CONDITION_ORDER,
                ylim=(0, 200),
            )

    if REPORT_STATS:
        anova_r2 = run_two_way_anova(runs_df, value_col="r2")
        anova_aic = run_two_way_anova(runs_df, value_col="aic")
        print("ANOVA r2:")
        for k, v in anova_r2.items():
            print("\\n===", k, "===")
            print(v)
        print("\\nANOVA AIC:")
        for k, v in anova_aic.items():
            print("\\n===", k, "===")
            print(v)
    return metrics_summary_df


if __name__ == "__main__":
    df = run_behavioral_model_fit()
    print(f"Saved runs with params to: {RUNS_JSON}")
    print(df)
