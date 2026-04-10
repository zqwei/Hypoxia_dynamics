from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


DEFAULT_SPEC_ORDER = ["o2:8", "o2:8, swim:1", "swim:9", "o2:8, swim:9"]
DEFAULT_CONDITION_ORDER = ["normoxia", "hypoxia"]


def load_runs_json(path: str | Path) -> pd.DataFrame:
    runs = json.loads(Path(path).read_text())
    return pd.DataFrame(runs)


def aggregate_per_fish(
    runs_df: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    return (
        runs_df.groupby(["spec_label", "condition", "fish_idx"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )


def plot_models_with_lines(
    runs_df: pd.DataFrame,
    value_col: str,
    output_path: str | Path,
    spec_order: list[str] | None = None,
    condition_order: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fish_df = aggregate_per_fish(runs_df, value_col)
    spec_order = spec_order or DEFAULT_SPEC_ORDER
    condition_order = condition_order or DEFAULT_CONDITION_ORDER

    plt.figure(figsize=(5, 3))
    ax = sns.barplot(
        data=fish_df,
        x="spec_label",
        y=value_col,
        hue="condition",
        order=spec_order,
        hue_order=condition_order,
        errorbar="se",
    )

    # Per-fish lines across 8 points (2 conditions x 4 models)
    x_offsets = {"normoxia": -0.2, "hypoxia": 0.2}
    for fish_idx, fish_group in fish_df.groupby("fish_idx"):
        xs = []
        ys = []
        for spec in spec_order:
            for condition in condition_order:
                row = fish_group[
                    (fish_group["spec_label"] == spec)
                    & (fish_group["condition"] == condition)
                ]
                if row.empty:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                else:
                    xs.append(spec_order.index(spec) + x_offsets[condition])
                    ys.append(float(row[value_col].iloc[0]))
        ax.plot(xs, ys, color="k", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_models_no_lines(
    runs_df: pd.DataFrame,
    value_col: str,
    output_path: str | Path,
    spec_order: list[str] | None = None,
    condition_order: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fish_df = aggregate_per_fish(runs_df, value_col)
    spec_order = spec_order or DEFAULT_SPEC_ORDER
    condition_order = condition_order or DEFAULT_CONDITION_ORDER

    plt.figure(figsize=(5, 3))
    ax = sns.barplot(
        data=fish_df,
        x="spec_label",
        y=value_col,
        hue="condition",
        order=spec_order,
        hue_order=condition_order,
        errorbar="se",
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_two_way_anova(
    runs_df: pd.DataFrame,
    value_col: str,
    spec_order: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    spec_order = spec_order or DEFAULT_SPEC_ORDER

    fish_df = (
        runs_df.groupby(["spec_label", "condition", "fish_idx"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )

    results: dict[str, pd.DataFrame] = {}
    for condition in DEFAULT_CONDITION_ORDER:
        cond_df = fish_df[fish_df["condition"] == condition].copy()
        cond_df = cond_df[cond_df["spec_label"].isin(spec_order)]
        model = ols(f"{value_col} ~ C(spec_label) + C(fish_idx)", data=cond_df).fit()
        results[condition] = sm.stats.anova_lm(model, typ=2)

    comb_df = (
        fish_df.groupby(["spec_label", "fish_idx"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )
    comb_df = comb_df[comb_df["spec_label"].isin(spec_order)]
    model = ols(f"{value_col} ~ C(spec_label) + C(fish_idx)", data=comb_df).fit()
    results["combined"] = sm.stats.anova_lm(model, typ=2)
    return results

