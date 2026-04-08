from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 6000.0
MATCH_HZ = 10.0

Numeric = int | float


@dataclass(frozen=True)
class TrvpExampleData:
    csv_path: Path
    example_row_idx: int
    drug_timing_df: pd.DataFrame
    example_row: pd.Series
    target_folder: Path
    start_xml_path: Path
    chflt_path: Path
    file_content: np.memmap
    rawdata: dict[str, np.ndarray]
    visual_input_keys: list[str]
    nch: int
    npoint: int
    xml_numeric_fields: dict[str, list[Numeric]]
    match_df: pd.DataFrame
    alias_summary_df: pd.DataFrame
    alias_map: dict[str, str]
    loaded_signals: dict[str, np.ndarray]
    timing_fields: dict[str, float]
    stimulus_fields: dict[str, list[Numeric]]
    summary_df: pd.DataFrame


def parse_xml_value(value: str) -> Numeric | str:
    if value == "":
        return value
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def extract_ordered_numeric_field_sequences(path: Path) -> dict[str, list[Numeric]]:
    root = ET.parse(path).getroot()
    field_sequences: dict[str, list[Numeric | str]] = {}
    for epoch_idx, entry in enumerate(root):
        entry_values: dict[str, Numeric | str] = {}
        for child in entry:
            field_name = child.tag.split("}")[-1]
            entry_values[field_name] = parse_xml_value((child.text or "").strip())
            if field_name not in field_sequences:
                field_sequences[field_name] = [""] * epoch_idx
        for field_name, values in field_sequences.items():
            values.append(entry_values.get(field_name, ""))

    return {
        field_name: values
        for field_name, values in field_sequences.items()
        if values and all(isinstance(value, (int, float)) for value in values)
    }


def load_chflt_rawdata(path: Path) -> tuple[np.memmap, dict[str, np.ndarray], list[str], int, int]:
    match = re.search(r"(\d+)chFlt$", path.name)
    if match is None:
        raise ValueError(f"Could not infer channel count from {path.name}")

    nch = int(match.group(1))
    file_content = np.memmap(path, dtype=np.float32, mode="r")
    nlen = file_content.size
    npoint = int(nlen / nch)
    nlen = npoint * nch

    rawdata: dict[str, np.ndarray] = {
        "swim1": file_content[0:nlen:nch],
        "swim2": file_content[1:nlen:nch],
        "swim3": file_content[2:nlen:nch],
    }
    visual_input_keys = []
    for i in range(3, nch):
        visual_key = f"visualInput{i - 2}"
        rawdata[visual_key] = file_content[i:nlen:nch]
        visual_input_keys.append(visual_key)

    return file_content, rawdata, visual_input_keys, nch, npoint


def build_expected_trace(
    field_values: list[Numeric],
    duration_values: list[Numeric],
    sample_idx: np.ndarray,
    offset_samples: int,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, int]:
    durations_samples = np.round(np.asarray(duration_values, dtype=float) * sample_rate_hz).astype(int)
    cycle_samples = int(durations_samples.sum())
    cycle_edges = np.cumsum(durations_samples)
    phase = (sample_idx - offset_samples) % cycle_samples
    epoch_idx = np.searchsorted(cycle_edges, phase, side="right")
    expected = np.asarray(field_values, dtype=float)[epoch_idx]
    return expected, cycle_samples


def match_field_to_channel(
    field_values: list[Numeric],
    duration_values: list[Numeric],
    channel: np.ndarray,
    npoint: int,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    match_hz: float = MATCH_HZ,
) -> dict[str, float | bool]:
    step_samples = max(1, int(sample_rate_hz / match_hz))
    sample_idx = np.arange(0, npoint, step_samples)
    channel_sampled = np.asarray(channel[sample_idx], dtype=float)
    _, cycle_samples = build_expected_trace(
        field_values,
        duration_values,
        sample_idx[:1],
        0,
        sample_rate_hz=sample_rate_hz,
    )
    offset_candidates = np.arange(0, cycle_samples, step_samples)

    best: dict[str, float | bool] | None = None
    field_range = float(np.max(field_values) - np.min(field_values)) if field_values else 0.0
    tolerance = max(1e-6, 0.02 * field_range)
    direct_mae_threshold = max(1e-6, 0.10 * field_range)

    for offset_samples in offset_candidates:
        expected, _ = build_expected_trace(
            field_values,
            duration_values,
            sample_idx,
            int(offset_samples),
            sample_rate_hz=sample_rate_hz,
        )
        mae = float(np.mean(np.abs(channel_sampled - expected)))
        match_fraction = float(np.mean(np.isclose(channel_sampled, expected, atol=tolerance)))
        if best is None or mae < float(best["mae"]):
            best = {
                "mae": mae,
                "match_fraction": match_fraction,
                "offset_seconds": offset_samples / sample_rate_hz,
            }

    assert best is not None
    best["direct_match"] = bool(
        float(best["match_fraction"]) >= 0.80 and float(best["mae"]) <= direct_mae_threshold
    )
    return best


def infer_channel_aliases(
    rawdata: dict[str, np.ndarray],
    visual_input_keys: list[str],
    xml_numeric_fields: dict[str, list[Numeric]],
    npoint: int,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    match_hz: float = MATCH_HZ,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    fields_to_match = ["gain", "velocity", "duration", "brightness"]
    duration_values = xml_numeric_fields["duration"]
    match_rows = []
    for field_name in fields_to_match:
        field_values = xml_numeric_fields[field_name]
        for channel_name in visual_input_keys:
            best = match_field_to_channel(
                field_values,
                duration_values,
                rawdata[channel_name],
                npoint,
                sample_rate_hz=sample_rate_hz,
                match_hz=match_hz,
            )
            match_rows.append(
                {
                    "field_name": field_name,
                    "channel_name": channel_name,
                    "field_values": field_values,
                    **best,
                }
            )

    match_df = pd.DataFrame(match_rows).sort_values(
        ["field_name", "direct_match", "mae"],
        ascending=[True, False, True],
    )

    chosen_rows = []
    used_channels: set[str] = set()
    for field_name in ["gain", "velocity", "brightness", "duration"]:
        field_matches = match_df[match_df["field_name"] == field_name]
        available_matches = field_matches[~field_matches["channel_name"].isin(used_channels)]
        chosen = available_matches.iloc[0] if not available_matches.empty else field_matches.iloc[0]
        chosen_rows.append(chosen.to_dict())
        used_channels.add(str(chosen["channel_name"]))

    best_match_df = pd.DataFrame(chosen_rows)
    alias_map = {row.field_name: row.channel_name for row in best_match_df.itertuples(index=False)}
    return match_df, best_match_df, alias_map


def load_drug_timing_table(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def build_loaded_signals(rawdata: dict[str, np.ndarray], alias_map: dict[str, str]) -> dict[str, np.ndarray]:
    return {
        "swim1": rawdata["swim1"],
        "swim2": rawdata["swim2"],
        "camTicks": rawdata["visualInput1"],
        "gratingVel": rawdata["visualInput2"],
        "gain_trace": rawdata[alias_map["gain"]],
        "velocity_trace": rawdata[alias_map["velocity"]],
        "duration_trace": rawdata[alias_map["duration"]],
        "brightness_trace": rawdata[alias_map["brightness"]],
    }


def load_trvp_example(
    csv_path: str | Path,
    example_row_idx: int,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    match_hz: float = MATCH_HZ,
) -> TrvpExampleData:
    path = Path(csv_path)
    drug_timing_df = load_drug_timing_table(path)
    example_row = drug_timing_df.iloc[example_row_idx].copy()
    target_folder = Path(example_row["target_folder"])
    start_xml_path = sorted(target_folder.glob("*start*.xml"))[0]
    chflt_path = sorted(target_folder.glob("*chFlt*"))[0]

    xml_numeric_fields = extract_ordered_numeric_field_sequences(start_xml_path)
    file_content, rawdata, visual_input_keys, nch, npoint = load_chflt_rawdata(chflt_path)
    match_df, best_match_df, alias_map = infer_channel_aliases(
        rawdata,
        visual_input_keys,
        xml_numeric_fields,
        npoint,
        sample_rate_hz=sample_rate_hz,
        match_hz=match_hz,
    )

    loaded_signals = build_loaded_signals(rawdata, alias_map)
    stimulus_fields = {
        field_name: xml_numeric_fields[field_name]
        for field_name in ["gain", "velocity", "duration", "brightness"]
    }
    timing_fields = example_row[
        ["add_start_s", "add_end_s", "washout_start_s", "washout_end_s"]
    ].to_dict()
    alias_summary_df = best_match_df[
        [
            "field_name",
            "field_values",
            "channel_name",
            "mae",
            "match_fraction",
            "offset_seconds",
            "direct_match",
        ]
    ].copy()
    summary_df = pd.DataFrame([{**timing_fields, **stimulus_fields}])

    return TrvpExampleData(
        csv_path=path,
        example_row_idx=example_row_idx,
        drug_timing_df=drug_timing_df,
        example_row=example_row,
        target_folder=target_folder,
        start_xml_path=start_xml_path,
        chflt_path=chflt_path,
        file_content=file_content,
        rawdata=rawdata,
        visual_input_keys=visual_input_keys,
        nch=nch,
        npoint=npoint,
        xml_numeric_fields=xml_numeric_fields,
        match_df=match_df,
        alias_summary_df=alias_summary_df,
        alias_map=alias_map,
        loaded_signals=loaded_signals,
        timing_fields=timing_fields,
        stimulus_fields=stimulus_fields,
        summary_df=summary_df,
    )
