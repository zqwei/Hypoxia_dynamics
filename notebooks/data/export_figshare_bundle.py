from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TARGET_ROOT = Path("/nrs/ahrens/Ziqiang/Motor_clamp/figshare")
ARCHIVE_PATH = TARGET_ROOT / "hypoxia_dynamics_figshare_recordings.tar.gz"

REQUIRED_FILES = [
    (
        "cell_center.npy",
        "cell coordinates (z, y, x) in pixel space; pixel spacing is 5 x 0.812 x 0.812 um",
    ),
    ("Y_ave.npy", "anatomical brain volume of the recording"),
    (
        "cell_dff.npz",
        "cell dynamics, including baseline dynamics [baseline] and dF/F dynamics [dFF]",
    ),
    (
        "locs_cam.npy",
        "frame triggers (2-3 Hz) in the ephys recording (6000 Hz)",
    ),
    ("swim_ds.npy", "processed ephys data downsampled to imaging frames"),
    (
        "data.mat",
        "raw ephys data: left-side swimming signal [ch1], right-side swimming signal [ch2], "
        "and swim detection results [swimStartInd*], [swimEndInd*], [swimPower*]",
    ),
]

COHORTS = [
    (
        "neuronal_recordings",
        Path("notebooks/data/datalist_huc_h2b_gc7f.csv"),
        range(0, 9),
    ),
    (
        "glial_recordings",
        Path("notebooks/data/datalist_gfap_gc6f.csv"),
        range(0, 6),
    ),
]


@dataclass
class RecordingReport:
    cohort: str
    index: int
    recording_name: str
    source_root: str
    destination_root: str
    total_bytes: int
    missing_files: list[str]
    copied: bool


def build_recording_reports() -> list[RecordingReport]:
    reports: list[RecordingReport] = []
    for cohort, csv_path, indices in COHORTS:
        df = pd.read_csv(csv_path)
        for idx in indices:
            source_root = Path(df.loc[idx, "save_root"])
            recording_name = source_root.name
            destination_root = TARGET_ROOT / cohort / recording_name
            missing_files = [
                filename
                for filename, _ in REQUIRED_FILES
                if not (source_root / filename).exists()
            ]
            total_bytes = 0
            for filename, _ in REQUIRED_FILES:
                file_path = source_root / filename
                if file_path.exists():
                    total_bytes += file_path.stat().st_size
            reports.append(
                RecordingReport(
                    cohort=cohort,
                    index=idx,
                    recording_name=recording_name,
                    source_root=str(source_root),
                    destination_root=str(destination_root),
                    total_bytes=total_bytes,
                    missing_files=missing_files,
                    copied=False,
                )
            )
    return reports


def write_readme() -> None:
    lines = [
        "# Figshare export bundle",
        "",
        "This folder contains the subset of neuronal and glial recording files prepared for Figshare sharing.",
        "",
        "## Cohorts",
        "- neuronal_recordings: 9 recordings",
        "- glial_recordings: 6 recordings",
        "",
        "Each recording folder preserves the original recording basename from `save_root` and contains only the files listed below.",
        "",
        "## Files included in each recording folder",
    ]
    for filename, description in REQUIRED_FILES:
        lines.append(f"- `{filename}`: {description}")
    lines.extend(
        [
            "",
            "## Notes",
            "- `locs_cam.npy` maps imaging-frame triggers into the 6000 Hz ephys recording.",
            "- `data.mat` contains raw ephys channels and swim detection outputs.",
        ]
    )
    (TARGET_ROOT / "README.md").write_text("\n".join(lines) + "\n")


def write_copy_report(reports: list[RecordingReport]) -> None:
    csv_path = TARGET_ROOT / "copy_report.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cohort",
                "index",
                "recording_name",
                "source_root",
                "destination_root",
                "total_bytes",
                "missing_files",
                "copied",
                *[filename for filename, _ in REQUIRED_FILES],
            ],
        )
        writer.writeheader()
        for report in reports:
            row = {
                "cohort": report.cohort,
                "index": report.index,
                "recording_name": report.recording_name,
                "source_root": report.source_root,
                "destination_root": report.destination_root,
                "total_bytes": report.total_bytes,
                "missing_files": ";".join(report.missing_files),
                "copied": report.copied,
            }
            for filename, _ in REQUIRED_FILES:
                row[filename] = filename not in report.missing_files
            writer.writerow(row)

    totals_by_cohort = {}
    for cohort, _, _ in COHORTS:
        totals_by_cohort[cohort] = sum(
            report.total_bytes for report in reports if report.cohort == cohort
        )
    combined_total = sum(report.total_bytes for report in reports)
    missing_count = sum(bool(report.missing_files) for report in reports)

    lines = [
        "# Copy report",
        "",
        f"- Recordings checked: {len(reports)}",
        f"- Recordings with missing files: {missing_count}",
        f"- Neuronal total bytes: {totals_by_cohort['neuronal_recordings']}",
        f"- Glial total bytes: {totals_by_cohort['glial_recordings']}",
        f"- Combined total bytes: {combined_total}",
        "",
    ]
    if missing_count == 0:
        lines.append("- Validation status: all required files were present before copy.")
    else:
        lines.append("- Validation status: missing files were found; copy was aborted.")
    (TARGET_ROOT / "copy_report.md").write_text("\n".join(lines) + "\n")


def copy_files(reports: list[RecordingReport]) -> None:
    for report in reports:
        dest_root = Path(report.destination_root)
        dest_root.mkdir(parents=True, exist_ok=True)
        source_root = Path(report.source_root)
        for filename, _ in REQUIRED_FILES:
            shutil.copy2(source_root / filename, dest_root / filename)
        report.copied = True


def create_archive() -> None:
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    subprocess.run(
        [
            "tar",
            "-czf",
            str(ARCHIVE_PATH),
            "-C",
            str(TARGET_ROOT),
            "README.md",
            "neuronal_recordings",
            "glial_recordings",
        ],
        check=True,
    )


def main() -> None:
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    reports = build_recording_reports()
    write_readme()
    write_copy_report(reports)

    if any(report.missing_files for report in reports):
        raise SystemExit("Missing required files detected. See copy_report.csv and copy_report.md.")

    copy_files(reports)
    write_copy_report(reports)
    create_archive()

    archive_size = ARCHIVE_PATH.stat().st_size
    neuronal_total = sum(
        report.total_bytes for report in reports if report.cohort == "neuronal_recordings"
    )
    glial_total = sum(
        report.total_bytes for report in reports if report.cohort == "glial_recordings"
    )
    combined_total = sum(report.total_bytes for report in reports)
    print("Export complete.")
    print(f"Neuronal total bytes: {neuronal_total}")
    print(f"Glial total bytes: {glial_total}")
    print(f"Combined total bytes: {combined_total}")
    print(f"Archive size bytes: {archive_size}")
    print(f"Archive path: {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
