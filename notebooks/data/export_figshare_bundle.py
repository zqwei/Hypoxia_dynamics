from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TARGET_ROOT = Path("/nrs/ahrens/Ziqiang/Motor_clamp/figshare")
ARCHIVE_PATH = TARGET_ROOT / "hypoxia_dynamics_figshare_recordings.tar.gz"
ARCHIVE_INPUTS = [
    "README.md",
    "neuronal_recordings",
    "glial_recordings",
]

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


def staged_archive_input_bytes() -> int:
    total_bytes = 0
    for relative_path in ARCHIVE_INPUTS:
        path = TARGET_ROOT / relative_path
        if path.is_file():
            total_bytes += path.stat().st_size
            continue
        for child in path.rglob("*"):
            if child.is_file():
                total_bytes += child.stat().st_size
    return total_bytes


def create_archive() -> None:
    total_input_bytes = staged_archive_input_bytes()
    tmp_archive_path = ARCHIVE_PATH.with_name(f"{ARCHIVE_PATH.name}.tmp")
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    if tmp_archive_path.exists():
        tmp_archive_path.unlink()

    tar_cmd = [
        "tar",
        "-C",
        str(TARGET_ROOT),
        "-cf",
        "-",
        *ARCHIVE_INPUTS,
    ]
    use_pv = shutil.which("pv") is not None

    with tmp_archive_path.open("wb") as archive_file:
        tar_proc = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
        if tar_proc.stdout is None:
            raise RuntimeError("Failed to open tar stdout pipe.")

        prev_stdout = tar_proc.stdout
        pv_proc = None
        if use_pv:
            pv_proc = subprocess.Popen(
                [
                    "pv",
                    "-s",
                    str(total_input_bytes),
                ],
                stdin=prev_stdout,
                stdout=subprocess.PIPE,
            )
            prev_stdout.close()
            if pv_proc.stdout is None:
                raise RuntimeError("Failed to open pv stdout pipe.")
            prev_stdout = pv_proc.stdout

        pigz_proc = subprocess.Popen(
            ["pigz"],
            stdin=prev_stdout,
            stdout=archive_file,
        )
        prev_stdout.close()

        pigz_returncode = pigz_proc.wait()
        tar_returncode = tar_proc.wait()
        pv_returncode = pv_proc.wait() if pv_proc is not None else 0

    if pigz_returncode != 0:
        raise subprocess.CalledProcessError(pigz_returncode, ["pigz"])
    if pv_returncode != 0:
        raise subprocess.CalledProcessError(pv_returncode, ["pv"])
    if tar_returncode != 0:
        raise subprocess.CalledProcessError(tar_returncode, tar_cmd)

    os.replace(tmp_archive_path, ARCHIVE_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage and archive the Figshare export bundle."
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Rebuild only the archive from an existing staged folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    if args.archive_only:
        create_archive()
        archive_size = ARCHIVE_PATH.stat().st_size
        print("Archive rebuilt from existing staged folders.")
        print(f"Archive size bytes: {archive_size}")
        print(f"Archive path: {ARCHIVE_PATH}")
        return

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
