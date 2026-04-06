from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_CANDIDATES = (
    REPO_ROOT / "notebooks" / "data",
    REPO_ROOT / "data",
)


def data_file(name: str) -> Path:
    for base_dir in DATA_DIR_CANDIDATES:
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {name!r} in notebooks/data or data under {REPO_ROOT}"
    )


def load_datalist(name: str) -> pd.DataFrame:
    return pd.read_csv(data_file(name), index_col=0)


def ensure_directory(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
