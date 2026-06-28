"""Data loading helpers for preserving identifier-like strings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_preserve_codes(path_or_buffer: str | Path | Any, **kwargs: Any) -> pd.DataFrame:
    """Read CSV while preserving codes such as ISO2 ``NA``."""
    kwargs.setdefault("keep_default_na", False)
    return pd.read_csv(path_or_buffer, **kwargs)
