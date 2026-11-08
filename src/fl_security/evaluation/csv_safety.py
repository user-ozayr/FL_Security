"""CSV output hardening against spreadsheet formula injection."""
from __future__ import annotations

import pandas as pd


_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _safe_csv_cell(value):
    if not isinstance(value, str):
        return value
    # Spreadsheet applications may ignore leading whitespace/control chars
    # before parsing a formula. Prefix an apostrophe to force text treatment.
    probe = value.lstrip(" \t\r\n\ufeff")
    if value.startswith(("\t", "\r", "\n")) or probe.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def sanitize_csv_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with formula-like text cells forced to spreadsheet text."""
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column].dtype) or pd.api.types.is_string_dtype(
            safe[column].dtype
        ):
            safe[column] = safe[column].map(_safe_csv_cell)
    return safe
