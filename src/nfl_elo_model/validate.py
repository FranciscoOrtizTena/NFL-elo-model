from __future__ import annotations

import pandas as pd

from .schema import SchemaCheck


def require_columns(df: pd.DataFrame, schema: SchemaCheck) -> None:
    missing = [c for c in schema.required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{schema.name}: missing required columns: {missing}")


def require_non_empty(df: pd.DataFrame, name: str) -> None:
    if len(df) == 0:
        raise ValueError(f"{name}: dataframe is empty")


def require_no_nulls(df: pd.DataFrame, cols: list[str], name: str) -> None:
    nulls = {c: int(df[c].isna().sum()) for c in cols}
    bad = {c: n for c, n in nulls.items() if n > 0}
    if bad:
        raise ValueError(f"{name}: nulls found in columns: {bad}")

