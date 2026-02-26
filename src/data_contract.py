"""Data contract and lightweight validation for transaction fraud training/inference."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


LABEL_COLUMN = "Class"
TIME_COLUMN = "Time"

# Kaggle mlg-ulb/creditcardfraud columns: Time, V1..V28, Amount, Class
REQUIRED_COLUMNS = {
    TIME_COLUMN,
    "Amount",
    LABEL_COLUMN,
    *{f"V{i}" for i in range(1, 29)},
}

# Fields that should never be present as model inputs.
LEAKAGE_COLUMNS = {LABEL_COLUMN}


def validate_dataframe(df: pd.DataFrame, *, require_label: bool = True) -> None:
    """Raise ValueError if schema does not match expected contract."""
    required = set(REQUIRED_COLUMNS)
    if not require_label:
        required.discard(LABEL_COLUMN)

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def feature_columns(columns: Iterable[str]) -> list[str]:
    """Return safe feature columns after removing leakage fields."""
    safe = [c for c in columns if c not in LEAKAGE_COLUMNS]
    return sorted(safe)
