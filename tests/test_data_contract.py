from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_contract import LABEL_COLUMN, REQUIRED_COLUMNS, feature_columns, validate_dataframe


def make_valid_dataframe(rows: int = 3, include_label: bool = True) -> pd.DataFrame:
    data: dict[str, list[float] | list[int]] = {}
    for col in sorted(REQUIRED_COLUMNS):
        if col == LABEL_COLUMN:
            if include_label:
                data[col] = [0 for _ in range(rows)]
        else:
            data[col] = [0.0 for _ in range(rows)]
    return pd.DataFrame(data)


class DataContractTests(unittest.TestCase):
    def test_validate_dataframe_accepts_expected_schema(self) -> None:
        df = make_valid_dataframe()
        validate_dataframe(df, require_label=True)

    def test_validate_dataframe_raises_for_missing_required_columns(self) -> None:
        df = make_valid_dataframe()
        df = df.drop(columns=["V7"])
        with self.assertRaises(ValueError):
            validate_dataframe(df, require_label=True)

    def test_feature_columns_removes_label_and_sorts(self) -> None:
        cols = ["V2", "Class", "Amount", "Time", "V1"]
        self.assertEqual(feature_columns(cols), ["Amount", "Time", "V1", "V2"])


if __name__ == "__main__":
    unittest.main()
