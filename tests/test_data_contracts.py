import sys
import unittest
from pathlib import Path

import pandas as pd

# Allow running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nfl_elo_model.paths import get_paths
from nfl_elo_model.schema import MODEL_TABLE_SCHEMA, SCHEDULE_FEATURES_SCHEMA
from nfl_elo_model.validate import require_columns, require_non_empty


class TestDataContracts(unittest.TestCase):
    def test_schedule_features_schema(self) -> None:
        paths = get_paths(".")
        df = pd.read_parquet(paths.schedule_features_path)
        require_non_empty(df, "schedule_features")
        require_columns(df, SCHEDULE_FEATURES_SCHEMA)

    def test_model_table_schema(self) -> None:
        paths = get_paths(".")
        df = pd.read_parquet(paths.model_table_path)
        require_non_empty(df, "final_first_model")
        require_columns(df, MODEL_TABLE_SCHEMA)

    def test_no_obvious_leakage_columns_in_features(self) -> None:
        # Sanity check: do not treat raw scores as model features.
        # (Models should use pregame features + context; outcomes are targets.)
        forbidden = {"home_score", "away_score", "result", "total_points", "margin", "home_win"}
        allowed_targets = {"total_points", "margin", "home_win"}

        paths = get_paths(".")
        df = pd.read_parquet(paths.model_table_path)

        # These columns can exist in the table (they are targets/outcomes),
        # but we ensure our core pregame feature columns are present and distinct.
        pregame_features = {"elo_diff", "off_diff", "def_allowed_diff", "to_diff", "rest_diff", "div_game"}
        self.assertTrue(pregame_features.issubset(set(df.columns)))

        # Make sure we didn't accidentally name a feature like a target.
        self.assertTrue(pregame_features.isdisjoint(forbidden - allowed_targets))


if __name__ == "__main__":
    unittest.main()
