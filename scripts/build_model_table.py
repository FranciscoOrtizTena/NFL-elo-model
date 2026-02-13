from __future__ import annotations

import argparse

import pandas as pd

from nfl_elo_model.paths import get_paths
from nfl_elo_model.schema import MODEL_TABLE_SCHEMA, SCHEDULE_FEATURES_SCHEMA
from nfl_elo_model.validate import require_columns, require_non_empty


def main() -> None:
    ap = argparse.ArgumentParser(description="Build final_first_model.parquet from schedule features.")
    ap.add_argument("--repo-root", default=".", help="Repository root (default: .)")
    args = ap.parse_args()

    paths = get_paths(args.repo_root)

    df = pd.read_parquet(paths.schedule_features_path)
    require_non_empty(df, "schedule_features")
    require_columns(df, SCHEDULE_FEATURES_SCHEMA)

    # Keep only completed games.
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    # Targets.
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]

    # Differentials (pregame).
    df["elo_diff"] = df["r_home"] - df["r_away"]
    df["off_diff"] = df["off_epa_pp_home"] - df["off_epa_pp_away"]
    df["def_allowed_diff"] = df["def_epa_pp_away"] - df["def_epa_pp_home"]
    df["to_diff"] = df["turnover_rate_away"] - df["turnover_rate_home"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    # Environment flags used by modeling notebook.
    roof = df["roof"].astype(str).str.strip().str.lower()
    surface = df["surface"].astype(str).str.strip().str.lower()
    df["is_grass"] = (surface == "grass").astype(int)
    df["is_dome"] = (roof == "dome").astype(int)
    df["is_outdoors"] = (roof == "outdoors").astype(int)
    df["is_retractable"] = roof.isin(["open", "closed"]).astype(int)

    require_columns(df, MODEL_TABLE_SCHEMA)

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.model_table_path, index=False)
    print(f"Wrote: {paths.model_table_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

