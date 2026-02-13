from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

# Allow running without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nfl_elo_model.paths import get_paths


def delta(r_home: float, r_away: float, points_home: float, points_away: float, elo_hfa: float = 65, k: float = 20) -> tuple[float, float]:
    s_home = 1.0 if points_home > points_away else (0.5 if points_home == points_away else 0.0)
    expected_home = 1.0 / (1.0 + 10 ** (-((r_home - r_away + elo_hfa) / 400.0)))
    mov = np.log(abs(points_home - points_away) + 1.0)
    d = k * mov * (s_home - expected_home)
    return r_home + d, r_away - d


def main() -> None:
    ap = argparse.ArgumentParser(description="Build schedule_features_2016_2025.parquet from nflreadpy.")
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root (default: repo root)")
    args = ap.parse_args()

    paths = get_paths(args.repo_root)
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 2015 priors (Elo + EPA)
    # -------------------------
    df_ls_2015 = nfl.load_schedules(seasons=2015).to_pandas()

    cols_sched = ["game_type", "week", "away_team", "away_score", "home_team", "home_score", "result"]
    df_ls_2015_sub = df_ls_2015[cols_sched]

    elo_init = 1500.0
    rho = 0.75
    elos: dict[str, float] = {}

    for _, row in df_ls_2015_sub.iterrows():
        home = row.home_team
        away = row.away_team
        r_home = elos.get(home, elo_init)
        r_away = elos.get(away, elo_init)
        k_row = 24 if row.game_type != "REG" else 20
        r_home_new, r_away_new = delta(r_home, r_away, row.home_score, row.away_score, k=k_row)
        elos[home] = r_home_new
        elos[away] = r_away_new

    rename_map = {"SD": "LAC", "OAK": "LV", "STL": "LA"}
    for old, new in rename_map.items():
        if old in elos:
            elos[new] = elos.pop(old)

    df_lts_2015 = nfl.load_team_stats(seasons=2015).to_pandas()

    cols_off_epa = ["team", "attempts", "passing_epa", "carries", "rushing_epa"]
    df_lts_2015_off = df_lts_2015[cols_off_epa].groupby("team").sum()
    df_lts_2015_off["off_epa_pp_2015"] = (
        (df_lts_2015_off["passing_epa"] + df_lts_2015_off["rushing_epa"])
        / (df_lts_2015_off["attempts"] + df_lts_2015_off["carries"])
    )

    cols_def_epa = ["opponent_team", "attempts", "passing_epa", "carries", "rushing_epa"]
    df_lts_2015_def = df_lts_2015[cols_def_epa].groupby("opponent_team").sum()
    df_lts_2015_def["def_epa_allowed_pp_2015"] = (
        (df_lts_2015_def["passing_epa"] + df_lts_2015_def["rushing_epa"])
        / (df_lts_2015_def["attempts"] + df_lts_2015_def["carries"])
    )

    cols_to = [
        "team",
        "passing_interceptions",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "sack_fumbles_lost",
        "attempts",
        "carries",
    ]
    df_lts_2015_to = df_lts_2015[cols_to].groupby("team").sum()
    df_lts_2015_to["turnover_rate_2015"] = (
        df_lts_2015_to["passing_interceptions"]
        + df_lts_2015_to["rushing_fumbles_lost"]
        + df_lts_2015_to["receiving_fumbles_lost"]
        + df_lts_2015_to["sack_fumbles_lost"]
    ) / (df_lts_2015_to["attempts"] + df_lts_2015_to["carries"])

    off_epa_pp = df_lts_2015_off["off_epa_pp_2015"].to_dict()
    def_epa_pp = df_lts_2015_def["def_epa_allowed_pp_2015"].to_dict()
    turnover_rate = df_lts_2015_to["turnover_rate_2015"].to_dict()

    # -------------------------
    # 2016-2025 schedules/stats
    # -------------------------
    seasons = list(range(2016, 2026))
    df_ls = nfl.load_schedules(seasons=seasons).to_pandas()
    df_lts = nfl.load_team_stats(seasons=seasons).to_pandas()

    cols_support = [
        "season",
        "week",
        "team",
        "opponent_team",
        "attempts",
        "carries",
        "passing_epa",
        "rushing_epa",
        "passing_interceptions",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "sack_fumbles_lost",
    ]
    df_lts_sub = df_lts.loc[:, cols_support].copy()
    plays = df_lts_sub["attempts"] + df_lts_sub["carries"]
    off_epa = df_lts_sub["passing_epa"] + df_lts_sub["rushing_epa"]
    turnover_count = (
        df_lts_sub["passing_interceptions"]
        + df_lts_sub["rushing_fumbles_lost"]
        + df_lts_sub["receiving_fumbles_lost"]
        + df_lts_sub["sack_fumbles_lost"]
    )
    df_lts_sub = df_lts_sub.assign(
        plays=plays,
        off_epa=off_epa,
        off_epa_pp=off_epa / plays,
        turnover_count=turnover_count,
        turnover_rate=turnover_count / plays,
    )[
        [
            "season",
            "week",
            "team",
            "opponent_team",
            "plays",
            "off_epa",
            "off_epa_pp",
            "turnover_count",
            "turnover_rate",
        ]
    ]

    cols_model = [
        "game_id",
        "season",
        "game_type",
        "week",
        "away_team",
        "away_score",
        "home_team",
        "home_score",
        "result",
        "total",
        "overtime",
        "away_rest",
        "home_rest",
        "div_game",
        "roof",
        "surface",
    ]
    df_ls_sub = df_ls[cols_model].copy()

    team_map = {"SD": "LAC", "OAK": "LV"}
    df_ls_sub["home_team"] = df_ls_sub["home_team"].replace(team_map)
    df_ls_sub["away_team"] = df_ls_sub["away_team"].replace(team_map)

    df_ls_sub = df_ls_sub.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    r_home_list: list[float] = []
    r_away_list: list[float] = []
    off_epa_pp_home_list: list[float] = []
    off_epa_pp_away_list: list[float] = []
    def_epa_pp_home_list: list[float] = []
    def_epa_pp_away_list: list[float] = []
    turnover_rate_home_list: list[float] = []
    turnover_rate_away_list: list[float] = []

    for _, row in df_ls_sub.iterrows():
        home = row.home_team
        away = row.away_team

        r_home = elos.get(home)
        r_away = elos.get(away)
        off_epa_pp_home = off_epa_pp.get(home)
        off_epa_pp_away = off_epa_pp.get(away)
        def_epa_pp_home = def_epa_pp.get(home)
        def_epa_pp_away = def_epa_pp.get(away)
        turnover_rate_home = turnover_rate.get(home)
        turnover_rate_away = turnover_rate.get(away)

        if row.week == 1:
            r_home = elo_init + rho * (r_home - elo_init)
            r_away = elo_init + rho * (r_away - elo_init)

        r_home_list.append(r_home)
        r_away_list.append(r_away)
        off_epa_pp_home_list.append(off_epa_pp_home)
        off_epa_pp_away_list.append(off_epa_pp_away)
        def_epa_pp_home_list.append(def_epa_pp_home)
        def_epa_pp_away_list.append(def_epa_pp_away)
        turnover_rate_home_list.append(turnover_rate_home)
        turnover_rate_away_list.append(turnover_rate_away)

        if pd.isna(row["away_score"]):
            continue

        k_row = 24 if row.game_type != "REG" else 20
        r_home_new, r_away_new = delta(r_home, r_away, row.home_score, row.away_score, k=k_row)

        epa_lambda = 0.35 if row.week < 5 else 0.28
        off_epa_pp_home_new = (1 - epa_lambda) * off_epa_pp_home + epa_lambda * (
            df_lts_sub[
                (df_lts_sub.team == row.home_team)
                & (df_lts_sub.season == row.season)
                & (df_lts_sub.week == row.week)
            ].off_epa_pp.sum()
        )
        off_epa_pp_away_new = (1 - epa_lambda) * off_epa_pp_away + epa_lambda * (
            df_lts_sub[
                (df_lts_sub.team == row.away_team)
                & (df_lts_sub.season == row.season)
                & (df_lts_sub.week == row.week)
            ].off_epa_pp.sum()
        )

        def_epa_pp_home_new = (1 - epa_lambda) * def_epa_pp_home + epa_lambda * (
            df_lts_sub[
                (df_lts_sub.opponent_team == row.home_team)
                & (df_lts_sub.season == row.season)
                & (df_lts_sub.week == row.week)
            ].off_epa_pp.sum()
        )
        def_epa_pp_away_new = (1 - epa_lambda) * def_epa_pp_away + epa_lambda * (
            df_lts_sub[
                (df_lts_sub.opponent_team == row.away_team)
                & (df_lts_sub.season == row.season)
                & (df_lts_sub.week == row.week)
            ].off_epa_pp.sum()
        )

        turnover_rate_home_new = (1 - epa_lambda) * turnover_rate_home + epa_lambda * (
            df_lts_sub[
                (df_lts_sub.team == row.home_team)
                & (df_lts_sub.season == row.season)
                & (df_lts_sub.week == row.week)
            ].turnover_rate.sum()
        )
        turnover_rate_away_new = (1 - epa_lambda) * turnover_rate_away + epa_lambda * (
            df_lts_sub[
                (df_lts_sub.team == row.away_team)
                & (df_lts_sub.season == row.season)
                & (df_lts_sub.week == row.week)
            ].turnover_rate.sum()
        )

        elos[home] = r_home_new
        elos[away] = r_away_new
        off_epa_pp[home] = off_epa_pp_home_new
        off_epa_pp[away] = off_epa_pp_away_new
        def_epa_pp[home] = def_epa_pp_home_new
        def_epa_pp[away] = def_epa_pp_away_new
        turnover_rate[home] = turnover_rate_home_new
        turnover_rate[away] = turnover_rate_away_new

    df_ls_sub["r_away"] = r_away_list
    df_ls_sub["r_home"] = r_home_list
    df_ls_sub["off_epa_pp_away"] = off_epa_pp_away_list
    df_ls_sub["off_epa_pp_home"] = off_epa_pp_home_list
    df_ls_sub["def_epa_pp_away"] = def_epa_pp_away_list
    df_ls_sub["def_epa_pp_home"] = def_epa_pp_home_list
    df_ls_sub["turnover_rate_away"] = turnover_rate_away_list
    df_ls_sub["turnover_rate_home"] = turnover_rate_home_list

    df_ls_sub.to_parquet(paths.schedule_features_path, index=False)
    print(f"Wrote: {paths.schedule_features_path} ({len(df_ls_sub)} rows)")


if __name__ == "__main__":
    main()
