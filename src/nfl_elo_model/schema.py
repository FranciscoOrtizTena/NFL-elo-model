from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaCheck:
    name: str
    required_cols: tuple[str, ...]


SCHEDULE_FEATURES_SCHEMA = SchemaCheck(
    name="schedule_features",
    required_cols=(
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "away_rest",
        "home_rest",
        "div_game",
        "roof",
        "surface",
        "r_away",
        "r_home",
        "off_epa_pp_away",
        "off_epa_pp_home",
        "def_epa_pp_away",
        "def_epa_pp_home",
        "turnover_rate_away",
        "turnover_rate_home",
    ),
)


MODEL_TABLE_SCHEMA = SchemaCheck(
    name="final_first_model",
    required_cols=(
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "home_win",
        "margin",
        "total_points",
        "elo_diff",
        "off_diff",
        "def_allowed_diff",
        "to_diff",
        "rest_diff",
        "div_game",
        "is_dome",
        "is_outdoors",
        "is_retractable",
        "is_grass",
    ),
)

