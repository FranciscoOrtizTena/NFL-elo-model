from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import joblib

from .paths import get_paths, ProjectPaths


TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


@dataclass(frozen=True)
class Models:
    spread_model: object
    total_model: object
    win_model: object


def load_models(paths: ProjectPaths) -> Models:
    models_dir = paths.data_dir / "models"
    spread_model = joblib.load(models_dir / "spread_ridge_timecv.joblib")
    total_model = joblib.load(models_dir / "total_ridge.joblib")
    win_model = joblib.load(models_dir / "home_win_logit.joblib")
    return Models(spread_model=spread_model, total_model=total_model, win_model=win_model)


def load_schedule(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_parquet(paths.schedule_features_path)


def load_priors(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_parquet(paths.data_dir / "team_priors_latest.parquet")


def load_stadium(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_parquet(paths.data_dir / "team_stadium_latest.parquet")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["elo_diff"] = x["r_home"] - x["r_away"]
    x["off_diff"] = x["off_epa_pp_home"] - x["off_epa_pp_away"]
    x["def_allowed_diff"] = x["def_epa_pp_away"] - x["def_epa_pp_home"]
    x["to_diff"] = x["turnover_rate_away"] - x["turnover_rate_home"]
    x["rest_diff"] = x["home_rest"] - x["away_rest"]

    x["roof"] = x["roof"].astype(str).str.strip().str.lower()
    x["surface"] = x["surface"].astype(str).str.strip().str.lower()
    x["is_grass"] = (x["surface"] == "grass").astype(int)
    x["is_dome"] = (x["roof"] == "dome").astype(int)
    x["is_outdoors"] = (x["roof"] == "outdoors").astype(int)
    x["is_retractable"] = x["roof"].isin(["open", "closed"]).astype(int)

    x["off_sum"] = x["off_epa_pp_home"] + x["off_epa_pp_away"]
    x["def_sum_allowed"] = x["def_epa_pp_home"] + x["def_epa_pp_away"]
    x["to_sum"] = x["turnover_rate_home"] + x["turnover_rate_away"]
    x["abs_elo_diff"] = x["elo_diff"].abs()
    return x


def make_synthetic_row(home: str, away: str, priors: pd.DataFrame, stadium: pd.DataFrame) -> pd.DataFrame:
    home_prior = priors[priors["team"] == home]
    away_prior = priors[priors["team"] == away]
    if len(home_prior) == 0 or len(away_prior) == 0:
        raise ValueError("Team not found in team_priors_latest.parquet")

    home_stadium = stadium[stadium["team"] == home]
    roof = ""
    surface = ""
    if len(home_stadium) > 0:
        roof = str(home_stadium.iloc[0]["roof"])
        surface = str(home_stadium.iloc[0]["surface"])

    season = int(home_prior.iloc[0]["season"]) if pd.notna(home_prior.iloc[0]["season"]) else None
    week = int(home_prior.iloc[0]["week"]) if pd.notna(home_prior.iloc[0]["week"]) else 1

    return pd.DataFrame(
        [
            {
                "game_id": None,
                "season": season,
                "week": week,
                "away_team": away,
                "home_team": home,
                "away_score": None,
                "home_score": None,
                "result": None,
                "total": None,
                "overtime": None,
                "away_rest": 0,
                "home_rest": 0,
                "div_game": 0,
                "roof": roof,
                "surface": surface,
                "r_away": float(away_prior.iloc[0]["r_elo"]),
                "r_home": float(home_prior.iloc[0]["r_elo"]),
                "off_epa_pp_away": float(away_prior.iloc[0]["off_epa_pp"]),
                "off_epa_pp_home": float(home_prior.iloc[0]["off_epa_pp"]),
                "def_epa_pp_away": float(away_prior.iloc[0]["def_epa_allowed_pp"]),
                "def_epa_pp_home": float(home_prior.iloc[0]["def_epa_allowed_pp"]),
                "turnover_rate_away": float(away_prior.iloc[0]["turnover_rate"]),
                "turnover_rate_home": float(home_prior.iloc[0]["turnover_rate"]),
            }
        ]
    )


def predict_row(row: pd.Series, models: Models) -> dict[str, float]:
    feat_spread = ["elo_diff", "off_diff", "def_allowed_diff", "to_diff", "rest_diff", "div_game"]
    feat_total = [
        "off_epa_pp_home",
        "off_epa_pp_away",
        "def_epa_pp_home",
        "def_epa_pp_away",
        "turnover_rate_home",
        "turnover_rate_away",
        "off_sum",
        "def_sum_allowed",
        "to_sum",
        "abs_elo_diff",
        "week",
        "div_game",
        "rest_diff",
        "is_dome",
        "is_outdoors",
        "is_retractable",
        "is_grass",
    ]
    feat_win = feat_spread

    pred_margin = float(models.spread_model.predict(pd.DataFrame([row[feat_spread]]))[0])
    pred_total = float(models.total_model.predict(pd.DataFrame([row[feat_total]]))[0])
    p_home_win = float(models.win_model.predict_proba(pd.DataFrame([row[feat_win]]))[:, 1][0])

    return {
        "pred_margin": pred_margin,
        "pred_total": pred_total,
        "p_home_win": p_home_win,
    }


def get_paths_default() -> ProjectPaths:
    return get_paths()
