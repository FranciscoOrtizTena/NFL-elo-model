from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow running without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nfl_elo_model.paths import get_paths


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict spread/total/home-win for a single game row.")
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root (default: repo root)")
    ap.add_argument("--game-id", default=None, help="Filter by game_id (exact match).")
    ap.add_argument("--home", default=None, help="Home team (e.g., NE).")
    ap.add_argument("--away", default=None, help="Away team (e.g., BAL).")
    ap.add_argument("--tail", type=int, default=1, help="Use last N rows if no game-id provided.")
    args = ap.parse_args()

    paths = get_paths(args.repo_root)
    schedule_path = paths.schedule_features_path
    models_dir = paths.data_dir / "models"

    spread_model = joblib.load(models_dir / "spread_ridge_timecv.joblib")
    total_model = joblib.load(models_dir / "total_ridge.joblib")
    win_model = joblib.load(models_dir / "home_win_logit.joblib")

    df = pd.read_parquet(schedule_path)
    if args.game_id:
        df = df[df["game_id"] == args.game_id].copy()
    elif args.home and args.away:
        # Try to find scheduled row first.
        df_sched = df[(df["home_team"] == args.home) & (df["away_team"] == args.away)].copy()
        if len(df_sched) > 0:
            df = df_sched.tail(1).copy()
        else:
            # Build a synthetic row from latest priors + latest home stadium.
            priors_path = paths.data_dir / "team_priors_latest.parquet"
            stadium_path = paths.data_dir / "team_stadium_latest.parquet"
            priors = pd.read_parquet(priors_path)
            stadium = pd.read_parquet(stadium_path)

            home_prior = priors[priors["team"] == args.home]
            away_prior = priors[priors["team"] == args.away]
            if len(home_prior) == 0 or len(away_prior) == 0:
                raise SystemExit("Team not found in team_priors_latest.parquet")

            home_stadium = stadium[stadium["team"] == args.home]
            roof = ""
            surface = ""
            if len(home_stadium) > 0:
                roof = str(home_stadium.iloc[0]["roof"])
                surface = str(home_stadium.iloc[0]["surface"])

            df = pd.DataFrame(
                [
                    {
                        "game_id": None,
                        "season": int(home_prior.iloc[0]["season"]) if pd.notna(home_prior.iloc[0]["season"]) else None,
                        "week": int(home_prior.iloc[0]["week"]) if pd.notna(home_prior.iloc[0]["week"]) else 1,
                        "away_team": args.away,
                        "home_team": args.home,
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
    else:
        df = df.tail(args.tail).copy()

    if len(df) == 0:
        raise SystemExit("No rows found for prediction.")

    x = build_features(df)

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

    for _, row in x.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        pred_margin = float(spread_model.predict(pd.DataFrame([row[feat_spread]]))[0])
        pred_total = float(total_model.predict(pd.DataFrame([row[feat_total]]))[0])
        p_home_win = float(win_model.predict_proba(pd.DataFrame([row[feat_win]]))[:, 1][0])
        pick_home = int(p_home_win >= 0.5)

        print(f"{away} @ {home}")
        print(f"Pred home win prob: {p_home_win:.3f}  (pick: {'HOME' if pick_home else 'AWAY'})")
        print(f"Pred spread (home margin): {pred_margin:.2f}  -> suggested line: {home} {(-pred_margin):.2f}")
        print(f"Pred total points: {pred_total:.2f}")


if __name__ == "__main__":
    main()
