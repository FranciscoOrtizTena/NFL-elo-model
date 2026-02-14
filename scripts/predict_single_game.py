from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nfl_elo_model.paths import get_paths
from nfl_elo_model.predict import build_features, load_models, load_priors, load_schedule, load_stadium, make_synthetic_row, predict_row


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict spread/total/home-win for a single game row.")
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root (default: repo root)")
    ap.add_argument("--game-id", default=None, help="Filter by game_id (exact match).")
    ap.add_argument("--home", default=None, help="Home team (e.g., NE).")
    ap.add_argument("--away", default=None, help="Away team (e.g., BAL).")
    ap.add_argument("--tail", type=int, default=1, help="Use last N rows if no game-id provided.")
    args = ap.parse_args()

    paths = get_paths(args.repo_root)
    models = load_models(paths)
    df = load_schedule(paths)
    if args.game_id:
        df = df[df["game_id"] == args.game_id].copy()
    elif args.home and args.away:
        # Try to find scheduled row first.
        df_sched = df[(df["home_team"] == args.home) & (df["away_team"] == args.away)].copy()
        if len(df_sched) > 0:
            df = df_sched.tail(1).copy()
        else:
            # Build a synthetic row from latest priors + latest home stadium.
            priors = load_priors(paths)
            stadium = load_stadium(paths)
            df = make_synthetic_row(args.home, args.away, priors, stadium)
    else:
        df = df.tail(args.tail).copy()

    if len(df) == 0:
        raise SystemExit("No rows found for prediction.")

    x = build_features(df)

    for _, row in x.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        preds = predict_row(row, models)
        pred_margin = preds["pred_margin"]
        pred_total = preds["pred_total"]
        p_home_win = preds["p_home_win"]
        pick_home = int(p_home_win >= 0.5)

        print(f"{away} @ {home}")
        print(f"Pred home win prob: {p_home_win:.3f}  (pick: {'HOME' if pick_home else 'AWAY'})")
        print(f"Pred spread (home margin): {pred_margin:.2f}  -> suggested line: {home} {(-pred_margin):.2f}")
        print(f"Pred total points: {pred_total:.2f}")


if __name__ == "__main__":
    main()
