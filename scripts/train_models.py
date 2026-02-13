from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow running without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nfl_elo_model.paths import get_paths
from nfl_elo_model.schema import MODEL_TABLE_SCHEMA
from nfl_elo_model.validate import require_columns, require_non_empty


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate spread, total, and win-prob baselines.")
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root (default: repo root)")
    ap.add_argument("--train-through", type=int, default=2023, help="Last season included in train split.")
    ap.add_argument("--save-models", action="store_true", help="Save trained models to data/models/")
    args = ap.parse_args()

    paths = get_paths(args.repo_root)
    df = pd.read_parquet(paths.model_table_path)
    require_non_empty(df, "final_first_model")
    require_columns(df, MODEL_TABLE_SCHEMA)

    train = df[df["season"] <= args.train_through].copy()
    test = df[df["season"] > args.train_through].copy()
    print("Train rows:", len(train), "| Test rows:", len(test))

    # ----------------
    # Spread (margin)
    # ----------------
    feat_spread = ["elo_diff", "off_diff", "def_allowed_diff", "to_diff", "rest_diff", "div_game"]
    Xtr_s, ytr_s = train[feat_spread], train["margin"]
    Xte_s, yte_s = test[feat_spread], test["margin"]

    base_spread = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    base_spread.fit(Xtr_s, ytr_s)
    pred = base_spread.predict(Xte_s)
    print("SPREAD Ridge | MAE:", round(mean_absolute_error(yte_s, pred), 3))

    # Time-aware tuning.
    train_s = train.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    gs = GridSearchCV(pipe, {"ridge__alpha": [0.1, 0.3, 1, 3, 10, 30, 100]}, scoring="neg_mean_absolute_error", cv=tscv)
    gs.fit(train_s[feat_spread], train_s["margin"])
    tuned_spread = gs.best_estimator_
    pred = tuned_spread.predict(Xte_s)
    print("SPREAD Tuned Ridge (TimeCV) | MAE:", round(mean_absolute_error(yte_s, pred), 3), "| best:", gs.best_params_)

    # ----------------
    # Total points
    # ----------------
    df2 = df.copy()
    df2["off_sum"] = df2["off_epa_pp_home"] + df2["off_epa_pp_away"]
    df2["def_sum_allowed"] = df2["def_epa_pp_home"] + df2["def_epa_pp_away"]
    df2["to_sum"] = df2["turnover_rate_home"] + df2["turnover_rate_away"]
    df2["abs_elo_diff"] = df2["elo_diff"].abs()

    train_t = df2[df2["season"] <= args.train_through].copy()
    test_t = df2[df2["season"] > args.train_through].copy()

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
    Xtr_t, ytr_t = train_t[feat_total], train_t["total_points"]
    Xte_t, yte_t = test_t[feat_total], test_t["total_points"]

    total = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
    total.fit(Xtr_t, ytr_t)
    pred = total.predict(Xte_t)
    print("TOTAL Ridge | MAE:", round(mean_absolute_error(yte_t, pred), 3))

    # -----------------------
    # Home win probability
    # -----------------------
    ytr_w = train["home_win"].astype(int)
    yte_w = test["home_win"].astype(int)
    feat_win = feat_spread

    win_model = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))])
    win_model.fit(train[feat_win], ytr_w)
    p = win_model.predict_proba(test[feat_win])[:, 1]
    pred = (p >= 0.5).astype(int)

    print(
        "WIN Logit | AUC:",
        round(roc_auc_score(yte_w, p), 4),
        "| LogLoss:",
        round(log_loss(yte_w, p), 4),
        "| Brier:",
        round(brier_score_loss(yte_w, p), 4),
        "| Acc:",
        round(accuracy_score(yte_w, pred), 4),
    )

    frac_pos, mean_pred = calibration_curve(yte_w, p, n_bins=10, strategy="quantile")
    print("Calibration points (mean_pred, frac_pos):", list(zip(np.round(mean_pred, 3), np.round(frac_pos, 3))))

    if args.save_models:
        models_dir = paths.data_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(tuned_spread, models_dir / "spread_ridge_timecv.joblib")
        joblib.dump(total, models_dir / "total_ridge.joblib")
        joblib.dump(win_model, models_dir / "home_win_logit.joblib")
        print(f"Saved models to: {models_dir}")


if __name__ == "__main__":
    main()
