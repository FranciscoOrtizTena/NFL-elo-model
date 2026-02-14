from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow running without installing the package.
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nfl_elo_model.paths import get_paths
from nfl_elo_model.predict import (
    TEAM_NAMES,
    build_features,
    load_models,
    load_priors,
    load_schedule,
    load_stadium,
    make_synthetic_row,
    predict_row,
)


def main() -> None:
    st.set_page_config(page_title="NFL Outcome Predictor", layout="centered")
    st.title("NFL Outcome Predictor")

    paths = get_paths(".")

    try:
        models = load_models(paths)
    except Exception as exc:
        st.error(f"Failed to load models in data/models: {exc}")
        st.stop()

    schedule = load_schedule(paths)
    priors = load_priors(paths)
    stadium = load_stadium(paths)

    st.markdown(
        """
        <style>
        .matchup-card {
            background: #0f172a;
            color: #f8fafc;
            padding: 16px 20px;
            border-radius: 12px;
            margin: 8px 0 16px 0;
            border: 1px solid #1e293b;
        }
        .subtle {
            color: #94a3b8;
            font-size: 0.9rem;
        }
        .pick {
            font-weight: 600;
            color: #22c55e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio("Prediction mode", ["Game ID", "Imaginary matchup"])

    if mode == "Game ID":
        game_id = st.text_input("Game ID", "")
        if st.button("Predict"):
            df = schedule[schedule["game_id"] == game_id].copy()
            if len(df) == 0:
                st.error("Game ID not found in schedule_features.")
                st.stop()
            x = build_features(df)
            row = x.iloc[0]
            preds = predict_row(row, models)

            home = row["home_team"]
            away = row["away_team"]
            home_name = TEAM_NAMES.get(home, home)
            away_name = TEAM_NAMES.get(away, away)
            p_home = preds["p_home_win"]
            pick = home_name if p_home >= 0.5 else away_name
            pred_margin = preds["pred_margin"]
            pred_total = preds["pred_total"]
            home_pts = (pred_total + pred_margin) / 2.0
            away_pts = pred_total - home_pts

            st.markdown(
                f"""
                <div class="matchup-card">
                  <div><strong>{away_name}</strong> @ <strong>{home_name}</strong></div>
                  <div class="subtle">Source: schedule row</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**Projected Winner:** <span class='pick'>{pick}</span>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Home win probability", f"{p_home:.3f}")
            c2.metric("Pred spread (home margin)", f"{pred_margin:.2f}")
            c3.metric("Pred total points", f"{pred_total:.2f}")

            st.markdown(
                f"**Implied score:** {away_name} {away_pts:.1f} — {home_name} {home_pts:.1f}"
            )

    else:
        teams = sorted(TEAM_NAMES.keys())
        home = st.selectbox("Home team", teams, format_func=lambda x: TEAM_NAMES.get(x, x))
        away_options = [t for t in teams if t != home]
        away = st.selectbox("Away team", away_options, format_func=lambda x: TEAM_NAMES.get(x, x))
        if st.button("Predict"):
            df = schedule[(schedule["home_team"] == home) & (schedule["away_team"] == away)].copy()
            if len(df) == 0:
                df = make_synthetic_row(home, away, priors, stadium)
                source = "latest priors + latest home stadium"
            else:
                df = df.tail(1).copy()
                source = "schedule row"
            x = build_features(df)
            row = x.iloc[0]
            preds = predict_row(row, models)

            home_name = TEAM_NAMES.get(home, home)
            away_name = TEAM_NAMES.get(away, away)
            p_home = preds["p_home_win"]
            pick = home_name if p_home >= 0.5 else away_name
            pred_margin = preds["pred_margin"]
            pred_total = preds["pred_total"]
            home_pts = (pred_total + pred_margin) / 2.0
            away_pts = pred_total - home_pts

            st.markdown(
                f"""
                <div class="matchup-card">
                  <div><strong>{away_name}</strong> @ <strong>{home_name}</strong></div>
                  <div class="subtle">Source: {source}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**Projected Winner:** <span class='pick'>{pick}</span>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Home win probability", f"{p_home:.3f}")
            c2.metric("Pred spread (home margin)", f"{pred_margin:.2f}")
            c3.metric("Pred total points", f"{pred_total:.2f}")

            st.markdown(
                f"**Implied score:** {away_name} {away_pts:.1f} — {home_name} {home_pts:.1f}"
            )


if __name__ == "__main__":
    main()
