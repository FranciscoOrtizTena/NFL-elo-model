from __future__ import annotations

import sys
import base64
import json
import math
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_elo_model.paths import get_paths
from nfl_elo_model.export_image import weekly_predictions_png
from nfl_elo_model.predict import (
    TEAM_NAMES, build_features, load_models, load_priors, load_schedule,
    load_stadium, make_synthetic_row, predict_row,
)


def default_round(schedule: pd.DataFrame) -> tuple[int, int]:
    """Find the first pending round in the latest available season."""
    season = int(schedule["season"].max())
    current = schedule[schedule["season"] == season]
    pending = current[current[["home_score", "away_score"]].isna().any(axis=1)]
    week = int(pending["week"].min() if not pending.empty else current["week"].max())
    return season, week


def matchup(row: pd.Series) -> str:
    return f"{TEAM_NAMES.get(row['away_team'], row['away_team'])} @ {TEAM_NAMES.get(row['home_team'], row['home_team'])}"


def team_identity(team: str) -> tuple[str, str]:
    """Use bundled logos so matchup details do not need network requests."""
    logo = REPO_ROOT / "assets" / "teams" / f"{team}.png"
    metadata = REPO_ROOT / "assets" / "teams" / "metadata.json"
    colors = json.loads(metadata.read_text()) if metadata.exists() else {}
    color = colors.get(team, {}).get("color", "#475569")
    if not logo.exists():
        return f'<span class="team-fallback">{escape(team)}</span>', color
    encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
    name = escape(TEAM_NAMES.get(team, team), quote=True)
    return f'<img class="team-logo" src="data:image/png;base64,{encoded}" alt="{name} logo">', color


def pick_outcome(row: pd.Series, winner: str) -> str:
    """Grade the predicted winner only when a complete, non-tied result exists."""
    if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
        return "pending"
    if row["home_score"] == row["away_score"]:
        return "tie"
    actual = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
    return "correct" if winner == actual else "incorrect"


def rounded_score(score: float) -> int:
    """Round projected points to the nearest integer, with halves rounded up."""
    lower = math.floor(score)
    return lower if score - lower < 0.5 else math.ceil(score)


def weekly_table_html(table: pd.DataFrame, outcomes: list[str] | None = None) -> str:
    """Render an accessible weekly scoreboard with escaped schedule values."""
    headers = "".join(f'<th scope="col">{escape(column)}</th>' for column in table.columns)
    rows = []
    for position, (_, record) in enumerate(table.iterrows()):
        cells = []
        for column, value in record.items():
            content = escape(f"{value:.2f}" if column == "Projected total" else str(value))
            css = ""
            if column == "Status":
                state = "final" if value == "Final" else "pending"
                content = f'<span class="game-status {state}">{content}</span>'
            elif column in ("Result", "Projected score"):
                css = ' class="result-cell"'
                if value == "—":
                    content = '<span class="pending-result" aria-label="Result pending">—</span>'
            elif column == "Win probability":
                probability = float(str(value).rstrip("%"))
                content = f'<span class="win-value">{content}</span><span class="win-track" aria-hidden="true"><span style="width:{probability:.1f}%"></span></span>'
            elif column in ("Home spread", "Projected total"):
                css = ' class="numeric-cell"'
            elif column == "Projected winner":
                outcome = outcomes[position] if outcomes is not None else "pending"
                labels = {"correct": "Correct pick", "incorrect": "Incorrect pick", "pending": "Result pending", "tie": "Game tied"}
                css = f' class="pick-cell {outcome}"'
                content = f'<span title="{labels[outcome]}">{content}</span>'
            cells.append(f"<td{css}>{content}</td>")
        rows.append(f'<tr>{"".join(cells)}</tr>')
    return f'<div class="weekly-table-wrap" role="region" aria-label="Weekly matchups and results" tabindex="0"><table class="weekly-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def show_prediction(row: pd.Series, preds: dict[str, float]) -> None:
    home, away = row["home_team"], row["away_team"]
    p_home = preds["p_home_win"]
    winner = home if p_home >= 0.5 else away
    margin, total = preds["pred_margin"], preds["pred_total"]
    home_logo, home_color = team_identity(home)
    away_logo, away_color = team_identity(away)
    home_name, away_name = (escape(TEAM_NAMES.get(t, t)) for t in (home, away))
    st.markdown(f"""
<div class="matchup-panel" style="--away-color:{away_color};--home-color:{home_color}">
  <div class="panel-label">MATCHUP SPOTLIGHT <span>PROJECTED SCORE</span></div>
  <div class="scoreboard">
    <div class="team-block">{away_logo}<div class="venue-label">AWAY</div><div class="team-name">{away_name}</div><div class="projected-score">{rounded_score((total - margin) / 2)}</div></div>
    <div class="versus">AT</div>
    <div class="team-block">{home_logo}<div class="venue-label">HOME</div><div class="team-name">{home_name}</div><div class="projected-score">{rounded_score((total + margin) / 2)}</div></div>
  </div>
  <div class="probability-labels"><span>{escape(away)} <strong>{1 - p_home:.1%}</strong></span><span class="probability-title">WIN PROBABILITY</span><span>{escape(home)} <strong>{p_home:.1%}</strong></span></div>
  <div class="probability-track" role="img" aria-label="{away_name} win probability {1 - p_home:.1%}; {home_name} win probability {p_home:.1%}"><div style="width:{(1 - p_home) * 100:.3f}%"></div></div>
</div>
""", unsafe_allow_html=True)
    st.markdown(f"**Projected winner: {TEAM_NAMES.get(winner, winner)}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Winner probability", f"{max(p_home, 1 - p_home):.1%}")
    c2.metric("Home spread", f"{home} {-margin:+.2f}")
    c3.metric("Projected total", f"{total:.2f}")
    st.caption("Home spread: a negative line favors the home team; a positive line favors the away team.")
    if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
        st.success(f"Recorded result: {away} {row['away_score']:.0f} — {home} {row['home_score']:.0f}")


def main() -> None:
    st.set_page_config(page_title="NFL Outcome Predictor", page_icon="🏈", layout="wide")
    st.markdown(f"<style>{(REPO_ROOT / 'assets' / 'app.css').read_text()}</style>", unsafe_allow_html=True)
    st.markdown("""
<div class="app-hero"><div class="eyebrow">NFL · WEEKLY FORECAST</div>
<h1>NFL Outcome Predictor<span class="accent-dot">.</span></h1>
<p>Every matchup. A clearer picture of game day.</p>
<div class="hero-tags"><span>WIN PROBABILITIES</span><span>PROJECTED SPREADS</span><span>EXPECTED SCORES</span></div></div>
""", unsafe_allow_html=True)
    paths = get_paths(REPO_ROOT)
    try:
        models = load_models(paths)
        schedule = load_schedule(paths)
    except Exception as exc:
        st.error(f"Unable to load data or models: {exc}")
        st.stop()
    if schedule.empty:
        st.info("No games found in the saved schedule.")
        return

    mode = st.radio("Prediction mode", ["Weekly predictions", "Custom matchup"], horizontal=True)
    if mode == "Custom matchup":
        teams = sorted(TEAM_NAMES)
        home = st.selectbox("Home team", teams, format_func=lambda t: TEAM_NAMES[t])
        away = st.selectbox("Away team", [t for t in teams if t != home], format_func=lambda t: TEAM_NAMES[t])
        if st.button("Predict matchup"):
            df = schedule[(schedule["home_team"] == home) & (schedule["away_team"] == away)].tail(1)
            if df.empty:
                df = make_synthetic_row(home, away, load_priors(paths), load_stadium(paths))
            row = build_features(df).iloc[0]
            show_prediction(row, predict_row(row, models))
        return

    season_default, week_default = default_round(schedule)
    seasons = sorted(schedule["season"].astype(int).unique(), reverse=True)
    c1, c2 = st.columns(2)
    season = c1.selectbox("Season", seasons, index=seasons.index(season_default))
    season_rows = schedule[schedule["season"] == season]
    weeks = sorted(season_rows["week"].astype(int).unique())
    selected_default = week_default if season == season_default else weeks[0]
    week = c2.selectbox("Week", weeks, index=weeks.index(selected_default), key=f"week_{season}")
    games = season_rows[season_rows["week"] == week].copy()
    sort_cols = [c for c in ["gameday", "gametime", "game_id"] if c in games]
    games = games.sort_values(sort_cols)
    completed = games[["home_score", "away_score"]].notna().all(axis=1)
    st.subheader(f"Week {week} · {season} season")
    st.caption(f"{len(games)} games · {int((~completed).sum())} upcoming · {int(completed.sum())} with results · Based on the saved schedule")
    if season == season_default and not season_rows[["home_score", "away_score"]].isna().any(axis=1).any():
        st.info("No upcoming games in this season. Showing the latest available week.")

    features = build_features(games)
    predictions = {}
    records = []
    outcomes = []
    errors = []
    for _, row in features.iterrows():
        try:
            pred = predict_row(row, models)
        except ValueError:
            errors.append(matchup(row))
            continue
        predictions[row["game_id"]] = pred
        p_home = pred["p_home_win"]
        winner = row["home_team"] if p_home >= 0.5 else row["away_team"]
        outcomes.append(pick_outcome(row, winner))
        record = {
            "Matchup": matchup(row),
            "Status": "Final" if completed.loc[row.name] else "Upcoming",
            "Result": (
                f"{row['away_team']} {row['away_score']:.0f} — {row['home_team']} {row['home_score']:.0f}"
                if completed.loc[row.name] else "—"
            ),
            "Projected score": (
                f"{row['away_team']} {rounded_score((pred['pred_total'] - pred['pred_margin']) / 2)}"
                f" — {row['home_team']} {rounded_score((pred['pred_total'] + pred['pred_margin']) / 2)}"
            ),
            "Projected winner": TEAM_NAMES.get(winner, winner),
            "Win probability": f"{max(p_home, 1 - p_home):.1%}",
            "Home spread": f"{row['home_team']} {-pred['pred_margin']:+.2f}",
            "Projected total": round(pred["pred_total"], 2),
        }
        if "gameday" in games:
            record = {"Date": row["gameday"], **record}
        records.append(record)
    if errors:
        st.warning("Predictions unavailable for these games; check their input data: " + "; ".join(errors))
    if not records:
        return
    table = pd.DataFrame(records)
    st.markdown(weekly_table_html(table, outcomes), unsafe_allow_html=True)
    st.caption("Result: away team score — home team score. A dash means no complete result is saved yet.")
    st.caption("Projected score: model estimates rounded to whole points (.5 rounds up), away — home. Projected winner: green = correct pick; red = incorrect pick; neutral = pending or tied.")
    st.caption("Home spread: a negative line favors the home team; a positive line favors the away team.")
    csv_col, png_col = st.columns(2)
    csv_col.download_button("Download weekly picks · CSV", table.to_csv(index=False).encode("utf-8-sig"), file_name=f"predictions_{season}_week_{week:02d}.csv", mime="text/csv")
    png_col.download_button("Download weekly picks · PNG", weekly_predictions_png(table, season, week), file_name=f"predictions_{season}_week_{week:02d}.png", mime="image/png")

    st.divider()
    st.subheader("Game breakdown")
    available = features[features["game_id"].isin(predictions)].set_index("game_id", drop=False)
    ids = available.index.tolist()
    pending_ids = available.index[available[["home_score", "away_score"]].isna().any(axis=1)].tolist()
    selected = st.selectbox("Choose a matchup", ids, index=ids.index(pending_ids[0]) if pending_ids else 0, format_func=lambda gid: matchup(available.loc[gid]), key=f"game_{season}_{week}")
    show_prediction(available.loc[selected], predictions[selected])


if __name__ == "__main__":
    main()
