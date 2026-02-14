# NFL Pregame Modeling (Elo + EPA) — Analysis Game Plan

This repository contains an end-to-end NFL pregame modeling workflow:

1. Build **pregame** features (Elo + EPA + turnovers) for every game on the schedule.
2. Perform EDA and create a clean modeling table with outcome targets and differential features.
3. Train baseline models for spread, total points, and home win probability, then run a single-game prediction example.

The main workflow lives in `analysis/` and writes outputs to `data/`.
For reproducible runs without notebooks, use the scripts in `scripts/`.

## Folder Overview

- `analysis/`
  - Notebooks for feature building, EDA, and modeling.
- `data/`
  - Parquet outputs used across notebooks:
    - `data/schedule_features_2016_2025.parquet`
    - `data/final_first_model.parquet`
- `scripts/`
  - CLI scripts to build features and train models without notebooks.

## Notebooks (analysis/)

### `analysis/build_nfl_pregame_features.ipynb`

Creates the pregame feature table and exports it to:

- `data/schedule_features_2016_2025.parquet`

High-level steps:

- Pull schedules for 2016-2025 with `nflreadpy`.
- Pull team-level weekly stats for the same seasons with `nflreadpy`.
- Initialize team priors from the 2015 season (Elo + EPA + turnover rate).
- Walk forward game-by-game:
  - Save each team's **current** (pregame) priors into the game row.
  - If final scores exist, update priors using the game result and same-week stats.

### CLI (scripts/)

If you want to reproduce the full pipeline from scratch without notebooks:

1. Build schedule features directly from `nflreadpy`:
   - `python scripts/build_schedule_features.py --repo-root .`
2. Build the final modeling table:
   - `python scripts/build_model_table.py --repo-root .`
3. Train/evaluate baselines:
   - `python scripts/train_models.py --repo-root .`
4. Predict single games (scheduled or imaginary):
   - `python scripts/predict_single_game.py --repo-root . --home NE --away BAL`

### `analysis/eda_feature_diagnostics.ipynb`

Turns schedule-level features into a modeling table and exports it to:

- `data/final_first_model.parquet`

What it does:

- Loads `data/schedule_features_2016_2025.parquet`.
- Drops rows without final scores (so targets like margin/total are defined).
- Engineers targets:
  - `home_win` (binary)
  - `margin` (home_score - away_score)
  - `total_points` (home_score + away_score)
- Engineers key differential features used later in modeling:
  - `elo_diff`, `off_diff`, `def_allowed_diff`, `to_diff`, `rest_diff`
- Produces quick diagnostic plots/tables to understand calibration, ranges, and outliers.

### `analysis/nfl_game_outcomes_modeling.ipynb`

Trains and evaluates three modeling targets using a temporal split:

- Spread (home margin)
- Total points
- Home win probability

It loads:

- `data/final_first_model.parquet`

And uses:

- Ridge regression baselines (linear, regularized) for spread and total.
- Logistic regression baselines for home win probability.
- Optional gradient boosting benchmarks to sanity-check whether nonlinearity helps.

It also includes a single-game inference example using:

- `data/schedule_features_2016_2025.parquet`

## How Elo Is Built (Step-By-Step)

This project constructs a simple, transparent Elo that is updated sequentially.

### 1) Bootstrap (2015 -> 2016 prior)

- Every team starts with `elo_init = 1500` before 2015.
- The 2015 schedule is replayed to produce end-of-season ratings.
- Team abbreviations are normalized (e.g., `SD -> LAC`, `OAK -> LV`, `STL -> LA`) so priors match later seasons.

### 2) Pregame Elo Feature

For each 2016-2025 game row:

- `r_home` and `r_away` are recorded **before** updating with the game result.
- These stored values are what downstream notebooks treat as the pregame Elo feature.

### 3) Elo Update Formula

After a game with final score:

- Determine home outcome `s_home`:
  - `1.0` if home wins, `0.5` if tie, `0.0` if home loses
- Compute the expected score with a standard Elo logistic curve plus home-field advantage:

```
E_home = 1 / (1 + 10 ** ( - ( (r_home - r_away + elo_hfa) / 400 ) ))
```

- Apply a margin-of-victory multiplier:

```
mov = ln(|points_home - points_away| + 1)
```

- Update ratings (home increases when it outperforms expectation; away decreases symmetrically):

```
delta = K * mov * (s_home - E_home)
r_home_new = r_home + delta
r_away_new = r_away - delta
```

Parameters used:

- `elo_hfa = 65` (home-field advantage in Elo points)
- `K = 20` for regular season
- `K = 24` for non-regular games (a small increase for higher-stakes games)

### 4) Season-to-Season Regression

At the start of each season (`week == 1`), Elo is regressed toward the league mean:

```
r_week1 = elo_init + rho * (r_prev - elo_init)
```

With:

- `rho = 0.75`

This prevents ratings from drifting too far year-over-year and helps reflect roster/coach turnover.

## How EPA Features Are Built (Offense + Defense Allowed)

EPA inputs come from `nflreadpy` team-level stats and are converted into simple per-play rates.

### Offensive EPA per play (team strength)

For each team:

- Define plays as `attempts + carries`
- Define offensive EPA as `passing_epa + rushing_epa`
- Compute offensive efficiency:

```
off_epa_pp = (passing_epa + rushing_epa) / (attempts + carries)
```

This is stored separately for home and away teams as:

- `off_epa_pp_home`
- `off_epa_pp_away`

### Defensive EPA allowed per play (opponent perspective)

To approximate defensive quality, the same EPA signals are aggregated by `opponent_team`:

```
def_epa_allowed_pp = (passing_epa + rushing_epa) / (attempts + carries)
```

Grouped by `opponent_team` so the resulting value represents what that defense allowed.

This is stored as:

- `def_epa_pp_home`
- `def_epa_pp_away`

(Downstream EDA uses a differential form that compares the two.)

### Turnover rate (lost turnovers per play)

Turnovers are tracked as lost events per offensive play:

```
turnover_count =
  passing_interceptions
  + rushing_fumbles_lost
  + receiving_fumbles_lost
  + sack_fumbles_lost

turnover_rate = turnover_count / (attempts + carries)
```

Saved as:

- `turnover_rate_home`
- `turnover_rate_away`

### Updating EPA/turnovers Over Time (in-season smoothing)

The feature builder maintains running priors per team and updates them after games with a weighted blend:

- Early weeks (week < 5) use a higher update weight: `epa_lambda = 0.35`
- Later weeks use: `epa_lambda = 0.28`

Conceptually:

```
new_prior = (1 - epa_lambda) * old_prior + epa_lambda * current_week_value
```

This behaves like simple exponential smoothing: a stable prior that reacts faster early in the season and steadies later.

## Outputs (Data Contracts)

The notebooks produce and consume two main parquet files:

- `data/schedule_features_2016_2025.parquet`
  - One row per scheduled game
  - Contains pregame Elo/EPA/turnover priors for home/away plus schedule context fields
- `data/final_first_model.parquet`
  - Only completed games (final scores present)
  - Adds targets (`home_win`, `margin`, `total_points`) and differential features used for modeling
