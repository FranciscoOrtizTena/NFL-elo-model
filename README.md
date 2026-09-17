# NFL Outcome Predictor — Elo + EPA

An NFL forecasting project that combines team Elo ratings, offensive and defensive
EPA, turnovers, and schedule context to estimate game outcomes. The workflow
covers feature engineering, exploratory analysis, model training, and a Streamlit
app for browsing and exporting weekly predictions.

The app provides projected winners, win probabilities, home spreads, total points,
and implied scores. Models stay fixed during the season while team ratings and
pregame inputs are refreshed as new results become available.

## Quick start

Use Python 3.12 or later. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead of the `source` command.

Open [the local app](http://localhost:8501). In GitHub Codespaces, open port **8501**
from the **Ports** panel. Stop the app with `Ctrl+C` in its terminal.

The repository includes saved Parquet files and model artifacts, so you can open
the app without rebuilding the dataset or training models first. Model artifacts
were saved with scikit-learn 1.8.0; use that version when reproducing those artifacts
if your environment reports a version mismatch. The dependency files currently
allow newer releases.

## Explore the app

The interface and CSV exports are in English.

### Weekly predictions

On startup, the app selects the latest season in the saved calendar and its first
week with pending games. Use **Season** and **Week** to browse another round.

The table displays:

| Column | Meaning |
| --- | --- |
| Matchup | Away team at home team |
| Status | `Upcoming` if either score is missing; `Result available` if both are recorded |
| Projected winner | Team selected by the win-probability model |
| Win probability | Probability assigned to the projected winner |
| Home spread | Model-implied line for the home team |
| Projected total | Expected combined points |

A **Date** column is also displayed if `gameday` exists in the input file. The
current feature-building script does not retain that column.

A partially completed week remains selected until all its games have scores in
the saved calendar. The app then selects the next pending week. If the latest
season has no pending games, it displays that season's final available week.

### Game breakdown

Choose a matchup below the table to see:

- Both team logos and team colors.
- Home and away labels with the projected score.
- A bar comparing both teams' win probabilities.
- The projected winner, winner probability, home spread, and total.
- The recorded score when results are available.

All 32 team logos are bundled in `assets/teams/`; selecting a game does not
download them. The table remains text-only. The layout includes mobile styling.

Select **Download weekly picks · CSV** to save the displayed predictions as
`predictions_<season>_week_<week>.csv`, or **Download weekly picks · PNG** to save
the same table as `predictions_<season>_week_<week>.png`. Exports are downloaded on demand; the app
does not maintain an automatic prediction archive.

### Custom matchup

Select **Custom matchup**, choose the home and away teams, and click
**Predict matchup**. This mode uses the last matching row in the saved calendar
when one exists. Otherwise, it builds a synthetic matchup using the latest saved
team priors and home-stadium information. Use **Weekly predictions** to select a
specific scheduled meeting.

## Read the predictions

The spread model predicts the home scoring margin:

```text
predicted_margin = expected home points - expected away points
home_spread = -predicted_margin
```

For example, a predicted home margin of `+3.0` is displayed as a home spread of
`-3.0`. A negative home spread favors the home team; a positive spread favors the
away team.

Projected scores combine the margin and total models:

```text
home_points = (predicted_total + predicted_margin) / 2
away_points = (predicted_total - predicted_margin) / 2
```

The winner comes from a separate logistic regression model. The probability shown
in the weekly table belongs to that selected team; the game breakdown shows both
teams' probabilities.

## Update data during the season

Starting Streamlit reads local data and runs inference. It does **not** download
results, rebuild team ratings, or retrain models.

To refresh the inputs:

```bash
python scripts/build_schedule_features.py --repo-root .
```

After the script finishes, reload Streamlit. The script overwrites:

- `data/schedule_features_2016_2025.parquet`
- `data/team_priors_latest.parquet`
- `data/team_stadium_latest.parquet`

**The schedule filename is retained for compatibility.** It can contain 2026
records despite ending in `2025`. The script currently requests schedules for
2016–2026 with `range(2016, 2027)`, using 2015 to initialize team priors.

Each run rebuilds the configured history. It requests team statistics only for
seasons where the downloaded calendar has at least one game with both scores
recorded. This avoids requesting a new season's statistics before it has any
recorded results. It is not an incremental download or an availability check for
the statistics file: if results arrive before that file is published, the request
can still return a `404`. Missing per-game statistics are not yet explicitly
validated by the builder.

The weekly workflow keeps the existing files in `data/models/` unchanged. You do
not need to rebuild the training table or retrain models to refresh predictions.
There is currently no scheduled updater or update button in the app; updates are
started with the command above.

## Data files and project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, weekly selection, matchup details, and CSV export |
| `assets/app.css` | App layout and visual styling |
| `assets/teams/` | Bundled logos, colors, and source metadata |
| `src/nfl_elo_model/` | Shared paths, schemas, validation, feature transforms, and inference |
| `scripts/` | Feature building, modeling-table construction, training, and CLI prediction |
| `analysis/` | Feature engineering, exploratory analysis, and modeling notebooks |
| `tests/` | Data-contract checks |

| Saved artifact | Contents |
| --- | --- |
| `data/schedule_features_2016_2025.parquet` | One row per scheduled game, with pregame ratings, context, and available results |
| `data/team_priors_latest.parquet` | One row per team with its latest post-game Elo, EPA, turnover rate, and `as_of_game_id` |
| `data/team_stadium_latest.parquet` | Most recent recorded home-game roof and surface for each team |
| `data/final_first_model.parquet` | Completed games with targets and engineered training features |
| `data/models/spread_ridge_timecv.joblib` | Ridge model for home margin |
| `data/models/total_ridge.joblib` | Ridge model for combined points |
| `data/models/home_win_logit.joblib` | Logistic regression model for home win probability |

Scheduled predictions use the pregame values in the selected schedule row.
`team_priors_latest.parquet` records the state after each team's last processed
game; `as_of_game_id` identifies that game.

## Run with Docker

The image includes the app, scripts, dependencies, styles, and logos. Parquet
files and model artifacts are excluded from the image by `.dockerignore`, so
mount the local `data/` directory when running it.

Build and run from the repository root (Bash):

```bash
docker build -t nfl-elo-ui .
docker run --rm --name nfl-elo-ui -p 8501:8501 \
  -v "$(pwd)/data:/app/data" nfl-elo-ui
```

Open [the local app](http://localhost:8501). Stop any other app using port 8501
before starting the container. To stop this container from another terminal:

```bash
docker stop nfl-elo-ui
```

You can refresh data using the same image in a separate, temporary container:

```bash
docker run --rm -v "$(pwd)/data:/app/data" nfl-elo-ui \
  python scripts/build_schedule_features.py --repo-root /app
```

Reload the app after the update completes. Rebuild the image when app code,
assets, or dependencies change. Data-only updates use the mounted folder and do
not require rebuilding the image. Docker installs `requirements.txt`; it does
not use `uv.lock`.

## Reproduce the analysis and training

These steps are separate from the routine data refresh. The examples below use
Bash; `PYTHONPATH=src` makes the package available to the modeling-table script
without installing it.

```bash
# Rebuild schedule features and latest team state.
python scripts/build_schedule_features.py --repo-root .

# Create the completed-game training table.
PYTHONPATH=src python scripts/build_model_table.py --repo-root .

# Train and evaluate using a temporal split, without replacing saved models.
python scripts/train_models.py --repo-root . --train-through 2023
```

In PowerShell, set `$env:PYTHONPATH = "src"` before running
`python scripts/build_model_table.py --repo-root .`.

The training table adds `home_win`, `margin`, and `total_points`, along with Elo,
EPA, turnover, and rest differentials and roof/surface flags. Training uses:

- Ridge regression with scaling and time-series cross-validation for spread.
- Ridge regression with scaling for total points.
- Logistic regression with scaling for home win probability.

The default split trains through 2023 and evaluates subsequent seasons in the
training table. The script reports margin/total MAE and win-model AUC, log loss,
Brier score, accuracy, and calibration points. Both train and test periods must
contain data; the script is not a separate full-history production-refit command.

To intentionally replace the three saved model artifacts, add `--save-models`:

```bash
python scripts/train_models.py --repo-root . --train-through 2023 --save-models
```

### Single-game CLI

Game IDs follow `SEASON_WEEK_AWAY_HOME`, with a two-digit week:

```bash
python scripts/predict_single_game.py --repo-root . --game-id 2026_01_NE_SEA
```

This example selects New England at Seattle in Week 1 of 2026. You can also pass
`--home` and `--away`; that mode uses the last matching schedule row or falls back
to a synthetic matchup, just like the app's custom mode.

### Notebooks

- `analysis/build_nfl_pregame_features.ipynb`: builds pregame ratings and features.
- `analysis/eda_feature_diagnostics.ipynb`: explores features and constructs the modeling table.
- `analysis/nfl_game_outcomes_modeling.ipynb`: trains and evaluates models, including optional gradient-boosting comparisons.

The scripts provide the command-line workflow; notebook cells retain their own
configuration and must be checked when reproducing a particular run.

## How team ratings are calculated

### Elo

Teams begin at `1500` before the 2015 bootstrap season. The schedule is replayed,
and subsequent games store each team's rating before incorporating their result.
The script maps historical franchise abbreviations to current ones for Elo and
the main schedule.

```text
expected_home = 1 / (1 + 10 ** (-(r_home - r_away + 65) / 400))
mov = ln(abs(home_score - away_score) + 1)
delta = K * mov * (home_outcome - expected_home)
r_home_new = r_home + delta
r_away_new = r_away - delta
```

`home_outcome` is `1` for a home win, `0.5` for a tie, and `0` for a home loss.
`K` is `20` for regular-season games and `24` for other game types.

For Week 1, the script regresses Elo toward the starting value:

```text
week1_elo = 1500 + 0.75 * (previous_elo - 1500)
```

This adjusted Elo is stored in the Week 1 schedule row. The latest-priors file
continues to hold the previous post-game state until a new result is processed.
Currently, before a team's Week 1 result is processed, later pending weeks use
its unadjusted prior Elo. Rebuilding after that result and its statistics are
available gives Week 2 the newly updated rating.

### EPA and turnovers

The builder uses these rates from team statistics:

```text
plays = attempts + carries
off_epa_pp = (passing_epa + rushing_epa) / plays
turnovers = passing_interceptions + rushing_fumbles_lost
            + receiving_fumbles_lost + sack_fumbles_lost
turnover_rate = turnovers / plays
```

Defensive EPA allowed is derived from opponents' offensive EPA and grouped by
`opponent_team`. Offensive EPA, defensive EPA allowed, and turnover rates are
initialized from 2015 and updated with exponential smoothing:

```text
new_prior = (1 - update_weight) * old_prior + update_weight * game_value
```

The update weight is `0.35` in weeks below 5 and `0.28` thereafter. These rates
carry over into the next season without an additional offseason adjustment.

## Validation

Run the existing data-contract checks against the local Parquet files:

```bash
python -m unittest discover -s tests -v
```

These checks verify required columns, nonempty tables, and core feature names.

## Data and logo sources

Schedules and team statistics are loaded through
[nflreadpy](https://nflreadpy.nflverse.com/). Team colors and logo URLs come from
its teams dataset. ESPN logos are bundled locally; source URLs and attribution
are recorded in [assets/teams/README.md](assets/teams/README.md) and
[assets/teams/metadata.json](assets/teams/metadata.json).
