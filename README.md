# NHL Expected Goals (xG) Model

An expected goals model for NHL shots built with XGBoost, covering regular season and playoff data from 2012–13 to 2024–25.

Expected goals (xG) estimates the probability that a given shot results in a goal based on the shot's context — location, type, situation, and game state — independent of the shooter or goalie. This gives a more stable measure of shot quality than raw shot counts (whic are highly skewed).

## Results

| Split | Seasons | ROC-AUC | Log Loss | Brier Score |
|-------|---------|---------|----------|-------------|
| Train | 2012–13 to 2022–23 | 0.841 | 0.333 | 0.108 |
| Validation | 2023–24 | 0.834 | 0.322 | 0.105 |
| Test (held-out) | 2024–25 | 0.833 | 0.319 | 0.103 |

The model is evaluated using probability-based metrics rather than accuracy. With goals occurring on ~5% of shots, a naive "never predict a goal" classifier would score 95% accuracy — AUC, log loss, and Brier score measure the full predicted probability distribution instead.

Post-calibration, mean predicted xG (0.053) matches the observed goal rate (0.052).

## Project Structure

```
nhl_xgoals/
├── data/
│   ├── nhl_play_by_play_data.parquet   # Raw play-by-play input
│   └── game_xg.parquet                 # Output: per-game xG totals
├── scripts/
│   ├── nhl_xgoals_features.py          # Data loading and feature engineering
│   └── nhl_xgoals_model.py             # Model training, evaluation, and outputs
├── models/
│   ├── xgboost_xgoals.pkl              # Trained XGBoost classifier
│   └── isotonic_calibrator.pkl         # Isotonic regression calibrator
└── documentation/
    ├── calibration_plot.png            # Reliability diagram + xG distribution
    └── feature_importance.png          # Top 20 features by XGBoost gain
```

## Features

| Feature | Description |
|---------|-------------|
| `shot_distance` | Distance from goal (feet) |
| `shot_angle` | Angle from centre ice (degrees) |
| `x_adj`, `y_adj` | Absolute ice coordinates |
| `is_rebound` | Shot within 3 events of a prior shot |
| `shot_type_*` | One-hot encoded shot type (wrist, slap, etc.) |
| `home_strength` / `away_strength` | Skater counts from situation code |
| `is_empty_net` | Goalie not in net |
| `score_diff` | Score differential from the shooting team's perspective |
| `seconds_in_period` | Time elapsed in the period |
| `is_playoff` | Regular season vs. playoff |

## Methodology

**Train/val/test split** is chronological by season to prevent data leakage — the model never sees future seasons during training or validation.

**Calibration** is applied via isotonic regression fit on validation set predictions. Raw XGBoost probabilities are systematically overestimated (mean ~0.24 vs. true rate of ~0.05 due to `scale_pos_weight`); isotonic regression maps these to the true probability scale.

**`scale_pos_weight=9`** upweights goal events during training to help the model learn from the minority class. This is a training aid only — calibration corrects the output probabilities afterwards.

## Usage

```bash
# From the project root
python scripts/nhl_xgoals_model.py
```

This will:
1. Load and engineer features from `data/nhl_play_by_play_data.parquet`
2. Train an XGBoost model with early stopping on the validation season
3. Print diagnostics for train, validation, and test sets
4. Save the model and calibrator to `models/`
5. Save calibration and feature importance plots to `documentation/`
6. Save per-game xG totals to `data/game_xg.parquet`

## Dependencies

```
pandas
numpy
xgboost
scikit-learn
matplotlib
pyarrow
```
