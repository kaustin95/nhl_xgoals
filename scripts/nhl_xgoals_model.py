import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from nhl_xgoals_features import build_features, BASE_DIR

MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "documentation"
MODELS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


def chronological_split(X, y, shots):
    seasons = sorted(shots['season'].unique())
    if len(seasons) < 3:
        raise ValueError(f"Need at least 3 seasons, found {len(seasons)}")

    train_seasons = seasons[:-2]
    val_season = seasons[-2]
    test_season = seasons[-1]

    print(f"Train seasons: {train_seasons[0]} – {train_seasons[-1]}")
    print(f"Val season:    {val_season}")
    print(f"Test season:   {test_season}")

    train_idx = shots['season'].isin(train_seasons)
    val_idx = shots['season'] == val_season
    test_idx = shots['season'] == test_season

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


def train_model(X_train, y_train, X_val, y_val):
    model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=5,
        colsample_bytree=0.8,
        scale_pos_weight=9,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def print_diagnostics(model, X, y, label):
    """
    Probability-based metrics only. Accuracy at a 0.5 threshold is misleading
    for a 5% positive-class event — a naive 'always no goal' model scores 95%.
    AUC, log loss, and Brier score measure the full probability distribution.
    """
    proba = model.predict_proba(X)[:, 1]
    print(f"\n=== {label} ===")
    print(f"  Samples:      {len(y):,}")
    print(f"  ROC-AUC:      {roc_auc_score(y, proba):.4f}")
    print(f"  Log Loss:     {log_loss(y, proba):.4f}")
    print(f"  Brier Score:  {brier_score_loss(y, proba):.4f}")
    print(f"  Mean xG:      {proba.mean():.4f}  (actual goal rate: {y.mean():.4f})")
    return proba


def fit_calibration(model, X_val, y_val):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(model.predict_proba(X_val)[:, 1], y_val)
    return iso


def plot_calibration(raw_proba, cal_proba, y_true):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for proba, label, color in [
        (raw_proba, 'Raw XGBoost', 'steelblue'),
        (cal_proba, 'Calibrated (Isotonic)', 'darkorange'),
    ]:
        frac_pos, mean_pred = calibration_curve(
            y_true, proba, n_bins=20, strategy='quantile'
        )
        ax.plot(mean_pred, frac_pos, marker='o', label=label, color=color)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    ax.set_xlabel('Mean predicted xG')
    ax.set_ylabel('Observed goal rate')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(raw_proba, bins=60, alpha=0.6, label='Raw XGBoost', density=True,
            color='steelblue', range=(0, 0.8))
    ax.hist(cal_proba, bins=60, alpha=0.6, label='Calibrated', density=True,
            color='darkorange', range=(0, 0.8))
    ax.axvline(y_true.mean(), color='black', linestyle='--',
               label=f'Actual goal rate ({y_true.mean():.3f})')
    ax.set_xlabel('Predicted xG')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Predicted xG')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('NHL xGoals Model — Calibration Diagnostics', fontsize=13, y=1.02)
    plt.tight_layout()
    out = DOCS_DIR / "calibration_plot.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCalibration plot saved to {out}")


def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    top_n = min(20, len(feature_names))
    idx = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([feature_names[i] for i in idx], importance[idx], color='steelblue')
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title(f'Top {top_n} Feature Importances — XGBoost xGoals')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    out = DOCS_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {out}")


def build_game_xg(shots):
    game_xg = (
        shots.groupby(
            ['game_id', 'game_date', 'season', 'game_type', 'venue_name', 'home_team', 'away_team'],
            observed=True,
        )
        .apply(
            lambda df: pd.Series({
                'home_xGF':   df.loc[df['is_home_team'] == 1, 'xG'].sum(),
                'away_xGF':   df.loc[df['is_home_team'] == 0, 'xG'].sum(),
                'home_goals': df.loc[df['is_home_team'] == 1, 'is_goal'].sum(),
                'away_goals': df.loc[df['is_home_team'] == 0, 'is_goal'].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    return game_xg


def main():
    print("Loading and engineering features...")
    X, y, shots = build_features()
    print(f"  Total shots: {len(X):,}  |  Goal rate: {y.mean():.4f}")

    print("\nChronological split...")
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y, shots)
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    print("\nTraining XGBoost model...")
    model = train_model(X_train, y_train, X_val, y_val)
    print(f"  Best iteration: {model.best_iteration}")

    raw_train_proba = print_diagnostics(model, X_train, y_train, "Train")
    raw_val_proba   = print_diagnostics(model, X_val,   y_val,   "Validation")
    raw_test_proba  = print_diagnostics(model, X_test,  y_test,  "Test (held-out)")

    print("\nFitting isotonic calibration on validation set...")
    iso = fit_calibration(model, X_val, y_val)

    raw_all = model.predict_proba(X)[:, 1]
    cal_all = iso.transform(raw_all)

    shots = shots.copy()
    shots['xG_raw'] = raw_all
    shots['xG'] = cal_all

    print(f"\nPost-calibration (all data):")
    print(f"  Mean xG raw:        {shots['xG_raw'].mean():.4f}")
    print(f"  Mean xG calibrated: {shots['xG'].mean():.4f}")
    print(f"  Actual goal rate:   {shots['is_goal'].mean():.4f}")

    # Save model artifacts
    # pickle used instead of model.save_model() because the sklearn wrapper
    # errors on _estimator_type when early stopping has been used
    model_path = MODELS_DIR / "xgboost_xgoals.pkl"
    cal_path   = MODELS_DIR / "isotonic_calibrator.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(cal_path, "wb") as f:
        pickle.dump(iso, f)
    print(f"\nModel saved to      {model_path}")
    print(f"Calibrator saved to {cal_path}")

    # Plots
    plot_calibration(shots['xG_raw'].values, shots['xG'].values, y.values)
    plot_feature_importance(model, list(X.columns))

    # Game-level xG output
    game_xg = build_game_xg(shots)
    game_xg_path = BASE_DIR / "data" / "game_xg.parquet"
    game_xg.to_parquet(game_xg_path, index=False)
    print(f"Game-level xG saved to {game_xg_path}")
    print(f"\nDone. {len(game_xg):,} games processed.")


if __name__ == '__main__':
    main()
