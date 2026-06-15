from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "nhl_play_by_play_data.parquet"

TEAM_ABBREV_TO_ID = {
    'NJD': 1,  'NYI': 2,  'NYR': 3,  'PHI': 4,  'PIT': 5,
    'BOS': 6,  'BUF': 7,  'MTL': 8,  'OTT': 9,  'TOR': 10,
    'CAR': 12, 'FLA': 13, 'TBL': 14, 'WSH': 15, 'CHI': 16,
    'DET': 17, 'NSH': 18, 'STL': 19, 'CGY': 20, 'COL': 21,
    'EDM': 22, 'VAN': 23, 'ANA': 24, 'DAL': 25, 'LAK': 26,
    'SJS': 28, 'CBJ': 29, 'MIN': 30, 'WPG': 52, 'VGK': 54,
    'SEA': 55, 'UTA': 59,
}

SHOT_EVENTS = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
GOAL_LINE_X = 89

BASE_FEATURE_COLS = [
    'x_adj', 'y_adj', 'shot_distance', 'shot_angle', 'is_rebound',
    'home_strength', 'away_strength', 'score_diff', 'seconds_in_period',
    'is_empty_net', 'is_playoff',
]


def load_shots(data_path=DATA_PATH):
    data = pd.read_parquet(data_path)
    data = data.sort_values(['game_id', 'event_id'])
    data['time_since_last_event'] = data.groupby('game_id')['event_id'].diff()

    shots = data[data['type_desc_key'].isin(SHOT_EVENTS)].copy()
    shots['is_goal'] = (shots['type_desc_key'] == 'goal').astype(int)
    shots['home_team_id'] = shots['home_team'].map(TEAM_ABBREV_TO_ID)
    shots['away_team_id'] = shots['away_team'].map(TEAM_ABBREV_TO_ID)
    return shots


def engineer_features(shots):
    shots = shots.copy()

    # Spatial features
    shots['x_adj'] = shots['x_coord'].abs()
    shots['y_adj'] = shots['y_coord'].abs()
    shots['shot_distance'] = np.sqrt(
        (GOAL_LINE_X - shots['x_adj']) ** 2 + shots['y_adj'] ** 2
    )
    shots['shot_angle'] = np.degrees(
        np.arctan2(shots['y_adj'], GOAL_LINE_X - shots['x_adj'])
    )

    # Shot context
    shots['is_rebound'] = (
        shots.groupby('game_id')['event_id'].diff() <= 3
    ).astype(int)
    shots['is_home_team'] = (
        shots['event_owner_team_id'] == shots['home_team_id']
    ).astype(int)

    # Game state
    shots['situation_str'] = shots['situation_code'].astype(str)
    shots['home_strength'] = pd.to_numeric(
        shots['situation_str'].str[-2], errors='coerce'
    )
    shots['away_strength'] = pd.to_numeric(
        shots['situation_str'].str[-1], errors='coerce'
    )
    shots['is_empty_net'] = shots['goalie_in_net_id'].isna().astype(int)
    shots['is_playoff'] = (shots['game_type'] == 3).astype(int)
    shots['seconds_in_period'] = (
        shots['time_in_period'].str.split(':').str[0].astype(int) * 60
        + shots['time_in_period'].str.split(':').str[1].astype(int)
    )

    # Score differential from the shooting team's perspective
    shots['home_score_before'] = (
        shots.groupby('game_id')['home_score'].shift(1).fillna(0)
    )
    shots['away_score_before'] = (
        shots.groupby('game_id')['away_score'].shift(1).fillna(0)
    )
    shots['score_diff'] = np.where(
        shots['is_home_team'] == 1,
        shots['home_score_before'] - shots['away_score_before'],
        shots['away_score_before'] - shots['home_score_before'],
    )

    shot_dummies = pd.get_dummies(shots['shot_type'], prefix='shot', dtype=int)

    X = pd.concat([shots[BASE_FEATURE_COLS], shot_dummies], axis=1)
    y = shots['is_goal']

    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask], shots[mask]


def build_features(data_path=DATA_PATH):
    shots = load_shots(data_path)
    return engineer_features(shots)


if __name__ == '__main__':
    X, y, shots = build_features()
    print(f"Shots: {len(X):,}")
    print(f"Goal rate: {y.mean():.4f}")
    print(f"Seasons: {sorted(shots['season'].unique())}")
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
