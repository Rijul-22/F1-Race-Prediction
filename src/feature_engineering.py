import pandas as pd
import os

def load_clean(path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, '..', 'data', 'processed', 'cleaned_results.csv')
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    return df

def add_baseline_features(df):
    # Sort correctly — driver first, then chronological
    df.sort_values(by=["driver", "season", "round"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Driver recent form — rolling avg finish over last 5 races
    df['driver_recent_form'] = (
        df.groupby('driver')['finish_position']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Team performance — rolling avg finish over last 3 races
    df['team_performance'] = (
        df.groupby('team')['finish_position']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    print("Baseline features added: driver_recent_form, team_performance")
    return df

def add_dnf_probability(df):
    df.sort_values(by=["driver", "season", "round"], inplace=True)

    df['dnf_rate_last5'] = (
        df.groupby('driver')['dnf']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    print("Added: dnf_rate_last5")
    return df

def add_driver_points(df):
    df.sort_values(by=["driver", "season", "round"], inplace=True)

    df['driver_points_before_race'] = (
        df.groupby(['driver', 'season'])['points']
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    print("Added: driver_points_before_race")
    return df

def add_standings_position(df):
    df.sort_values(by=["season", "round", "driver"], inplace=True)

    df['driver_standings_pos'] = df.groupby(['season', 'round'])['driver_points_before_race']\
        .rank(ascending=False, method='min')

    print("Added: driver_standings_pos")
    return df

def save_features(df, path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, '..', 'data', 'features', 'features_v5.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")

if __name__ == "__main__":
    df = load_clean()
    df = add_baseline_features(df)
    df = add_dnf_probability(df)
    df = add_driver_points(df)
    df = add_standings_position(df)
    # Re-sort chronologically before saving
    df.sort_values(by=["season", "round", "driver"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    save_features(df)
    print("\nSample:")
    print(df[['driver', 'season', 'round', 'grid_position',
              'driver_recent_form', 'team_performance', 'finish_position']].head(10))