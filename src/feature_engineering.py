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

def save_features(df, path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, '..', 'data', 'features', 'features_v2.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")

def add_circuit_performance(df):
    df.sort_values(by=["driver", "circuit", "season", "round"], inplace=True)

    df['circuit_avg_finish'] = (
        df.groupby(['driver', 'circuit'])['finish_position']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    print("Added: circuit_avg_finish")
    return df

if __name__ == "__main__":
    df = load_clean()
    df = add_baseline_features(df)
    df = add_circuit_performance(df)
    # Re-sort chronologically before saving
    df.sort_values(by=["season", "round", "driver"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    save_features(df)
    print("\nSample:")
    print(df[['driver', 'season', 'round', 'grid_position',
              'driver_recent_form', 'team_performance', 'finish_position']].head(10))