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

def add_weighted_form(df):
    df.sort_values(by=["driver", "season", "round"], inplace=True)

    df['weighted_finish_form'] = (
        df.groupby('driver')['finish_position']
        .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )

    print("Added: weighted_finish_form")
    return df

def add_position_gain(df):
    df.sort_values(by=["season", "round", "driver"], inplace=True)

    df['position_gain'] = df['grid_position'] - df['finish_position']
    df['avg_position_gain'] = (
        df.groupby('driver')['position_gain']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    print("Added: position_gain")
    return df

def add_driver_vs_field(df):
    df.sort_values(by=["driver", "season", "round"], inplace=True)
    race_avg = df.groupby(['season', 'round'])['finish_position'].transform('mean')
    df['driver_vs_field'] = df['finish_position'] - race_avg
    df['driver_vs_field'] = (
        df.groupby('driver')['driver_vs_field']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    print("Added: driver_vs_field")
    return df

def add_constructor_standings(df):
    df.sort_values(by=["team", "season", "round"], inplace=True)

    df['team_points_before_race'] = (
        df.groupby(['team', 'season'])['points']
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    df.sort_values(by=["season", "round", "team"], inplace=True)

    df['constructor_standings_pos'] = df.groupby(['season', 'round'])['team_points_before_race']\
        .rank(ascending=False, method='min')

    print("Added: team_points_before_race, constructor_standings_pos")
    return df

def add_qualifying_gap(df):
    base = os.path.dirname(__file__)
    qual_path = os.path.join(base, '..', 'data', 'processed', 'qualifying_data.csv')
    qual = pd.read_csv(qual_path)
    qual = qual.rename(columns={'Driver': 'driver'})

    df = df.merge(qual[['season', 'round', 'driver', 'gap_to_pole']],
                  on=['season', 'round', 'driver'],
                  how='left')

    # Fill missing with per-race median
    df['gap_to_pole'] = (
        df.groupby(['season', 'round'])['gap_to_pole']
        .transform(lambda x: x.fillna(x.median()))
    )

    print(f"Added: gap_to_pole | nulls remaining: {df['gap_to_pole'].isnull().sum()}")
    return df

def add_teammate_delta(df):
    df.sort_values(by=["driver", "season", "round"], inplace=True)

    team_avg = df.groupby(['season', 'round', 'team'])['finish_position'].transform('mean')
    df['teammate_delta'] = df['finish_position'] - team_avg

    df['teammate_delta'] = (
        df.groupby('driver')['teammate_delta']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    print("Added: teammate_delta")
    return df

def add_weather(df):
    base = os.path.dirname(__file__)
    weather_path = os.path.join(base, '..', 'data', 'processed', 'weather_data.csv')
    weather = pd.read_csv(weather_path)

    df = df.merge(weather, on=['season', 'round'], how='left')

    print(f"Added: is_wet_race, air_temp | nulls: {df[['is_wet_race', 'air_temp']].isnull().sum().sum()}")
    return df

def save_features(df, path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, '..', 'data', 'features', 'features_v11.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")

if __name__ == "__main__":
    df = load_clean()
    df = add_baseline_features(df)
    df = add_dnf_probability(df)
    df = add_driver_points(df)
    df = add_standings_position(df)
    df = add_weighted_form(df)
    df = add_position_gain(df)
    df = add_driver_vs_field(df)
    df = add_constructor_standings(df)
    df = add_qualifying_gap(df)
    df = add_teammate_delta(df)
    df = add_weather(df)
    # Re-sort chronologically before saving
    df.sort_values(by=["season", "round", "driver"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    save_features(df)
    print("\nSample:")
    print(df[['driver', 'season', 'round', 'grid_position',
              'driver_recent_form', 'team_performance', 'finish_position']].head(10))