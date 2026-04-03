import pandas as pd
import os

# ── 1. Load ───────────────────────────────────────────────────────
def load_raw(path=None):
    if path is None:
        base = os.path.dirname(__file__)  # src/ folder
        path = os.path.join(base, '..', 'data', 'processed', 'raw_results.csv')

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


# ── 2. Clean ──────────────────────────────────────────────────────
def clean(df):

    # Drop Q1/Q2/Q3 — all NaN, not useful
    df = df.drop(columns=['Q1', 'Q2', 'Q3'])

    # Rename to snake_case
    df = df.rename(columns={
        'DriverNumber'  : 'driver_number',
        'Abbreviation'  : 'driver',
        'FullName'      : 'driver_name',
        'TeamName'      : 'team',
        'GridPosition'  : 'grid_position',
        'Position'      : 'finish_position',
        'Points'        : 'points',
        'Status'        : 'status',
        'Season'        : 'season',
        'Round'         : 'round',
        'EventName'     : 'event_name',
        'CircuitName'   : 'circuit'
    })

    # DNF flag — anything that isn't Finished/Lapped/+N Laps
    finished_statuses = ['Finished', 'Lapped', '+1 Lap', '+2 Laps',
                         '+3 Laps', '+4 Laps', '+5 Laps']
    df['dnf'] = (~df['status'].isin(finished_statuses)).astype(int)

    # Fill missing grid_position (pit lane starts) with 20
    df['grid_position'] = df['grid_position'].fillna(20)

    # Fill missing finish_position for DNFs with 20 (last place)
    df['finish_position'] = df['finish_position'].fillna(20)

    # Convert to int
    df['grid_position']   = df['grid_position'].astype(int)
    df['finish_position'] = df['finish_position'].astype(int)

    # Drop "Did not start" rows — not useful for prediction
    df = df[df['status'] != 'Did not start'].reset_index(drop=True)

    print(f"After cleaning: {len(df)} rows")
    print(f"DNF count: {df['dnf'].sum()} ({df['dnf'].mean()*100:.1f}%)")
    return df


# ── 3. Save ───────────────────────────────────────────────────────
def save(df, path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, '..', 'data', 'processed', 'cleaned_results.csv')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")

# ── 4. Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    save(df)
    print("\nSample:")
    print(df.head())
    print("\nDtypes:")
    print(df.dtypes)