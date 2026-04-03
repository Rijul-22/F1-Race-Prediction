import fastf1
import pandas as pd
import os
import time
# after session.load(...)
# ── 1. Cache setup ────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── 2. Config ─────────────────────────────────────────────────────
SEASONS = [2022, 2023, 2024, 2025]

# ── 3. Collect results ────────────────────────────────────────────
def collect_all_races(seasons=SEASONS):
    all_results = []

    for season in seasons:
        schedule = fastf1.get_event_schedule(season, include_testing=False)

        for _, event in schedule.iterrows():
            round_num = event['RoundNumber']
            gp_name   = event['EventName']

            # Skip rounds that haven't happened yet
            if round_num == 0:
                continue

            try:
                session = fastf1.get_session(season, round_num, 'R')
                session.load(telemetry=False, weather=False, messages=False)

                results = session.results[['DriverNumber', 'Abbreviation',
                                           'FullName', 'TeamName',
                                           'GridPosition', 'Position',
                                           'Points', 'Status', 'Q1', 'Q2', 'Q3']]
                results = results.copy()
                results['Season']      = season
                results['Round']       = round_num
                results['EventName']   = gp_name
                results['CircuitName'] = event['Location']

                all_results.append(results)
                print(f"  ✅ {season} Round {round_num}: {gp_name}")
                time.sleep(3)  # ← what value should go here?
            except Exception as e:
                print(f"  ❌ {season} Round {round_num}: {gp_name} — {e}")
                continue
    return pd.concat(all_results, ignore_index=True)


# ── 4. Save ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🏎️  Starting F1 data collection...\n")
    df = collect_all_races()

    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'raw_results.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n✅ Saved {len(df)} rows → {out_path}")
    print(df.head())