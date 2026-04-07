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
                session = fastf1.get_session(season, round_num, 'Q')
                session.load(telemetry=False, weather=False, messages=False)

                laps = session.laps
                best_laps = laps.groupby('Driver')['LapTime'].min().reset_index()
                pole_time = best_laps['LapTime'].min()
                best_laps['gap_to_pole'] = (best_laps['LapTime'] - pole_time).dt.total_seconds()
                best_laps['season'] = season
                best_laps['round'] = round_num
                best_laps = best_laps[['Driver', 'gap_to_pole', 'season', 'round']]

                all_results.append(best_laps)
                print(f"  ✅ {season} Round {round_num}: {gp_name}")
                time.sleep(4)

            except Exception as e:
                print(f"  ❌ {season} Round {round_num}: {gp_name} — {e}")
                continue
    return pd.concat(all_results, ignore_index=True)


# ── 4. Save ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🏎️  Starting F1 data collection...\n")
    df = collect_all_races()

    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'qualifying_data.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n✅ Saved {len(df)} rows → {out_path}")
    print(df.head())