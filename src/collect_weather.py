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
                session.load(telemetry=False, weather=True, messages=False, laps=False)

                wd = session.weather_data
                is_wet = int(bool(wd['Rainfall'].any()))
                air_temp = float(wd['AirTemp'].mean())
                all_results.append({
                    'season': season,
                    'round': round_num,
                    'is_wet_race': is_wet,
                    'air_temp': air_temp
                })
                print(f"  ✅ {season} Round {round_num}: {gp_name}")
                time.sleep(4)

            except Exception as e:
                print(f"  ❌ {season} Round {round_num}: {gp_name} — {e}")
                continue
    return pd.DataFrame(all_results)

# ── 4. Save ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🏎️  Starting F1 data collection...\n")
    df = collect_all_races()

    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'weather_data.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n✅ Saved {len(df)} rows → {out_path}")
    print(df.head())