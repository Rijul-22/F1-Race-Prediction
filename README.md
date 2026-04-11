# 🏎️ F1 Race Prediction Model

A machine learning pipeline that predicts Formula 1 race finishing positions using historical data from the FastF1 API.

---

## 📌 Project Overview

This project builds a **lean, interpretable, and honest** F1 race prediction system. Unlike other approaches that predict lap times or use qualifying order as a proxy, this model:

- Predicts **actual finishing positions** (1–20) for each driver
- Trains on **2022–2024 seasons** and tests on **2025 season**
- Adds features **one at a time**, measuring improvement at each step
- Strictly avoids **data leakage** — no future information used

---

## 📁 Project Structure
F1 Race Prediction/
├── data/
│   ├── raw/          # FastF1 cache (gitignored)
│   ├── processed/    # Cleaned dataframes
│   └── features/     # Engineered feature tables
├── notebooks/        # EDA & experimentation
├── src/
│   ├── data_collection.py    # FastF1 data collection
│   ├── preprocessing.py      # Data cleaning
│   ├── feature_engineering.py # Feature computation
│   ├── train.py              # Model training & evaluation
│   └── evaluate.py           # Evaluation metrics
├── dashboard/
│   └── app.py        # Streamlit dashboard
├── outputs/
│   ├── models/       # Saved model files
│   ├── plots/        # EDA & result charts
│   └── reports/      # Final report assets
├── requirements.txt
└── README.md

---

## ⚙️ Setup
```bash
# Clone the repo
git clone https://github.com/Rijul-22/F1-Race-Prediction
cd F1-Race-Prediction

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Pipeline

Run scripts in this order:
```bash
# 1. Collect data (takes time — downloads 4 seasons)
python src/data_collection.py

# 2. Clean and preprocess
python src/preprocessing.py

# 3. Engineer features
python src/feature_engineering.py

# 4. Train and evaluate model
python src/train.py
```

---

## 📊 Features

Features are added **one at a time** and kept only if they improve MAE without causing data loss.

| Feature | Description | Status |
|---|---|---|
| `grid_position` | Starting grid position | ✅ Baseline |
| `driver_recent_form` | Rolling avg finish (last 5 races) | ✅ Kept |
| `team_performance` | Team rolling avg finish (last 3 races) | ✅ Kept |
| `dnf_rate_last5` | Driver DNF rate over last 5 races | ✅ Kept |
| `driver_points_before_race` | Cumulative driver points before each race | ✅ Kept |
| `driver_target_enc` | Driver target encoded past finishes | ✅ Kept |
| `team_target_enc` | Team target encoded past finishes | ✅ Kept |
| `gap_to_pole_norm` | Normalized gap to pole per race | ✅ Kept |
| `intra_season_form` | Rolling avg of current season (last 3 races) | ✅ Kept |
| `circuit_avg_finish` | Driver avg finish at specific circuit | ❌ Discarded (data loss) |

### Golden rule — no data leakage:
```python
# Always: sort → shift(1) → rolling
df.groupby('driver')['finish_position']
  .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
```

---

## 📈 Model Performance

| Model | MAE |
|---|---|
| Baseline — grid position only | ~2.780 |
| Linear Regression | ~2.338 |
| Random Forest | ~2.363 |
| XGBoost | ~2.410 |
| Stacking v1 (8 features) | ~2.352 |
| Stacking v2 (12 features + Optuna tuning) | ~2.375 |
| Delta Regression + Rank Normalization | **2.211** |

> MAE = Mean Absolute Error in finishing positions. 
> To ensure a high-fidelity continuous metric, drivers who DNF (Did Not Finish) are excluded from the test evaluation, and final predictions use a Delta Regression approach with in-race rank normalization.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `fastf1` | F1 data source |
| `pandas` | Data manipulation |
| `scikit-learn` | ML models & metrics |
| `xgboost` / `lightgbm` / `catboost` | Gradient boosting models |
| `optuna` | Hyperparameter tuning |
| `matplotlib/seaborn` | Visualisation |
| `streamlit` | Interactive Dashboard |

---

## 🗺️ Roadmap

- [x] Data collection & preprocessing
- [x] Baseline feature engineering
- [x] Linear Regression baseline model
- [x] Feature selection with Random Forest importance
- [x] Integration of XGBoost, LightGBM, and CatBoost
- [x] Hyperparameter tuning pipeline (Optuna)
- [x] Advanced methods (Delta Regression + Rank Normalization)
- [x] Interactive Streamlit dashboard
- [ ] Per-race prediction reports

---

## ⚠️ Limitations

- DNF outcomes add noise — a driver in P1 who retires is classified as P20
- 2025 rookies (Antonelli, Hadjar etc.) have limited historical data
- Weather and safety car events are not modelled

---

## 📜 License

MIT License