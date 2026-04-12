# 🏎️ F1 Race Position Prediction

A machine learning pipeline that predicts Formula 1 race finishing positions (1–20) using historical data from the FastF1 API.

🔗 **Live Dashboard:** [f1-race-prediction-rijul-mittal.streamlit.app](https://f1-race-prediction-rijul-mittal.streamlit.app/)

---

## 📌 Project Overview

This project builds a **lean, interpretable, and honest** F1 race prediction system. Unlike other approaches that predict lap times or use qualifying order as a proxy, this model:

- Predicts **actual finishing positions (1–20)** for each driver
- Trains on **2022–2024 seasons** and tests on the **unseen 2025 season**
- Adds features **one at a time**, measuring MAE improvement at each step
- Strictly avoids **data leakage** — no future information used at training time
- Uses a **Delta Regression + Rank Normalisation** approach as the best model

---

## 📁 Project Structure

```
F1-Race-Prediction/
├── data/
│   ├── raw/                      # FastF1 cache (gitignored)
│   ├── processed/                # Cleaned dataframes
│   └── features/                 # Engineered feature tables (features_v10.csv)
├── src/
│   ├── data_collection.py        # FastF1 data collection (2022–2025)
│   ├── preprocessing.py          # Data cleaning & DNF handling
│   ├── feature_engineering.py    # Feature computation (v10)
│   ├── collect_qualifying.py     # gap_to_pole per race
│   ├── train.py                  # Model training, stacking & delta regression
│   ├── hyperparameter_tuning.py  # GridSearchCV tuning for all models
│   └── eda.py                    # EDA visualisations (5 plots)
├── dashboard/
│   └── app.py                    # 4-tab Streamlit dashboard
├── outputs/                      # Plots, charts, report assets
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/Rijul-22/F1-Race-Prediction
cd F1-Race-Prediction

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Pipeline

Run scripts in this order:

```bash
# 1. Collect race data (downloads 4 seasons — takes time)
python src/data_collection.py

# 2. Collect qualifying gap-to-pole data
python src/collect_qualifying.py

# 3. Clean and preprocess
python src/preprocessing.py

# 4. Engineer features
python src/feature_engineering.py

# 5. Hyperparameter tuning (GridSearchCV, 5-fold CV)
python src/hyperparameter_tuning.py

# 6. Train and evaluate all models
python src/train.py

# 7. Launch dashboard
streamlit run dashboard/app.py
```

---

## 📊 Features

Features are added **one at a time** and kept only if they improve MAE without causing data loss. Final dataset: **1,829 rows · 8 features · 0 missing values**.

| Feature | Description | Status |
|---|---|---|
| `grid_position` | Starting grid position (qualifying proxy) | ✅ Kept |
| `weighted_finish_form` | Exponentially-weighted rolling 5-race mean finish (shift(1) applied) | ✅ Kept |
| `team_performance` | Team rolling 5-race mean finish position | ✅ Kept |
| `dnf_rate_last5` | Driver DNF rate over last 5 races | ✅ Kept |
| `driver_points_before_race` | Cumulative championship points before each race | ✅ Kept |
| `driver_standings_pos` | Driver championship standing at race time | ✅ Kept |
| `avg_position_gain` | Driver's mean positions gained (grid − finish) over season | ✅ Kept |
| `gap_to_pole` | Qualifying time gap to pole position in seconds | ✅ Kept |
| `dnf_flag` | 1 = Did Not Finish; finish position imputed as 20 for training | ✅ Kept |
| `circuit_avg_finish` | Driver avg finish at specific circuit | ❌ Discarded (563 rows lost, only 0.068 MAE gain) |

### Golden rule — no data leakage:

```python
# Always: sort → shift(1) → rolling — never use current-race data
df.sort_values(by=["driver", "season", "round"], inplace=True)
df.groupby('driver')['finish_position']
  .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
```

### Train / Test split — always season-based, never random:

```python
train = df[df['season'] < 2025]   # 2022–2024
test  = df[df['season'] == 2025]  # unseen test set

# Season weights — recent seasons weighted higher
weight_map = {2022: 0.2, 2023: 0.5, 2024: 1.0}
```

---

## 📈 Model Performance

All models evaluated on the **2025 season test set** with DNFs excluded from evaluation.

| Model | MAE | vs Baseline |
|---|---|---|
| Baseline — grid position only | 2.780 | — |
| Linear Regression | 2.338 | −15.9% |
| Random Forest *(GridSearchCV tuned)* | 2.363 | −15.0% |
| XGBoost *(GridSearchCV tuned)* | 2.410 | −13.3% |
| CatBoost *(GridSearchCV tuned)* | 2.347 | −14.5% |
| LightGBM *(GridSearchCV tuned)* | 2.346 | −14.6% |
| Stacking Ensemble (LR + RF + XGB) | 2.352 | −15.4% |
| **Delta Regression + Rank Normalisation** | **2.211** | **−20.5%** ✅ Best |

> **DNF exclusion:** Drivers who Did Not Finish are excluded from test evaluation. Finish position is imputed as 20 in training so the `dnf_flag` and `dnf_rate_last5` features remain informative.

### Why Delta Regression works best

Instead of predicting absolute finish position, the model predicts **position gain** (grid − finish). This signal is stationary — a driver who consistently gains 3 positions does so regardless of their starting slot. The predicted gain is added back to grid position and rank-normalised within each race to produce valid integer positions 1–20 with no ties.

---

## 🔧 Hyperparameter Tuning

GridSearchCV with **5-fold cross-validation** was run separately for each model using `neg_mean_absolute_error`. Each grid contained **5 candidate values per hyperparameter**.

| Model | Key Params Tuned | Best CV MAE |
|---|---|---|
| Random Forest | n_estimators, max_depth, min_samples_leaf | 2.271 |
| XGBoost | n_estimators, max_depth, learning_rate, subsample | 2.248 |
| LightGBM | n_estimators, max_depth, learning_rate, subsample | 2.267 |
| CatBoost | iterations, depth, learning_rate | **2.239** ✅ Best CV |

---

## 📊 Dashboard

The interactive **4-tab Streamlit dashboard** is publicly deployed at:

🔗 [f1-race-prediction-rijul-mittal.streamlit.app](https://f1-race-prediction-rijul-mittal.streamlit.app/)

| Tab | Content |
|---|---|
| **Tab 1 — Model Comparison** | MAE bar chart, model deep-dive (scatter + error histogram), best/worst predicted races, hyperparameter table |
| **Tab 2 — Feature Importance** | Per-model feature importance bar chart + Pearson correlation with finish position |
| **Tab 3 — Race Predictor** | Select any 2025 race, choose a model, view full 20-driver predicted vs actual finishing order |
| **Tab 4 — Hyperparameter Tuning** | GridSearchCV results per model — best CV MAE bar chart + tuned parameter deep dive |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `fastf1` | Official F1 data source |
| `pandas` | Data manipulation |
| `scikit-learn` | ML models, GridSearchCV, metrics |
| `xgboost` | Gradient boosting |
| `lightgbm` | Light gradient boosting |
| `catboost` | Categorical-aware boosting |
| `optuna` | Bayesian hyperparameter tuning (Stacking v2) |
| `matplotlib` / `seaborn` | EDA visualisations |
| `streamlit` / `plotly` | Interactive dashboard |

---

## 🗺️ Roadmap

- [x] Data collection & preprocessing (FastF1, 2022–2025)
- [x] Baseline feature engineering (one-at-a-time validation)
- [x] Linear Regression baseline
- [x] Random Forest, XGBoost, LightGBM, CatBoost
- [x] Stacking ensemble (meta-learner)
- [x] GridSearchCV hyperparameter tuning (5-fold, 5 values per param)
- [x] Delta Regression + Rank Normalisation (best model, MAE 2.211)
- [x] 4-tab interactive Streamlit dashboard (publicly deployed)
- [x] Full project report
- [ ] Per-race prediction reports
- [ ] Live race updater (lap-by-lap position estimates)

---

## ⚠️ Limitations

- DNF outcomes add noise — a driver in P1 who retires is classified as P20 in training
- 2025 rookies (Antonelli, Hadjar, etc.) have limited historical data, reducing prediction accuracy for them specifically
- Weather and safety car events are not modelled — a well-timed safety car can invalidate an otherwise good prediction
- No circuit-specific features — track characteristics such as overtaking difficulty and tyre degradation are not encoded

---

## 📜 License

MIT License