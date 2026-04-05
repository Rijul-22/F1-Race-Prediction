import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# ── 1. Load ───────────────────────────────────────────────────────
def load_features(path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, '..', 'data', 'features', 'features_v5.csv')
    return pd.read_csv(path)

# ── 2. Prepare ────────────────────────────────────────────────────
def prepare(df):
    features = ['grid_position', 'driver_recent_form', 'team_performance', 'dnf_rate_last5', 'driver_points_before_race', 'driver_standings_pos']
    target   = 'finish_position'

    train = df[df['season'] < 2025].copy()
    test  = df[df['season'] == 2025].copy()

    # Drop rows where any feature or target is NaN
    cols = features + [target]
    train = train.dropna(subset=cols)
    test  = test.dropna(subset=cols)

    X_train = train[features]
    y_train = train[target]
    X_test  = test[features]
    y_test  = test[target]

    return X_train, y_train, X_test, y_test, test

# ── 3. Evaluate baselines ─────────────────────────────────────────
def evaluate_baselines(test):
    print("=" * 40)
    print("BASELINE MODELS")
    print("=" * 40)

    # Baseline 1: predict grid position
    mae1 = mean_absolute_error(test['finish_position'], test['grid_position'])
    print(f"Baseline 1 — grid_position only:      MAE = {mae1:.3f}")

    # Baseline 2: predict driver recent form
    mae2 = mean_absolute_error(test['finish_position'], test['driver_recent_form'])
    print(f"Baseline 2 — driver_recent_form only: MAE = {mae2:.3f}")

    return mae1, mae2

# ── 4. Train Linear Regression ────────────────────────────────────
def train_linear(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    print()
    print("=" * 40)
    print("LINEAR REGRESSION (3 features)")
    print("=" * 40)
    print(f"MAE = {mae:.3f}")
    print()
    print("Coefficients:")
    for name, coef in zip(X_train.columns, model.coef_):
        print(f"  {name:25s}: {coef:.4f}")
    print(f"  {'intercept':25s}: {model.intercept_:.4f}")

    return model, mae

# ── 5. Train Random Forest ────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    print(f"Train MAE: {mean_absolute_error(y_train, train_preds):.3f}")
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    print()
    print("=" * 40)
    print("RANDOM FOREST")
    print("=" * 40)
    print(f"MAE = {mae:.3f}")
    print()
    print("Feature Importances:")
    for name, imp in sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:25s}: {imp:.4f}")

    return model, mae

# ── 6. Train XGBoost ────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test):
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    print(f"Train MAE: {mean_absolute_error(y_train, train_preds):.3f}")
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    print()
    print("=" * 40)
    print("XGBOOST")
    print("=" * 40)
    print(f"MAE = {mae:.3f}")
    print()
    print("Feature Importances:")
    for name, imp in sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:25s}: {imp:.4f}")

    return model, mae

# ── 7. Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_features()
    X_train, y_train, X_test, y_test, test = prepare(df)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    mae1, mae2 = evaluate_baselines(test)
    model, mae_lr = train_linear(X_train, y_train, X_test, y_test)
    model_rf, mae_rf = train_random_forest(X_train, y_train, X_test, y_test)
    model_xg, mae_xg = train_xgboost(X_train, y_train, X_test, y_test)

    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Baseline 1 (grid pos):     MAE = {mae1:.3f}")
    print(f"Baseline 2 (recent form):  MAE = {mae2:.3f}")
    print(f"Linear Regression:         MAE = {mae_lr:.3f}")
    print(f"Random Forest:             MAE = {mae_rf:.3f}")
    print(f"XGBoost:                   MAE = {mae_xg:.3f}")