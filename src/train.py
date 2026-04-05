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
        path = os.path.join(base, '..', 'data', 'features', 'features_v7.csv')
    return pd.read_csv(path)

# ── 2. Prepare ────────────────────────────────────────────────────
def prepare(df):
    features = ['grid_position', 'weighted_finish_form', 'team_performance',
                'dnf_rate_last5', 'driver_points_before_race',
                'driver_standings_pos', 'avg_position_gain']
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

    return X_train, y_train, X_test, y_test, test,train, features

# ── 3. Evaluate baselines ─────────────────────────────────────────
def evaluate_baselines(test):
    print("=" * 40)
    print("BASELINE MODELS")
    print("=" * 40)

    # Baseline 1: predict grid position
    mae1 = mean_absolute_error(test['finish_position'], test['grid_position'])
    print(f"Baseline 1 — grid_position only:      MAE = {mae1:.3f}")

    # Baseline 2: predict driver recent form
    mae2 = mean_absolute_error(test['finish_position'], test['weighted_finish_form'])
    print(f"Baseline 2 — weighted_finish_form only: MAE = {mae2:.3f}")

    return mae1, mae2

# ── 4. Train Linear Regression ────────────────────────────────────
def train_linear(X_train, y_train, X_test, y_test, train):
    model = LinearRegression()
    weights = train['season'].map({2022: 0.2, 2023: 0.5, 2024: 1.0}).values
    model.fit(X_train, y_train, sample_weight=weights)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    print()
    print("=" * 40)
    print("LINEAR REGRESSION")
    print("=" * 40)
    print(f"MAE = {mae:.3f}")
    print()
    print("Coefficients:")
    for name, coef in zip(X_train.columns, model.coef_):
        print(f"  {name:25s}: {coef:.4f}")
    print(f"  {'intercept':25s}: {model.intercept_:.4f}")

    return model, mae, preds

# ── 5. Train Random Forest ────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test, train):
    model = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)
    weights = train['season'].map({2022: 0.2, 2023: 0.5, 2024: 1.0}).values
    model.fit(X_train, y_train, sample_weight=weights)
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
def train_xgboost(X_train, y_train, X_test, y_test, train):
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
    weights = train['season'].map({2022: 0.2, 2023: 0.5, 2024: 1.0}).values
    model.fit(X_train, y_train, sample_weight=weights)
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

# ── 7. Optuna ────────────────────────────────────────────────────────
def tune_xgboost(df, features):
    import optuna
    from xgboost import XGBRegressor

    # Optuna train/val split — never touch 2025 test set
    opt_train = df[df['season'] <= 2023][features + ['finish_position','season']].dropna()
    opt_val   = df[df['season'] == 2024][features + ['finish_position']].dropna()

    X_opt_tr  = opt_train[features]
    y_opt_tr  = opt_train['finish_position']
    X_opt_val = opt_val[features]
    y_opt_val = opt_val['finish_position']
    w_opt_tr = opt_train['season'].map({2022: 0.2, 2023: 0.5, 2024: 1.0}).values

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 100, 800),
            max_depth        = trial.suggest_int('max_depth', 2, 6),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        )
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X_opt_tr, y_opt_tr, sample_weight=w_opt_tr)
        return mean_absolute_error(y_opt_val, model.predict(X_opt_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print()
    print("=" * 40)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 40)
    print(f"Best val MAE (2024): {study.best_value:.3f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params

def train_xgboost_tuned(X_train, y_train, X_test, y_test, best_params):
    from xgboost import XGBRegressor
    model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    print()
    print("=" * 40)
    print("XGBOOST TUNED (2025 test)")
    print("=" * 40)
    print(f"MAE = {mae:.3f}")
    return model, mae, preds

def rank_predictions(test, raw_preds):
    tmp = test[['season', 'round']].copy()
    tmp['raw'] = raw_preds
    tmp['ranked'] = (tmp.groupby(['season', 'round'])['raw']
                        .rank(method='first')
                        .astype(float))
    return tmp['ranked'].values

# ── 8. Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_features()
    X_train, y_train, X_test, y_test, test, train, features = prepare(df)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    mae1, mae2 = evaluate_baselines(test)
    model, mae_lr, lr_preds= train_linear(X_train, y_train, X_test, y_test, train)
    lr_preds = model.predict(X_test)
    model_rf, mae_rf = train_random_forest(X_train, y_train, X_test, y_test, train)
    model_xg, mae_xg = train_xgboost(X_train, y_train, X_test, y_test, train)
    best_params = tune_xgboost(df, features)
    model_xg_tuned, mae_xg_tuned, xg_tuned_preds = train_xgboost_tuned(X_train, y_train, X_test, y_test, best_params)

    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Baseline 1 (grid pos):     MAE = {mae1:.3f}")
    print(f"Baseline 2 (recent form):  MAE = {mae2:.3f}")
    print(f"Linear Regression:         MAE = {mae_lr:.3f}")
    print(f"Random Forest:             MAE = {mae_rf:.3f}")
    print(f"XGBoost:                   MAE = {mae_xg:.3f}")
    print(f"XGBoost Tuned:             MAE = {mae_xg_tuned:.3f}")
