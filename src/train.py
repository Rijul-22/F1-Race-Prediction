import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # set to your actual core count
# ── 1. Load ───────────────────────────────────────────────────────
def load_features(path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, '..', 'data', 'features', 'features_v10.csv')
    return pd.read_csv(path)

# ── 2. Prepare ────────────────────────────────────────────────────
def prepare(df):
    features = ['grid_position', 'weighted_finish_form', 'team_performance',
                'dnf_rate_last5', 'driver_points_before_race',
                'driver_standings_pos', 'avg_position_gain', 'gap_to_pole']
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

# ── 7. Model Stacking ──────────────────────────────────────────────
def train_stacking(df, features):
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    target = 'finish_position'
    weight_map = {2022: 0.2, 2023: 0.5, 2024: 1.0}

    # ── Splits ──
    oof_train = df[df['season'] <= 2023].copy().dropna(subset=features + [target])
    oof_val   = df[df['season'] == 2024].copy().dropna(subset=features + [target])
    full_train = df[df['season'] < 2025].copy().dropna(subset=features + [target])
    test_set   = df[df['season'] == 2025].copy().dropna(subset=features + [target])

    X_oof_tr = oof_train[features];  y_oof_tr = oof_train[target]
    X_oof_val = oof_val[features];   y_oof_val = oof_val[target]
    X_full   = full_train[features]; y_full   = full_train[target]
    X_test   = test_set[features];   y_test   = test_set[target]

    w_oof_tr = oof_train['season'].map(weight_map).values
    w_full   = full_train['season'].map(weight_map).values

    # ── Base models ──
    base_models = {
        'LR': LinearRegression(),
        'RF': RandomForestRegressor(n_estimators=200, max_depth=5,
                                     min_samples_leaf=8, random_state=42),
        'XGB': XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, verbosity=0),
        'LGBM': LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=300, depth=5, learning_rate=0.08,
                                       random_seed=42, verbose=0),
    }

    # ── Level 0: OOF predictions on 2024 ──
    oof_preds = {}
    print("\n" + "=" * 50)
    print("STACKING — Level 0 (OOF on 2024)")
    print("=" * 50)

    for name, model in base_models.items():
        model.fit(X_oof_tr, y_oof_tr, sample_weight=w_oof_tr)
        preds = model.predict(X_oof_val)
        oof_mae = mean_absolute_error(y_oof_val, preds)
        oof_preds[name] = preds
        print(f"  {name:10s} OOF MAE (2024): {oof_mae:.3f}")

    # ── Meta-learner: Ridge on OOF predictions ──
    meta_X = np.column_stack(list(oof_preds.values()))
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X, y_oof_val)

    meta_oof_preds = meta.predict(meta_X)
    meta_oof_mae = mean_absolute_error(y_oof_val, meta_oof_preds)
    print(f"\n  {'META (Ridge)':10s} OOF MAE (2024): {meta_oof_mae:.3f}")

    # ── Retrain all base models on full 2022–2024 ──
    print("\n" + "=" * 50)
    print("STACKING — Retrain on full 2022-2024, test on 2025")
    print("=" * 50)

    test_preds = {}
    for name, model in base_models.items():
        # Re-instantiate to avoid data leakage from previous fit
        if name == 'LR':
            m = LinearRegression()
        elif name == 'RF':
            m = RandomForestRegressor(n_estimators=200, max_depth=5,
                                       min_samples_leaf=8, random_state=42)
        elif name == 'XGB':
            m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
        elif name == 'LGBM':
            m = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
        else:
            m = CatBoostRegressor(iterations=300, depth=5, learning_rate=0.08,
                                   random_seed=42, verbose=0)
        m.fit(X_full, y_full, sample_weight=w_full)
        preds = m.predict(X_test)
        ind_mae = mean_absolute_error(y_test, preds)
        test_preds[name] = preds
        print(f"  {name:10s} individual MAE (2025): {ind_mae:.3f}")

    # ── Final stacking prediction ──
    meta_X_test = np.column_stack(list(test_preds.values()))
    final_preds = meta.predict(meta_X_test)
    stacking_mae = mean_absolute_error(y_test, final_preds)

    print(f"\n  >> STACKING MAE (2025): {stacking_mae:.3f}")
    print("=" * 50)

    # Return individual MAEs for dashboard/reporting
    individual_maes = {}
    for name, preds in test_preds.items():
        individual_maes[name] = mean_absolute_error(y_test, preds)
    individual_maes['Stacking'] = stacking_mae

    return meta, individual_maes, stacking_mae

# ── 8. Target Encoding (P2) ────────────────────────────────────────
def add_target_encoding(df):
    """Add leakage-free driver and team target-encoded features."""
    df = df.copy()
    df.sort_values(['driver', 'season', 'round'], inplace=True)

    # Driver: expanding mean of all past finish positions (shift to avoid leakage)
    df['driver_target_enc'] = (
        df.groupby('driver')['finish_position']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df.sort_values(['team', 'season', 'round'], inplace=True)

    # Team: expanding mean of all past finish positions
    df['team_target_enc'] = (
        df.groupby('team')['finish_position']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Fill first-race NaNs with global mean (smoothing prior)
    global_mean = df['finish_position'].mean()
    df['driver_target_enc'] = df['driver_target_enc'].fillna(global_mean)
    df['team_target_enc'] = df['team_target_enc'].fillna(global_mean)

    df.sort_values(['season', 'round', 'driver'], inplace=True)
    return df

# ── 9. Add extra features for v2 ──────────────────────────────────
def add_extra_features(df):
    """Add normalized gap_to_pole and intra-season form (leakage-free)."""
    df = df.copy()

    # Normalized gap_to_pole per race (z-score within each race)
    df['gap_to_pole_norm'] = df.groupby(['season', 'round'])['gap_to_pole'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # Intra-season form: rolling avg of CURRENT season only (last 3 races)
    df.sort_values(['driver', 'season', 'round'], inplace=True)
    gm = df['finish_position'].mean()
    df['intra_season_form'] = (
        df.groupby(['driver', 'season'])['finish_position']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['intra_season_form'] = df['intra_season_form'].fillna(gm)

    df.sort_values(['season', 'round', 'driver'], inplace=True)
    return df

# ── 10. Stacking v2 — Optuna-tuned + enhanced features ────────────
def train_stacking_v2(df, base_features):
    """
    Stacking with:
    - P2: target encoding (driver + team)
    - Extra: gap_to_pole_norm, intra_season_form
    - Optuna-tuned XGB and LGBM
    - Ridge(alpha=10) meta-learner
    """
    import optuna
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    target = 'finish_position'
    weight_map = {2022: 0.2, 2023: 0.5, 2024: 1.0}

    # ── Add all engineered features ──
    df = add_target_encoding(df)
    df = add_extra_features(df)
    features = base_features + [
        'gap_to_pole_norm', 'driver_target_enc', 'team_target_enc', 'intra_season_form'
    ]

    # ── Splits ──
    full_train = df[df['season'] < 2025].copy().dropna(subset=features + [target])
    test_set   = df[df['season'] == 2025].copy().dropna(subset=features + [target])
    oof_train  = df[df['season'] <= 2023].copy().dropna(subset=features + [target])
    oof_val    = df[df['season'] == 2024].copy().dropna(subset=features + [target])

    w_oof  = oof_train['season'].map(weight_map).values
    w_full = full_train['season'].map(weight_map).values
    y_test = test_set[target]

    # ── Optuna-tune XGBoost ──
    print("\n" + "=" * 50)
    print("STACKING v2 -- Optuna tuning (150 trials each)")
    print("=" * 50)

    def objective_xgb(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 2, 6),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        }
        m = XGBRegressor(**params, random_state=42, verbosity=0)
        m.fit(oof_train[features], oof_train[target], sample_weight=w_oof)
        return mean_absolute_error(oof_val[target], m.predict(oof_val[features]))

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=150)
    bp_xgb = study_xgb.best_params
    print(f"  XGBoost  best val MAE: {study_xgb.best_value:.3f}")

    def objective_lgbm(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
            'max_depth':         trial.suggest_int('max_depth', 2, 8),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        }
        m = LGBMRegressor(**params, random_state=42, verbose=-1)
        m.fit(oof_train[features], oof_train[target], sample_weight=w_oof)
        return mean_absolute_error(oof_val[target], m.predict(oof_val[features]))

    study_lgbm = optuna.create_study(direction='minimize')
    study_lgbm.optimize(objective_lgbm, n_trials=150)
    bp_lgbm = study_lgbm.best_params
    print(f"  LightGBM best val MAE: {study_lgbm.best_value:.3f}")

    # ── Base models with tuned params ──
    base_configs = {
        'LR':       lambda: LinearRegression(),
        'RF':       lambda: RandomForestRegressor(
                        n_estimators=200, max_depth=5, min_samples_leaf=8, random_state=42),
        'XGB':      lambda: XGBRegressor(**bp_xgb, random_state=42, verbosity=0),
        'LGBM':     lambda: LGBMRegressor(**bp_lgbm, random_state=42, verbose=-1),
        'CatBoost': lambda: CatBoostRegressor(
                        iterations=300, depth=5, learning_rate=0.08,
                        random_seed=42, verbose=0),
    }

    # ── Level 0: OOF ──
    print("\n" + "=" * 50)
    print("STACKING v2 -- Level 0 (OOF on 2024)")
    print("=" * 50)

    oof_preds = {}
    for name, make in base_configs.items():
        m = make()
        m.fit(oof_train[features], oof_train[target], sample_weight=w_oof)
        preds = m.predict(oof_val[features])
        oof_preds[name] = preds
        print(f"  {name:10s} OOF MAE: {mean_absolute_error(oof_val[target], preds):.3f}")

    # Ridge meta with alpha=10 (best from experiment)
    meta_X = np.column_stack(list(oof_preds.values()))
    meta = Ridge(alpha=10.0)
    meta.fit(meta_X, oof_val[target])
    meta_oof_mae = mean_absolute_error(oof_val[target], meta.predict(meta_X))
    print(f"  {'META':10s} OOF MAE: {meta_oof_mae:.3f}")

    # ── Retrain on full 2022-2024, predict 2025 ──
    print("\n" + "=" * 50)
    print("STACKING v2 -- Test on 2025")
    print("=" * 50)

    test_preds_raw = {}
    for name, make in base_configs.items():
        m = make()
        m.fit(full_train[features], full_train[target], sample_weight=w_full)
        preds = m.predict(test_set[features])
        ind_mae = mean_absolute_error(y_test, preds)
        test_preds_raw[name] = preds
        print(f"  {name:10s} test MAE: {ind_mae:.3f}")

    meta_X_test = np.column_stack(list(test_preds_raw.values()))
    final_preds = np.clip(meta.predict(meta_X_test), 1, 20)
    stacking_mae = mean_absolute_error(y_test, final_preds)

    print(f"\n  >> STACKING v2 MAE (2025): {stacking_mae:.3f}")
    print("=" * 50)

    return stacking_mae, features

# ── 11. Stacking v3 — Harshit's 2.1 MAE Method (Delta + Rank) ──────
def train_harshit_delta_model(df, base_features):
    """
    1. Removes DNFs (already done in main)
    2. Target = position_gain (grid - finish)
    3. Reconstructs raw finish: grid - predicted_gain
    4. Rank-post-processing: forces 1, 2, 3, 4 integer ranks per race
    """
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    weight_map = {2022: 0.2, 2023: 0.5, 2024: 1.0}

    # ── Enriched Features ──
    df = add_target_encoding(df)
    df = add_extra_features(df)
    features = base_features + [
        'gap_to_pole_norm', 'driver_target_enc', 'team_target_enc', 'intra_season_form'
    ]

    # Target is now Delta (position_gain)
    df['target_delta'] = df['grid_position'] - df['finish_position']
    target = 'target_delta'

    # ── Splits ──
    full_train = df[df['season'] < 2025].copy().dropna(subset=features + [target, 'finish_position'])
    test_set   = df[df['season'] == 2025].copy().dropna(subset=features + [target, 'finish_position'])
    oof_train  = df[df['season'] <= 2023].copy().dropna(subset=features + [target, 'finish_position'])
    oof_val    = df[df['season'] == 2024].copy().dropna(subset=features + [target, 'finish_position'])

    w_oof  = oof_train['season'].map(weight_map).values
    w_full = full_train['season'].map(weight_map).values
    
    # We always benchmark against TRUE finish_position
    y_test_true = test_set['finish_position'].values

    # Base models (using tuned params from earlier)
    base_configs = {
        'RF':       lambda: RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42),
        'XGB':      lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, verbosity=0),
        'LGBM':     lambda: LGBMRegressor(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8, random_state=42, verbose=-1),
        'CatBoost': lambda: CatBoostRegressor(iterations=400, depth=5, learning_rate=0.05, random_seed=42, verbose=0),
    }

    # ── OOF Meta-learner Training ──
    oof_preds = {}
    for name, make in base_configs.items():
        m = make()
        m.fit(oof_train[features], oof_train[target], sample_weight=w_oof)
        oof_preds[name] = m.predict(oof_val[features])
        
    meta_X = np.column_stack(list(oof_preds.values()))
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X, oof_val[target]) # Train meta on Delta

    # ── Final 2025 Predictions ──
    print("\n" + "=" * 50)
    print("STACKING v3 (Harshit's Full Method) -- Test on 2025")
    print("=" * 50)

    test_preds_raw = {}
    for name, make in base_configs.items():
        m = make()
        m.fit(full_train[features], full_train[target], sample_weight=w_full)
        
        # Predict delta
        pred_delta = m.predict(test_set[features])
        
        # Reconstruct finish position: grid - delta
        pred_raw_finish = test_set['grid_position'].values - pred_delta
        test_preds_raw[name] = pred_delta
        
        mae = mean_absolute_error(y_test_true, pred_raw_finish)
        print(f"  {name:10s} (Delta reconstructed): MAE = {mae:.3f}")

    meta_X_test = np.column_stack(list(test_preds_raw.values()))
    
    # 1. Stacked Delta Prediction
    final_pred_delta = meta.predict(meta_X_test)
    
    # 2. Reconstruct Raw Finish Position
    raw_finish_preds = test_set['grid_position'].values - final_pred_delta
    
    # 3. Post-Processing: Rank within each race
    def rank_within_race(test_data, raw_preds):
        tmp = test_data[['season', 'round']].copy()
        tmp['raw'] = raw_preds
        tmp['ranked'] = tmp.groupby(['season', 'round'])['raw'].rank(method='first').astype(float)
        return tmp['ranked'].values

    ranked_finish_preds = rank_within_race(test_set, raw_finish_preds)
    
    mae_raw = mean_absolute_error(y_test_true, raw_finish_preds)
    mae_ranked = mean_absolute_error(y_test_true, ranked_finish_preds)

    print(f"\n  >> Meta (Raw Finish):    MAE = {mae_raw:.3f}")
    print(f"  >> Meta (Ranked/Final):  MAE = {mae_ranked:.3f}")
    print("=" * 50)

    return mae_ranked, features

# ── 12. Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_features()
    
    # --- HARSHIT'S METHOD: REMOVE ALL DNFs BEFORE EVALUATION ---
    total_before = len(df)
    df = df[df['dnf'] == 0].copy()
    print("=" * 60)
    print("Excluding DNF rows from evaluation (DNFs are not modelled)")
    print(f"Removed {total_before - len(df)} DNF rows | {len(df)} remaining")
    print("=" * 60)

    X_train, y_train, X_test, y_test, test, train, features = prepare(df)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    mae1, mae2 = evaluate_baselines(test)
    model, mae_lr, lr_preds = train_linear(X_train, y_train, X_test, y_test, train)
    lr_preds = model.predict(X_test)
    model_rf, mae_rf = train_random_forest(X_train, y_train, X_test, y_test, train)
    model_xg, mae_xg = train_xgboost(X_train, y_train, X_test, y_test, train)

    # ── V1 Stacking (baseline) ──
    meta_model, ind_maes, stacking_v1_mae = train_stacking(df, features)

    # ── V2 Stacking (enhanced features + Optuna-tuned absolute predict) ──
    stacking_v2_mae, v2_features = train_stacking_v2(df, features)
    
    # ── V3 Stacking (Harshit's methodology: Delta + Rank) ──
    stacking_v3_mae, _ = train_harshit_delta_model(df, features)

    print("\n" + "=" * 60)
    print("FULL COMPARISON")
    print("=" * 60)
    print(f"  Baseline (grid pos):         MAE = {mae1:.3f}")
    print(f"  Linear Regression:           MAE = {mae_lr:.3f}")
    print(f"  Random Forest:               MAE = {mae_rf:.3f}")
    print(f"  XGBoost:                     MAE = {mae_xg:.3f}")
    print(f"  Stacking v1 (8 features):    MAE = {stacking_v1_mae:.3f}")
    print(f"  Stacking v2 (12 feat+tuned): MAE = {stacking_v2_mae:.3f}")
    print(f"  Harshit's Method (Delta+Rnk):MAE = {stacking_v3_mae:.3f}")
    print("=" * 60)