import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4" # set to actual core count dynamically

# ── 1. Data Prep Setup ──────────────────────────────────────────────
def load_features(path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, '..', 'data', 'features', 'features_v10.csv')
    return pd.read_csv(path)

def prepare_tuning_data(df):
    features = ['grid_position', 'weighted_finish_form', 'team_performance',
                'dnf_rate_last5', 'driver_points_before_race',
                'driver_standings_pos', 'avg_position_gain', 'gap_to_pole']
    target   = 'finish_position'

    # Remove DNF rows
    df = df[df['dnf'] == 0].copy()

    # We use < 2025 for cross-validation training
    train = df[df['season'] < 2025].copy()

    # Drop NaNs
    cols = features + [target]
    train = train.dropna(subset=cols)

    X_train = train[features]
    y_train = train[target]

    # Weights
    weights = train['season'].map({2022: 0.2, 2023: 0.5, 2024: 1.0}).values

    return X_train, y_train, weights

# ── 2. Param Grids ──────────────────────────────────────────────────
param_grid_rf = {
    'n_estimators':    [50, 100, 200, 300, 500],
    'max_depth':       [3, 4, 6, 8, 10],
    'min_samples_leaf':[2, 5, 10, 15, 20],
}

param_grid_xgb = {
    'n_estimators':  [50, 100, 200, 300, 500],
    'max_depth':     [2, 3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample':     [0.6, 0.7, 0.8, 0.9, 1.0],
}

param_grid_lgbm = {
    'n_estimators':  [100, 200, 300, 400, 500],
    'max_depth':     [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.2],
    'subsample':     [0.6, 0.7, 0.8, 0.9, 1.0],
}

param_grid_cat = {
    'iterations':    [100, 200, 300, 400, 500],
    'depth':         [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.2],
}

# ── 3. Models Config ────────────────────────────────────────────────
models = {
    'Random Forest': {
        'estimator': RandomForestRegressor(random_state=42),
        'grid': param_grid_rf
    },
    'XGBoost': {
        'estimator': XGBRegressor(random_state=42, verbosity=0),
        'grid': param_grid_xgb
    },
    'LightGBM': {
        'estimator': LGBMRegressor(random_state=42, verbose=-1),
        'grid': param_grid_lgbm
    },
    'CatBoost': {
        'estimator': CatBoostRegressor(random_seed=42, verbose=0, allow_writing_files=False),
        'grid': param_grid_cat
    }
}

# ── 4. Main Sequence ────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_features()
    X_train, y_train, weights = prepare_tuning_data(df)

    print(f"Loaded {len(X_train)} samples for GridSearchCV.")
    print("Beginning Tuning. This will take a while...")

    results = []
    plot_data = []

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    for name, config in models.items():
        print(f"\n[{name}] Tuning...")
        grid_search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['grid'],
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Pass weights to fit
        grid_search.fit(X_train, y_train, sample_weight=weights)
        
        best_mae = -grid_search.best_score_
        best_params = grid_search.best_params_
        
        # Get stats for plotting (difference across combos)
        cv_scores = -grid_search.cv_results_['mean_test_score']
        min_mae = np.min(cv_scores)  # best
        mean_mae = np.mean(cv_scores)
        max_mae = np.max(cv_scores)  # worst
        
        plot_data.append({
            'Model': name,
            'Best MAE': min_mae,
            'Mean MAE': mean_mae,
            'Worst MAE': max_mae
        })

        print(f"[{name}] Best MAE: {best_mae:.3f} with params: {best_params}")

        # Build CSV rows
        for param_name, best_val in best_params.items():
            vals_tried = config['grid'][param_name]
            results.append({
                'Model': name,
                'Param': param_name,
                'Values Tried': str(vals_tried),
                'Best Value': best_val,
                'Best MAE': best_mae
            })

    # Save CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, 'hyperparam_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV results to {csv_path}")

    # Generate Chart
    plot_df = pd.DataFrame(plot_data)
    
    # We want a grouped bar chart where the groups are the Models
    # and the bars are Best MAE, Mean MAE, Worst MAE
    plot_df_melted = plot_df.melt(id_vars='Model', var_name='Metric', value_name='CV MAE')
    
    plt.figure(figsize=(10, 6))
    
    # Use seaborn colors matching a generic nice palette
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=plot_df_melted, 
        x='Model', 
        y='CV MAE', 
        hue='Metric',
        palette=['#2ca02c', '#1f77b4', '#d62728'] # Green (best), Blue (mean), Red (worst)
    )
    
    plt.title('Hyperparameter Grid Search: CV MAE Across Combinations', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Mean Absolute Error (Lower is Better)', fontsize=12)
    plt.legend(title='')
    
    # Add values on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
    plt.tight_layout()
    chart_path = os.path.join(out_dir, 'hyperparam_comparison.png')
    plt.savefig(chart_path, dpi=300)
    print(f"Saved Grouped Bar Chart to {chart_path}\n")
    print("Done!")
