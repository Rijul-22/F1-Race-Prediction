"""
EDA Visualizations for F1 Race Prediction Report
Generates 5 plots and saves them to outputs/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'features', 'features_v10.csv')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ['grid_position', 'weighted_finish_form', 'team_performance',
            'dnf_rate_last5', 'driver_points_before_race',
            'driver_standings_pos', 'avg_position_gain', 'gap_to_pole']
TARGET = 'finish_position'

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150


def load():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows for EDA")
    return df


# ── 1. Finish Position Distribution ───────────────────────────────
def plot_finish_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['finish_position'], bins=20, kde=True, color='#e10600',
                 edgecolor='white', alpha=0.8, ax=ax)
    ax.set_title('Distribution of Finish Positions (2022–2025)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Finish Position')
    ax.set_ylabel('Count')
    ax.axvline(df['finish_position'].mean(), color='#1a1a2e', ls='--', lw=2,
               label=f'Mean = {df["finish_position"].mean():.1f}')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'finish_position_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ── 2. Feature Correlation Heatmap ────────────────────────────────
def plot_correlation_heatmap(df):
    cols = FEATURES + [TARGET]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'correlation_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ── 3. MAE Progression Chart ─────────────────────────────────────
def plot_mae_progression():
    """
    Uses MAE values from training runs.
    Update these after running train.py with stacking.
    """
    models = ['Baseline\n(Grid Pos)', 'Linear\nRegression', 'Random\nForest',
              'XGBoost', 'Stacking\nv1', 'Stacking\nv2', "Delta+\nRank Norm"]
    maes = [2.780, 2.338, 2.363, 2.410, 2.352, 2.375, 2.211]
    colors = ['#95a5a6'] * 4 + ['#3498db', '#3498db'] + ['#e10600']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, maes, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{mae:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('MAE Progression Across Models', fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(maes) + 0.5)
    ax.axhline(y=2.0, color='green', ls='--', lw=1.5, alpha=0.7, label='Target MAE = 2.0')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'mae_progression.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ── 4. Feature Importance Chart ───────────────────────────────────
def plot_feature_importance():
    # Values taken directly from train.py output (RF trained on 2022-2024, no DNFs)
    importances = {
        'weighted_finish_form':      0.4277,
        'grid_position':             0.3196,
        'driver_standings_pos':      0.1213,
        'gap_to_pole':               0.0820,
        'driver_points_before_race': 0.0386,
        'avg_position_gain':         0.0036,
        'team_performance':          0.0070,
        'dnf_rate_last5':            0.0002,
    }

    imp = pd.Series(importances).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#e10600' if v == imp.max() else '#2c3e50' for v in imp.values]
    imp.plot.barh(ax=ax, color=colors, edgecolor='white')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importances', fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'feature_importance.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ── 5. Grid Position vs Finish Position Scatter ──────────────────
def plot_grid_vs_finish(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df['grid_position'], df['finish_position'],
               alpha=0.15, s=30, color='#2c3e50', edgecolors='none')

    # Regression line
    z = np.polyfit(df['grid_position'], df['finish_position'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, 20, 100)
    ax.plot(x_line, p(x_line), color='#e10600', lw=2.5,
            label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

    # Perfect prediction line
    ax.plot([1, 20], [1, 20], ls='--', color='gray', lw=1, alpha=0.6, label='Perfect (y=x)')

    ax.set_xlabel('Grid Position')
    ax.set_ylabel('Finish Position')
    ax.set_title('Grid Position vs Finish Position', fontweight='bold', fontsize=14)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'grid_vs_finish.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 40)
    print("GENERATING EDA PLOTS")
    print("=" * 40)

    df = load()
    plot_finish_distribution(df)
    plot_correlation_heatmap(df)
    plot_mae_progression()
    plot_feature_importance()
    plot_grid_vs_finish(df)

    print("\n[DONE] All 5 EDA plots saved to outputs/")
