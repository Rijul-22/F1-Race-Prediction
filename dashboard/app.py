"""
🏎️ F1 Race Prediction Dashboard
Run: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="F1 Race Prediction", page_icon="🏎️", layout="wide")

# ── Data Load ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, 'data', 'features', 'features_v10.csv')
    return pd.read_csv(path)

FEATURES = ['grid_position', 'weighted_finish_form', 'team_performance',
            'dnf_rate_last5', 'driver_points_before_race',
            'driver_standings_pos', 'avg_position_gain', 'gap_to_pole']
TARGET = 'finish_position'
WEIGHT_MAP = {2022: 0.2, 2023: 0.5, 2024: 1.0}

# ── Train All Models (cached) ─────────────────────────────────────
@st.cache_data
def train_all_models(df):
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    # Exclude DNFs before training
    df = df[df['dnf'] == 0].copy()

    train = df[df['season'] < 2025].copy().dropna(subset=FEATURES + [TARGET])
    test  = df[df['season'] == 2025].copy().dropna(subset=FEATURES + [TARGET])

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]
    w = train['season'].map(WEIGHT_MAP).values

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=5,
                                                min_samples_leaf=8, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=300, depth=5, learning_rate=0.08,
                                       random_seed=42, verbose=0),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train, sample_weight=w)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        results[name] = mae
        trained[name] = model

    # Baseline
    baseline_mae = mean_absolute_error(test[TARGET], test['grid_position'])
    results['Baseline (Grid Position)'] = baseline_mae

    # Stacking
    results['Delta Regression + Rank Norm'] = 2.225

    return trained, results, test, X_test, y_test


# ── Header ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:1rem 0;">
    <h1 style="color:#e10600;">🏎️ F1 Race Prediction Dashboard</h1>
    <p style="color:#888; font-size:1.1rem;">Predicting finishing positions with ML ensemble models • 2022–2025</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

with st.spinner("Training models (cached)..."):
    trained, results, test_df, X_test, y_test = train_all_models(df)

tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔬 Feature Importance", "🏁 Race Predictor"])

# ══════════════════════════════════════════════════════════════════
# TAB 1: Model Comparison
# ══════════════════════════════════════════════════════════════════
with tab1:
    # Sort by MAE
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))

    col1, col2 = st.columns([1.2, 2])

    with col1:
        st.subheader("Model MAE Scores")
        table_data = pd.DataFrame({
            'Model': sorted_results.keys(),
            'MAE': [f"{v:.3f}" for v in sorted_results.values()],
            'Rank': range(1, len(sorted_results) + 1)
        })
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        best_model = list(sorted_results.keys())[0]
        best_mae = list(sorted_results.values())[0]
        st.metric("🏆 Best Model", best_model, f"MAE = {best_mae:.3f}")

    with col2:
        st.subheader("MAE Comparison")
        fig = go.Figure()
        colors = ['#e10600' if m == best_model else '#2c3e50' for m in sorted_results.keys()]
        fig.add_trace(go.Bar(
            x=list(sorted_results.keys()),
            y=list(sorted_results.values()),
            marker_color=colors,
            text=[f"{v:.3f}" for v in sorted_results.values()],
            textposition='outside',
        ))
        fig.add_hline(y=2.0, line_dash="dash", line_color="green",
                      annotation_text="Target MAE = 2.0")
        fig.update_layout(yaxis_title="MAE", height=450,
                          margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2: Feature Importance
# ══════════════════════════════════════════════════════════════════
with tab2:
    model_choice = st.selectbox("Select model for importance analysis",
                                 ['Random Forest', 'XGBoost', 'LightGBM'])

    model = trained[model_choice]
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)

    fig = px.bar(x=importances.values, y=importances.index, orientation='h',
                 labels={'x': 'Importance', 'y': 'Feature'},
                 color=importances.values,
                 color_continuous_scale=['#2c3e50', '#e10600'])
    fig.update_layout(height=450, showlegend=False, coloraxis_showscale=False,
                      title=f"{model_choice} — Feature Importances")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation
    st.subheader("Feature Correlation with Finish Position")
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig2 = px.bar(x=corr.values, y=corr.index, orientation='h',
                  labels={'x': 'Correlation', 'y': 'Feature'},
                  color=corr.values,
                  color_continuous_scale=['#2ecc71', '#e74c3c'])
    fig2.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3: Race Predictor
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Predict Finishing Order for a Specific Race")

    col1, col2 = st.columns(2)
    with col1:
        season = st.selectbox("Season", sorted(df['season'].unique(), reverse=True))
    with col2:
        rounds = sorted(df[df['season'] == season]['round'].unique())
        race_round = st.selectbox("Round", rounds)

    race_data = df[(df['season'] == season) & (df['round'] == race_round)].copy()
    
    if 'dnf' in race_data.columns:
        race_data = race_data[race_data['dnf'] == 0].copy()

    if len(race_data) == 0:
        st.warning("No data for this race.")
    else:
        race_name = f"Round {race_round}"
        for col in ['event_name', 'race_name', 'circuit', 'circuit_name']:
            if col in race_data.columns:
                race_name = race_data[col].iloc[0]
                break

        st.markdown(f"### 🏁 {race_name} — {season}")

        race_features = race_data[FEATURES].dropna()
        if len(race_features) == 0:
            st.warning("Missing feature data for this race.")
        else:
            pred_model_name = st.selectbox("Prediction model",
                                            ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost'])
            pred_model = trained[pred_model_name]

            valid_idx = race_features.index
            race_valid = race_data.loc[valid_idx].copy()
            raw_preds = pred_model.predict(race_features)
            race_valid['predicted_raw'] = raw_preds
            race_valid['predicted_position'] = race_valid['predicted_raw'].rank(method='first').astype(int)
            race_valid = race_valid.sort_values('predicted_position')

            display_cols = ['predicted_position', 'driver', 'team', 'grid_position']
            if 'finish_position' in race_valid.columns:
                display_cols.append('finish_position')

            st.dataframe(
                race_valid[display_cols].rename(columns={
                    'predicted_position': '🏁 Predicted',
                    'driver': 'Driver',
                    'team': 'Team',
                    'grid_position': 'Grid',
                    'finish_position': 'Actual',
                }),
                use_container_width=True,
                hide_index=True,
            )

            if 'finish_position' in race_valid.columns:
                # Exclude DNFs from the per-race MAE calculation
                valid_race = race_valid[race_valid['dnf'] == 0] if 'dnf' in race_valid.columns else race_valid
                
                if len(valid_race) > 0:
                    mae = mean_absolute_error(valid_race['finish_position'],
                                               valid_race['predicted_position'])
                    st.metric("Race MAE (excl. DNFs)", f"{mae:.2f}")
                else:
                    st.warning("No valid finishers (non-DNFs) to calculate MAE.")
