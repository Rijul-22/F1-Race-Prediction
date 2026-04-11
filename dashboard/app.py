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
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6,
                                                min_samples_leaf=2, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.2,
                                subsample=0.6, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=500, max_depth=3, learning_rate=0.01,
                                   subsample=0.6, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05,
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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🔬 Feature Importance", "🏁 Race Predictor", "⚙️ Hyperparameter Tuning"])

# ══════════════════════════════════════════════════════════════════
# TAB 1: Model Comparison
# ══════════════════════════════════════════════════════════════════
with tab1:
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
    best_model = list(sorted_results.keys())[0]

    st.subheader("Model MAE Comparison")
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
    fig.update_layout(yaxis_title="MAE", height=400, margin=dict(t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Per-model deep dive ───────────────────────────────────────
    st.subheader("Model Deep Dive")
    selected_model = st.selectbox(
        "Select a model to inspect",
        [m for m in sorted_results.keys() if m not in
         ['Baseline (Grid Position)', 'Delta Regression + Rank Norm']]
    )

    model = trained[selected_model]
    preds = model.predict(X_test)

    # Rank predictions within each race
    tmp = test_df[['season', 'round']].copy()
    tmp['raw'] = preds
    tmp['ranked'] = tmp.groupby(['season', 'round'])['raw'].rank(method='first').astype(float)
    ranked_preds = tmp['ranked'].values
    actual = y_test.values
    errors = ranked_preds - actual

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{sorted_results[selected_model]:.3f}")
    col2.metric("Mean Error", f"{errors.mean():.3f}", help="Positive = predicting too high")
    col3.metric("Std of Errors", f"{errors.std():.3f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Predicted vs Actual Finish Position**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=actual, y=ranked_preds,
            mode='markers',
            marker=dict(color='#2c3e50', opacity=0.4, size=5),
            name='Predictions'
        ))
        fig2.add_trace(go.Scatter(
            x=[1, 20], y=[1, 20],
            mode='lines',
            line=dict(color='#e10600', dash='dash', width=1.5),
            name='Perfect prediction'
        ))
        fig2.update_layout(
            xaxis_title="Actual Position",
            yaxis_title="Predicted Position",
            height=380,
            margin=dict(t=20, b=20),
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("**Prediction Error Distribution**")
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            marker_color='#e10600',
            opacity=0.8,
            name='Errors'
        ))
        fig3.add_vline(x=0, line_dash="dash", line_color="#2c3e50", line_width=1.5)
        fig3.update_layout(
            xaxis_title="Prediction Error (Predicted − Actual)",
            yaxis_title="Count",
            height=380,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Best and worst predicted races ───────────────────────────
    st.markdown("**Best and Worst Predicted Races (2025)**")
    tmp2 = test_df[['season', 'round', 'driver']].copy()
    tmp2['actual'] = actual
    tmp2['predicted'] = ranked_preds
    tmp2['abs_error'] = np.abs(errors)

    race_mae = (tmp2.groupby(['season', 'round'])['abs_error']
                .mean()
                .reset_index()
                .rename(columns={'abs_error': 'race_mae'})
                .sort_values('race_mae'))

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("🟢 **5 Best Predicted Races**")
        st.dataframe(race_mae.head(5).rename(columns={
            'season': 'Season', 'round': 'Round', 'race_mae': 'MAE'
        }).assign(MAE=lambda x: x['MAE'].round(3)),
        hide_index=True, use_container_width=True)

    with col_d:
        st.markdown("🔴 **5 Worst Predicted Races**")
        st.dataframe(race_mae.tail(5).sort_values('race_mae', ascending=False).rename(columns={
            'season': 'Season', 'round': 'Round', 'race_mae': 'MAE'
        }).assign(MAE=lambda x: x['MAE'].round(3)),
        hide_index=True, use_container_width=True)
    # ── Hyperparameter Comparison Table ──────────────────────────
    st.divider()
    st.subheader("Hyperparameter Comparison")

    import pandas as pd
    hyperparam_data = {
        "Hyperparameter":    ["n_estimators/iterations", "max_depth/depth", "learning_rate", "subsample", "min_samples_leaf", "sample_weight"],
        "Linear Regression": ["N/A",                      "N/A",             "N/A",           "N/A",       "N/A",              "Season weights"],
        "Random Forest":     ["100",                      "6",               "N/A",           "N/A",       "2",                "Season weights"],
        "XGBoost":           ["50",                       "2",               "0.2",           "0.6",       "N/A",              "Season weights"],
        "LightGBM":          ["500",                      "3",               "0.01",          "0.6",       "N/A",              "Season weights"],
        "CatBoost":          ["200",                      "4",               "0.05",          "N/A",       "N/A",              "Season weights"],
    }
    hp_df = pd.DataFrame(hyperparam_data).set_index("Hyperparameter")
    st.dataframe(hp_df, use_container_width=True)


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

# ══════════════════════════════════════════════════════════════════
# TAB 4: Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Hyperparameter Tuning Results")
    st.markdown("Interactive comparison of our tuned models and optimal parameters discovered via `GridSearchCV`.")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'outputs', 'hyperparam_results.csv')
    
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        
        # ── 1. Interactive Bar Chart ──
        model_best_maes = results_df.groupby('Model')['Best MAE'].first().sort_values()
        
        # Identify the absolute best among tuned models
        best_tuned_model = model_best_maes.index[0]
        
        fig_hp = go.Figure()
        colors_hp = ['#e10600' if m == best_tuned_model else '#2c3e50' for m in model_best_maes.index]
        
        fig_hp.add_trace(go.Bar(
            x=model_best_maes.index,
            y=model_best_maes.values,
            marker_color=colors_hp,
            text=[f"{v:.3f}" for v in model_best_maes.values],
            textposition='outside',
        ))
        
        fig_hp.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="Target MAE")
        fig_hp.update_layout(yaxis_title="Best CV MAE", height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig_hp, use_container_width=True)
        
        st.divider()
        
        # ── 2. Parameter Deep Dive ──
        st.subheader("Tuned Parameter Deep Dive")
        st.markdown("Select a tuned model to examine its optimal configuration.")
        
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            selected_model_hp = st.selectbox("Model", results_df['Model'].unique(), key='hp_select', label_visibility='collapsed')
        
        model_df = results_df[results_df['Model'] == selected_model_hp]
        best_mae_val = model_df['Best MAE'].iloc[0]
        
        # Draw metric cards for the hyperparameters
        st.markdown(f"#### Optimal Params found for {selected_model_hp} <span style='font-size:0.9em;color:#888;'>(Score: {best_mae_val:.3f})</span>", unsafe_allow_html=True)
        
        param_cols = st.columns(len(model_df))
        for i, (_, row) in enumerate(model_df.iterrows()):
            param_cols[i].metric(label=row['Param'], value=str(row['Best Value']))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Search Grid Space Evaluated**")
        
        # Pretty table for the evaluated grid
        st.dataframe(
            model_df[['Param', 'Values Tried', 'Best Value']], 
            use_container_width=True, 
            hide_index=True
        )
        
    else:
        st.info("Hyperparameter results CSV not found. Ensure the tuning script was run.")
