"""
app/app.py — Streamlit dashboard for Global Fuel Crisis Analyzer.

Layout
------
Sidebar : user inputs (country, fuel usage, crisis severity)
Tab 1   : Supply Shock Simulator
Tab 2   : Historical Oil Price Explorer
Tab 3   : Model Comparison
Tab 4   : News Sentiment
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from config import COUNTRIES, PROC_DIR, RAW_DIR
from simulation import simulate_supply_shock, run_scenario_sweep
from visualization import (
    plotly_country_impact,
    plotly_shock_heatmap,
    plotly_scenario_sweep,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Fuel Crisis Analyzer",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (works in both light and dark mode) ─────────────────────────────
st.markdown("""
<style>
    .metric-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .metric-card h2 { color: #FF6F00; font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.85rem; margin: 0; opacity: 0.75; }
    .crisis-badge {
        border: 1px solid #D32F2F; color: #D32F2F;
        border-radius: 20px; padding: 2px 10px; font-size: 0.78rem;
    }
    .sidebar-header {
        font-size: 1.1rem; font-weight: 700;
        color: #FF6F00; letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    """Load the processed master dataset or fall back to synthetic data."""
    master_path = PROC_DIR / "master.csv"
    if master_path.exists():
        df = pd.read_csv(master_path, index_col="date", parse_dates=True)
        return df
    logger.warning("master.csv not found — generating synthetic oil price data.")
    return _synthetic_master()


def _synthetic_master() -> pd.DataFrame:
    """
    Generate realistic synthetic oil price data 2000–2026.

    Geopolitical events modelled:
      • 2008 GFC spike & crash
      • 2014–2016 OPEC price war
      • 2020 COVID crash & recovery
      • 2022 Russia–Ukraine war spike
      • 2023 Israel–Hamas war (Oct 7 attack)
      • 2024 Iran–Israel direct strikes
      • 2025 US–Iran tensions / Red Sea disruption
      • 2025–2026 gradual normalisation
    """
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", "2026-03-01", freq="MS")
    n     = len(dates)

    price = np.zeros(n)
    price[0] = 30.0
    for i in range(1, n):
        shock  = rng.normal(0, 2.8)
        revert = -0.04 * (price[i - 1] - 72)
        price[i] = max(price[i - 1] + 0.10 + revert + shock, 10)

    events = [
        ("2007-06-01", "2008-07-31",  55),
        ("2008-08-01", "2008-12-31", -60),
        ("2011-01-01", "2012-06-30",  25),
        ("2014-07-01", "2016-03-31", -45),
        ("2018-10-01", "2018-12-31",  15),
        ("2020-02-01", "2020-04-30", -48),
        ("2020-05-01", "2021-06-30",  22),
        ("2022-02-01", "2022-08-31",  40),
        ("2022-09-01", "2022-12-31",  -8),
        ("2023-10-01", "2023-12-31",  12),
        ("2024-01-01", "2024-03-31",   8),
        ("2024-04-01", "2024-04-30",  14),
        ("2024-05-01", "2024-07-31",  -6),
        ("2024-08-01", "2024-09-30",  -5),
        ("2024-10-01", "2024-10-31",  10),
        ("2024-11-01", "2024-12-31",  -8),
        ("2025-01-01", "2025-03-31",   6),
        ("2025-04-01", "2025-06-30",  -4),
        ("2025-07-01", "2025-09-30",   3),
        ("2025-10-01", "2025-12-31",  -5),
        ("2026-01-01", "2026-03-01",  -3),
    ]
    for start, end, bump in events:
        mask = (dates >= start) & (dates <= end)
        idx  = np.where(mask)[0]
        if len(idx):
            price[idx] += np.linspace(0, bump, len(idx))

    price = np.clip(price, 15, 145)

    df = pd.DataFrame({
        "brent_crude":   price,
        "wti_crude":     np.clip(price - rng.uniform(1.5, 4.5, n), 12, 140),
        "natural_gas":   np.abs(rng.normal(3.2, 1.4, n)),
        "us_cpi_energy": 180 + np.cumsum(rng.normal(0.35, 0.9, n)),
        "crisis_flag":   0,
        "crisis_name":   "normal",
    }, index=dates)

    crisis_periods = [
        ("2003-02-01", "2003-06-30", "gulf_war_2"),
        ("2007-06-01", "2008-12-31", "gfc_spike"),
        ("2011-01-01", "2012-06-30", "arab_spring"),
        ("2014-07-01", "2016-03-31", "opec_price_war"),
        ("2020-01-01", "2020-06-30", "covid_crash"),
        ("2022-02-01", "2022-12-31", "ukraine_war"),
        ("2023-10-01", "2024-04-30", "israel_hamas_iran"),
        ("2025-01-01", "2025-06-30", "us_iran_tensions"),
    ]
    for s, e, name in crisis_periods:
        mask = (df.index >= s) & (df.index <= e)
        df.loc[mask, "crisis_flag"] = 1
        df.loc[mask, "crisis_name"] = name

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build features from raw master df (used by Model Comparison tab)
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all lag, rolling, and pct-change features needed by modeling.py.
    Works on both real (from master.csv) and synthetic DataFrames.
    """
    df = df.copy()

    for col in ["brent_crude", "wti_crude"]:
        if col not in df.columns:
            continue
        for lag in [1, 2, 3, 6, 12]:
            df[f"{col}_lag{lag}m"] = df[col].shift(lag)
        for w in [3, 6, 12]:
            df[f"{col}_roll{w}m_mean"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}m_std"]  = df[col].rolling(w, min_periods=1).std()
        for p in [1, 3, 12]:
            df[f"{col}_pct{p}m"] = df[col].pct_change(p) * 100

    # Spread
    if "brent_crude" in df.columns and "wti_crude" in df.columns:
        df["brent_wti_spread"] = df["brent_crude"] - df["wti_crude"]

    # Calendar
    df["month"]   = df.index.month
    df["quarter"] = df.index.quarter

    return df


def _split_train_test(df: pd.DataFrame, split_date: str = "2022-01-01"):
    """Split on date, drop rows with NaN in core lag columns."""
    cols_needed = [
        "brent_crude", "brent_crude_lag1m", "brent_crude_lag3m",
        "brent_crude_roll3m_mean", "wti_crude_lag1m",
    ]
    existing = [c for c in cols_needed if c in df.columns]
    df_clean = df.dropna(subset=existing)
    train = df_clean[df_clean.index < split_date]
    test  = df_clean[df_clean.index >= split_date]
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⛽ Fuel Crisis Analyzer")
        st.markdown("---")
        st.markdown('<p class="sidebar-header">🌍 Country Settings</p>', unsafe_allow_html=True)

        country_options = {v["name"]: k for k, v in COUNTRIES.items()}
        selected_country_name = st.selectbox(
            "Select Country",
            options=sorted(country_options.keys()),
            index=sorted(country_options.keys()).index("United States"),
        )
        selected_iso3 = country_options[selected_country_name]

        st.markdown('<p class="sidebar-header">🔥 Crisis Parameters</p>', unsafe_allow_html=True)
        crisis_severity = st.slider(
            "Supply Drop (%)", min_value=1, max_value=50, value=15, step=1,
            help="Percentage reduction in global crude oil supply",
        )
        base_price = st.number_input(
            "Current Brent Price (USD/bbl)",
            min_value=20.0, max_value=200.0, value=85.0, step=0.5,
        )

        st.markdown('<p class="sidebar-header">🚗 Household Fuel Usage</p>', unsafe_allow_html=True)
        monthly_litres = st.number_input(
            "Monthly Fuel Consumption (litres)",
            min_value=10.0, max_value=500.0, value=60.0, step=5.0,
        )

        st.markdown("---")
        run_sim = st.button("🚀 Run Simulation", use_container_width=True, type="primary")

    return {
        "country_name":    selected_country_name,
        "iso3":            selected_iso3,
        "crisis_severity": crisis_severity,
        "base_price":      base_price,
        "monthly_litres":  monthly_litres,
        "run_sim":         run_sim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Supply Shock Simulator
# ─────────────────────────────────────────────────────────────────────────────

def tab_simulator(params: dict, master_df: pd.DataFrame) -> None:
    st.header("🔥 Supply Shock Simulator")
    st.caption(
        "Model a global crude oil supply disruption and see how it propagates "
        "to retail fuel prices and household costs across countries."
    )

    if not params["run_sim"]:
        st.info("👈 Configure parameters in the sidebar and click **Run Simulation**.")
        if "brent_crude" in master_df.columns:
            latest = master_df["brent_crude"].dropna().iloc[-1]
            st.metric("Latest Brent Crude Price (Mar 2026)", f"${latest:.2f}/bbl")
        return

    with st.spinner("Running simulation …"):
        result = simulate_supply_shock(
            percent_drop=params["crisis_severity"],
            base_price=params["base_price"],
            monthly_litres=params["monthly_litres"],
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <p>Brent Crude (Shocked)</p>
          <h2>${result.shocked_brent_price:.2f}</h2>
          <p>was ${result.base_brent_price:.2f}/bbl</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        sign = "+" if result.brent_price_delta > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
          <p>Price Change</p>
          <h2>{sign}{result.brent_price_delta:.2f}</h2>
          <p>{sign}{result.brent_price_pct:.1f}% USD/bbl</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
          <p>Global CPI Impact</p>
          <h2>+{result.global_inflation_proxy:.3f} pp</h2>
          <p>Average across countries</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        country_impact = next(
            (c for c in result.country_impacts if c.iso3 == params["iso3"]), None
        )
        if country_impact:
            st.markdown(f"""
            <div class="metric-card">
              <p>{country_impact.name} Monthly Cost Δ</p>
              <h2>${country_impact.monthly_cost_increase:+.2f}</h2>
              <p>per {params['monthly_litres']:.0f} L/month</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        impact_df = result.to_dataframe()
        fig = plotly_country_impact(
            impact_df,
            f"{params['crisis_severity']}% Supply Drop | Base: ${params['base_price']:.0f}/bbl"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Country Detail")
        if country_impact:
            detail_data = {
                "Metric": [
                    "Base Retail (USD/L)", "New Retail (USD/L)",
                    "Retail Δ (USD/L)",    "Retail Δ (%)",
                    "Monthly Cost Δ (USD)","CPI Contribution (pp)",
                ],
                "Value": [
                    f"${country_impact.base_retail_price_usd:.4f}",
                    f"${country_impact.new_retail_price_usd:.4f}",
                    f"${country_impact.retail_price_delta:+.4f}",
                    f"{country_impact.retail_price_pct:+.2f}%",
                    f"${country_impact.monthly_cost_increase:+.2f}",
                    f"{country_impact.inflation_contribution:+.4f}",
                ],
            }
            st.dataframe(pd.DataFrame(detail_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Multi-Scenario Sensitivity Analysis")
    with st.spinner("Running sweep across 5–30% disruption range …"):
        sweep = run_scenario_sweep(drops=[5, 10, 15, 20, 25, 30], base_price=params["base_price"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plotly_scenario_sweep(sweep), use_container_width=True)
    with c2:
        st.plotly_chart(plotly_shock_heatmap(sweep), use_container_width=True)

    with st.expander("📄 View Full Impact Table"):
        st.dataframe(
            impact_df.sort_values("retail_price_pct", ascending=False)
            .style.format({
                "base_retail_price_usd": "${:.4f}",
                "new_retail_price_usd":  "${:.4f}",
                "retail_price_delta":    "${:+.4f}",
                "retail_price_pct":      "{:+.2f}%",
                "monthly_cost_increase": "${:+.2f}",
                "inflation_contribution":"{:+.4f}",
            }),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Historical Explorer
# ─────────────────────────────────────────────────────────────────────────────

def tab_history(master_df: pd.DataFrame) -> None:
    st.header("📈 Historical Oil Price Explorer")

    if "brent_crude" not in master_df.columns:
        st.warning("No historical data loaded.")
        return

    min_date = master_df.index.min().date()
    max_date = master_df.index.max().date()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

    df_filtered = master_df.loc[str(start):str(end)]

    fig = go.Figure()
    if "brent_crude" in df_filtered.columns:
        fig.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["brent_crude"],
            name="Brent Crude", line=dict(color="#D32F2F", width=2),
        ))
    if "wti_crude" in df_filtered.columns:
        fig.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["wti_crude"],
            name="WTI Crude", line=dict(color="#1565C0", width=1.5, dash="dash"),
        ))

    crisis_meta = [
        ("2003-02-01", "2003-06-30", "Gulf War II",        "rgba(211,47,47,0.12)"),
        ("2007-06-01", "2008-12-31", "GFC Spike",          "rgba(255,111,0,0.12)"),
        ("2011-01-01", "2012-06-30", "Arab Spring",        "rgba(21,101,192,0.12)"),
        ("2014-07-01", "2016-03-31", "OPEC Price War",     "rgba(46,125,50,0.12)"),
        ("2020-01-01", "2020-06-30", "COVID Crash",        "rgba(106,27,154,0.12)"),
        ("2022-02-01", "2022-12-31", "Ukraine War",        "rgba(191,54,12,0.12)"),
        ("2023-10-01", "2024-04-30", "Israel–Hamas–Iran",  "rgba(183,28,28,0.15)"),
        ("2025-01-01", "2025-06-30", "US–Iran Tensions",   "rgba(130,0,0,0.12)"),
    ]
    for s, e, label, color in crisis_meta:
        try:
            if pd.Timestamp(s) <= df_filtered.index.max() and pd.Timestamp(e) >= df_filtered.index.min():
                fig.add_vrect(
                    x0=max(s, str(df_filtered.index.min().date())),
                    x1=min(e, str(df_filtered.index.max().date())),
                    fillcolor=color, layer="below",
                    annotation_text=label, annotation_position="top left",
                    annotation=dict(font_size=9),
                )
        except Exception:
            pass

    fig.update_layout(
        title="Brent & WTI Crude Oil Prices (2000–2026)",
        yaxis_title="USD / Barrel",
        height=480,
        legend=dict(x=0, y=1.1, orientation="h"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📅 Key Geopolitical Events Timeline"):
        events_df = pd.DataFrame([
            {"Date": "Feb 2022", "Event": "Russia invades Ukraine — Brent spikes to $127/bbl"},
            {"Date": "Oct 2023", "Event": "Hamas attacks Israel (Oct 7) — Middle East risk premium returns"},
            {"Date": "Jan 2024", "Event": "Houthi attacks on Red Sea shipping — global freight costs surge"},
            {"Date": "Apr 2024", "Event": "Iran launches 300+ drones & missiles at Israel — Brent jumps ~$3"},
            {"Date": "Oct 2024", "Event": "Israel strikes Iranian military sites — short-term spike"},
            {"Date": "Jan 2025", "Event": "US re-imposes maximum pressure sanctions on Iran"},
            {"Date": "Apr 2025", "Event": "OPEC+ surprise output hike weighs on prices"},
            {"Date": "2026",     "Event": "Gradual de-escalation — diplomatic channels open"},
        ])
        st.dataframe(events_df, hide_index=True, use_container_width=True)

    st.subheader("Descriptive Statistics (filtered range)")
    show_cols = [c for c in ["brent_crude", "wti_crude", "natural_gas", "us_cpi_energy"]
                 if c in df_filtered.columns]
    st.dataframe(df_filtered[show_cols].describe().round(2), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Model Comparison  (FIXED: builds features inline, no get_train_test)
# ─────────────────────────────────────────────────────────────────────────────

def tab_models(master_df: pd.DataFrame) -> None:
    st.header("🤖 Model Comparison")
    st.caption("Train and compare Baseline Ridge, ARIMA, Random Forest, and XGBoost models.")

    train_button = st.button("🧠 Train All Models", type="primary")

    if train_button:
        with st.spinner("Training models … this may take 2–3 minutes."):
            try:
                from modeling import ModelComparison

                # Build lag/rolling features directly from master_df
                df_feat = _build_features(master_df)

                # Split chronologically — no dependency on get_train_test
                train, test = _split_train_test(df_feat, split_date="2022-01-01")

                if len(train) == 0 or len(test) == 0:
                    st.error(f"Split produced empty set — train:{len(train)} test:{len(test)}")
                    return

                mc = ModelComparison()
                metrics_df, preds_df = mc.run(train, test, fit_arima=False)

                st.session_state["metrics_df"] = metrics_df
                st.session_state["preds_df"]   = preds_df
                st.session_state["mc"]         = mc

            except Exception as exc:
                st.error(f"Training failed: {exc}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "metrics_df" in st.session_state:
        metrics_df = st.session_state["metrics_df"]
        preds_df   = st.session_state["preds_df"]
        mc         = st.session_state.get("mc")

        st.subheader("Model Performance Metrics")
        st.dataframe(
            metrics_df.style
                .highlight_min(color="#1b5e20", subset=["RMSE", "MAE"])
                .highlight_max(color="#1b5e20", subset=["R2"]),
            use_container_width=True,
        )

        st.subheader("Predictions vs Actual (Test Set)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=preds_df.index, y=preds_df["actual"],
            name="Actual", line=dict(color="#37474F", width=2.5),
        ))
        colors = ["#D32F2F", "#1565C0", "#2E7D32", "#FF6F00"]
        for i, col in enumerate([c for c in preds_df.columns if c != "actual"]):
            fig.add_trace(go.Scatter(
                x=preds_df.index, y=preds_df[col],
                name=col, line=dict(color=colors[i % len(colors)], width=1.5, dash="dot"),
            ))
        fig.update_layout(height=420, hovermode="x unified", yaxis_title="Brent (USD/bbl)")
        st.plotly_chart(fig, use_container_width=True)

        if mc and "XGBoost" in mc.models:
            xgb = mc.models["XGBoost"]
            if hasattr(xgb, "feature_importances_"):
                st.subheader("XGBoost Feature Importances")
                top = xgb.feature_importances_.head(15)
                fi_fig = go.Figure(go.Bar(
                    x=top.values[::-1], y=top.index[::-1],
                    orientation="h", marker_color="#1565C0",
                ))
                fi_fig.update_layout(height=350, yaxis_title="", xaxis_title="Importance")
                st.plotly_chart(fi_fig, use_container_width=True)
    else:
        st.info("Click **Train All Models** to train and compare models on historical data.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Sentiment Analysis
# ─────────────────────────────────────────────────────────────────────────────

def tab_sentiment() -> None:
    st.header("📰 News Sentiment Analysis")
    st.caption("Analyse recent oil market headlines and track sentiment over time.")

    run_sentiment = st.button("📡 Fetch & Score Headlines", type="primary")
    if run_sentiment:
        with st.spinner("Fetching and scoring headlines …"):
            try:
                from sentiment import fetch_headlines, score_headlines, aggregate_daily_sentiment
                import os
                api_key   = os.getenv("GNEWS_API_KEY", "")
                headlines = fetch_headlines(api_key=api_key)
                scored    = score_headlines(headlines)
                daily     = aggregate_daily_sentiment(scored)
                st.session_state["scored_headlines"] = scored
                st.session_state["daily_sentiment"]  = daily
            except Exception as exc:
                st.error(f"Sentiment analysis failed: {exc}")
                return

    if "scored_headlines" in st.session_state:
        scored = st.session_state["scored_headlines"]
        daily  = st.session_state["daily_sentiment"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Headlines Analysed", len(scored))
        c2.metric("Avg Polarity",       f"{scored['polarity'].mean():.3f}")
        c3.metric("Dominant Label",     scored["label"].mode()[0].title())

        label_counts = scored["label"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=label_counts.index, values=label_counts.values,
            marker_colors=["#2E7D32", "#FF6F00", "#D32F2F"],
            hole=0.4,
        ))
        fig_pie.update_layout(title="Sentiment Distribution", height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

        if not daily.empty:
            fig_line = go.Figure(go.Scatter(
                x=daily.index, y=daily["mean_polarity"],
                fill="tozeroy", line=dict(color="#1565C0"),
                name="Daily Mean Polarity",
            ))
            fig_line.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_line.update_layout(title="Daily Sentiment Trend", height=320,
                                   yaxis_title="Polarity Score")
            st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Headline Details")
        display_cols = [c for c in ["date", "source", "headline", "polarity", "label"]
                        if c in scored.columns]
        st.dataframe(
            scored[display_cols]
            .sort_values("date", ascending=False)
            .style.applymap(
                lambda v: "color: #4CAF50" if v == "positive" else
                          ("color: #F44336" if v == "negative" else ""),
                subset=["label"],
            ),
            use_container_width=True,
        )
    else:
        st.info("Click **Fetch & Score Headlines** to load and analyse recent news.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    params    = render_sidebar()
    master_df = load_master()

    st.title("⛽ Global Fuel Crisis Analyzer")
    st.caption(
        "End-to-end data science system for analysing and predicting the impact "
        "of global fuel crises on oil prices and country-level fuel costs."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Supply Shock Simulator",
        "📈 Historical Explorer",
        "🤖 Model Comparison",
        "📰 News Sentiment",
    ])

    with tab1:
        tab_simulator(params, master_df)
    with tab2:
        tab_history(master_df)
    with tab3:
        tab_models(master_df)
    with tab4:
        tab_sentiment()


if __name__ == "__main__":
    main()
