"""
visualization.py — All chart generation for the Global Fuel Crisis Analyzer.

Produces:
  1. oil_price_trend        — Brent / WTI historical time series with crisis bands
  2. crisis_comparison      — Box-and-whisker: crisis vs normal periods
  3. model_predictions      — Actual vs predicted for all models
  4. feature_importance     — Top N features from RF / XGBoost
  5. country_impact_bar     — Simulation retail price impact by country
  6. supply_shock_heatmap   — Country × severity shock heatmap
  7. scenario_sweep_lines   — Price trajectory across supply drop scenarios

All functions accept an `ax` / `fig` parameter for embedding in notebooks
or return a standalone Plotly figure for the Streamlit dashboard.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from loguru import logger
from matplotlib.patches import FancyArrowPatch

# ── Global Matplotlib style ────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {
    "brent":    "#D32F2F",
    "wti":      "#1565C0",
    "crisis":   "#FF6F00",
    "normal":   "#2E7D32",
    "forecast": "#6A1B9A",
    "actual":   "#37474F",
}

CRISIS_BANDS = [
    ("2003-02-01", "2003-06-30", "Gulf War II"),
    ("2007-06-01", "2008-12-31", "GFC Spike"),
    ("2011-01-01", "2012-06-30", "Arab Spring"),
    ("2014-07-01", "2016-03-31", "OPEC Price War"),
    ("2020-01-01", "2020-06-30", "COVID Crash"),
    ("2022-02-01", "2022-12-31", "Ukraine War"),
]


def _add_crisis_bands(ax: plt.Axes, alpha: float = 0.12) -> None:
    """Overlay named crisis periods as shaded rectangles on a time-series axes."""
    colors = ["#FF6F00", "#C62828", "#283593", "#1B5E20", "#4A148C", "#BF360C"]
    for i, (start, end, label) in enumerate(CRISIS_BANDS):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=alpha, color=colors[i % len(colors)], label=f"■ {label}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Oil price trend
# ─────────────────────────────────────────────────────────────────────────────

def plot_oil_price_trend(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot Brent and WTI crude prices as overlapping time series with crisis bands.

    Parameters
    ----------
    df        : master DataFrame with 'brent_crude' and/or 'wti_crude' columns
    ax        : optional matplotlib Axes to draw on
    save_path : if provided, save figure to this path
    """
    fig_created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    if "brent_crude" in df.columns:
        ax.plot(df.index, df["brent_crude"], color=PALETTE["brent"], lw=1.8,
                label="Brent Crude (USD/bbl)", alpha=0.9)
    if "wti_crude" in df.columns:
        ax.plot(df.index, df["wti_crude"], color=PALETTE["wti"], lw=1.2,
                label="WTI Crude (USD/bbl)", alpha=0.7, ls="--")

    _add_crisis_bands(ax)

    ax.set_title("Global Crude Oil Prices 2000–Present", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("USD / Barrel", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(loc="upper left", fontsize=9, ncol=4, framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%g"))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {save_path}")

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 2. Crisis vs normal comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_crisis_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Box plots comparing Brent crude price distribution during crisis vs normal
    periods, broken out by named crisis event.
    """
    col = "brent_crude"
    if col not in df.columns:
        raise ValueError("DataFrame must contain 'brent_crude' column.")

    plot_df = df[[col, "crisis_name"]].dropna()
    order = (
        ["normal"]
        + [c[2].lower().replace(" ", "_") for c in CRISIS_BANDS
           if c[2].lower().replace(" ", "_") in plot_df["crisis_name"].unique()]
    )
    order = [o for o in order if o in plot_df["crisis_name"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: box plot by crisis name
    sns.boxplot(
        data=plot_df,
        x="crisis_name",
        y=col,
        order=order,
        palette=["#2E7D32" if o == "normal" else "#D32F2F" for o in order],
        ax=axes[0],
    )
    axes[0].set_title("Price Distribution: Crisis vs Normal", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Brent (USD/bbl)")
    axes[0].tick_params(axis="x", rotation=30)

    # Right: violin plot of crisis vs normal (binary)
    binary_df = plot_df.copy()
    binary_df["period"] = np.where(binary_df["crisis_name"] == "normal", "Normal", "Crisis")
    sns.violinplot(
        data=binary_df,
        x="period",
        y=col,
        palette={"Normal": PALETTE["normal"], "Crisis": PALETTE["crisis"]},
        inner="quartile",
        ax=axes[1],
    )
    axes[1].set_title("Crisis vs Normal Periods (Violin)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Brent (USD/bbl)")

    plt.suptitle("Oil Price Behaviour During Crisis Periods", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model predictions vs actual
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_predictions(
    preds_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel chart: actual vs predicted for each model.

    Parameters
    ----------
    preds_df : DataFrame with 'actual' column and one column per model name.
    """
    model_cols = [c for c in preds_df.columns if c != "actual"]
    n          = len(model_cols)
    fig, axes  = plt.subplots(n, 1, figsize=(13, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = ["#6A1B9A", "#1565C0", "#2E7D32", "#BF360C"]
    for i, (col, ax) in enumerate(zip(model_cols, axes)):
        ax.plot(preds_df.index, preds_df["actual"],
                color=PALETTE["actual"], lw=2.0, label="Actual", alpha=0.9)
        ax.plot(preds_df.index, preds_df[col],
                color=colors[i % len(colors)], lw=1.5, ls="--",
                label=f"Predicted ({col})", alpha=0.85)
        ax.fill_between(preds_df.index, preds_df["actual"], preds_df[col],
                        alpha=0.10, color=colors[i % len(colors)])
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("USD/bbl")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%g"))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    plt.suptitle("Model Predictions vs Actual Oil Price (Test Set)", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    importance_series: pd.Series,
    model_name: str = "Model",
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    top = importance_series.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    bars = ax.barh(top.index, top.values, color="#1565C0", alpha=0.85, edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", ha="left", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Country impact bar chart (Plotly — used in dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def plotly_country_impact(impact_df: pd.DataFrame, scenario_name: str = "") -> go.Figure:
    """
    Interactive Plotly bar chart of retail fuel price impact per country.

    Parameters
    ----------
    impact_df     : result.to_dataframe() from ShockResult
    scenario_name : title label
    """
    df = impact_df.sort_values("retail_price_pct", ascending=True)

    fig = px.bar(
        df,
        x="retail_price_pct",
        y="name",
        orientation="h",
        color="retail_price_pct",
        color_continuous_scale="RdYlGn_r",
        labels={"retail_price_pct": "Retail Price Change (%)", "name": "Country"},
        title=f"Retail Fuel Price Impact by Country<br><sup>{scenario_name}</sup>",
        text=df["retail_price_pct"].apply(lambda v: f"{v:+.1f}%"),
        hover_data=["iso3", "base_retail_price_usd", "new_retail_price_usd",
                    "monthly_cost_increase", "inflation_contribution"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(400, len(df) * 30),
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Retail Price Change (%)",
        margin=dict(l=130, r=40, t=80, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Supply shock heatmap (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def plotly_shock_heatmap(sweep_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap: rows = countries, columns = supply drop %, values = retail Δ%.

    Parameters
    ----------
    sweep_df : output of simulation.run_scenario_sweep()
    """
    pivot = sweep_df.pivot_table(
        index="name",
        columns="supply_drop_pct",
        values="retail_price_pct",
        aggfunc="first",
    )
    fig = px.imshow(
        pivot,
        labels=dict(x="Supply Drop (%)", y="Country", color="Retail Price Δ (%)"),
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Fuel Price Impact Heatmap: Countries × Supply Shock Severity",
    )
    fig.update_layout(height=500, margin=dict(l=130, r=40, t=80, b=60))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Scenario sweep lines (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def plotly_scenario_sweep(sweep_df: pd.DataFrame, top_countries: int = 8) -> go.Figure:
    """
    Line chart: for each country, plot retail price change across shock severities.
    """
    # Select countries with widest impact range for clarity
    range_by_country = (
        sweep_df.groupby("name")["retail_price_pct"].max()
        - sweep_df.groupby("name")["retail_price_pct"].min()
    ).nlargest(top_countries).index.tolist()

    df_top = sweep_df[sweep_df["name"].isin(range_by_country)]

    fig = px.line(
        df_top,
        x="supply_drop_pct",
        y="retail_price_pct",
        color="name",
        markers=True,
        labels={
            "supply_drop_pct":  "Supply Drop (%)",
            "retail_price_pct": "Retail Price Δ (%)",
            "name":             "Country",
        },
        title="Retail Fuel Price Impact vs Supply Shock Severity",
    )
    fig.update_layout(height=400, margin=dict(t=60))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sentiment over time (optional, requires NLP module)
# ─────────────────────────────────────────────────────────────────────────────

def plotly_sentiment_trend(sentiment_df: pd.DataFrame) -> go.Figure:
    """
    Line chart of news sentiment score alongside oil price.

    Parameters
    ----------
    sentiment_df : DataFrame with columns ['date', 'sentiment', 'brent_crude']
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sentiment_df["date"], y=sentiment_df["brent_crude"],
        name="Brent Crude (USD/bbl)", yaxis="y1",
        line=dict(color=PALETTE["brent"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sentiment_df["date"], y=sentiment_df["sentiment"],
        name="News Sentiment Score", yaxis="y2",
        line=dict(color=PALETTE["forecast"], width=1.5, dash="dot"),
    ))
    fig.update_layout(
        title="News Sentiment vs Oil Price",
        yaxis=dict(title="Brent Crude (USD/bbl)", side="left"),
        yaxis2=dict(title="Sentiment Score", side="right", overlaying="y"),
        legend=dict(x=0, y=1.15, orientation="h"),
        height=400,
    )
    return fig
