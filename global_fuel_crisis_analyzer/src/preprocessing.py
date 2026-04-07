"""
preprocessing.py — Data cleaning, merging, and feature engineering.

Pipeline stages:
  1. Load raw CSVs (or accept DataFrames directly).
  2. Align to a common monthly DatetimeIndex.
  3. Impute / forward-fill missing values sensibly.
  4. Engineer features: rolling stats, % changes, lag windows.
  5. Produce a model-ready master DataFrame saved to data/processed/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Optional

from config import PROC_DIR, RAW_DIR, TRAIN_TEST_SPLIT_DATE

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_fred(path: Optional[Path] = None) -> pd.DataFrame:
    """Load FRED raw CSV into a monthly DatetimeIndex DataFrame."""
    path = path or RAW_DIR / "fred_data.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df.resample("MS").mean()   # resample to Month-Start for alignment
    logger.info(f"Loaded FRED: {df.shape}  ({df.index.min()} → {df.index.max()})")
    return df


def load_eia(path: Optional[Path] = None) -> pd.DataFrame:
    """Load EIA raw CSV."""
    path = path or RAW_DIR / "eia_data.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df.resample("MS").mean()
    logger.info(f"Loaded EIA: {df.shape}")
    return df


def load_worldbank(path: Optional[Path] = None) -> pd.DataFrame:
    """Load World Bank panel data (iso3 × year)."""
    path = path or RAW_DIR / "worldbank_data.csv"
    df = pd.read_csv(path, index_col=["iso3", "year"])
    logger.info(f"Loaded World Bank: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """
    Replace extreme outliers with NaN using IQR fencing.

    factor=3.0 is intentionally permissive — oil prices CAN have genuine
    large swings (e.g., 2020 crash).  We only remove *data errors*, not
    real economic events.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    cleaned = series.where(series.between(lower, upper), other=np.nan)
    n_removed = series.notna().sum() - cleaned.notna().sum()
    if n_removed:
        logger.debug(f"  outlier removal [{series.name}]: {n_removed} values replaced with NaN")
    return cleaned


def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-column cleaning:
      - Remove implausible negatives (prices must be ≥ 0)
      - IQR outlier fencing
      - Forward-fill short gaps (≤ 3 months)
      - Remaining NaN → linear interpolation
    """
    df = df.copy()
    for col in df.columns:
        # Non-negative prices
        df[col] = df[col].clip(lower=0)
        # Outlier removal
        df[col] = _remove_outliers_iqr(df[col])
        # Fill short gaps
        df[col] = df[col].ffill(limit=3)
        # Interpolate residual gaps
        df[col] = df[col].interpolate(method="time")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Add rolling mean and rolling std for selected columns.

    New column naming convention:
      {col}_roll{w}m_mean  —  w-month rolling average
      {col}_roll{w}m_std   —  w-month rolling standard deviation (volatility)
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll{w}m_mean"] = df[col].rolling(w, min_periods=max(1, w // 2)).mean()
            df[f"{col}_roll{w}m_std"]  = df[col].rolling(w, min_periods=max(1, w // 2)).std()
    return df


def add_pct_change_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: tuple[int, ...] = (1, 3, 12),
) -> pd.DataFrame:
    """
    Add percentage-change features at multiple horizons.

    New columns: {col}_pct{p}m — % change over p months
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for p in periods:
            df[f"{col}_pct{p}m"] = df[col].pct_change(periods=p) * 100
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
) -> pd.DataFrame:
    """
    Add lagged versions of selected columns.

    New columns: {col}_lag{l}m — value l months prior
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for l in lags:
            df[f"{col}_lag{l}m"] = df[col].shift(l)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month-of-year and year as integer features for seasonality encoding."""
    df = df.copy()
    df["month"]   = df.index.month
    df["year"]    = df.index.year
    df["quarter"] = df.index.quarter
    return df


def add_crisis_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label known major oil-price-shock windows with a binary flag.

    These are used for model conditioning and in visualisations.
    """
    df = df.copy()
    crises = {
        "gulf_war_2":    ("2003-02-01", "2003-06-30"),
        "gfc_spike":     ("2007-06-01", "2008-12-31"),
        "arab_spring":   ("2011-01-01", "2012-06-30"),
        "opec_price_war":("2014-07-01", "2016-03-31"),
        "covid_crash":   ("2020-01-01", "2020-06-30"),
        "ukraine_war":   ("2022-02-01", "2022-12-31"),
    }
    df["crisis_flag"] = 0
    df["crisis_name"] = "normal"
    for name, (start, end) in crises.items():
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "crisis_flag"] = 1
        df.loc[mask, "crisis_name"] = name
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_master_dataset(
    fred_df: Optional[pd.DataFrame] = None,
    eia_df:  Optional[pd.DataFrame] = None,
    save:    bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline that produces the model-ready master dataset.

    Parameters
    ----------
    fred_df : Pre-loaded FRED DataFrame (or None to load from CSV)
    eia_df  : Pre-loaded EIA DataFrame  (or None to load from CSV)
    save    : Whether to persist to data/processed/master.csv

    Returns
    -------
    pd.DataFrame — clean, feature-rich time series indexed by date
    """
    logger.info("Building master dataset …")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    fred = fred_df if fred_df is not None else load_fred()
    eia  = eia_df  if eia_df  is not None else (
        load_eia() if (RAW_DIR / "eia_data.csv").exists() else pd.DataFrame()
    )

    # ── 2. Merge ──────────────────────────────────────────────────────────────
    df = fred.copy()
    if not eia.empty:
        df = df.join(eia, how="left")

    # ── 3. Restrict to date range ──────────────────────────────────────────────
    df = df.loc["2000-01-01":]

    # ── 4. Clean ──────────────────────────────────────────────────────────────
    price_cols   = [c for c in df.columns if c in ("brent_crude", "wti_crude", "natural_gas")]
    macro_cols   = [c for c in df.columns if c not in price_cols]
    df[price_cols] = clean_timeseries(df[price_cols])
    df[macro_cols] = df[macro_cols].ffill(limit=6).interpolate(method="time")

    # ── 5. Feature engineering ─────────────────────────────────────────────────
    key_features = [c for c in ["brent_crude", "wti_crude", "us_cpi_energy", "world_crude_supply"] if c in df.columns]

    df = add_rolling_features(df, columns=key_features, windows=(3, 6, 12))
    df = add_pct_change_features(df, columns=key_features, periods=(1, 3, 12))
    df = add_lag_features(df, columns=key_features, lags=(1, 2, 3, 6, 12))
    df = add_calendar_features(df)
    df = add_crisis_flags(df)

    # ── 6. Brent-WTI spread (market-stress indicator) ─────────────────────────
    if "brent_crude" in df.columns and "wti_crude" in df.columns:
        df["brent_wti_spread"] = df["brent_crude"] - df["wti_crude"]

    # ── 7. Normalised supply disruption index ────────────────────────────────
    if "world_crude_supply" in df.columns:
        supply_mean = df["world_crude_supply"].mean()
        supply_std  = df["world_crude_supply"].std()
        df["supply_zscore"] = (df["world_crude_supply"] - supply_mean) / supply_std

    # ── 8. Train / test split column ─────────────────────────────────────────
    df["split"] = np.where(df.index < TRAIN_TEST_SPLIT_DATE, "train", "test")

    # ── 9. Drop rows with NaN in the primary target ──────────────────────────
    if "brent_crude" in df.columns:
        pre_drop = len(df)
        df.dropna(subset=["brent_crude"], inplace=True)
        logger.debug(f"  Dropped {pre_drop - len(df)} rows with NaN brent_crude")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    if save:
        out = PROC_DIR / "master.csv"
        df.to_csv(out)
        logger.success(f"Master dataset saved → {out}  shape={df.shape}")

    logger.success(f"Master dataset ready. Shape: {df.shape}")
    logger.info(f"  Date range : {df.index.min().date()} → {df.index.max().date()}")
    logger.info(f"  Columns    : {list(df.columns)}")
    return df


def get_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split master dataset into train and test sets based on the date configured
    in TRAIN_TEST_SPLIT_DATE.
    """
    train = df[df["split"] == "train"].drop(columns=["split", "crisis_name"], errors="ignore")
    test  = df[df["split"] == "test"].drop(columns=["split", "crisis_name"], errors="ignore")
    logger.info(f"Train: {len(train)} samples | Test: {len(test)} samples")
    return train, test


if __name__ == "__main__":
    master = build_master_dataset()
    print(master.head())
    print(master.describe())
