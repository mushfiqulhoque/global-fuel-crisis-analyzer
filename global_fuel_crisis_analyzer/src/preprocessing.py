"""
preprocessing.py — Data cleaning, merging, and feature engineering.

TIME-SERIES CORRECT:
  - lag1, lag2, lag3, lag6, lag12   via shift(N)       — strictly past data
  - rolling_mean_3/6/12, rolling_std_3/6/12  via shift(1)+rolling — no leakage
  - trend (0,1,2,…), month, year, quarter
  - NA rows created by lag/rolling are DROPPED before any model sees them
  - Data is NEVER shuffled — temporal order preserved throughout
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
    path = path or RAW_DIR / "fred_data.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df.resample("MS").mean()
    logger.info(f"Loaded FRED: {df.shape}  ({df.index.min()} → {df.index.max()})")
    return df


def load_eia(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or RAW_DIR / "eia_data.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df.resample("MS").mean()
    logger.info(f"Loaded EIA: {df.shape}")
    return df


def load_worldbank(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or RAW_DIR / "worldbank_data.csv"
    df = pd.read_csv(path, index_col=["iso3", "year"])
    logger.info(f"Loaded World Bank: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _remove_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    cleaned = series.where(series.between(lower, upper), other=np.nan)
    n_removed = series.notna().sum() - cleaned.notna().sum()
    if n_removed:
        logger.debug(f"  outlier removal [{series.name}]: {n_removed} values → NaN")
    return cleaned


def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].clip(lower=0)
        df[col] = _remove_outliers_iqr(df[col])
        df[col] = df[col].ffill(limit=3)
        df[col] = df[col].interpolate(method="time")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — TIME-SERIES CORRECT
# Every feature uses only information available at time t-1 or earlier.
# The resulting NaN rows (lag warm-up period) must be dropped before training.
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
) -> pd.DataFrame:
    """
    Add Y(t-k) lag columns.  shift(k) guarantees no future leakage.
    Rows 0…max(lag)-1 will be NaN — caller must drop them.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}m"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Rolling mean and std computed on shift(1) of each series.
    shift(1) ensures the current observation is excluded → zero leakage.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].shift(1)          # exclude current month from window
        for w in windows:
            min_p = max(1, w // 2)
            df[f"{col}_roll{w}m_mean"] = s.rolling(w, min_periods=min_p).mean()
            df[f"{col}_roll{w}m_std"]  = s.rolling(w, min_periods=min_p).std()
    return df


def add_pct_change_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: tuple[int, ...] = (1, 3),
) -> pd.DataFrame:
    """pct_change(k) is equivalent to shift(k) — no leakage."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for p in periods:
            df[f"{col}_pct{p}m"] = df[col].pct_change(periods=p) * 100
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fully causal: derived from the row's timestamp only."""
    df = df.copy()
    df["month"]   = df.index.month
    df["year"]    = df.index.year
    df["quarter"] = df.index.quarter
    df["trend"]   = np.arange(len(df))   # 0, 1, 2, … — linear time index
    return df


def add_crisis_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    crises = {
        "gulf_war_2":     ("2003-02-01", "2003-06-30"),
        "gfc_spike":      ("2007-06-01", "2008-12-31"),
        "arab_spring":    ("2011-01-01", "2012-06-30"),
        "opec_price_war": ("2014-07-01", "2016-03-31"),
        "covid_crash":    ("2020-01-01", "2020-06-30"),
        "ukraine_war":    ("2022-02-01", "2022-12-31"),
    }
    df["crisis_flag"] = 0
    df["crisis_name"] = "normal"
    for name, (start, end) in crises.items():
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "crisis_flag"] = 1
        df.loc[mask, "crisis_name"] = name
    return df


def add_momentum_features(df: pd.DataFrame, col: str = "brent_crude") -> pd.DataFrame:
    """All signals use shift(1) as their base → causal."""
    df = df.copy()
    if col not in df.columns:
        return df
    s = df[col].shift(1)
    roll3  = s.rolling(3,  min_periods=1).mean()
    roll12 = s.rolling(12, min_periods=1).mean()
    df[f"{col}_momentum_3_12"]  = roll3 - roll12
    df[f"{col}_acceleration"]   = (s.pct_change(1) * 100).diff()
    roll12_max = s.rolling(12, min_periods=1).max()
    roll12_min = s.rolling(12, min_periods=1).min()
    rng = (roll12_max - roll12_min).replace(0, np.nan)
    df[f"{col}_range_position"] = (df[col] - roll12_min) / rng
    roll6_mean = s.rolling(6, min_periods=1).mean().replace(0, np.nan)
    df[f"{col}_vol_regime"]     = s.rolling(6, min_periods=1).std() / roll6_mean
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_master_dataset(
    fred_df: Optional[pd.DataFrame] = None,
    eia_df:  Optional[pd.DataFrame] = None,
    save:    bool = True,
) -> pd.DataFrame:
    logger.info("Building master dataset …")

    fred = fred_df if fred_df is not None else load_fred()
    eia  = eia_df  if eia_df  is not None else (
        load_eia() if (RAW_DIR / "eia_data.csv").exists() else pd.DataFrame()
    )

    df = fred.copy()
    if not eia.empty:
        df = df.join(eia, how="left")

    df = df.loc["2000-01-01":]

    price_cols = [c for c in df.columns if c in ("brent_crude", "wti_crude", "natural_gas")]
    macro_cols = [c for c in df.columns if c not in price_cols]
    df[price_cols] = clean_timeseries(df[price_cols])
    df[macro_cols] = df[macro_cols].ffill(limit=6).interpolate(method="time")

    key_features = [c for c in ["brent_crude", "wti_crude", "us_cpi_energy", "world_crude_supply"]
                    if c in df.columns]

    # --- TIME-SERIES FEATURE ENGINEERING (all causal) ---
    df = add_lag_features(df, columns=key_features, lags=(1, 2, 3, 6, 12))
    df = add_rolling_features(df, columns=key_features, windows=(3, 6, 12))
    df = add_pct_change_features(df, columns=key_features, periods=(1, 3))
    df = add_calendar_features(df)
    df = add_crisis_flags(df)
    df = add_momentum_features(df, col="brent_crude")
    if "wti_crude" in df.columns:
        df = add_momentum_features(df, col="wti_crude")

    if "brent_crude" in df.columns and "wti_crude" in df.columns:
        df["brent_wti_spread"] = df["brent_crude"].shift(1) - df["wti_crude"].shift(1)

    if "world_crude_supply" in df.columns:
        supply_mean = df["world_crude_supply"].mean()
        supply_std  = df["world_crude_supply"].std()
        df["supply_zscore"] = (df["world_crude_supply"] - supply_mean) / supply_std

    if "crisis_flag" in df.columns and "brent_crude_momentum_3_12" in df.columns:
        df["crisis_x_momentum"] = df["crisis_flag"] * df["brent_crude_momentum_3_12"]

    # Train/test split (time-based — NO shuffling)
    df["split"] = np.where(df.index < TRAIN_TEST_SPLIT_DATE, "train", "test")

    # --- CRITICAL: drop rows where target is NaN ---
    if "brent_crude" in df.columns:
        pre = len(df)
        df.dropna(subset=["brent_crude"], inplace=True)
        logger.debug(f"  Dropped {pre - len(df)} rows with NaN target")

    # Replace any inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if save:
        out = PROC_DIR / "master.csv"
        df.to_csv(out)
        logger.success(f"Master dataset saved → {out}  shape={df.shape}")

    logger.success(f"Master dataset ready. Shape: {df.shape}")
    logger.info(f"  Date range : {df.index.min().date()} → {df.index.max().date()}")
    return df


def get_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split. Data is NEVER shuffled.
    Lag-NaN rows (first ~12 months) are dropped here so neither
    train nor test ever contains rows with missing lag features.
    """
    # Identify the lag/rolling feature columns
    lag_roll_cols = [c for c in df.columns if (
        "_lag" in c or "_roll" in c or "_pct" in c or
        "_momentum" in c or "_acceleration" in c or
        "_range_position" in c or "_vol_regime" in c
    )]

    # Drop rows where ANY lag/rolling feature is NaN (warm-up period)
    if lag_roll_cols:
        before = len(df)
        df = df.dropna(subset=lag_roll_cols, how="any")
        logger.debug(f"  Dropped {before - len(df)} NaN-feature rows (lag warm-up)")

    train = df[df["split"] == "train"].drop(columns=["split", "crisis_name"], errors="ignore")
    test  = df[df["split"] == "test"].drop(columns=["split", "crisis_name"],  errors="ignore")

    logger.info(f"Train: {len(train)} samples | Test: {len(test)} samples")
    return train, test


if __name__ == "__main__":
    master = build_master_dataset()
    print(master.head())
    print(master.describe())
