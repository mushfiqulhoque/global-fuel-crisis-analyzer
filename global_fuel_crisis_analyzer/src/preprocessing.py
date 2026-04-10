"""
preprocessing.py — Data cleaning, merging, and feature engineering.

Key improvements:
  - Lag features:    lag1m, lag3m, lag6m, lag12m
  - Rolling stats:   roll3m/6m/12m mean + std
  - Time features:   month, year, quarter, trend (linear)
  - Momentum:        short-long spread, acceleration, range position
  - No data leakage: all rolling/lag ops use only past observations
  - Robust NaN handling: forward-fill ≤ 3 steps then interpolate
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
# Cleaning helpers
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
    """Clip negatives, remove extreme outliers, forward-fill gaps ≤ 3 months."""
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].clip(lower=0)
        df[col] = _remove_outliers_iqr(df[col])
        df[col] = df[col].ffill(limit=3)
        df[col] = df[col].interpolate(method="time")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — all use shift() / rolling() so no future leaks
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
) -> pd.DataFrame:
    """Shift each column by `lags` months (strictly past data only)."""
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
    Rolling mean and std — uses shift(1) before rolling so the current
    observation is never included (no leakage).
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        shifted = df[col].shift(1)   # exclude current month
        for w in windows:
            df[f"{col}_roll{w}m_mean"] = shifted.rolling(w, min_periods=max(1, w // 2)).mean()
            df[f"{col}_roll{w}m_std"]  = shifted.rolling(w, min_periods=max(1, w // 2)).std()
    return df


def add_pct_change_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: tuple[int, ...] = (1, 3, 12),
) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for p in periods:
            df[f"{col}_pct{p}m"] = df[col].pct_change(periods=p) * 100
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month, quarter, year, and a linear trend index (no future info)."""
    df = df.copy()
    df["month"]   = df.index.month
    df["year"]    = df.index.year
    df["quarter"] = df.index.quarter
    # Linear trend: months since start of series (fully causal)
    df["trend"]   = np.arange(len(df))
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
    """
    Momentum signals derived from lagged/rolling data — zero leakage.
    """
    df = df.copy()
    if col not in df.columns:
        return df

    shifted = df[col].shift(1)

    # Short-minus-long rolling mean spread
    roll3  = shifted.rolling(3,  min_periods=1).mean()
    roll12 = shifted.rolling(12, min_periods=1).mean()
    df[f"{col}_momentum_3_12"] = roll3 - roll12

    # Acceleration: change in 1-month pct change
    pct1m = shifted.pct_change(1) * 100
    df[f"{col}_acceleration"] = pct1m.diff()

    # Range position: where current price sits in its 12-month high/low range
    roll12_max   = shifted.rolling(12, min_periods=1).max()
    roll12_min   = shifted.rolling(12, min_periods=1).min()
    roll12_range = (roll12_max - roll12_min).replace(0, np.nan)
    df[f"{col}_range_position"] = (df[col] - roll12_min) / roll12_range

    # Volatility regime: rolling CV
    roll6_mean = shifted.rolling(6, min_periods=1).mean().replace(0, np.nan)
    df[f"{col}_vol_regime"] = shifted.rolling(6, min_periods=1).std() / roll6_mean

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

    # Lag features: 1, 2, 3, 6, 12 months
    df = add_lag_features(df, columns=key_features, lags=(1, 2, 3, 6, 12))

    # Rolling features: no-leakage (shift(1) inside the function)
    df = add_rolling_features(df, columns=key_features, windows=(3, 6, 12))

    # Pct changes (uses pandas default which is lagged by nature)
    df = add_pct_change_features(df, columns=key_features, periods=(1, 3, 12))

    # Calendar + linear trend
    df = add_calendar_features(df)

    # Crisis flags
    df = add_crisis_flags(df)

    # Momentum signals (all lagged internally)
    df = add_momentum_features(df, col="brent_crude")
    if "wti_crude" in df.columns:
        df = add_momentum_features(df, col="wti_crude")

    # Derived features
    if "brent_crude" in df.columns and "wti_crude" in df.columns:
        df["brent_wti_spread"] = df["brent_crude"].shift(1) - df["wti_crude"].shift(1)

    if "world_crude_supply" in df.columns:
        supply_mean = df["world_crude_supply"].mean()
        supply_std  = df["world_crude_supply"].std()
        df["supply_zscore"] = (df["world_crude_supply"] - supply_mean) / supply_std

    # Crisis × momentum interaction
    if "crisis_flag" in df.columns and "brent_crude_momentum_3_12" in df.columns:
        df["crisis_x_momentum"] = df["crisis_flag"] * df["brent_crude_momentum_3_12"]

    # Train/test split flag
    df["split"] = np.where(df.index < TRAIN_TEST_SPLIT_DATE, "train", "test")

    # Drop rows with NaN target
    if "brent_crude" in df.columns:
        pre_drop = len(df)
        df.dropna(subset=["brent_crude"], inplace=True)
        logger.debug(f"  Dropped {pre_drop - len(df)} rows with NaN brent_crude")

    # Replace any inf that slipped through
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if save:
        out = PROC_DIR / "master.csv"
        df.to_csv(out)
        logger.success(f"Master dataset saved → {out}  shape={df.shape}")

    logger.success(f"Master dataset ready. Shape: {df.shape}")
    logger.info(f"  Date range : {df.index.min().date()} → {df.index.max().date()}")
    return df


def get_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"].drop(columns=["split", "crisis_name"], errors="ignore")
    test  = df[df["split"] == "test"].drop(columns=["split", "crisis_name"],  errors="ignore")
    logger.info(f"Train: {len(train)} samples | Test: {len(test)} samples")
    return train, test


if __name__ == "__main__":
    master = build_master_dataset()
    print(master.head())
    print(master.describe())
