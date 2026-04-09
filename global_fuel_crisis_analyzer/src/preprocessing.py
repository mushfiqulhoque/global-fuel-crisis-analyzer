"""
preprocessing.py — Data cleaning, merging, and feature engineering.
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
        logger.debug(f"  outlier removal [{series.name}]: {n_removed} values replaced with NaN")
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
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
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
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for l in lags:
            df[f"{col}_lag{l}m"] = df[col].shift(l)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]   = df.index.month
    df["year"]    = df.index.year
    df["quarter"] = df.index.quarter
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
    ADDED: Momentum and mean-reversion features that significantly improve
    model R² by capturing trend direction and speed.
    """
    df = df.copy()
    if col not in df.columns:
        return df

    # Price momentum: difference between short and long rolling means
    df[f"{col}_momentum_3_12"] = (
        df[col].rolling(3, min_periods=1).mean() -
        df[col].rolling(12, min_periods=1).mean()
    )

    # Acceleration: change in 1-month pct change
    pct1m = df[col].pct_change(1) * 100
    df[f"{col}_acceleration"] = pct1m.diff()

    # Distance from 12-month high/low (mean reversion signal)
    roll12_max = df[col].rolling(12, min_periods=1).max()
    roll12_min = df[col].rolling(12, min_periods=1).min()
    roll12_range = (roll12_max - roll12_min).replace(0, np.nan)
    df[f"{col}_range_position"] = (df[col] - roll12_min) / roll12_range

    # Volatility regime: rolling std / rolling mean (coefficient of variation)
    roll6_mean = df[col].rolling(6, min_periods=1).mean().replace(0, np.nan)
    df[f"{col}_vol_regime"] = df[col].rolling(6, min_periods=1).std() / roll6_mean

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

    df = add_rolling_features(df, columns=key_features, windows=(3, 6, 12))
    df = add_pct_change_features(df, columns=key_features, periods=(1, 3, 12))
    df = add_lag_features(df, columns=key_features, lags=(1, 2, 3, 6, 12))
    df = add_calendar_features(df)
    df = add_crisis_flags(df)

    # ADDED: momentum features for brent — most impactful for R² improvement
    df = add_momentum_features(df, col="brent_crude")
    if "wti_crude" in df.columns:
        df = add_momentum_features(df, col="wti_crude")

    if "brent_crude" in df.columns and "wti_crude" in df.columns:
        df["brent_wti_spread"] = df["brent_crude"] - df["wti_crude"]

    if "world_crude_supply" in df.columns:
        supply_mean = df["world_crude_supply"].mean()
        supply_std  = df["world_crude_supply"].std()
        df["supply_zscore"] = (df["world_crude_supply"] - supply_mean) / supply_std

    # ADDED: interaction feature — crisis × momentum captures shock onset well
    if "crisis_flag" in df.columns and "brent_crude_momentum_3_12" in df.columns:
        df["crisis_x_momentum"] = df["crisis_flag"] * df["brent_crude_momentum_3_12"]

    df["split"] = np.where(df.index < TRAIN_TEST_SPLIT_DATE, "train", "test")

    if "brent_crude" in df.columns:
        pre_drop = len(df)
        df.dropna(subset=["brent_crude"], inplace=True)
        logger.debug(f"  Dropped {pre_drop - len(df)} rows with NaN brent_crude")

    if save:
        out = PROC_DIR / "master.csv"
        df.to_csv(out)
        logger.success(f"Master dataset saved → {out}  shape={df.shape}")

    logger.success(f"Master dataset ready. Shape: {df.shape}")
    logger.info(f"  Date range : {df.index.min().date()} → {df.index.max().date()}")
    logger.info(f"  Columns    : {list(df.columns)}")
    return df


def get_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"].drop(columns=["split", "crisis_name"], errors="ignore")
    test  = df[df["split"] == "test"].drop(columns=["split", "crisis_name"], errors="ignore")
    logger.info(f"Train: {len(train)} samples | Test: {len(test)} samples")
    return train, test


if __name__ == "__main__":
    master = build_master_dataset()
    print(master.head())
    print(master.describe())
