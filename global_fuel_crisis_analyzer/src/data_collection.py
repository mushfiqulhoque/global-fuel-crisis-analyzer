"""
data_collection.py — Unified API client for all external data sources.

Integrates:
  • FRED   (Federal Reserve Economic Data) — oil prices, CPI
  • EIA    (U.S. Energy Information Administration) — world crude supply
  • World Bank — country-level economic & energy indicators

All HTTP responses are cached to disk (SQLite via requests-cache) to avoid
hammering rate limits during development and to make runs reproducible.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import requests_cache
from loguru import logger

from config import (
    CACHE_DIR,
    CACHE_EXPIRE_HOURS,
    COUNTRIES,
    EIA_API_KEY,
    EIA_SERIES,
    FRED_API_KEY,
    FRED_SERIES,
    RAW_DIR,
    WB_INDICATORS,
)

# ── Install a persistent SQLite cache ─────────────────────────────────────────
requests_cache.install_cache(
    str(CACHE_DIR / "api_cache"),
    backend="sqlite",
    expire_after=timedelta(hours=CACHE_EXPIRE_HOURS),
)
logger.info(f"Request cache active at {CACHE_DIR / 'api_cache'}.sqlite")


# ══════════════════════════════════════════════════════════════════════════════
# FRED Client
# ══════════════════════════════════════════════════════════════════════════════

class FREDClient:
    """Thin wrapper around the FRED REST API v1."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str = FRED_API_KEY):
        if not api_key:
            logger.warning(
                "FRED_API_KEY not set — unauthenticated requests may be "
                "rate-limited. Set FRED_API_KEY in your .env file."
            )
        self.api_key = api_key

    def fetch(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        frequency: str = "m",       # d=daily, w=weekly, m=monthly, q=quarterly
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series and return a tidy DataFrame.

        Parameters
        ----------
        series_id : FRED series identifier (e.g., 'DCOILBRENTEU')
        start     : ISO date string, inclusive lower bound
        end       : ISO date string, defaults to today
        frequency : Aggregation frequency ('d', 'w', 'm', 'q', 'a')

        Returns
        -------
        pd.DataFrame with columns ['date', series_id]
        """
        end = end or datetime.today().strftime("%Y-%m-%d")
        params = {
            "series_id":             series_id,
            "observation_start":     start,
            "observation_end":       end,
            "frequency":             frequency,
            "aggregation_method":    "avg",
            "file_type":             "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        logger.info(f"FRED fetch → {series_id} ({start} → {end}, freq={frequency})")
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()

        observations = resp.json().get("observations", [])
        df = pd.DataFrame(observations)[["date", "value"]]
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.rename(columns={"value": series_id}, inplace=True)
        df.set_index("date", inplace=True)
        logger.success(f"  ✓ {len(df)} observations for {series_id}")
        return df

    def fetch_all(self, start: str = "2000-01-01") -> pd.DataFrame:
        """Fetch every configured FRED series and merge into one DataFrame."""
        frames: list[pd.DataFrame] = []
        for label, sid in FRED_SERIES.items():
            try:
                df = self.fetch(sid, start=start)
                df.columns = [label]          # rename to human-friendly label
                frames.append(df)
            except Exception as exc:
                logger.error(f"FRED: failed to fetch {sid}: {exc}")

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for df in frames[1:]:
            merged = merged.join(df, how="outer")
        merged.index.name = "date"
        return merged


# ══════════════════════════════════════════════════════════════════════════════
# EIA Client
# ══════════════════════════════════════════════════════════════════════════════

class EIAClient:
    """Client for the EIA Open Data API v2."""

    BASE_URL_V2 = "https://api.eia.gov/v2"
    BASE_URL_V1 = "https://api.eia.gov/series/"   # fallback for legacy series IDs

    def __init__(self, api_key: str = EIA_API_KEY):
        if not api_key:
            logger.warning(
                "EIA_API_KEY not set — EIA requests require a free key. "
                "Register at https://www.eia.gov/opendata/register.php"
            )
        self.api_key = api_key

    def fetch_series_v1(self, series_id: str) -> pd.DataFrame:
        """
        Fetch a time series using the legacy EIA v1 API endpoint.

        Returns
        -------
        pd.DataFrame indexed by date with one column named series_id.
        """
        if not self.api_key:
            logger.warning(f"Skipping EIA series {series_id} — no API key.")
            return pd.DataFrame()

        url = self.BASE_URL_V1
        params = {
            "series_id": series_id,
            "api_key":   self.api_key,
            "out":       "json",
        }
        logger.info(f"EIA v1 fetch → {series_id}")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        payload = resp.json()
        try:
            raw_data = payload["series"][0]["data"]
        except (KeyError, IndexError) as exc:
            logger.error(f"EIA: unexpected response structure for {series_id}: {exc}")
            return pd.DataFrame()

        records = [{"date": r[0], "value": r[1]} for r in raw_data]
        df = pd.DataFrame(records)
        df["date"]  = pd.to_datetime(df["date"], format="%Y%m", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.rename(columns={"value": series_id}, inplace=True)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        logger.success(f"  ✓ {len(df)} observations for {series_id}")
        return df

    def fetch_all(self) -> pd.DataFrame:
        """Fetch every configured EIA series and merge into one DataFrame."""
        frames: list[pd.DataFrame] = []
        for label, sid in EIA_SERIES.items():
            try:
                df = self.fetch_series_v1(sid)
                if not df.empty:
                    df.columns = [label]
                    frames.append(df)
            except Exception as exc:
                logger.error(f"EIA: failed to fetch {sid}: {exc}")

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for df in frames[1:]:
            merged = merged.join(df, how="outer")
        merged.index.name = "date"
        return merged


# ══════════════════════════════════════════════════════════════════════════════
# World Bank Client
# ══════════════════════════════════════════════════════════════════════════════

class WorldBankClient:
    """Client for the World Bank Data API v2."""

    BASE_URL = "https://api.worldbank.org/v2"

    def fetch_indicator(
        self,
        indicator: str,
        countries: list[str],
        start_year: int = 2000,
        end_year: int   = 2023,
    ) -> pd.DataFrame:
        """
        Fetch a single World Bank indicator for a list of ISO-3 country codes.

        Returns
        -------
        pd.DataFrame with columns: ['country', 'iso3', 'year', indicator_label]
        """
        country_str = ";".join(countries)
        url = f"{self.BASE_URL}/country/{country_str}/indicator/{indicator}"
        params = {
            "format":   "json",
            "per_page": 1000,
            "mrv":      end_year - start_year + 1,
            "date":     f"{start_year}:{end_year}",
        }
        logger.info(f"World Bank → {indicator} for {len(countries)} countries")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        payload = resp.json()
        if len(payload) < 2 or not payload[1]:
            logger.warning(f"World Bank: no data for {indicator}")
            return pd.DataFrame()

        records = []
        for entry in payload[1]:
            if entry.get("value") is not None:
                records.append(
                    {
                        "country": entry["country"]["value"],
                        "iso3":    entry["countryiso3code"],
                        "year":    int(entry["date"]),
                        indicator: float(entry["value"]),
                    }
                )

        df = pd.DataFrame(records)
        logger.success(f"  ✓ {len(df)} records for {indicator}")
        return df

    def fetch_all(self, countries: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Fetch all configured World Bank indicators and merge into a wide panel.

        Returns
        -------
        pd.DataFrame indexed by (iso3, year) with one column per indicator.
        """
        if countries is None:
            countries = list(COUNTRIES.keys())

        indicator_frames: list[pd.DataFrame] = []
        for label, indicator_id in WB_INDICATORS.items():
            try:
                df = self.fetch_indicator(indicator_id, countries)
                if not df.empty:
                    df.rename(columns={indicator_id: label}, inplace=True)
                    indicator_frames.append(df)
            except Exception as exc:
                logger.error(f"World Bank: failed to fetch {indicator_id}: {exc}")

        if not indicator_frames:
            return pd.DataFrame()

        # Merge all indicators on (iso3, year)
        base = indicator_frames[0][["country", "iso3", "year"]].drop_duplicates()
        for df in indicator_frames:
            label = [c for c in df.columns if c not in ("country", "iso3", "year")][0]
            base = base.merge(df[["iso3", "year", label]], on=["iso3", "year"], how="left")

        base.sort_values(["iso3", "year"], inplace=True)
        base.set_index(["iso3", "year"], inplace=True)
        return base


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — collect & persist all raw data
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_data(start: str = "2000-01-01") -> dict[str, pd.DataFrame]:
    """
    Run all three data collection pipelines and save raw CSVs to data/raw/.

    Returns
    -------
    dict with keys 'fred', 'eia', 'worldbank'
    """
    logger.info("═" * 60)
    logger.info("Starting full data collection pipeline")
    logger.info("═" * 60)

    results: dict[str, pd.DataFrame] = {}

    # ── FRED ──────────────────────────────────────────────────────────────────
    logger.info("[1/3] Collecting FRED data …")
    fred_df = FREDClient().fetch_all(start=start)
    if not fred_df.empty:
        path = RAW_DIR / "fred_data.csv"
        fred_df.to_csv(path)
        logger.success(f"Saved FRED data → {path}  shape={fred_df.shape}")
    results["fred"] = fred_df

    # ── EIA ───────────────────────────────────────────────────────────────────
    logger.info("[2/3] Collecting EIA data …")
    eia_df = EIAClient().fetch_all()
    if not eia_df.empty:
        path = RAW_DIR / "eia_data.csv"
        eia_df.to_csv(path)
        logger.success(f"Saved EIA data → {path}  shape={eia_df.shape}")
    results["eia"] = eia_df

    # ── World Bank ─────────────────────────────────────────────────────────────
    logger.info("[3/3] Collecting World Bank data …")
    wb_df = WorldBankClient().fetch_all()
    if not wb_df.empty:
        path = RAW_DIR / "worldbank_data.csv"
        wb_df.to_csv(path)
        logger.success(f"Saved WB data → {path}  shape={wb_df.shape}")
    results["worldbank"] = wb_df

    logger.info("Data collection complete.")
    return results


if __name__ == "__main__":
    collect_all_data()
