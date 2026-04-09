"""
config.py — Centralized project configuration.

All API keys, paths, model hyperparameters, and country metadata
live here so every module imports from a single source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = DATA_DIR / "cache"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"

for _dir in (DATA_DIR, CACHE_DIR, RAW_DIR, PROC_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── API Keys (loaded from .env) ────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
EIA_API_KEY  = os.getenv("EIA_API_KEY",  "")

# ── FRED Series IDs ────────────────────────────────────────────────────────────
FRED_SERIES = {
    "brent_crude":   "DCOILBRENTEU",
    "wti_crude":     "DCOILWTICO",
    "us_cpi_energy": "CPIENGSL",
    "us_cpi_all":    "CPIAUCSL",
    "natural_gas":   "MHHNGSP",
}

# ── EIA Series IDs ─────────────────────────────────────────────────────────────
EIA_SERIES = {
    "world_crude_supply": "INTL.57-1-WORL-TBPD.M",
    "oecd_stocks":        "STEO.OECD_STK_CRUDE.M",
}

# ── World Bank Indicators ──────────────────────────────────────────────────────
WB_INDICATORS = {
    "gdp_per_capita":         "NY.GDP.PCAP.CD",
    "fuel_imports_pct_merch": "TM.VAL.FUEL.ZS.UN",
    "energy_use_per_cap":     "EG.USE.PCAP.KG.OE",
    "oil_rents_pct_gdp":      "NY.GDP.PETR.RT.ZS",
    "population":             "SP.POP.TOTL",
}

# ── Country universe ───────────────────────────────────────────────────────────
COUNTRIES = {
    "USA":   {"iso3": "USA", "name": "United States",   "region": "North America"},
    "DEU":   {"iso3": "DEU", "name": "Germany",          "region": "Europe"},
    "CHN":   {"iso3": "CHN", "name": "China",            "region": "Asia"},
    "IND":   {"iso3": "IND", "name": "India",            "region": "Asia"},
    "BRA":   {"iso3": "BRA", "name": "Brazil",           "region": "South America"},
    "BGD":   {"iso3": "BGD", "name": "Bangladesh",       "region": "Asia"},
    "PAK":   {"iso3": "PAK", "name": "Pakistan",         "region": "Asia"},
    "IDN":   {"iso3": "IDN", "name": "Indonesia",        "region": "Asia"},
    "TUR":   {"iso3": "TUR", "name": "Turkey",           "region": "Europe/Asia"},
    "ZAF":   {"iso3": "ZAF", "name": "South Africa",     "region": "Africa"},
    "NGA":   {"iso3": "NGA", "name": "Nigeria",          "region": "Africa"},
    "SAU":   {"iso3": "SAU", "name": "Saudi Arabia",     "region": "Middle East"},
    "RUS":   {"iso3": "RUS", "name": "Russia",           "region": "Europe/Asia"},
    "JPN":   {"iso3": "JPN", "name": "Japan",            "region": "Asia"},
    "GBR":   {"iso3": "GBR", "name": "United Kingdom",   "region": "Europe"},
}

# ── Fuel price multipliers ─────────────────────────────────────────────────────
COUNTRY_FUEL_MULTIPLIER = {
    "USA": 0.021,   "DEU": 0.038,   "CHN": 0.025,   "IND": 0.028,
    "BRA": 0.030,   "BGD": 0.032,   "PAK": 0.030,   "IDN": 0.022,
    "TUR": 0.040,   "ZAF": 0.026,   "NGA": 0.015,   "SAU": 0.008,
    "RUS": 0.012,   "JPN": 0.035,   "GBR": 0.042,
}

# ── Modeling ───────────────────────────────────────────────────────────────────
TRAIN_TEST_SPLIT_DATE = "2022-01-01"
ARIMA_ORDER           = (2, 1, 2)
ARIMA_SEASONAL_ORDER  = (1, 1, 1, 12)
RF_PARAMS = {
    "n_estimators":    400,
    "max_depth":       6,
    "min_samples_leaf": 2,
    "max_features":    "sqrt",
    "random_state":    42,
    "n_jobs":          -1,
}
XGB_PARAMS = {
    "n_estimators":      500,
    "max_depth":         4,
    "learning_rate":     0.03,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
}

# ── Simulation defaults ────────────────────────────────────────────────────────
# FIXED: elasticity was -0.08 causing 300%+ price jumps on large shocks.
# Literature-backed short-run elasticity range is -0.25 to -0.35.
SUPPLY_ELASTICITY     = -0.30   # % price change per 1% supply drop (short-run)
PASSTHROUGH_RATE      = 0.65    # fraction of oil price shock passed to retail pump
INFLATION_FUEL_WEIGHT = 0.055   # fuel's weight in CPI basket (global average)

# ── Cache TTL ──────────────────────────────────────────────────────────────────
CACHE_EXPIRE_HOURS = 24
