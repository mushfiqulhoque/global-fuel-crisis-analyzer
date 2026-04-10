"""
modeling.py — Multi-model forecasting for Brent crude oil prices.

FEATURE FIX:
  Features are HARDCODED explicitly using exact column names from preprocessing.
  No auto-selection. No crisis_flag. No raw prices.
  Only lag, rolling, and pct-change columns are used.
  NaN rows from lag warm-up are DROPPED before fitting.
  Temporal order is NEVER shuffled.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import ARIMA_ORDER, ARIMA_SEASONAL_ORDER, PROC_DIR

MODEL_DIR = PROC_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXPLICIT FEATURE LIST — hardcoded exact column names from preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    # Brent lag features — primary signal
    "brent_crude_lag1m",
    "brent_crude_lag2m",
    "brent_crude_lag3m",
    "brent_crude_lag6m",
    "brent_crude_lag12m",
    # Brent rolling features — trend/volatility
    "brent_crude_roll3m_mean",
    "brent_crude_roll6m_mean",
    "brent_crude_roll12m_mean",
    "brent_crude_roll3m_std",
    "brent_crude_roll6m_std",
    # Brent momentum
    "brent_crude_pct1m",
    "brent_crude_pct3m",
    # WTI lag — correlated market signal
    "wti_crude_lag1m",
    "wti_crude_lag3m",
    # Spread
    "brent_wti_spread",
    # Calendar — seasonality
    "month",
    "quarter",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    return {
        "RMSE": round(rmse, 4),
        "MAE":  round(mae,  4),
        "R2":   round(r2,   4),
        "MAPE": round(mape, 4),
    }


def _get_features(df: pd.DataFrame) -> list[str]:
    """Return only features that exist in df (guards against missing columns)."""
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"Missing features (will be skipped): {missing}")
    logger.info(f"Using {len(available)} features: {available}")
    return available


def _prepare_Xy(
    df: pd.DataFrame,
    features: list[str],
    target: str = "brent_crude",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract X and y, drop ALL rows where ANY value is NaN or inf.
    This removes the lag warm-up rows (first 12 months) that have NaN lags.
    Filling NaN with 0 causes flat predictions and negative R² — so we DROP.
    """
    X = df[features].copy().replace([np.inf, -np.inf], np.nan)
    y = df[target].copy().replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & y.notna()
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.debug(f"Dropped {n_dropped} NaN rows (lag warm-up)")
    X_clean = X[valid]
    y_clean = y[valid]

    # ── DEBUG PROOF ────────────────────────────────────────────────────────────
    logger.info(f"  X shape after NaN drop: {X_clean.shape}")
    logger.info(f"  Features in X: {list(X_clean.columns)}")
    logger.info(f"  X head (3 rows):\n{X_clean.head(3).to_string()}")
    logger.info(f"  y head: {y_clean.head(3).values}")
    # ──────────────────────────────────────────────────────────────────────────

    return X_clean, y_clean


# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline: Ridge Regression
# ─────────────────────────────────────────────────────────────────────────────

class BaselineLinear:
    def __init__(self, alpha: float = 10.0):
        self.alpha    = alpha
        self.model    = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=alpha)),
        ])
        self.features: list[str] = []
        self._is_fit  = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "BaselineLinear":
        self.features = _get_features(df_train)
        X, y = _prepare_Xy(df_train, self.features, target)
        if len(X) == 0:
            raise ValueError("No valid rows after NaN drop.")
        self.model.fit(X, y)
        self._is_fit = True
        logger.success(f"BaselineLinear trained: {len(X)} rows, {len(self.features)} features.")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = df_test[self.features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return self.model.predict(X)

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        return evaluate_predictions(df_test[target].values, self.predict(df_test))

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "baseline_linear.pkl"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BaselineLinear":
        return joblib.load(path or MODEL_DIR / "baseline_linear.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARIMA / SARIMA Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class ARIMAForecaster:
    def __init__(self):
        self.model     = None
        self.fit_order = None
        self._is_fit   = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "ARIMAForecaster":
        series = df_train[target].dropna()
        logger.info(f"ARIMA: fitting SARIMAX{ARIMA_ORDER}×{ARIMA_SEASONAL_ORDER}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = SARIMAX(
                series,
                order=ARIMA_ORDER,
                seasonal_order=ARIMA_SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model = sm.fit(disp=False)
        self.fit_order = (ARIMA_ORDER, ARIMA_SEASONAL_ORDER)
        self._is_fit   = True
        logger.success(f"ARIMA trained. Order: {self.fit_order}")
        return self

    def predict(self, steps: int) -> np.ndarray:
        return self.model.forecast(steps=steps).values

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        y_true = df_test[target].values
        y_pred = self.predict(len(df_test))
        n = min(len(y_true), len(y_pred))
        return evaluate_predictions(y_true[:n], y_pred[:n])

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "arima.pkl"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ARIMAForecaster":
        return joblib.load(path or MODEL_DIR / "arima.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Forest
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.6,
            random_state=42,
            n_jobs=-1,
        )
        self.features: list[str] = []
        self.feature_importances_: Optional[pd.Series] = None
        self._is_fit = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "RandomForestModel":
        self.features = _get_features(df_train)
        X, y = _prepare_Xy(df_train, self.features, target)
        if len(X) == 0:
            raise ValueError("No valid rows after NaN drop.")
        self.model.fit(X, y)
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)
        self._is_fit = True
        logger.success(f"RandomForest trained: {len(X)} rows. "
                       f"Top feature: {self.feature_importances_.idxmax()}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = df_test[self.features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return self.model.predict(X)

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        return evaluate_predictions(df_test[target].values, self.predict(df_test))

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "random_forest.pkl"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RandomForestModel":
        return joblib.load(path or MODEL_DIR / "random_forest.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 4. XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=4,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=42,
            verbosity=0,
        )
        self.features: list[str] = []
        self.feature_importances_: Optional[pd.Series] = None
        self._is_fit = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "XGBoostModel":
        self.features = _get_features(df_train)
        X, y = _prepare_Xy(df_train, self.features, target)
        if len(X) == 0:
            raise ValueError("No valid rows after NaN drop.")

        # Chronological validation split — last 15% of training rows
        val_size = max(12, int(len(X) * 0.15))
        X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)
        self._is_fit = True
        logger.success(f"XGBoost trained: {len(X_tr)} rows. "
                       f"Top feature: {self.feature_importances_.idxmax()}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = df_test[self.features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return self.model.predict(X)

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        return evaluate_predictions(df_test[target].values, self.predict(df_test))

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "xgboost.pkl"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "XGBoostModel":
        return joblib.load(path or MODEL_DIR / "xgboost.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ModelComparison:
    def __init__(self, target: str = "brent_crude"):
        self.target      = target
        self.models:      dict[str, object]     = {}
        self.metrics:     dict[str, dict]       = {}
        self.predictions: dict[str, np.ndarray] = {}

    def run(
        self,
        train: pd.DataFrame,
        test:  pd.DataFrame,
        fit_arima: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        y_true = test[self.target].values

        # ── Baseline Ridge ─────────────────────────────────────────────────────
        logger.info("Training Baseline Linear …")
        bl = BaselineLinear().fit(train, self.target)
        self.models["Baseline (Ridge)"]      = bl
        bl.save()
        preds_bl                              = bl.predict(test)
        self.predictions["Baseline (Ridge)"] = preds_bl
        self.metrics["Baseline (Ridge)"]     = evaluate_predictions(y_true, preds_bl)

        # ── ARIMA ──────────────────────────────────────────────────────────────
        if fit_arima:
            logger.info("Training ARIMA …")
            try:
                arima = ARIMAForecaster().fit(train, self.target)
                self.models["ARIMA"] = arima
                arima.save()
                preds_ar = arima.predict(len(test))
                n        = min(len(y_true), len(preds_ar))
                self.predictions["ARIMA"] = preds_ar[:n]
                self.metrics["ARIMA"]     = evaluate_predictions(y_true[:n], preds_ar[:n])
            except Exception as exc:
                logger.warning(f"ARIMA training failed: {exc}")

        # ── Random Forest ──────────────────────────────────────────────────────
        logger.info("Training Random Forest …")
        rf = RandomForestModel().fit(train, self.target)
        self.models["Random Forest"]      = rf
        rf.save()
        preds_rf                           = rf.predict(test)
        self.predictions["Random Forest"] = preds_rf
        self.metrics["Random Forest"]     = evaluate_predictions(y_true, preds_rf)

        # ── XGBoost ────────────────────────────────────────────────────────────
        logger.info("Training XGBoost …")
        xgb = XGBoostModel().fit(train, self.target)
        self.models["XGBoost"]      = xgb
        xgb.save()
        preds_xg                     = xgb.predict(test)
        self.predictions["XGBoost"] = preds_xg
        self.metrics["XGBoost"]     = evaluate_predictions(y_true, preds_xg)

        # ── Summary ────────────────────────────────────────────────────────────
        metrics_df = pd.DataFrame(self.metrics).T.sort_values("RMSE")
        logger.info("\n" + metrics_df.to_string())

        preds_df = pd.DataFrame({"actual": y_true}, index=test.index[:len(y_true)])
        for name, preds in self.predictions.items():
            preds_df[name] = preds[:len(preds_df)]

        return metrics_df, preds_df

    def best_model(self) -> tuple[str, object]:
        if not self.metrics:
            raise RuntimeError("No models trained yet.")
        best = min(self.metrics, key=lambda k: self.metrics[k]["RMSE"])
        return best, self.models[best]


# ─────────────────────────────────────────────────────────────────────────────
# Quick local test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import build_master_dataset, get_train_test
    master = build_master_dataset()
    train, test = get_train_test(master)
    mc = ModelComparison()
    metrics, preds = mc.run(train, test, fit_arima=False)
    print("\nModel Comparison:")
    print(metrics)
    print("\nPrediction sample (first 5 rows):")
    print(preds.head())
