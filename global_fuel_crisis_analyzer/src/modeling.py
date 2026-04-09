"""
modeling.py — Multi-model forecasting pipeline for Brent crude oil prices.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

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
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (
    ARIMA_ORDER,
    ARIMA_SEASONAL_ORDER,
    PROC_DIR,
    RF_PARAMS,
    XGB_PARAMS,
)

MODEL_DIR = PROC_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4), "MAPE": round(mape, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Feature list — EXPANDED with momentum & interaction features
# ─────────────────────────────────────────────────────────────────────────────

REGRESSION_FEATURES = [
    # Core lags — most predictive for time series
    "brent_crude_lag1m", "brent_crude_lag2m", "brent_crude_lag3m",
    "brent_crude_lag6m", "brent_crude_lag12m",
    # Rolling means
    "brent_crude_roll3m_mean", "brent_crude_roll6m_mean", "brent_crude_roll12m_mean",
    # Rolling volatility
    "brent_crude_roll3m_std", "brent_crude_roll6m_std",
    # Pct changes
    "brent_crude_pct1m", "brent_crude_pct3m",
    # Correlated commodities
    "wti_crude_lag1m", "wti_crude_lag2m",
    "natural_gas_lag1m",
    # Macro
    "us_cpi_energy",
    "brent_wti_spread",
    # Supply
    "supply_zscore",
    # Calendar
    "month", "year", "quarter",
    # Crisis
    "crisis_flag",
    # ADDED: momentum features from preprocessing
    "brent_crude_momentum_3_12",
    "brent_crude_acceleration",
    "brent_crude_range_position",
    "brent_crude_vol_regime",
    "wti_crude_momentum_3_12",
    # ADDED: crisis interaction
    "crisis_x_momentum",
]


def _safe_features(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    available = [f for f in wanted if f in df.columns]
    missing   = set(wanted) - set(available)
    if missing:
        logger.debug(f"  Features not found, skipping: {sorted(missing)}")
    return available


# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline: Ridge Regression
# ─────────────────────────────────────────────────────────────────────────────

class BaselineLinear:
    def __init__(self, alpha: float = 1.0):
        self.alpha   = alpha
        self.model   = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        self.features: list[str] = []
        self._is_fit = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "BaselineLinear":
        self.features = _safe_features(df_train, REGRESSION_FEATURES)
        X = df_train[self.features].dropna()
        y = df_train.loc[X.index, target]
        self.model.fit(X, y)
        self._is_fit = True
        logger.success(f"BaselineLinear trained on {len(X)} samples, {len(self.features)} features.")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = df_test[self.features].ffill().fillna(0)
        return self.model.predict(X)

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        y_true = df_test[target].values
        y_pred = self.predict(df_test)
        return evaluate_predictions(y_true, y_pred)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "baseline_linear.pkl"
        joblib.dump(self, path)
        logger.info(f"Saved BaselineLinear → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BaselineLinear":
        return joblib.load(path or MODEL_DIR / "baseline_linear.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARIMA / SARIMA Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class ARIMAForecaster:
    def __init__(self, auto: bool = False):
        self.auto      = auto
        self.model     = None
        self.fit_order = None
        self._is_fit   = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "ARIMAForecaster":
        series = df_train[target].dropna()

        if self.auto:
            logger.info("ARIMA: running auto_arima …")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = auto_arima(
                    series,
                    start_p=1, start_q=1, max_p=4, max_q=4,
                    d=1, seasonal=True, m=12,
                    start_P=0, start_Q=0, max_P=2, max_Q=2,
                    D=1, stepwise=True,
                    suppress_warnings=True, error_action="ignore",
                )
            self.fit_order = (self.model.order, self.model.seasonal_order)
        else:
            logger.info(f"ARIMA: fitting SARIMAX{ARIMA_ORDER}×{ARIMA_SEASONAL_ORDER}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sm_model = SARIMAX(
                    series,
                    order=ARIMA_ORDER,
                    seasonal_order=ARIMA_SEASONAL_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.model = sm_model.fit(disp=False)
            self.fit_order = (ARIMA_ORDER, ARIMA_SEASONAL_ORDER)

        self._train_series = series
        self._is_fit = True
        logger.success(f"ARIMA trained. Order: {self.fit_order}")
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.auto:
            return self.model.predict(n_periods=steps)
        else:
            return self.model.forecast(steps=steps).values

    def predict_in_sample(self, df_test: pd.DataFrame, target: str = "brent_crude") -> np.ndarray:
        n = len(df_test)
        return self.predict(n)

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        y_true = df_test[target].values
        y_pred = self.predict_in_sample(df_test, target)
        min_len = min(len(y_true), len(y_pred))
        return evaluate_predictions(y_true[:min_len], y_pred[:min_len])

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "arima.pkl"
        joblib.dump(self, path)
        logger.info(f"Saved ARIMAForecaster → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ARIMAForecaster":
        return joblib.load(path or MODEL_DIR / "arima.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Forest
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel:
    def __init__(self, params: dict = RF_PARAMS):
        self.params   = params
        self.model    = RandomForestRegressor(**params)
        self.features: list[str] = []
        self.feature_importances_: Optional[pd.Series] = None
        self._is_fit = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "RandomForestModel":
        self.features = _safe_features(df_train, REGRESSION_FEATURES)
        X = df_train[self.features].dropna()
        y = df_train.loc[X.index, target]
        self.model.fit(X, y)
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)
        self._is_fit = True
        logger.success(f"RandomForest trained. Top feature: {self.feature_importances_.idxmax()}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = df_test[self.features].ffill().fillna(0)
        return self.model.predict(X)

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        y_true = df_test[target].values
        y_pred = self.predict(df_test)
        return evaluate_predictions(y_true, y_pred)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "random_forest.pkl"
        joblib.dump(self, path)
        logger.info(f"Saved RandomForestModel → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RandomForestModel":
        return joblib.load(path or MODEL_DIR / "random_forest.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 4. XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    def __init__(self, params: dict = XGB_PARAMS):
        self.params   = params
        self.model    = XGBRegressor(verbosity=0, **params)
        self.features: list[str] = []
        self.feature_importances_: Optional[pd.Series] = None
        self._is_fit = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "XGBoostModel":
        self.features = _safe_features(df_train, REGRESSION_FEATURES)
        X = df_train[self.features].dropna()
        y = df_train.loc[X.index, target]

        # FIXED: use train/validation split for early stopping to prevent
        # overfitting that causes flat or extrapolating test predictions
        val_size = max(12, int(len(X) * 0.15))
        X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)
        self._is_fit = True
        logger.success(f"XGBoost trained. Top feature: {self.feature_importances_.idxmax()}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X = df_test[self.features].fillna(0)
        return self.model.predict(X)

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        y_true = df_test[target].values
        y_pred = self.predict(df_test)
        return evaluate_predictions(y_true, y_pred)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "xgboost.pkl"
        joblib.dump(self, path)
        logger.info(f"Saved XGBoostModel → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "XGBoostModel":
        return joblib.load(path or MODEL_DIR / "xgboost.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ModelComparison:
    def __init__(self, target: str = "brent_crude"):
        self.target  = target
        self.models: dict[str, object]          = {}
        self.metrics: dict[str, dict]           = {}
        self.predictions: dict[str, np.ndarray] = {}

    def run(
        self,
        train: pd.DataFrame,
        test:  pd.DataFrame,
        fit_arima: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        y_true = test[self.target].values

        logger.info("Training Baseline Linear …")
        bl = BaselineLinear().fit(train, self.target)
        self.models["Baseline (Ridge)"] = bl
        bl.save()
        preds_bl = bl.predict(test)
        self.predictions["Baseline (Ridge)"] = preds_bl
        self.metrics["Baseline (Ridge)"] = evaluate_predictions(y_true, preds_bl)

        if fit_arima:
            logger.info("Training ARIMA …")
            try:
                arima = ARIMAForecaster(auto=False).fit(train, self.target)
                self.models["ARIMA"] = arima
                arima.save()
                preds_ar = arima.predict(len(test))
                min_len  = min(len(y_true), len(preds_ar))
                self.predictions["ARIMA"] = preds_ar[:min_len]
                self.metrics["ARIMA"] = evaluate_predictions(y_true[:min_len], preds_ar[:min_len])
            except Exception as exc:
                logger.warning(f"ARIMA training failed: {exc}")

        logger.info("Training Random Forest …")
        rf = RandomForestModel().fit(train, self.target)
        self.models["Random Forest"] = rf
        rf.save()
        preds_rf = rf.predict(test)
        self.predictions["Random Forest"] = preds_rf
        self.metrics["Random Forest"] = evaluate_predictions(y_true, preds_rf)

        logger.info("Training XGBoost …")
        xgb = XGBoostModel().fit(train, self.target)
        self.models["XGBoost"] = xgb
        xgb.save()
        preds_xg = xgb.predict(test)
        self.predictions["XGBoost"] = preds_xg
        self.metrics["XGBoost"] = evaluate_predictions(y_true, preds_xg)

        metrics_df = pd.DataFrame(self.metrics).T.sort_values("RMSE")
        logger.info("\n" + metrics_df.to_string())

        preds_df = pd.DataFrame({"actual": y_true}, index=test.index[:len(y_true)])
        for name, preds in self.predictions.items():
            preds_df[name] = preds[:len(preds_df)]

        return metrics_df, preds_df

    def best_model(self) -> tuple[str, object]:
        if not self.metrics:
            raise RuntimeError("No models trained yet. Call .run() first.")
        best_name = min(self.metrics, key=lambda k: self.metrics[k]["RMSE"])
        return best_name, self.models[best_name]


if __name__ == "__main__":
    from preprocessing import build_master_dataset, get_train_test
    master = build_master_dataset()
    train, test = get_train_test(master)
    mc = ModelComparison()
    metrics, preds = mc.run(train, test)
    print("\nModel Comparison:")
    print(metrics)
