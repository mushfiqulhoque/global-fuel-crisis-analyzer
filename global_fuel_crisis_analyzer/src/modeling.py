"""
modeling.py — Multi-model forecasting for Brent crude oil prices.

TIME-SERIES CORRECT:
  - Lag features (lag1m, lag2m, lag3m, lag6m, lag12m) are the PRIMARY inputs
  - Rolling mean/std features included
  - Trend + calendar features included
  - NaN rows from lag warm-up are DROPPED (not filled with 0)
  - Data is NEVER shuffled — temporal order preserved
  - Validation split takes the last 15% of training rows (chronologically)

Why this fixes negative R²:
  SS_res > SS_tot happens when the model predicts worse than the mean.
  Root cause was _clean_X filling lag-NaN rows with 0, training the model
  on fake data, then predicting on real lags → catastrophic mismatch.
  Fix: drop all rows with NaN features BEFORE fitting.
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
# Metrics
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


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic feature selection
# Features are selected at fit-time from actual dataframe columns.
# This guarantees lag columns are always used when present and nothing
# leaks the current target value into X.
# ─────────────────────────────────────────────────────────────────────────────

# Explicit lightweight extras to include if present
_EXTRA_FEATURES = ["month", "trend", "crisis_flag"]

# Raw unlagged columns that must NEVER appear in X
_FORBIDDEN = {
    "brent_crude", "wti_crude", "natural_gas",
    "us_cpi_energy", "world_crude_supply",
    "crisis_name", "split",
}


def _select_features(df: pd.DataFrame, target: str = "brent_crude") -> list[str]:
    """
    Select features by column-name pattern — never includes raw target.

    INCLUDE  columns whose name contains:
               _lag  _roll  _momentum  _pct  _acceleration
               _range_position  _vol_regime  _spread  _zscore  _x_momentum
    INCLUDE  explicit extras: month, trend, crisis_flag
    EXCLUDE  target column and all raw (unlagged) price/macro columns
    """
    pattern_include = (
        "_lag", "_roll", "_momentum", "_pct",
        "_acceleration", "_range_position", "_vol_regime",
        "_spread", "_zscore", "_x_momentum",
    )

    selected = []
    for col in df.columns:
        if col == target or col in _FORBIDDEN:
            continue
        if col in _EXTRA_FEATURES or any(p in col for p in pattern_include):
            selected.append(col)

    if not selected:
        raise ValueError(
            "No valid features found. "
            "Run preprocessing.build_master_dataset() to create lag/rolling columns."
        )

    for req in ("brent_crude_lag1m", "brent_crude_lag2m", "brent_crude_lag3m"):
        if req not in selected:
            logger.warning(f"Required lag feature '{req}' missing — re-run preprocessing.")

    logger.info(f"  Selected {len(selected)} features. "
                f"First 8: {selected[:8]}")
    return selected


def _prepare_Xy(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract X, y and DROP any rows where X or y contains NaN.

    This is the critical fix: rows created by lag warm-up have NaN in the
    lag columns. Filling them with 0 teaches the model that 'price was 0
    k months ago', which is wrong and causes predictions to be mean-like
    or flat → negative R².

    We drop these rows instead. After a 12-month warm-up the dataset is
    still large enough for robust training.
    """
    X = df[features].copy()
    y = df[target].copy()

    # Replace inf before NaN check
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    # Combined mask: keep only rows where BOTH X and y are fully valid
    valid_mask = X.notna().all(axis=1) & y.notna()

    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.debug(f"  _prepare_Xy: dropped {n_dropped} NaN rows (lag warm-up)")

    return X[valid_mask], y[valid_mask]


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
        self._is_fit = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "BaselineLinear":
        self.features = _select_features(df_train, target)
        X, y = _prepare_Xy(df_train, self.features, target)
        if len(X) == 0:
            raise ValueError("No valid rows after dropping NaN — check preprocessing.")
        logger.info(f"  X_train shape: {X.shape}")
        logger.info(f"  X_train columns: {list(X.columns)}")
        logger.info(f"  X_train head:\n{X.head(3).to_string()}")
        self.model.fit(X, y)
        self._is_fit = True
        logger.success(f"BaselineLinear: trained on {len(X)} rows, {len(self.features)} features.")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X, _ = _prepare_Xy(df_test, self.features, list(df_test.columns)[0])
        # We need predictions for ALL test rows in order; use index alignment
        X_full = df_test[self.features].replace([np.inf, -np.inf], np.nan)
        # For test rows with any NaN feature, forward-fill within test set only
        # (there should be none after get_train_test drops warm-up rows)
        X_full = X_full.ffill().bfill()
        return self.model.predict(X_full)

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        return evaluate_predictions(df_test[target].values, self.predict(df_test))

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

    def predict_in_sample(self, df_test: pd.DataFrame, target: str = "brent_crude") -> np.ndarray:
        return self.predict(len(df_test))

    def evaluate(self, df_test: pd.DataFrame, target: str = "brent_crude") -> dict:
        y_true = df_test[target].values
        y_pred = self.predict_in_sample(df_test, target)
        n      = min(len(y_true), len(y_pred))
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

_RF_PARAMS = {
    "n_estimators":     500,
    "max_depth":        8,
    "min_samples_leaf": 3,
    "max_features":     0.4,   # 40% per split → distributes importance
    "random_state":     42,
    "n_jobs":           -1,
}


class RandomForestModel:
    def __init__(self, params: dict = _RF_PARAMS):
        self.params   = params
        self.model    = RandomForestRegressor(**params)
        self.features: list[str] = []
        self.feature_importances_: Optional[pd.Series] = None
        self._is_fit = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "RandomForestModel":
        self.features = _select_features(df_train, target)
        X, y = _prepare_Xy(df_train, self.features, target)
        if len(X) == 0:
            raise ValueError("No valid rows after dropping NaN.")
        self.model.fit(X, y)
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)
        self._is_fit = True
        logger.success(f"RandomForest: trained on {len(X)} rows. "
                       f"Top feature: {self.feature_importances_.idxmax()}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X_full = df_test[self.features].replace([np.inf, -np.inf], np.nan)
        X_full = X_full.ffill().bfill()
        return self.model.predict(X_full)

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

_XGB_PARAMS = {
    "n_estimators":     600,
    "max_depth":        4,
    "learning_rate":    0.025,
    "subsample":        0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 4,
    "reg_alpha":        0.05,
    "reg_lambda":       1.5,
    "random_state":     42,
    "verbosity":        0,
}


class XGBoostModel:
    def __init__(self, params: dict = _XGB_PARAMS):
        self.params   = params
        self.model    = XGBRegressor(**params)
        self.features: list[str] = []
        self.feature_importances_: Optional[pd.Series] = None
        self._is_fit = False

    def fit(self, df_train: pd.DataFrame, target: str = "brent_crude") -> "XGBoostModel":
        self.features = _select_features(df_train, target)
        X, y = _prepare_Xy(df_train, self.features, target)
        if len(X) == 0:
            raise ValueError("No valid rows after dropping NaN.")

        # Chronological validation split (last 15% of training rows)
        val_size = max(12, int(len(X) * 0.15))
        X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)
        self._is_fit = True
        logger.success(f"XGBoost: trained on {len(X_tr)} rows. "
                       f"Top feature: {self.feature_importances_.idxmax()}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        X_full = df_test[self.features].replace([np.inf, -np.inf], np.nan)
        X_full = X_full.ffill().bfill()
        return self.model.predict(X_full)

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

        logger.info("Training Baseline Linear …")
        bl = BaselineLinear().fit(train, self.target)
        self.models["Baseline (Ridge)"]      = bl
        bl.save()
        preds_bl                              = bl.predict(test)
        self.predictions["Baseline (Ridge)"] = preds_bl
        self.metrics["Baseline (Ridge)"]     = evaluate_predictions(y_true, preds_bl)

        if fit_arima:
            logger.info("Training ARIMA …")
            try:
                arima = ARIMAForecaster(auto=False).fit(train, self.target)
                self.models["ARIMA"] = arima
                arima.save()
                preds_ar = arima.predict(len(test))
                n        = min(len(y_true), len(preds_ar))
                self.predictions["ARIMA"] = preds_ar[:n]
                self.metrics["ARIMA"]     = evaluate_predictions(y_true[:n], preds_ar[:n])
            except Exception as exc:
                logger.warning(f"ARIMA training failed: {exc}")

        logger.info("Training Random Forest …")
        rf = RandomForestModel().fit(train, self.target)
        self.models["Random Forest"]      = rf
        rf.save()
        preds_rf                           = rf.predict(test)
        self.predictions["Random Forest"] = preds_rf
        self.metrics["Random Forest"]     = evaluate_predictions(y_true, preds_rf)

        logger.info("Training XGBoost …")
        xgb = XGBoostModel().fit(train, self.target)
        self.models["XGBoost"]      = xgb
        xgb.save()
        preds_xg                     = xgb.predict(test)
        self.predictions["XGBoost"] = preds_xg
        self.metrics["XGBoost"]     = evaluate_predictions(y_true, preds_xg)

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
