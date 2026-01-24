#!/usr/bin/env python3
"""
SLM Benchmark Program (Single-File)
-----------------------------------
Purpose:
  Evaluate classical SLM/ML baselines for next-day direction prediction using
  deterministic TA indicators + time-series walk-forward validation.

Models (classification + regression):
  - Logistic Regression (p_up) + Ridge Regression (expected_return)
  - Gradient Boosting (LightGBM/XGBoost if available, else sklearn HistGradientBoosting)
  - Random Forest (classification + regression)

Evaluation:
  - Rolling-origin walk-forward with monthly test blocks
  - Expanding window baseline
  - Optional rolling-window sensitivity (fixed lookback)

Outputs:
  - Per-ticker predictions CSVs
  - Per-ticker per-model metrics CSV
  - Console summary

Requirements:
  - pandas, numpy, scikit-learn
  - yfinance (to fetch data) OR provide your own CSV via --csv_dir
Optional:
  - lightgbm OR xgboost (faster/stronger gradient boosting)

Notes:
  - Uses Adjusted Close for targets and indicator computations
  - Ensures no lookahead leakage (features at t, targets at t+1)
  - Drops initial rows for indicator warmup (e.g., SMA50)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------
# Optional boosters
# -----------------------

def _try_import_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore
        return lgb
    except Exception:
        return None


def _try_import_xgboost():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


# -----------------------
# Utilities
# -----------------------

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _directional_accuracy(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    # Compares sign agreement (treat zero as 0)
    true_sign = np.sign(y_true_ret)
    pred_sign = np.sign(y_pred_ret)
    return float(np.mean(true_sign == pred_sign))


def _month_key(dt: pd.Timestamp) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


# -----------------------
# Data loading
# -----------------------

def load_ohlcv_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "yfinance is not installed. Install it (pip install yfinance) or use --csv_dir to load data from CSV."
        ) from e

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker and dates.")
    df = df.rename(columns=lambda c: c.strip())
    # Ensure expected columns exist
    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns from yfinance for {ticker}: {sorted(missing)}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Keep only required columns (avoid surprises)
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def load_ohlcv_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "yfinance is not installed. Install it (pip install yfinance) or use --csv_dir to load data from CSV."
        ) from e

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker and dates.")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # If MultiIndex columns (happens in some yfinance versions/settings), slice out the ticker
    if isinstance(df.columns, pd.MultiIndex):
        # Case 1: columns like (Ticker, Field)
        if ticker in df.columns.get_level_values(0):
            df = df[ticker].copy()
        # Case 2: columns like (Field, Ticker)
        elif ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, level=1, axis=1).copy()
        else:
            # Fall back: if only one ticker was requested, take the first block
            df = df.droplevel(-1, axis=1) if df.columns.nlevels > 1 else df

    # Normalize column names for robust matching
    df = df.rename(columns=lambda c: str(c).strip())
    colmap = {c.lower().replace(" ", ""): c for c in df.columns}

    def get_col(key: str) -> str:
        k = key.lower().replace(" ", "")
        if k in colmap:
            return colmap[k]
        raise KeyError(key)

    # Required fields (allow minor variations)
    try:
        open_c = get_col("Open")
        high_c = get_col("High")
        low_c = get_col("Low")
        close_c = get_col("Close")
        vol_c = get_col("Volume")
    except KeyError:
        raise RuntimeError(f"Unexpected yfinance columns for {ticker}: {list(df.columns)}")

    # Adj Close sometimes appears as "Adj Close" or "AdjClose"
    adj_c = None
    for cand in ["Adj Close", "AdjClose", "Adjclose", "adjclose"]:
        try:
            adj_c = get_col(cand)
            break
        except KeyError:
            continue

    # If Adj Close missing, fall back to Close (keeps pipeline running, but note it in logs)
    if adj_c is None:
        adj_c = close_c

    out = df[[open_c, high_c, low_c, close_c, adj_c, vol_c]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return out


# -----------------------
# Indicators (deterministic)
# -----------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes indicators using Adj Close for price-based calculations.
    Volume is used as-is.

    Returns a new DataFrame with indicator columns appended.
    """
    out = df.copy()

    px = out["Adj Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    close = out["Close"].astype(float)  # for ATR TR uses actual close
    vol = out["Volume"].astype(float)

    # Log returns
    out["lr_1"] = np.log(px / px.shift(1))
    out["lr_5"] = np.log(px / px.shift(5))
    out["lr_20"] = np.log(px / px.shift(20))

    # Rolling volatility of daily log returns
    out["vol_10"] = out["lr_1"].rolling(10).std()
    out["vol_20"] = out["lr_1"].rolling(20).std()

    # SMA / EMA
    out["sma_10"] = px.rolling(10).mean()
    out["sma_20"] = px.rolling(20).mean()
    out["sma_50"] = px.rolling(50).mean()

    out["ema_12"] = px.ewm(span=12, adjust=False).mean()
    out["ema_26"] = px.ewm(span=26, adjust=False).mean()

    # MACD
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI(14) using Wilder smoothing (equivalent to ewm alpha=1/14)
    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Stochastic (14,3)
    ll14 = px.rolling(14).min()
    hh14 = px.rolling(14).max()
    denom = (hh14 - ll14).replace(0, np.nan)
    out["stoch_k_14"] = 100 * (px - ll14) / denom
    out["stoch_d_3"] = out["stoch_k_14"].rolling(3).mean()

    # ATR(14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # OBV
    sign = np.sign(px.diff()).fillna(0.0)
    out["obv"] = (sign * vol).cumsum()

    # Volume z-score (20d)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    out["vol_z_20"] = (vol - vol_mean) / vol_std

    return out


# -----------------------
# Feature engineering
# -----------------------

def _slope_last_n(values: np.ndarray) -> float:
    """
    Slope of values vs time index 0..n-1 via least squares.
    """
    n = len(values)
    if n < 2 or np.all(np.isnan(values)):
        return float("nan")
    x = np.arange(n, dtype=float)
    y = values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    # slope from polyfit degree 1
    try:
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return float("nan")


def build_feature_table(
    df_with_indicators: pd.DataFrame,
    lookback_days: int = 20,
    include_window_summaries: bool = True,
) -> pd.DataFrame:
    df = df_with_indicators.copy()

    indicator_cols = [
        "lr_1", "lr_5", "lr_20",
        "vol_10", "vol_20",
        "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi_14",
        "stoch_k_14", "stoch_d_3",
        "atr_14",
        "obv",
        "vol_z_20",
    ]

    missing = [c for c in indicator_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Indicator columns missing: {missing}")

    feats: Dict[str, pd.Series] = {}

    # Snapshot
    for c in indicator_cols:
        feats[c] = df[c]

    if include_window_summaries:
        for c in indicator_cols:
            roll = df[c].rolling(lookback_days)
            feats[f"{c}_mean_{lookback_days}"] = roll.mean()
            feats[f"{c}_std_{lookback_days}"] = roll.std()
            feats[f"{c}_min_{lookback_days}"] = roll.min()
            feats[f"{c}_max_{lookback_days}"] = roll.max()
            feats[f"{c}_slope_{lookback_days}"] = df[c].rolling(lookback_days).apply(
                lambda x: _slope_last_n(np.asarray(x, dtype=float)), raw=False
            )

    feat_df = pd.DataFrame(feats, index=df.index)
    return feat_df


def build_targets(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    y_class[t] = 1 if AdjClose[t+1] > AdjClose[t] else 0
    y_ret[t] = next-day return percentage
    """
    px = df["Adj Close"].astype(float)
    next_px = px.shift(-1)

    y_class = (next_px > px).astype(int)
    y_ret = (next_px / px - 1.0) * 100.0  # percent

    # last row has no t+1
    y_class = y_class.iloc[:-1]
    y_ret = y_ret.iloc[:-1]
    return y_class, y_ret


def align_features_targets(X: pd.DataFrame, y_class: pd.Series, y_ret: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Align on index intersection and drop NaNs
    common_idx = X.index.intersection(y_class.index).intersection(y_ret.index)
    X2 = X.loc[common_idx].copy()
    yc = y_class.loc[common_idx].copy()
    yr = y_ret.loc[common_idx].copy()

    # Drop rows with any NaNs in features or targets
    mask = X2.notna().all(axis=1) & yc.notna() & yr.notna()
    X2 = X2.loc[mask]
    yc = yc.loc[mask]
    yr = yr.loc[mask]
    return X2, yc, yr


# -----------------------
# Walk-forward folds
# -----------------------

@dataclass
class Fold:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp   # inclusive


def build_monthly_folds(
    index: pd.DatetimeIndex,
    min_train_days: int = 252,
    rolling_years: Optional[int] = None,
) -> List[Fold]:
    """
    Monthly test blocks. For each month M, test = all dates in that month.
    Train = all dates strictly before test_start (expanding) OR within rolling window.

    rolling_years=None => expanding window
    rolling_years=int  => rolling window lookback in years for sensitivity
    """
    if len(index) == 0:
        return []

    idx = pd.DatetimeIndex(index).sort_values()
    # group dates by month key
    month_keys = pd.Series(idx).map(_month_key).to_numpy()
    unique_months = pd.unique(month_keys)

    folds: List[Fold] = []
    for mk in unique_months:
        month_dates = idx[month_keys == mk]
        test_start = month_dates.min()
        test_end = month_dates.max()
        train_end = test_start - pd.Timedelta(days=1)

        train_dates = idx[idx <= train_end]
        if len(train_dates) < min_train_days:
            continue

        if rolling_years is None:
            train_start = train_dates.min()
        else:
            cutoff = test_start - pd.DateOffset(years=rolling_years)
            train_dates = train_dates[train_dates >= cutoff]
            if len(train_dates) < min_train_days:
                continue
            train_start = train_dates.min()

        folds.append(
            Fold(
                fold_id=f"{mk}-roll{rolling_years if rolling_years is not None else 'exp'}",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return folds


# -----------------------
# Model factory
# -----------------------

@dataclass
class ModelSpec:
    name: str
    clf: BaseEstimator
    reg: BaseEstimator


def make_model_specs(random_state: int = 42) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # Logistic (classification) + Ridge (regression) with scaling
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state)),
    ])
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0, random_state=random_state)),
    ])
    specs.append(ModelSpec("logit_ridge", logit, ridge))

    # Gradient boosting (prefer LightGBM, else XGBoost, else sklearn HistGradientBoosting)
    lgb = _try_import_lightgbm()
    xgb = _try_import_xgboost()

    if lgb is not None:
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
        reg = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
        specs.append(ModelSpec("lightgbm", clf, reg))
    elif xgb is not None:
        clf = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=0,
        )
        reg = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=0,
        )
        specs.append(ModelSpec("xgboost", clf, reg))
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            random_state=random_state,
        )
        reg = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            random_state=random_state,
        )
        specs.append(ModelSpec("sklearn_hgb", clf, reg))

    # Random Forest (no scaling required)
    rf_clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    rf_reg = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=5,
        max_features=0.6,
        n_jobs=-1,
        random_state=random_state,
    )
    specs.append(ModelSpec("random_forest", rf_clf, rf_reg))

    return specs


# -----------------------
# Evaluation
# -----------------------

@dataclass
class FoldMetrics:
    ticker: str
    window_type: str   # expanding or rollingX
    model: str
    fold_id: str
    n_test: int

    # Classification
    roc_auc: float
    logloss: float
    brier: float
    accuracy: float
    f1: float

    # Regression
    mae: float
    rmse: float
    corr: float
    dir_acc: float


def evaluate_fold(
    ticker: str,
    window_type: str,
    spec: ModelSpec,
    fold: Fold,
    X: pd.DataFrame,
    y_class: pd.Series,
    y_ret: pd.Series,
) -> Tuple[pd.DataFrame, FoldMetrics]:
    # Train/test split by fold dates
    train_mask = (X.index >= fold.train_start) & (X.index <= fold.train_end)
    test_mask = (X.index >= fold.test_start) & (X.index <= fold.test_end)

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    yc_train = y_class.loc[train_mask]
    yc_test = y_class.loc[test_mask]
    yr_train = y_ret.loc[train_mask]
    yr_test = y_ret.loc[test_mask]

    if len(X_test) == 0 or len(X_train) == 0:
        raise RuntimeError("Empty train/test in fold; check fold construction.")

    # Fit models
    clf = spec.clf
    reg = spec.reg
    clf.fit(X_train, yc_train)
    reg.fit(X_train, yr_train)

    # Predict
    if hasattr(clf, "predict_proba"):
        p_up = clf.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision function -> sigmoid
        if hasattr(clf, "decision_function"):
            z = clf.decision_function(X_test)
            p_up = 1.0 / (1.0 + np.exp(-z))
        else:
            # last resort: predict labels and treat as prob
            p_up = clf.predict(X_test).astype(float)

    yhat_ret = reg.predict(X_test).astype(float)

    # Clip probabilities for stable log loss
    eps = 1e-12
    p_up_clip = np.clip(p_up, eps, 1.0 - eps)
    yhat_class = (p_up_clip >= 0.5).astype(int)

    # Metrics
    roc = roc_auc_score(y_true=yc_test, y_score=p_up_clip) if len(np.unique(yc_test)) > 1 else float("nan")
    ll = log_loss(y_true=yc_test, y_pred=p_up_clip, labels=[0, 1])
    br = brier_score_loss(y_true=yc_test, y_proba=p_up_clip)
    acc = accuracy_score(y_true=yc_test, y_pred=yhat_class)
    f1 = f1_score(y_true=yc_test, y_pred=yhat_class, zero_division=0)

    mae = mean_absolute_error(yr_test, yhat_ret)
    rmse = _rmse(yr_test.to_numpy(dtype=float), yhat_ret)
    corr = _safe_corr(yr_test.to_numpy(dtype=float), yhat_ret)
    dir_acc = _directional_accuracy(yr_test.to_numpy(dtype=float), yhat_ret)

    preds = pd.DataFrame({
        "ticker": ticker,
        "date": X_test.index,
        "fold_id": fold.fold_id,
        "window_type": window_type,
        "model": spec.name,
        "p_up": p_up_clip,
        "expected_return_pct": yhat_ret,
        "y_class": yc_test.to_numpy(dtype=int),
        "y_return_pct": yr_test.to_numpy(dtype=float),
    }).set_index("date")

    fm = FoldMetrics(
        ticker=ticker,
        window_type=window_type,
        model=spec.name,
        fold_id=fold.fold_id,
        n_test=len(X_test),
        roc_auc=float(roc),
        logloss=float(ll),
        brier=float(br),
        accuracy=float(acc),
        f1=float(f1),
        mae=float(mae),
        rmse=float(rmse),
        corr=float(corr),
        dir_acc=float(dir_acc),
    )

    return preds, fm


def aggregate_metrics(metrics: List[FoldMetrics]) -> pd.DataFrame:
    """
    Aggregate by (ticker, window_type, model), weighted by n_test.
    """
    if not metrics:
        return pd.DataFrame()

    rows = [dataclasses.asdict(m) for m in metrics]
    df = pd.DataFrame(rows)

    group_cols = ["ticker", "window_type", "model"]

    def wavg(col: str) -> float:
        w = df_g["n_test"].to_numpy()
        x = df_g[col].to_numpy(dtype=float)
        mask = ~np.isnan(x)
        if mask.sum() == 0:
            return float("nan")
        return float(np.average(x[mask], weights=w[mask]))

    out_rows = []
    for key, df_g in df.groupby(group_cols, sort=True):
        ticker, window_type, model = key
        out_rows.append({
            "ticker": ticker,
            "window_type": window_type,
            "model": model,
            "folds": int(df_g.shape[0]),
            "n_test_total": int(df_g["n_test"].sum()),
            # Classification
            "roc_auc_wavg": wavg("roc_auc"),
            "logloss_wavg": wavg("logloss"),
            "brier_wavg": wavg("brier"),
            "accuracy_wavg": wavg("accuracy"),
            "f1_wavg": wavg("f1"),
            # Regression
            "mae_wavg": wavg("mae"),
            "rmse_wavg": wavg("rmse"),
            "corr_wavg": wavg("corr"),
            "dir_acc_wavg": wavg("dir_acc"),
        })

    return pd.DataFrame(out_rows).sort_values(["ticker", "window_type", "model"])


# -----------------------
# Main
# -----------------------

def run_for_ticker(
    ticker: str,
    df_ohlcv: pd.DataFrame,
    lookback_days: int,
    min_train_days: int,
    run_rolling_sensitivity_years: Optional[int],
    include_window_summaries: bool,
    out_dir: str,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      predictions_df (all models, all folds)
      metrics_df (fold-level)
    """
    # Indicators & features
    df_ind = compute_indicators(df_ohlcv)
    X = build_feature_table(df_ind, lookback_days=lookback_days, include_window_summaries=include_window_summaries)
    y_class, y_ret = build_targets(df_ind)
    X, y_class, y_ret = align_features_targets(X, y_class, y_ret)

    if X.empty:
        raise RuntimeError(f"{ticker}: Feature table is empty after alignment. Check date range and indicator warmup.")

    # Build folds
    idx = X.index
    folds_exp = build_monthly_folds(idx, min_train_days=min_train_days, rolling_years=None)
    folds_all: List[Tuple[str, List[Fold]]] = [("expanding", folds_exp)]

    if run_rolling_sensitivity_years is not None:
        folds_roll = build_monthly_folds(idx, min_train_days=min_train_days, rolling_years=run_rolling_sensitivity_years)
        folds_all.append((f"rolling_{run_rolling_sensitivity_years}y", folds_roll))

    specs = make_model_specs(random_state=random_state)

    all_preds = []
    all_metrics: List[FoldMetrics] = []

    for window_type, folds in folds_all:
        if not folds:
            print(f"[WARN] {ticker} {window_type}: No folds generated (min_train_days too high or not enough data).")
            continue

        for spec in specs:
            for fold in folds:
                preds, fm = evaluate_fold(
                    ticker=ticker,
                    window_type=window_type,
                    spec=spec,
                    fold=fold,
                    X=X,
                    y_class=y_class,
                    y_ret=y_ret,
                )
                all_preds.append(preds)
                all_metrics.append(fm)

    preds_df = pd.concat(all_preds, axis=0).sort_index() if all_preds else pd.DataFrame()
    metrics_df = pd.DataFrame([dataclasses.asdict(m) for m in all_metrics]) if all_metrics else pd.DataFrame()

    # Save per-ticker artifacts
    os.makedirs(out_dir, exist_ok=True)
    if not preds_df.empty:
        preds_path = os.path.join(out_dir, f"{ticker}_predictions.csv")
        preds_df.to_csv(preds_path)
    if not metrics_df.empty:
        metrics_path = os.path.join(out_dir, f"{ticker}_fold_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

    return preds_df, metrics_df


def main():
    parser = argparse.ArgumentParser(description="SLM Walk-Forward Benchmark (TA features, monthly folds)")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "TSLA", "GOOG"])
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--lookback_days", type=int, default=20)
    parser.add_argument("--min_train_days", type=int, default=252, help="Minimum training days required before first test fold (default: 252)")
    parser.add_argument("--rolling_sensitivity_years", type=int, default=2, help="Rolling-window sensitivity lookback in years. Use 0 to disable.")
    parser.add_argument("--no_window_summaries", action="store_true", help="If set, use snapshot-only features (no 20-day summaries).")
    parser.add_argument("--csv_dir", default=None, help="If provided, load {TICKER}.csv from this directory instead of yfinance.")
    parser.add_argument("--out_dir", default="slm_results")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    rolling_years = None if args.rolling_sensitivity_years == 0 else args.rolling_sensitivity_years
    include_window_summaries = not args.no_window_summaries

    print("=== SLM Benchmark ===")
    print(f"Tickers: {args.tickers}")
    print(f"Date range: {args.start} -> {args.end}")
    print(f"Lookback days (feature window summaries): {args.lookback_days} (enabled={include_window_summaries})")
    print(f"Walk-forward: monthly test blocks; expanding window + rolling sensitivity ({rolling_years}y)")
    print(f"Output dir: {args.out_dir}")
    print("")

    t0 = time.time()

    all_metrics = []
    all_preds = []

    for ticker in args.tickers:
        print(f"--- {ticker} ---")
        # Load data
        if args.csv_dir:
            csv_path = os.path.join(args.csv_dir, f"{ticker}.csv")
            if not os.path.exists(csv_path):
                raise RuntimeError(f"CSV not found: {csv_path}")
            df = load_ohlcv_from_csv(csv_path)
        else:
            df = load_ohlcv_from_yfinance(ticker, start=args.start, end=args.end)

        # Safety: keep in range & drop duplicates
        df = df.loc[(df.index >= pd.to_datetime(args.start)) & (df.index <= pd.to_datetime(args.end))].copy()
        df = df[~df.index.duplicated(keep="first")].sort_index()

        preds_df, metrics_df = run_for_ticker(
            ticker=ticker,
            df_ohlcv=df,
            lookback_days=args.lookback_days,
            min_train_days=args.min_train_days,
            run_rolling_sensitivity_years=rolling_years,
            include_window_summaries=include_window_summaries,
            out_dir=args.out_dir,
            random_state=args.random_state,
        )

        if not metrics_df.empty:
            all_metrics.append(metrics_df)
        if not preds_df.empty:
            all_preds.append(preds_df)

        print(f"{ticker}: rows={len(df)}, feature_rows={len(preds_df)} preds_rows={len(preds_df)} folds={metrics_df['fold_id'].nunique() if not metrics_df.empty else 0}")
        print("")

    # Aggregate
    os.makedirs(args.out_dir, exist_ok=True)
    if all_metrics:
        fold_metrics_all = pd.concat(all_metrics, axis=0, ignore_index=True)
        fold_metrics_all.to_csv(os.path.join(args.out_dir, "ALL_fold_metrics.csv"), index=False)

        # Build aggregated summary
        fold_objs: List[FoldMetrics] = []
        for _, r in fold_metrics_all.iterrows():
            fold_objs.append(FoldMetrics(
                ticker=str(r["ticker"]),
                window_type=str(r["window_type"]),
                model=str(r["model"]),
                fold_id=str(r["fold_id"]),
                n_test=int(r["n_test"]),
                roc_auc=float(r["roc_auc"]),
                logloss=float(r["logloss"]),
                brier=float(r["brier"]),
                accuracy=float(r["accuracy"]),
                f1=float(r["f1"]),
                mae=float(r["mae"]),
                rmse=float(r["rmse"]),
                corr=float(r["corr"]),
                dir_acc=float(r["dir_acc"]),
            ))
        summary = aggregate_metrics(fold_objs)
        summary.to_csv(os.path.join(args.out_dir, "ALL_summary_metrics.csv"), index=False)

        # Console summary (compact)
        pd.set_option("display.width", 140)
        pd.set_option("display.max_columns", 50)
        print("=== Aggregated Summary (weighted by test size) ===")
        print(summary.to_string(index=False))

    if all_preds:
        preds_all = pd.concat(all_preds, axis=0).sort_index()
        preds_all.to_csv(os.path.join(args.out_dir, "ALL_predictions.csv"))

    dt = time.time() - t0
    print("")
    print(f"Done. Runtime: {dt:.1f}s")
    print(f"Artifacts written to: {args.out_dir}")
    print("Key files:")
    print("  - ALL_summary_metrics.csv")
    print("  - ALL_fold_metrics.csv")
    print("  - ALL_predictions.csv")
    for ticker in args.tickers:
        print(f"  - {ticker}_predictions.csv")
        print(f"  - {ticker}_fold_metrics.csv")


if __name__ == "__main__":
    main()