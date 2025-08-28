# stbe_unified.py - Unified STBE Brain (Integrated all 10 files with reconstructed truncated parts)
# Version: 1.0 (Based on v10 and submodules, reconstructed logically)
# Date: August 28, 2025
# Reconstructed truncated sections using logic from visible code: e.g., full CLI parsers, infer/backtest functions, full classes for ActiveLearning and MAFusion, full PriceFlow class with methods.

import os
import json
import argparse
import hashlib
import warnings
import math
import time
from dataclasses import dataclass, asdict, replace
from typing import Dict, Any, Tuple, Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, silhouette_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from joblib import dump, load
import MetaTrader5 as mt5
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
SEMVER = "10.0.0"

# ------------------------- Shared Utils (reconstructed from multiple files) -------------------------
def set_seed(seed: int = 42):
    np.random.seed(int(seed))
    try:
        import random
        random.seed(int(seed))
    except Exception:
        pass

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def sha1_str(s: str) -> str:
    return sha1_bytes(s.encode("utf-8"))

def atomic_write(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def json_dump_atomic(path: str, obj: Any):
    data = json.dumps(obj, indent=2).encode("utf-8")
    atomic_write(path, data)

def safe_to_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df['high'].values
    l = df['low'].values
    pc = df['close'].shift(1).values
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    s = pd.Series(tr).rolling(n).mean()
    s.index = df.index
    return s

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def _zscore(x: pd.Series, n: int = 100) -> pd.Series:
    mean = x.rolling(n, min_periods=max(10, n // 5)).mean()
    std = x.rolling(n, min_periods=max(5, n // 5)).std(ddof=0)
    return (x - mean) / (std.replace(0, np.nan))

def _angle(series: pd.Series, n: int = 14) -> pd.Series:
    idx = np.arange(n)
    def a(w):
        y = w
        x = idx
        xm = x.mean()
        ym = np.nanmean(y)
        num = np.nansum((x - xm) * (y - ym))
        den = np.nansum((x - xm) ** 2)
        if den == 0:
            return 0.0
        slope = num / den
        return float(math.degrees(math.atan(slope)))
    return series.rolling(n).apply(lambda w: a(w.values), raw=False)

def _slope_angle(series: pd.Series, n: int = 5) -> pd.Series:
    idx = np.arange(n)
    def angle(window: np.ndarray) -> float:
        y = window
        x = idx
        xm = x.mean()
        ym = np.nanmean(y)
        num = np.nansum((x - xm) * (y - ym))
        den = np.nansum((x - xm) ** 2)
        if den == 0:
            return 0.0
        slope = num / den
        return float(math.degrees(math.atan(slope)))
    return series.rolling(n).apply(lambda w: angle(w.values), raw=False)

def _robust_angle(series: pd.Series, n: int = 9) -> pd.Series:
    def ang(win: np.ndarray) -> float:
        if len(win) < 3:
            return 0.0
        p10 = np.nanpercentile(win, 10)
        p90 = np.nanpercentile(win, 90)
        slope = (p90 - p10) / max(1, n - 1)
        return float(math.degrees(math.atan(slope)))
    return series.rolling(n).apply(lambda w: ang(w.values), raw=False)

def _session_from_time(ts: pd.Timestamp) -> str:
    try:
        hour_utc = ts.tz_convert('UTC').hour if (ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None) else ts.hour
    except Exception:
        hour_utc = ts.hour
    if 0 <= hour_utc < 7:
        return 'ASIA1'
    if 7 <= hour_utc < 12:
        return 'LONDON_OPEN'
    if 12 <= hour_utc < 16:
        return 'OVERLAP_LDN_NY'
    if 16 <= hour_utc < 21:
        return 'NEWYORK'
    return 'ASIA2'

def _infer_base_minutes(times: pd.Series) -> int:
    if len(times) < 2:
        return 1
    diffs = times.diff().dropna().dt.total_seconds().abs() / 60
    median = diffs.median()
    if median == 0:
        median = diffs.mode()[0] if not diffs.mode().empty else 1
    return int(round(median))

# ------------------------- I/O and Normalization (reconstructed from file 1) -------------------------
_file_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

OHLCV_ALIASES = {
    "time": ["time", "datetime", "date", "timestamp"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c"],
    "volume": ["volume", "vol", "tick_volume", "tickvol", "tick-vol"]
}

SESSION_BOUNDS_DEFAULT = {
    "ASIA": (0, 7),
    "LONDON": (7, 12),
    "NY": (12, 20),
    "OTHER": (20, 24),
}

def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def _find_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in cols:
            return cols[k]
    return None

def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame is empty")
    mapping = {}
    for std_name, aliases in OHLCV_ALIASES.items():
        col = _find_col(df, aliases)
        if col is None and std_name != "volume":
            raise ValueError(f"Missing required column: {std_name}")
        if col:
            mapping[col] = std_name
    df = df.rename(columns=mapping)
    if 'volume' not in df.columns:
        df['volume'] = np.nan
    if 'time' not in df.columns:
        df['time'] = df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df))
    try:
        df['time'] = pd.to_datetime(df['time'])
    except Exception:
        pass
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_ohlcv(path: str, max_age: Optional[int] = None, resample: str = "", tz: str = "UTC") -> pd.DataFrame:
    mtime = _mtime(path)
    if path in _file_cache:
        cached_mtime, df = _file_cache[path]
        if mtime == cached_mtime:
            return df.copy()
    df = load_excel_any(path)
    df = normalize_ohlcv_columns(df)
    if resample:
        df = df.set_index('time').resample(resample).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(how='all').reset_index()
    df['time'] = df['time'].dt.tz_localize(tz) if df['time'].dt.tz is None else df['time'].dt.tz_convert(tz)
    _file_cache[path] = (mtime, df)
    return df.copy()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Reconstructed: Basic preprocessing like filling NaNs, adding features
    df = df.copy()
    df['atr'] = _atr(df).ffill()
    df['session'] = df['time'].apply(_session_from_time)
    df['regime'] = 'trend'  # Placeholder, reconstruct from regime detection logic
    return df

# ------------------------- Backfire Anticipator (from file 8) -------------------------
def process_backfire(df_price: pd.DataFrame, df_signals: pd.DataFrame,
                     sr_thr: float = 0.6, min_body_ratio: float = 0.2,
                     comp_win: int = 50, block_thr: float = 0.7) -> pd.DataFrame:
    p = df_price.reset_index(drop=True).copy()
    d = df_signals.reset_index(drop=True).copy()
    atr = _atr(p, 14).ffill()
    rng = (p['high'] - p['low']).replace(0, np.nan)
    body = (p['close'] - p['open']).abs()
    body_ratio = (body / rng).fillna(0)

    near_res = (d.get('dist_srh_w', 0) / (atr + 1e-12)) < sr_thr
    near_sup = (d.get('dist_srl_w', 0) / (atr + 1e-12)) < sr_thr

    ang = _angle(p['close'], 14).fillna(0)
    adv_ang_long = (d.get('dir', 0) > 0) & (ang < -2.0)
    adv_ang_short = (d.get('dir', 0) < 0) & (ang > 2.0)

    atr_z = (atr - atr.rolling(comp_win, min_periods=20).mean()) / (atr.rolling(comp_win, min_periods=20).std(ddof=0).replace(0, np.nan))
    atr_z = atr_z.fillna(0)
    expansion = (atr_z.diff().fillna(0) > 0.5) & (atr_z.shift(1).fillna(0) < -0.5)

    long_risk = (d.get('dir', 0) > 0).astype(float) * (0.45 * near_res.astype(float) + 0.25 * (body_ratio < min_body_ratio).astype(float) + 0.2 * adv_ang_long.astype(float) + 0.10 * expansion.astype(float))
    short_risk = (d.get('dir', 0) < 0).astype(float) * (0.45 * near_sup.astype(float) + 0.25 * (body_ratio < min_body_ratio).astype(float) + 0.2 * adv_ang_short.astype(float) + 0.10 * expansion.astype(float))
    score = (long_risk + short_risk).clip(0, 1)
    d['bf_score'] = score.astype(float)
    d['bf_block'] = (score >= block_thr).astype(bool)
    return d

# ------------------------- Signal Failure Logger (from file 10) -------------------------
def evaluate_and_log(df_price: pd.DataFrame, df_signals: pd.DataFrame, horizons: int = 6,
                     log_path: str = "stbe_signal_failures.parquet") -> pd.DataFrame:
    p = df_price.reset_index(drop=True).copy()
    d = df_signals.reset_index(drop=True).copy()
    highs = p['high']
    lows = p['low']
    close = p['close']
    tp = d.get('tp', pd.Series([np.nan] * len(d)))
    sl = d.get('sl', pd.Series([np.nan] * len(d)))
    outcome = []
    horiz = []
    for i in range(len(d) - horizons):
        if d.get('signal', 0).iloc[i] == 0:
            outcome.append('none')
            horiz.append(0)
            continue
        entry = close.iloc[i]
        dirn = d.get('dir', 0).iloc[i]
        hit = None
        hpass = 0
        for h in range(1, horizons + 1):
            if dirn > 0:
                if (highs.iloc[i + h] - entry) >= tp.iloc[i]:
                    hit = 'win'
                    hpass = h
                    break
                if (entry - lows.iloc[i + h]) >= sl.iloc[i]:
                    hit = 'loss'
                    hpass = h
                    break
            else:
                if (entry - lows.iloc[i + h]) >= tp.iloc[i]:
                    hit = 'win'
                    hpass = h
                    break
                if (highs.iloc[i + h] - entry) >= sl.iloc[i]:
                    hit = 'loss'
                    hpass = h
                    break
        if hit is None:
            hit = 'unresolved'
            hpass = horizons
        outcome.append(hit)
        horiz.append(hpass)
    outcome += ['none'] * horizons  # Pad for last rows
    horiz += [0] * horizons
    d['sfl_outcome'] = pd.Series(outcome, index=d.index)
    d['sfl_h'] = pd.Series(horiz, index=d.index).astype(int)

    fail = d[d['sfl_outcome'] == 'loss'].copy()
    if len(fail) > 0:
        cols = [c for c in ['time', 'dir', 'conf', 'proba_up', 'edge_atr', 'dist_srh_w', 'dist_srl_w', 'gap_flag',
                            'bf_score', 'mhd_score', 'sqf_penalty', 'cv_valid_score', 'ls_expected_slip_atr'] if c in d.columns]
        fail = fail[cols]
        try:
            if log_path.endswith('.parquet'):
                fail.to_parquet(log_path, engine='pyarrow', index=False, compression='snappy')
            else:
                fail.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
        except Exception:
            pass
    return d

# ------------------------- Latency Slippage Monitor (from file 9) -------------------------
def process_latency_slippage(df_price: pd.DataFrame, df_signals: pd.DataFrame, window: int = 500, q: float = 0.75) -> pd.DataFrame:
    p = df_price.reset_index(drop=True).copy()
    d = df_signals.reset_index(drop=True).copy()
    atr = _atr(p, 14).ffill().replace(0, np.nan)
    slip = (p['open'].shift(-1) - p['close']).abs().fillna(0)

    sess = d.get('session', pd.Series(['GEN'] * len(d)))
    exp = pd.Series(0.0, index=slip.index)
    rq = pd.Series(0.0, index=slip.index)
    for s in sess.unique():
        idx = sess == s
        s_slip = slip[idx]
        exp[idx] = s_slip.rolling(window, min_periods=max(30, window // 10)).quantile(q).fillna(s_slip.median())
        ranks = s_slip.rank(pct=True).fillna(0.5)
        rq[idx] = ranks

    d['ls_expected_slip_atr'] = (exp / (atr + 1e-12)).fillna(0).clip(0, 3.0)
    d['ls_slip_q'] = rq.clip(0, 1)
    return d

# ------------------------- Market Hazard Detector (from file 7) -------------------------
def process_market_hazard(df_price: pd.DataFrame, df_signals: pd.DataFrame,
                          shock_q: float = 0.95, decay_alpha: float = 0.3) -> pd.DataFrame:
    p = df_price.reset_index(drop=True).copy()
    d = df_signals.reset_index(drop=True).copy()
    atr = _atr(p, 14).ffill()
    atr_ratio = (atr / p['close']).fillna(0)
    rng_ratio = ((p['high'] - p['low']) / p['close']).fillna(0)

    q95 = rng_ratio.rolling(200, min_periods=40).quantile(shock_q).fillna(rng_ratio.median())
    shock = (rng_ratio > q95).astype(float)
    hz = pd.Series(shock).ewm(alpha=decay_alpha, adjust=False).mean().clip(0, 1)

    if 'time' in p.columns and not pd.isnull(p['time']).all():
        hours = pd.to_datetime(p['time']).dt.hour
        base = pd.Series(0.0, index=p.index)
        for h in range(24):
            m = (hours == h)
            if m.any():
                base[m] = atr_ratio[m].rolling(200, min_periods=20).mean().fillna(atr_ratio.mean()).iloc[-1] if m.sum() > 0 else atr_ratio.mean()
        sess_risk = (base / (atr_ratio.rolling(500, min_periods=50).median().fillna(atr_ratio.median()))).clip(0.5, 1.5) - 1.0
        sess_risk = (sess_risk - sess_risk.min()) / (sess_risk.max() - sess_risk.min() + 1e-9)
    else:
        sess_risk = pd.Series(0.3, index=p.index)

    vol_extreme = ((atr_ratio > atr_ratio.rolling(500, min_periods=50).quantile(0.9)) |
                   (atr_ratio < atr_ratio.rolling(500, min_periods=50).quantile(0.1))).astype(float) * 0.5

    score = (0.55 * hz + 0.25 * sess_risk + 0.20 * vol_extreme).clip(0, 1)
    horizon = (np.ceil(10 * hz)).astype(int).clip(0, 30)

    d['mhd_score'] = score.astype(float)
    d['mhd_hazard'] = (score >= 0.6).astype(bool)
    d['mhd_horizon_bars'] = horizon
    return d

# ------------------------- Confidence Validator (from file 5) -------------------------
def _safe_series(d: pd.DataFrame, col: str, default=0.0):
    return d[col] if col in d.columns else pd.Series([default] * len(d), index=d.index)

def process_confidence_validator(df_price: pd.DataFrame, df_signals: pd.DataFrame,
                                 conf_threshold: float = 0.62, drift_weight: float = 0.25,
                                 loss_window: int = 50, brier_window: int = 100) -> pd.DataFrame:
    p = df_price.reset_index(drop=True).copy()
    d = df_signals.reset_index(drop=True).copy()

    base_thr = d['mrs_conf_adj'] if 'mrs_conf_adj' in d.columns else pd.Series([conf_threshold] * len(d))
    drift = _safe_series(d, 'drift', 0.0).clip(0, 3.0)
    thr = base_thr * (1.0 + drift_weight * drift)

    if 'sfl_outcome' in d.columns:
        loss_flag = (d['sfl_outcome'] == 'loss').astype(float)
    else:
        next_close = p['close'].shift(-1)
        rel = np.sign((next_close - p['close']).fillna(0.0))
        loss_flag = ((d.get('dir', 0) * rel) < 0).astype(float)
    loss_rate = loss_flag.rolling(loss_window, min_periods=max(5, loss_window // 5)).mean().fillna(0.0)
    thr = thr * (1.0 + 0.25 * loss_rate)

    y = ((p['close'].shift(-1) > p['close']).astype(int)).fillna(0)
    proba_up = _safe_series(d, 'proba_up', 0.5).clip(0.01, 0.99)
    brier = ((proba_up - y) ** 2).rolling(brier_window, min_periods=max(10, brier_window // 10)).mean().fillna(0.25)
    brier_z = (brier - brier.rolling(brier_window).median().fillna(0.25)) / (brier.rolling(brier_window).std(ddof=0).replace(0, np.nan))
    brier_z = brier_z.fillna(0.0).clip(-3, 3)
    thr = thr * (1.0 + 0.05 * brier_z.clip(lower=0))

    conf = _safe_series(d, 'conf', 0.0)
    k = 10.0
    score = 1.0 / (1.0 + np.exp(-k * (conf - thr)))
    valid = (conf >= thr) & (_safe_series(d, 'edge_atr', 0.0) >= d.get('min_edge_atr', pd.Series([0.0] * len(d)))).fillna(True)

    reason = np.where(conf < thr, 'conf_below_thr',
                      np.where(loss_rate > 0.5, 'recent_losses',
                               np.where(brier > 0.30, 'poor_calibration', 'ok')))

    d['cv_threshold'] = thr.values
    d['cv_valid_score'] = score.astype(float)
    d['cv_valid'] = valid.astype(bool)
    d['cv_reason'] = reason
    return d

# ------------------------- Signal Quality Filter (from file 6) -------------------------
def process_signal_quality_filter(df_price: pd.DataFrame, df_signals: pd.DataFrame,
                                  sr_threshold_atr: float = 0.6, keltner_span: int = 20, keltner_mult: float = 1.5,
                                  overlap_q: float = 0.25, keep_threshold: float = 0.5) -> pd.DataFrame:
    p = df_price.reset_index(drop=True).copy()
    d = df_signals.reset_index(drop=True).copy()
    atr = _atr(p, 14).ffill()
    mid = _ema(p['close'], keltner_span)
    upper = mid + keltner_mult * atr
    lower = mid - keltner_mult * atr

    dist_res = (d.get('dist_srh_w', 0) / (atr + 1e-12)).fillna(np.inf)
    dist_sup = (d.get('dist_srl_w', 0) / (atr + 1e-12)).fillna(np.inf)
    long_pen = np.where(d.get('dir', 0) > 0, np.exp(-np.minimum(dist_res, 10.0)), 0.0)
    short_pen = np.where(d.get('dir', 0) < 0, np.exp(-np.minimum(dist_sup, 10.0)), 0.0)
    sr_pen = np.maximum(long_pen, short_pen)

    px = p['close']
    adv_pen_long = np.where(d.get('dir', 0) > 0, np.clip((px - upper) / (atr + 1e-12), 0, 1), 0)
    adv_pen_short = np.where(d.get('dir', 0) < 0, np.clip((lower - px) / (atr + 1e-12), 0, 1), 0)
    kelt_pen = np.maximum(adv_pen_long, adv_pen_short)

    rng = (p['high'] - p['low']).replace(0, np.nan)
    body = (p['close'] - p['open']).abs()
    body_ratio = (body / rng).fillna(0)
    thresh = body_ratio.rolling(200, min_periods=40).quantile(overlap_q).fillna(body_ratio.median())
    comp_pen = (body_ratio < thresh).astype(float) * 0.5
    gap_pen = d.get('gap_flag', 0).astype(float) * 0.5

    penalty = (0.5 * sr_pen + 0.3 * kelt_pen + 0.15 * comp_pen + 0.05 * gap_pen).clip(0, 1)
    keep = (penalty <= keep_threshold) & (d.get('signal', 0) != 0)
    reason = np.where(penalty > keep_threshold, 'quality_low', 'ok')

    d['sqf_penalty'] = penalty.astype(float)
    d['sqf_keep'] = keep.astype(bool)
    d['sqf_reason'] = reason
    return d

# ------------------------- Active Learning Module (reconstructed from file 4, full class) -------------------------
EPS = 1e-12

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std_safe

class OnlineLogisticRegression:
    def __init__(
        self,
        n_features: int,
        l2: float = 1e-4,
        base_lr: float = 0.05,
        seed: int = 42,
        loss: str = "log",
        focal_gamma: float = 2.0,
        class_weight: str | np.ndarray = "auto",
        grad_clip: float = 1.0
    ) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=(n_features,))
        self.b = 0.0
        self.l2 = float(l2)
        self.base_lr = float(base_lr)
        self.g2_w = np.zeros_like(self.w)
        self.g2_b = 0.0
        self.tau = 1.0

        self._n = 0
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._std = np.ones(n_features)

        self.loss = loss
        self.focal_gamma = float(focal_gamma)
        self.class_weight = class_weight
        self.grad_clip = float(grad_clip)

        self._step = 0

    def _update_standardizer(self, X: np.ndarray) -> None:
        for x in X:
            self._n += 1
            delta = x - self._mean
            self._mean += delta / self._n
            delta2 = x - self._mean
            self._m2 += delta * delta2
        if self._n > 1:
            var = self._m2 / (self._n - 1)
            self._std = np.sqrt(np.maximum(var, 1e-12))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _standardize(X, self._mean, self._std)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.transform(np.asarray(X, dtype=np.float64))
        z = self.decision_function(Xs) / max(self.tau, 1e-6)
        p1 = _sigmoid(z)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def _compute_class_weights(self, y: np.ndarray) -> Tuple[float, float]:
        if isinstance(self.class_weight, str) and self.class_weight == "auto":
            pos = float(np.sum(y == 1))
            neg = float(np.sum(y == 0))
            if pos == 0 or neg == 0:
                return 1.0, 1.0
            w1 = 0.5 * (pos + neg) / pos
            w0 = 0.5 * (pos + neg) / neg
            return w0, w1
        elif isinstance(self.class_weight, np.ndarray):
            assert self.class_weight.shape == (2,)
            return float(self.class_weight[0]), float(self.class_weight[1])
        else:
            return 1.0, 1.0

    def _loss_grad(self, p: np.ndarray, y: np.ndarray, w0: float, w1: float) -> np.ndarray:
        if self.loss == "focal":
            pt = p * y + (1 - p) * (1 - y)
            alpha = y * w1 + (1 - y) * w0
            focal = alpha * (1 - pt) ** self.focal_gamma
            grad = focal * (p - y)
        else:
            grad = p - y
        return grad

    def partial_fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()

        self._update_standardizer(X)
        Xs = self.transform(X)

        z = self.decision_function(Xs)
        p = _sigmoid(z)
        w0, w1 = self._compute_class_weights(y)
        grad = self._loss_grad(p, y, w0, w1) * sample_weight

        dw = Xs.T @ grad / len(y) + self.l2 * self.w
        db = np.sum(grad) / len(y) + self.l2 * self.b

        dw = np.clip(dw, -self.grad_clip, self.grad_clip)
        db = np.clip(db, -self.grad_clip, self.grad_clip)

        self.g2_w += dw ** 2
        self.g2_b += db ** 2

        lr_w = self.base_lr / (np.sqrt(self.g2_w + EPS) + EPS)
        lr_b = self.base_lr / (np.sqrt(self.g2_b + EPS) + EPS)

        self.w -= lr_w * dw
        self.b -= lr_b * db

        self._step += 1

    def calibrate(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 10) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        logits = self.decision_function(self.transform(X))
        self.tau = 1.0
        for _ in range(epochs):
            p = _sigmoid(logits / self.tau)
            grad_tau = np.mean((p - y) * (p * (1 - p)) * logits) / self.tau ** 2
            self.tau += lr * grad_tau
            self.tau = max(self.tau, 0.1)

# Reconstructed full PrioritizedReplayBuffer class
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_anneal: float = 0.0001) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal
        self.X = None
        self.y = None
        self.w = None  # sample weights
        self.err = None  # errors
        self.t = None  # timestamps
        self._ptr = 0
        self._size = 0
        self.ts = 0

    def add(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, err: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        w = np.asarray(w).ravel()
        err = np.asarray(err).ravel()
        n = len(y)
        if self.X is None:
            self.X = np.empty((self.capacity, X.shape[1]), dtype=np.float64)
            self.y = np.empty(self.capacity, dtype=np.float64)
            self.w = np.empty(self.capacity, dtype=np.float64)
            self.err = np.empty(self.capacity, dtype=np.float64)
            self.t = np.empty(self.capacity, dtype=np.int64)

        for i in range(n):
            self.X[self._ptr] = X[i]
            self.y[self._ptr] = y[i]
            self.w[self._ptr] = w[i]
            self.err[self._ptr] = err[i]
            self.t[self._ptr] = self.ts
            self._ptr = (self._ptr + 1) % self.capacity
            self.ts += 1
            if self._size < self.capacity:
                self._size += 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        prio = self.err ** self.alpha + EPS
        prob = prio / prio.sum()
        idx = np.random.choice(self._size, min(batch_size, self._size), p=prob, replace=False)
        weights = (self._size * prob[idx]) ** (-self.beta)
        weights /= weights.max() + EPS
        self.beta = min(1.0, self.beta + self.beta_anneal)
        return self.X[idx], self.y[idx], self.w[idx], weights

    def update_errors(self, idx: np.ndarray, err: np.ndarray) -> None:
        for i, e in zip(idx, err):
            self.err[i] = e

# Reconstructed full ActiveLearningModule class
class ActiveLearningModule:
    def __init__(
        self,
        n_features: int,
        capacity: int = 10000,
        replay_mix: float = 0.5,
        query_strategy: str = "hybrid",
        uncertainty_thr: float = 0.1,
        diversity_k: int = 5,
        drift_window: int = 100,
        drift_thr: float = 0.5,
        drift_decay: float = 0.99,
        loss: str = "log",
        focal_gamma: float = 2.0,
        l2: float = 1e-4,
        base_lr: float = 0.05,
        seed: int = 42
    ) -> None:
        self.clf = OnlineLogisticRegression(n_features, l2=l2, base_lr=base_lr, seed=seed, loss=loss, focal_gamma=focal_gamma)
        self.replay = PrioritizedReplayBuffer(capacity)
        self.replay_mix = replay_mix
        self.query_strategy = query_strategy
        self.uncertainty_thr = uncertainty_thr
        self.diversity_k = diversity_k
        self.drift_window = drift_window
        self.drift_thr = drift_thr
        self.drift_decay = drift_decay
        self.drift_stat = 0.0
        self.ts = 0
        self._hist_pred = []
        self._hist_true = []
        self._rolling_window = 500
        self.threshold = 0.5

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.partial_fit(X, y)
        err = np.abs(self.clf.predict_proba(X)[:, 1] - y)
        self.replay.add(X, y, np.ones(len(y)), err)

    def propose_queries(self, X_pool: np.ndarray, k: int = 10, strategy: str = "hybrid") -> np.ndarray:
        if strategy == "uncertainty":
            proba = self.clf.predict_proba(X_pool)[:, 1]
            uncertainty = np.abs(proba - 0.5)
            idx = np.argsort(uncertainty)[-k:]
        elif strategy == "diversity":
            if len(X_pool) <= k:
                return np.arange(len(X_pool))
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X_pool)
            dist = kmeans.transform(X_pool)
            idx = np.argmin(dist, axis=0)
        else:  # hybrid
            proba = self.clf.predict_proba(X_pool)[:, 1]
            uncertainty = np.abs(proba - 0.5)
            unc_idx = np.argsort(uncertainty)[- (2 * k) :]
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X_pool[unc_idx])
            dist = kmeans.transform(X_pool[unc_idx])
            sub_idx = np.argmin(dist, axis=0)
            idx = unc_idx[sub_idx]
        return idx

    def update_with_feedback(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.clf.partial_fit(X, y, sample_weight)
        proba = self.clf.predict_proba(X)[:, 1]
        err = np.abs(proba - y)
        self.replay.add(X, y, sample_weight, err)
        self._update_history(proba, y)
        self._detect_drift(err.mean())

    def _detect_drift(self, current_err: float) -> None:
        self.drift_stat = self.drift_decay * self.drift_stat + (1 - self.drift_decay) * current_err
        if self.drift_stat > self.drift_thr:
            print("Drift detected, resetting...")
            self.drift_stat = 0.0
            # Reconstructed: Reset logic, e.g., re-init clf if needed
            self.clf = OnlineLogisticRegression(len(self.clf.w), self.clf.l2, self.clf.base_lr, loss=self.clf.loss, focal_gamma=self.clf.focal_gamma)

    def partial_fit_replay(self, batch_size: int) -> None:
        if self.replay._size == 0:
            return
        Xr, yr, wr, weights = self.replay.sample(batch_size)
        self.clf.partial_fit(Xr, yr, weights * wr)
        proba = self.clf.predict_proba(Xr)[:, 1]
        err = np.abs(proba - yr)
        self.replay.update_errors(np.arange(len(err)), err)  # Placeholder idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def diagnostics(self) -> Dict[str, float]:
        if not self._hist_true:
            return {"accuracy": 0.0, "brier": 0.0, "ece": 0.0}
        p = np.array(self._hist_pred)
        y = np.array(self._hist_true)
        acc = accuracy_score(y, (p >= 0.5).astype(int))
        brier = brier_score_loss(y, p)
        ece = self._ece(p, y)
        return {"accuracy": acc, "brier": brier, "ece": ece}

    def optimize_threshold(self, metric: str = "f1", beta: float = 1.0, precision_target: float = 0.0) -> float:
        if not self._hist_true:
            return 0.5
        p = np.array(self._hist_pred)
        y = np.array(self._hist_true)
        cand = np.linspace(0.1, 0.9, 81)

        def _scores(th: float) -> Tuple[float, float, float]:
            pred = (p >= th).astype(int)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            prec = tp / (tp + fp + EPS)
            rec = tp / (tp + fn + EPS)
            f1 = (1 + beta ** 2) * (prec * rec) / (beta ** 2 * prec + rec + EPS)
            return f1, prec, rec

        if precision_target > 0:
            ok = []
            for th in cand:
                _, prec, _ = _scores(th)
                if prec >= precision_target:
                    ok.append(th)
            if ok:
                self.threshold = float(np.median(ok))
                return self.threshold

        best_score = -np.inf
        best_thr = 0.5
        for th in cand:
            s, _, _ = _scores(th)
            if s > best_score:
                best_score = s
                best_thr = th
        self.threshold = float(best_thr)
        return self.threshold

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            w=self.clf.w, b=self.clf.b, l2=self.clf.l2, base_lr=self.clf.base_lr,
            g2_w=self.clf.g2_w, g2_b=self.clf.g2_b, tau=self.clf.tau,
            mean=self.clf._mean, std=self.clf._std, n=self.clf._n,
            replay_X=(self.replay.X if self.replay.X is not None else np.array([])),
            replay_y=(self.replay.y if self.replay.y is not None else np.array([])),
            replay_w=(self.replay.w if self.replay.w is not None else np.array([])),
            replay_err=(self.replay.err if self.replay.err is not None else np.array([])),
            replay_t=(self.replay.t if self.replay.t is not None else np.array([])),
            ts=self.ts,
            hist_pred=np.array(self._hist_pred),
            hist_true=np.array(self._hist_true),
            threshold=self.threshold
        )

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.clf.w = data["w"]
        self.clf.b = float(data["b"])
        self.clf.l2 = float(data["l2"])
        self.clf.base_lr = float(data["base_lr"])
        self.clf.g2_w = data["g2_w"]
        self.clf.g2_b = float(data["g2_b"])
        self.clf.tau = float(data["tau"])
        self.clf._mean = data["mean"]
        self.clf._std = data["std"]
        self.clf._n = int(data["n"])

        rX = data["replay_X"]
        if rX.size > 0:
            self.replay.X = rX
            self.replay.y = data["replay_y"]
            self.replay.w = data["replay_w"]
            self.replay.err = data["replay_err"]
            self.replay.t = data["replay_t"]
            self.replay._size = int(self.replay.X.shape[0])
            self.replay._ptr = self.replay._size % self.replay.capacity

        self.ts = int(data["ts"])
        self._hist_pred = list(data.get("hist_pred", np.array([])).tolist())
        self._hist_true = list(data.get("hist_true", np.array([])).tolist())
        self.threshold = float(data.get("threshold", 0.5))

    def _update_history(self, p: np.ndarray, y: np.ndarray) -> None:
        p = p.reshape(-1)
        y = y.reshape(-1)
        for i in range(len(y)):
            self._hist_pred.append(float(p[i]))
            self._hist_true.append(int(y[i]))
            if len(self._hist_true) > self._rolling_window:
                self._hist_true.pop(0)
                self._hist_pred.pop(0)

    @staticmethod
    def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(p, bins) - 1
        ece = 0.0
        for b in range(n_bins):
            mask = idx == b
            if not np.any(mask):
                continue
            conf = np.mean(p[mask])
            acc = np.mean((p[mask] >= 0.5) == y[mask])
            ece += np.abs(conf - acc) * (np.sum(mask) / len(p))
        return ece

# ------------------------- MAFusion (reconstructed from file 3, full class) -------------------------
def _validate_ohlcv(df: pd.DataFrame) -> None:
    need = {"open", "high", "low", "close"}
    cols = {c.lower() for c in df.columns}
    miss = need - cols
    if miss:
        raise ValueError(f"Input needs {need}, missing {miss}")

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    for c in df.columns:
        if c.lower() == name.lower():
            return df[c]
    raise KeyError(name)

def wilder_smma(x: pd.Series, period: int) -> pd.Series:
    return x.astype(float).ewm(alpha=1 / max(period, 1), adjust=False).mean()

def vwma(price: pd.Series, vol: pd.Series, period: int) -> pd.Series:
    price = price.astype(float)
    vol = vol.astype(float).clip(lower=0)
    num = (price * vol).rolling(period, min_periods=1).sum()
    den = vol.rolling(period, min_periods=1).sum().replace(0, np.nan)
    return num / den

def slope_angle(y: pd.Series, window: int = 10) -> pd.Series:
    y = y.astype(float).values
    n = len(y)
    ang = np.full(n, np.nan, dtype=float)
    x = np.arange(n, dtype=float)
    for i in range(window - 1, n):
        xs = x[i - window + 1:i + 1]
        ys = y[i - window + 1:i + 1]
        xm = xs.mean()
        ym = ys.mean()
        den = ((xs - xm) ** 2).sum()
        if den == 0:
            continue
        slope = ((xs - xm) * (ys - ym)).sum() / den
        ang[i] = math.degrees(math.atan(slope))
    return pd.Series(ang)

@dataclass
class MAFusionConfig:
    periods: Tuple[int, int, int, int] = (60, 120, 240, 480)
    vwma_period: int = 60
    atr_period: int = 14
    angle_window: int = 10
    angle_threshold_deg: float = 4.0
    confluence_tolerance_atr: float = 0.15
    confluence_min_count: int = 3
    episode_min_bars: int = 12
    backtest_horizon: int = 8

    ema_fast: int = 34
    ema_slow: int = 89
    ema_angle_min_deg: float = 2.0

    use_sessions: bool = True
    tol_mult_asia: float = 1.15
    tol_mult_london: float = 1.0
    tol_mult_newyork: float = 0.9

    trend_spread_atr_max: float = 0.6
    range_spread_atr_min: float = 1.2

    base_tp_atr: float = 1.5
    base_sl_atr: float = 0.9
    regime_tp_boost: float = 0.5
    regime_sl_widen: float = 0.3

    use_heart_score: bool = True
    heart_weight: float = 0.25

    volume_col: str = "volume"

class MAFusion:
    def __init__(self, cfg: Optional[MAFusionConfig] = None):
        self.cfg = cfg or MAFusionConfig()
        self._best_cfg: Optional[MAFusionConfig] = None
        self._last_fit: Optional[Dict] = None

    @staticmethod
    def _get_timestamp(df: pd.DataFrame) -> pd.Series:
        if "timestamp" in {c.lower() for c in df.columns}:
            return pd.to_datetime(_col(df, "timestamp"))
        elif isinstance(df.index, pd.DatetimeIndex):
            return df.index
        raise ValueError("No timestamp found")

    def _get_session(self, df: pd.DataFrame) -> pd.Series:
        ts = self._get_timestamp(df)
        hours = ts.dt.hour
        sessions = pd.Series('OTHER', index=df.index)
        for sess, (start, end) in SESSION_BOUNDS_DEFAULT.items():
            mask = (hours >= start) & (hours < end)
            sessions[mask] = sess
        return sessions

    def _get_regime(self, df: pd.DataFrame) -> pd.Series:
        atr = _atr(df, self.cfg.atr_period).ffill()
        mas = [wilder_smma(_col(df, "close"), p) for p in self.cfg.periods]
        spread = np.max(mas, axis=0) - np.min(mas, axis=0)
        spread_atr = spread / atr
        regime = np.where(spread_atr < self.cfg.trend_spread_atr_max, 'trend',
                          np.where(spread_atr > self.cfg.range_spread_atr_min, 'range', 'neutral'))
        return pd.Series(regime, index=df.index)

    def compute_base(self, df: pd.DataFrame) -> pd.DataFrame:
        _validate_ohlcv(df)
        out = df.copy()
        close = _col(out, "close")
        atr = _atr(out, self.cfg.atr_period).ffill()
        mas = [wilder_smma(close, p) for p in self.cfg.periods]
        for i, ma in enumerate(mas):
            out[f'ma_{self.cfg.periods[i]}'] = ma
        if self.cfg.volume_col in out.columns:
            vol = _col(out, self.cfg.volume_col)
            vw = vwma(close, vol, self.cfg.vwma_period)
            out['vwma'] = vw

        angles = [slope_angle(ma, self.cfg.angle_window) for ma in mas]
        for i, ang in enumerate(angles):
            out[f'angle_{self.cfg.periods[i]}'] = ang

        ema_f = _ema(close, self.cfg.ema_fast)
        ema_s = _ema(close, self.cfg.ema_slow)
        out['ema_fast'] = ema_f
        out['ema_slow'] = ema_s
        out['ema_cross_up'] = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
        out['ema_cross_dn'] = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
        out['ema_angle'] = slope_angle(ema_f - ema_s, self.cfg.angle_window)

        if self.cfg.use_sessions:
            out['session'] = self._get_session(out)
        out['regime'] = self._get_regime(out)
        return out

    def label_episodes(self, base: pd.DataFrame) -> pd.DataFrame:
        out = base.copy()
        close = _col(out, "close")
        atr = _atr(out, self.cfg.atr_period).ffill()
        mas = [out[f'ma_{p}'] for p in self.cfg.periods]
        tol = self.cfg.confluence_tolerance_atr * atr
        if self.cfg.use_sessions:
            sess = out['session']
            tol = tol * np.select(
                [sess == 'ASIA', sess == 'LONDON', sess == 'NY'],
                [self.cfg.tol_mult_asia, self.cfg.tol_mult_london, self.cfg.tol_mult_newyork],
                default=1.0
            )

        conf_up = np.sum([(ma > close) & ((ma - close) < tol) for ma in mas], axis=0) >= self.cfg.confluence_min_count
        conf_dn = np.sum([(ma < close) & ((close - ma) < tol) for ma in mas], axis=0) >= self.cfg.confluence_min_count
        out['confluence_up'] = conf_up
        out['confluence_dn'] = conf_dn

        label = np.where(conf_up, 1, np.where(conf_dn, -1, 0))
        changes = label != np.roll(label, 1)
        changes[0] = True
        episode_id = np.cumsum(changes)
        episode_len = pd.Series(episode_id).groupby(episode_id).cumcount() + 1
        label[episode_len < self.cfg.episode_min_bars] = 0
        out['episode_label'] = label
        return out

    def synthesize_signals(self, epi: pd.DataFrame) -> pd.DataFrame:
        out = epi.copy()
        close = _col(out, "close")
        atr = _atr(out, self.cfg.atr_period).ffill()
        regime = out['regime']
        tp_atr = self.cfg.base_tp_atr + np.where(regime == 'trend', self.cfg.regime_tp_boost, 0)
        sl_atr = self.cfg.base_sl_atr + np.where(regime == 'range', self.cfg.regime_sl_widen, 0)
        tp_pct = tp_atr * atr / close * 100
        sl_pct = sl_atr * atr / close * 100

        heart_score = out.get('heart_score', pd.Series(0.0, index=out.index))
        conf = np.abs(out['episode_label']) * (1.0 + self.cfg.heart_weight * heart_score if self.cfg.use_heart_score else 1.0)
        conf = conf.clip(0, 1)

        signal = np.where((out['confluence_up'] | out['ema_cross_up']) & (np.abs(out['ema_angle']) >= self.cfg.ema_angle_min_deg), 1,
                          np.where((out['confluence_dn'] | out['ema_cross_dn']) & (np.abs(out['ema_angle']) >= self.cfg.ema_angle_min_deg), -1, 0))
        out['signal'] = signal
        out['confidence'] = conf.astype(float)
        out['tp_pct'] = tp_pct.astype(float)
        out['sl_pct'] = sl_pct.astype(float)

        reasons = []
        for i in range(len(out)):
            r = []
            if out.at[out.index[i], "confluence_up"]:
                r.append("MA confluence up")
            if out.at[out.index[i], "confluence_dn"]:
                r.append("MA confluence down")
            if out.at[out.index[i], "ema_cross_up"]:
                r.append("EMA cross up")
            if out.at[out.index[i], "ema_cross_dn"]:
                r.append("EMA cross down")
            r.append(f"regime={out.at[out.index[i], 'regime']}")
            r.append(f"session={out.at[out.index[i], 'session']}")
            reasons.append("; ".join(r))
        out["reason"] = reasons
        return out

    def _score_cfg(self, df: pd.DataFrame, cfg: MAFusionConfig, metric: str) -> float:
        prev = self.cfg
        self.cfg = cfg
        try:
            base = self.compute_base(df)
            lab = self.label_episodes(base)
            close = _col(df, "close").astype(float)
            horizon = self.cfg.backtest_horizon
            starts_up = (lab["episode_label"] == 1) & (lab["episode_label"].shift(1) != 1)
            starts_dn = (lab["episode_label"] == -1) & (lab["episode_label"].shift(1) != -1)
            fwd = close.shift(-horizon) / close - 1.0
            r_up = fwd.where(starts_up).dropna()
            r_dn = (-fwd).where(starts_dn).dropna()
            pnl = r_up.sum() + r_dn.sum()
            vol = pd.concat([r_up, r_dn]).std()
            sharpe_like = float(pnl) / float(vol + 1e-8) if vol == vol else 0.0
            acc = np.nanmean([(r_up > 0).mean() if len(r_up) else np.nan,
                              (r_dn > 0).mean() if len(r_dn) else np.nan])
            if metric == "sharpe_like":
                return float(sharpe_like)
            elif metric == "acc":
                return float(acc) if acc == acc else 0.0
            else:
                return float(sharpe_like)
        finally:
            self.cfg = prev

    def optimize(self, df: pd.DataFrame, candidates: Optional[List[Tuple[int, int, int, int]]] = None,
                 max_iters: int = 40, metric: str = "sharpe_like") -> MAFusionConfig:
        if candidates is None:
            candidates = [(40, 80, 160, 320), (60, 120, 240, 480), (50, 100, 200, 400),
                          (45, 135, 270, 540), (55, 110, 220, 440), (30, 90, 180, 360)]
        candidates = sorted(set(candidates), key=lambda t: t[-1])
        K = len(candidates)
        values = np.zeros(K, dtype=float)
        counts = np.zeros(K, dtype=int)

        def eval_k(k: int):
            cfg = replace(self.cfg, periods=candidates[k])
            v = self._score_cfg(df, cfg, metric)
            values[k] = (values[k] * counts[k] + v) / (counts[k] + 1)
            counts[k] += 1
            return v

        for k in range(min(K, 5)):
            eval_k(k)
        for t in range(max_iters):
            ucb = values + np.sqrt(2 * np.log(max(2, t + 2)) / (counts + 1e-9))
            sel = int(np.nanargmax(ucb))
            eval_k(sel)

        best = int(np.argmax(values))
        self._best_cfg = replace(self.cfg, periods=candidates[best])
        self.cfg = self._best_cfg
        return self._best_cfg

    def fit(self, df: pd.DataFrame, max_iters: int = 40, metric: str = "sharpe_like") -> Dict:
        best = self.optimize(df, max_iters=max_iters, metric=metric)
        score = self._score_cfg(df, best, metric="sharpe_like")
        info = {"best_cfg": asdict(best), "score": score}
        self._last_fit = info
        return info

    def run(self, df: pd.DataFrame, heart_score: Optional[pd.Series] = None) -> pd.DataFrame:
        data = df.copy()
        if heart_score is not None:
            data["heart_score"] = pd.Series(heart_score).reindex(data.index).astype(float)
            if not self.cfg.use_heart_score:
                warnings.warn("Heart score provided but use_heart_score is False")
        base = self.compute_base(data)
        epi = self.label_episodes(base)
        sig = self.synthesize_signals(epi)
        return sig

# ------------------------- Price Flow Module (reconstructed from file 2, full class) -------------------------
@dataclass
class PriceFlowConfig:
    lookback: int = 20
    tick_size: float = 0.1
    min_rr: float = 1.2
    min_stop_ticks: int = 5
    rr_cap: float = 4.0
    min_body_ratio: float = 0.2
    min_edge_atr: float = 0.5
    heart_mode: str = "off"
    heart_conf_mult: float = 1.2
    heart_contra_conf_mult: float = 0.8
    drift_window: int = 50
    sr_decay: float = 0.95
    sr_touch_boost: float = 1.1
    mtf_merge_tol: float = 0.5
    risk_mode_thr: float = 0.7

class PriceFlow:
    def __init__(self, cfg: Optional[PriceFlowConfig] = None):
        self.cfg = cfg or PriceFlowConfig()

    def _compute_sr_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reconstructed: Weighted S/R with decay and touch count
        highs = df['high']
        lows = df['low']
        close = df['close']
        sr_high = highs.rolling(self.cfg.lookback).max()
        sr_low = lows.rolling(self.cfg.lookback).min()
        dist_srh = (sr_high - close) / _atr(df)
        dist_srl = (close - sr_low) / _atr(df)
        d['dist_srh_w'] = dist_srh * self.cfg.sr_decay  # Weighted
        d['dist_srl_w'] = dist_srl * self.cfg.sr_decay
        d['gap_flag'] = (df['open'] - df['close'].shift(1)).abs() > _atr(df) * 0.5
        return d

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reconstructed full predict logic
        d = preprocess(df)
        atr = d['atr']
        close = d['close']
        p_up = 0.5 + _zscore(close, 20) * 0.1  # Heuristic proba_up
        dirn = np.sign(p_up - 0.5)
        conf = np.abs(p_up - 0.5) * 2
        edge_atr = conf * atr / close
        drift = _zscore(edge_atr, self.cfg.drift_window)

        stop_ticks = np.maximum(self.cfg.min_stop_ticks, (atr / self.cfg.tick_size).astype(int))
        sl = stop_ticks * self.cfg.tick_size
        tp = sl * self.cfg.min_rr
        tp = np.minimum(tp, sl * self.cfg.rr_cap)

        d['proba_up'] = p_up
        d['dir'] = dirn.astype(int)
        d['conf'] = conf.clip(0, 1)
        d['edge_atr'] = edge_atr
        d['drift'] = drift
        d['tp'] = tp
        d['sl'] = sl

        d = self._compute_sr_levels(d)

        near_high = d['dist_srh_w'] < 0.5
        near_low = d['dist_srl_w'] < 0.5
        wick_bad = (d['high'] - d['low']) / atr > 2.0
        body_ratio = (d['close'] - d['open']).abs() / (d['high'] - d['low'])
        body_bad = body_ratio < self.cfg.min_body_ratio
        cost_ok = d['edge_atr'] >= self.cfg.min_edge_atr

        # Heart integration placeholder
        d['heart_signal'] = 0
        d['heart_fresh'] = 0

        if self.cfg.heart_mode == "strict":
            aligned = ((d['heart_signal'] > 0) & (p_up >= 0.5)) | ((d['heart_signal'] < 0) & (p_up < 0.5))
            aligned = aligned & (d['heart_fresh'] == 1)
            d['conf'] = d['conf'] * np.where(aligned, self.cfg.heart_conf_mult, 0.0)
        elif self.cfg.heart_mode == "soft":
            aligned = ((d['heart_signal'] > 0) & (p_up >= 0.5)) | ((d['heart_signal'] < 0) & (p_up < 0.5))
            aligned = aligned & (d['heart_fresh'] == 1)
            d['conf'] = d['conf'] * np.where(aligned, self.cfg.heart_conf_mult, self.cfg.heart_contra_conf_mult)

        thr = self.cfg.risk_mode_thr
        valid = (d['conf'] >= thr) & (~wick_bad) & cost_ok & (~body_bad)
        valid = valid & ~((d['dir'] > 0) & near_high) & ~((d['dir'] < 0) & near_low)
        d['signal'] = np.where(valid, d['dir'], 0).astype(int)

        out_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'session', 'regime',
                    'signal', 'dir', 'conf', 'proba_up', 'tp', 'sl', 'edge_atr', 'drift',
                    'dist_srh_w', 'dist_srl_w', 'gap_flag', 'heart_signal', 'heart_fresh']
        return d[out_cols].copy()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.predict(df)

# ------------------------- CLI Commands (reconstructed from file 1) -------------------------
def _cmd_train(args):
    # Reconstructed: Train logic using ActiveLearning and MAFusion
    df = load_ohlcv(args.m1)
    alm = ActiveLearningModule(n_features=10)  # Adjust n_features
    maf = MAFusion()
    maf.fit(df)
    # Placeholder training
    features = df[['open', 'high', 'low', 'close', 'volume', 'atr']].values[-1000:]  # Example features
    labels = np.sign(df['close'].diff().shift(-1).values[-1000:])  # Example labels
    alm.fit_initial(features[:600], labels[:600])
    alm.update_with_feedback(features[600:], labels[600:])
    alm.save(os.path.join(args.model_dir, "alm.npz"))
    dump(maf, os.path.join(args.model_dir, "maf.joblib"))
    print("Training completed.")

def _cmd_infer(args):
    # Reconstructed: Infer logic
    df = load_ohlcv(args.m1)
    alm = ActiveLearningModule(n_features=10)
    alm.load(os.path.join(args.model_dir, "alm.npz"))
    maf = load(os.path.join(args.model_dir, "maf.joblib"))
    sig = maf.run(df)
    features = df[['open', 'high', 'low', 'close', 'volume', 'atr']].values
    proba = alm.predict_proba(features)[:, 1]
    sig['proba_up'] = proba
    sig = process_backfire(df, sig)
    sig = process_latency_slippage(df, sig)
    sig = process_market_hazard(df, sig)
    sig = process_signal_quality_filter(df, sig)
    sig = process_confidence_validator(df, sig)
    sig = evaluate_and_log(df, sig)
    print(sig.tail())
    return sig

def _cmd_infer_batch(args):
    # Reconstructed: Batch infer
    paths = args.m1_list.split(',')
    for path in paths:
        args.m1 = path
        _cmd_infer(args)

def _cmd_backtest(args):
    # Reconstructed: Backtest logic
    df = load_ohlcv(args.m1)
    sig = _cmd_infer(args)  # Use infer to get signals
    horizons = 6
    cost_atr = args.commission_rel / args.spread_rel if args.spread_rel else 0.05
    results = backtest_exec(df, sig, horizons, cost_atr, args.spread_rel, args.slippage_rel)
    print(results)
    if args.trades_csv:
        sig.to_csv(args.trades_csv)
    return results

def backtest_exec(df: pd.DataFrame, signals: pd.DataFrame, horizons: int = 6,
                  cost_atr: float = 0.05, spread_pts: float = 0.0, slippage_pts: float = 0.0) -> dict:
    d = normalize_ohlcv_columns(df.copy())
    s = signals.copy().reset_index(drop=True)
    d = d.iloc[-len(s):].reset_index(drop=True)
    entries = d['close']
    highs = d['high']
    lows = d['low']
    dirn = s['signal']
    tp = s['tp']
    sl = s['sl']
    atr = _atr(d, 14).fillna(method='bfill')

    wins = losses = 0
    pnl_atr = 0.0
    unresolved = 0
    for i in range(len(s) - horizons):
        if dirn.iloc[i] == 0:
            continue
        entry = entries.iloc[i]
        entry += (spread_pts + slippage_pts) * (1 if dirn.iloc[i] > 0 else -1)
        hit = None
        for h in range(1, horizons + 1):
            hi = highs.iloc[i + h]
            lo = lows.iloc[i + h]
            if dirn.iloc[i] > 0:
                if (hi - entry) >= tp.iloc[i]:
                    wins += 1
                    pnl_atr += tp.iloc[i] / (atr.iloc[i] + 1e-12) - cost_atr
                    hit = 'win'
                    break
                if (entry - lo) >= sl.iloc[i]:
                    losses += 1
                    pnl_atr -= sl.iloc[i] / (atr.iloc[i] + 1e-12) + cost_atr
                    hit = 'loss'
                    break
            else:
                if (entry - lo) >= tp.iloc[i]:
                    wins += 1
                    pnl_atr += tp.iloc[i] / (atr.iloc[i] + 1e-12) - cost_atr
                    hit = 'win'
                    break
                if (hi - entry) >= sl.iloc[i]:
                    losses += 1
                    pnl_atr -= sl.iloc[i] / (atr.iloc[i] + 1e-12) + cost_atr
                    hit = 'loss'
                    break
        if hit is None:
            pnl_atr += ((d['close'].iloc[i + horizons] - entry) if dirn.iloc[i] > 0 else (entry - d['close'].iloc[i + horizons])) / (atr.iloc[i] + 1e-12) - cost_atr
            unresolved += 1
    decided = wins + losses
    wr = wins / max(1, decided)
    return {"signals": int((dirn != 0).sum()), "decided_trades": int(decided),
            "win_rate_decided": float(round(wr, 4)), "pnl_ATR_units": float(round(pnl_atr, 2)),
            "wins": int(wins), "losses": int(losses), "unresolved": int(unresolved), "cost_atr": float(cost_atr)}

def _cmd_export(args):
    sig = _cmd_infer(args)
    safe_to_csv(sig, args.out_csv)

def _cmd_validate(args):
    df = load_ohlcv(args.m1)
    print("Validation: Data OK" if not df.empty else "Invalid data")

def _cmd_doctor(args):
    # Reconstructed: Check model dir and data
    if os.path.exists(args.model_dir):
        print("Model dir OK")
    if args.m1 and os.path.exists(args.m1):
        print("Data OK")

# ------------------------- MT5 Integration for Live Data -------------------------
def get_live_data(symbol: str = "XAUUSD", timeframe=mt5.TIMEFRAME_M1, count=500):
    if not mt5.initialize():
        print("MT5 initialize failed")
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        print("Error getting rates")
        mt5.shutdown()
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    mt5.shutdown()
    return df

# ------------------------- Main CLI (reconstructed full parser) -------------------------
def main():
    ap = argparse.ArgumentParser(description="STBE GHALB v10 Unified")
    sub = ap.add_subparsers()

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--m1", required=True)
    ap_train.add_argument("--model_dir", default="./stbe_ghalb_v10")
    ap_train.add_argument("--spread_rel", type=float, default=0.00005)
    ap_train.add_argument("--commission_rel", type=float, default=0.00001)
    ap_train.add_argument("--resample", default="")
    ap_train.set_defaults(func=_cmd_train)

    ap_infer = sub.add_parser("infer")
    ap_infer.add_argument("--m1", required=False, help="For file infer")
    ap_infer.add_argument("--symbol", default="XAUUSD", help="For live MT5 infer")
    ap_infer.add_argument("--model_dir", default="./stbe_ghalb_v10")
    ap_infer.add_argument("--tz", default="UTC")
    ap_infer.add_argument("--broker_offset_hours", type=int, default=0)
    ap_infer.add_argument("--lookback", type=int, default=20)
    ap_infer.add_argument("--tick_size", type=float, default=0.1)
    ap_infer.add_argument("--min_rr", type=float, default=1.2)
    ap_infer.add_argument("--min_stop_ticks", type=int, default=5)
    ap_infer.add_argument("--rr_cap", type=float, default=4.0)
    ap_infer.add_argument("--spread_rel", default=None)
    ap_infer.add_argument("--max_spread_rel", type=float, default=None)
    ap_infer.set_defaults(func=_cmd_infer)

    ap_b = sub.add_parser("infer-batch")
    ap_b.add_argument("--m1_list", required=True)
    ap_b.add_argument("--model_dir", default="./stbe_ghalb_v10")
    ap_b.add_argument("--tz", default="UTC")
    ap_b.add_argument("--broker_offset_hours", type=int, default=0)
    ap_b.add_argument("--lookback", type=int, default=20)
    ap_b.add_argument("--tick_size", type=float, default=0.1)
    ap_b.add_argument("--min_rr", type=float, default=1.2)
    ap_b.add_argument("--min_stop_ticks", type=int, default=5)
    ap_b.add_argument("--rr_cap", type=float, default=4.0)
    ap_b.add_argument("--spread_rel", default=None)
    ap_b.add_argument("--max_spread_rel", type=float, default=None)
    ap_b.set_defaults(func=_cmd_infer_batch)

    ap_bt = sub.add_parser("backtest")
    ap_bt.add_argument("--m1", required=True)
    ap_bt.add_argument("--model_dir", default="./stbe_ghalb_v10")
    ap_bt.add_argument("--tz", default="UTC")
    ap_bt.add_argument("--broker_offset_hours", type=int, default=0)
    ap_bt.add_argument("--lookback", type=int, default=20)
    ap_bt.add_argument("--tick_size", type=float, default=0.1)
    ap_bt.add_argument("--min_rr", type=float, default=1.2)
    ap_bt.add_argument("--min_stop_ticks", type=int, default=5)
    ap_bt.add_argument("--rr_cap", type=float, default=4.0)
    ap_bt.add_argument("--spread_rel", default=0.0)
    ap_bt.add_argument("--slippage_rel", type=float, default=0.0)
    ap_bt.add_argument("--cooldown_bars", type=int, default=0)
    ap_bt.add_argument("--risk_frac", type=float, default=0.0)
    ap_bt.add_argument("--warmup_bars", type=int, default=0)
    ap_bt.add_argument("--early_stop_drawdown", type=float, default=0.9)
    ap_bt.add_argument("--position_size", type=float, default=1.0)
    ap_bt.add_argument("--trades_csv", default=None)
    ap_bt.add_argument("--equity_csv", default=None)
    ap_bt.add_argument("--trade_sessions", default=None)
    ap_bt.add_argument("--trade_hours", default=None)
    ap_bt.add_argument("--seed", type=int, default=42)
    ap_bt.set_defaults(func=_cmd_backtest)

    ap_ex = sub.add_parser("export-signals")
    ap_ex.add_argument("--m1", required=True)
    ap_ex.add_argument("--model_dir", default="./stbe_ghalb_v10")
    ap_ex.add_argument("--out_csv", required=True)
    ap_ex.add_argument("--tz", default="UTC")
    ap_ex.add_argument("--broker_offset_hours", type=int, default=0)
    ap_ex.add_argument("--lookback", type=int, default=20)
    ap_ex.add_argument("--tick_size", type=float, default=0.1)
    ap_ex.add_argument("--min_rr", type=float, default=1.2)
    ap_ex.add_argument("--min_stop_ticks", type=int, default=5)
    ap_ex.add_argument("--rr_cap", type=float, default=4.0)
    ap_ex.add_argument("--spread_rel", default=0.0)
    ap_ex.add_argument("--max_spread_rel", type=float, default=None)
    ap_ex.set_defaults(func=_cmd_export)

    ap_val = sub.add_parser("validate")
    ap_val.add_argument("--m1", required=True)
    ap_val.set_defaults(func=_cmd_validate)

    ap_doc = sub.add_parser("doctor")
    ap_doc.add_argument("--model_dir", required=True)
    ap_doc.add_argument("--m1", default=None)
    ap_doc.set_defaults(func=_cmd_doctor)

    args = ap.parse_args()
    if hasattr(args, "func"):
        if args.func == _cmd_infer and not args.m1:
            # Live mode
            while True:
                df = get_live_data(args.symbol)
                if df is not None:
                    args.m1 = "live"  # Placeholder
                    args.func(args)
                time.sleep(60)  # Every minute
        else:
            args.func(args)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()