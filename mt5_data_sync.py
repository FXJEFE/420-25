# -*- coding: utf-8 -*-
"""
mt5_data_sync.py
Pipeline entry for MT5 feature data.

Priority:
  1) Live pull via MetaTrader5 API → write FXJEFE_Features.csv
  2) Copy newest valid CSV from MT5 MQL5\\Files / Common\\Files / all terminals
  3) Build features from HISTORIC--DATA OHLCV exports
  4) Copy best existing non-corrupt feature CSV from known project folders

Default: single-shot (pipeline). Use --daemon for 60s loop.
"""
from __future__ import annotations
import os as _os_utf8, sys as _sys_utf8
_os_utf8.environ.setdefault('PYTHONUTF8','1')
_os_utf8.environ.setdefault('PYTHONIOENCODING','utf-8')
for _s in (getattr(_sys_utf8,'stdout',None), getattr(_sys_utf8,'stderr',None)):
    try:
        if _s is not None and hasattr(_s,'reconfigure'):
            _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import glob
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

LOG_DIR = config.get("log_path") or os.path.join(os.path.dirname(CONFIG_PATH), "logs")
DATA_DIR = config.get("data_path") or os.path.join(
    os.path.dirname(CONFIG_PATH), "FXJEFE_Project", "data"
)
PROJECT_ROOT = config.get("project_root") or os.path.join(
    os.path.dirname(CONFIG_PATH), "FXJEFE_Project"
)
HISTORIC_DIR = (
    config.get("historical_data_path")
    or config.get("historic_data_path")
    or os.path.join(PROJECT_ROOT, "HISTORIC--DATA")
)
MT5_MQL5 = config.get("mt5_mql5_path") or config.get("mt5_path") or ""
MT5_FILES = config.get("mt5_files_path") or (
    os.path.join(MT5_MQL5, "Files") if MT5_MQL5 else ""
)
MT5_COMMON = config.get("mt5_common_path") or ""

FILENAME = "FXJEFE_Features.csv"
DEST_PATH = os.path.join(DATA_DIR, FILENAME)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
if MT5_FILES:
    os.makedirs(MT5_FILES, exist_ok=True)
if MT5_COMMON:
    os.makedirs(MT5_COMMON, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "mt5_data_sync.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("mt5_data_sync")

FOREX = list(config.get("forex_symbols") or ["EURUSD", "USDJPY", "AUDUSD", "GBPUSD", "USDCAD"])
CRYPTO = list(config.get("crypto_symbols") or ["BTCUSD", "ETHUSD", "XRPUSD"])
EXTRA = list(config.get("extra_symbols") or ["XAUUSD"])
SYMBOLS = list(dict.fromkeys(FOREX + CRYPTO + EXTRA + ["XAUUSD"]))

HISTORY_BARS = int(config.get("history_bars") or 2000)
# M15 is the live prediction contract; historic + live both feed this TF
TF_NAME = str(config.get("preferred_timeframe") or config.get("mt5_timeframe") or "M15").upper()

FEATURE_COLS = [
    "time",
    "symbol",
    "price",
    "atr",
    "ema_diff",
    "rsi",
    "garch_vol",
    "macd_diff",
    "vwap",
    "price_vwap_diff",
    "bb_position",
    "roc",
    "stochastic",
    "cci",
    "williams",
    "momentum",
    "realized_vol",
    "chaikin_vol",
    "adx",
    "rvi",
    "obv",
    "volume_delta",
    "ad_line",
    "vol_osc",
    "supertrend",
    "hma",
    "ichimoku_tenkan",
    "sar",
    "dpo",
    "spread",
    "sentiment",
    "signal",
    "sma20",
    "sma50",
    "sma200",
    "close_sma_5",
    "close_sma_10",
    "price_lag1",
    "rsi_lag1",
    "macd_diff_lag1",
    "atr_lag1",
    "price_lag2",
    "rsi_lag2",
    "macd_diff_lag2",
    "atr_lag2",
    "price_lag3",
    "rsi_lag3",
    "macd_diff_lag3",
    "atr_lag3",
    "hour_of_day",
    "day_of_week",
    "volume_ratio",
    "candle_body_ratio",
    "candle_upper_shadow",
    "candle_lower_shadow",
    "zscore_20",
    "zscore_50",
    "price_above_sma50",
    "price_above_sma200",
    "sma_spread",
    "trend_filter",
]


# ── file helpers ──────────────────────────────────────────────────────────────
def _file_ok(path: str, min_bytes: int = 200) -> bool:
    try:
        if not path or not os.path.isfile(path) or os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as f:
            head = f.read(64)
        if not head or all(b == 0 for b in head):
            return False
        # must look like CSV text
        sample = head.lstrip()
        if sample[:1] in (b"\x00",):
            return False
        text = head.decode("utf-8", errors="ignore")
        if "," not in text and "\t" not in text and ";" not in text:
            # maybe header-only short; still reject pure binary
            if not any(c.isalpha() for c in text):
                return False
        return True
    except OSError:
        return False


def _copy(src: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    shutil.copy2(src, dest)
    log.info("Copied %s → %s (%.1f KB)", src, dest, os.path.getsize(dest) / 1024.0)
    return True


def _publish(src_csv: str, force: bool = True) -> None:
    """
    OG behavior: ALWAYS write FXJEFE_Features.csv to every destination.
    Never skip because dest exists or mtime is newer.
    """
    if not os.path.isfile(src_csv):
        log.error("publish source missing: %s", src_csv)
        return

    # Ensure primary data path has a real copy first
    if os.path.abspath(src_csv) != os.path.abspath(DEST_PATH):
        _copy(src_csv, DEST_PATH)

    # OG multi-target list (same as fxjefe_paths.feature_write_targets)
    targets = [
        DEST_PATH,
        os.path.join(DATA_DIR, FILENAME),
        os.path.join(PROJECT_ROOT, FILENAME),
        os.path.join(os.path.dirname(CONFIG_PATH), FILENAME),  # Documents root
        os.path.join(MT5_FILES, FILENAME) if MT5_FILES else "",
        os.path.join(MT5_COMMON, FILENAME) if MT5_COMMON else "",
    ]
    # de-dupe
    seen = set()
    for dest in targets:
        if not dest:
            continue
        ap = os.path.abspath(dest)
        if ap in seen:
            continue
        seen.add(ap)
        try:
            src = DEST_PATH if os.path.isfile(DEST_PATH) else src_csv
            if os.path.abspath(src) == ap:
                log.info(
                    "FEATURE CSV WRITE → %s (already primary, %.1f KB)",
                    dest,
                    os.path.getsize(dest) / 1024.0 if os.path.isfile(dest) else 0,
                )
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # force overwrite every time (OG does not skip)
            shutil.copy2(src, dest)
            log.info(
                "FEATURE CSV WRITE → %s (%.1f KB)",
                dest,
                os.path.getsize(dest) / 1024.0,
            )
        except PermissionError:
            log.warning("Could not write to %s (locked by MT5?)", dest)
        except Exception as e:
            log.warning("Publish to %s failed: %s", dest, e)


# ── feature math (shared live + historic) ─────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def calc_bb_position(close: pd.Series, period: int = 20) -> pd.Series:
    sma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std().replace(0, np.nan)
    return ((close - sma) / (2 * std) + 0.5).fillna(0.5)


def calc_roc(close: pd.Series, period: int = 10) -> pd.Series:
    prev = close.shift(period)
    return ((close - prev) / prev.replace(0, np.nan)).fillna(0.0)


def calc_stochastic(high, low, close, period: int = 14) -> pd.Series:
    lowest = low.rolling(period, min_periods=1).min()
    highest = high.rolling(period, min_periods=1).max()
    denom = (highest - lowest).replace(0, np.nan)
    return (100 * (close - lowest) / denom).fillna(50.0)


def calc_cci(high, low, close, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period, min_periods=1).mean()
    mad = tp.rolling(period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return ((tp - sma) / (0.015 * mad.replace(0, np.nan))).fillna(0.0)


def calc_williams(high, low, close, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    denom = (highest - lowest).replace(0, np.nan)
    return (-100 * (highest - close) / denom).fillna(-50.0)


def calc_adx(high, low, close, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = calc_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period, min_periods=1).mean().fillna(25.0)


def calc_rvi(close: pd.Series, period: int = 10) -> pd.Series:
    returns = close.pct_change()
    pos = returns.where(returns > 0, 0.0).rolling(period, min_periods=1).std()
    neg = returns.where(returns < 0, 0.0).rolling(period, min_periods=1).std()
    return (pos / (pos + neg).replace(0, np.nan)).fillna(0.0)


def calc_obv(close, volume) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def calc_ad_line(high, low, close, volume) -> pd.Series:
    mfm_denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / mfm_denom
    return (mfm.fillna(0.0) * volume).cumsum()


def calc_vol_osc(volume, short_p=5, long_p=20) -> pd.Series:
    short_ema = volume.ewm(span=short_p, adjust=False).mean()
    long_ema = volume.ewm(span=long_p, adjust=False).mean()
    return ((short_ema - long_ema) / long_ema.replace(0, np.nan) * 100).fillna(0.0)


def calc_supertrend(high, low, close, period=10, mult=3.0) -> pd.Series:
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st = pd.Series(np.nan, index=close.index, dtype=float)
    direction = 1
    for i in range(len(close)):
        if i == 0:
            st.iloc[i] = lower.iloc[i]
            continue
        if close.iloc[i] > upper.iloc[i - 1]:
            direction = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction = -1
        st.iloc[i] = lower.iloc[i] if direction == 1 else upper.iloc[i]
    return st.ffill().fillna(close)


def calc_hma(close, period=9) -> pd.Series:
    half = close.rolling(max(period // 2, 1), min_periods=1).mean()
    full = close.rolling(period, min_periods=1).mean()
    diff = 2 * half - full
    return diff.rolling(max(int(period ** 0.5), 1), min_periods=1).mean()


def calc_ichimoku_tenkan(high, low, period=9) -> pd.Series:
    return (high.rolling(period, min_periods=1).max() + low.rolling(period, min_periods=1).min()) / 2


def calc_sar(high, low, close, af_start=0.02, af_max=0.2) -> pd.Series:
    n = len(close)
    sar = pd.Series(0.0, index=close.index)
    if n == 0:
        return sar
    bull = True
    af = af_start
    ep = float(high.iloc[0])
    sar.iloc[0] = float(low.iloc[0])
    for i in range(1, n):
        sar.iloc[i] = float(sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1]))
        if bull:
            if float(low.iloc[i]) < sar.iloc[i]:
                bull = False
                sar.iloc[i] = ep
                ep = float(low.iloc[i])
                af = af_start
            elif float(high.iloc[i]) > ep:
                ep = float(high.iloc[i])
                af = min(af + af_start, af_max)
        else:
            if float(high.iloc[i]) > sar.iloc[i]:
                bull = True
                sar.iloc[i] = ep
                ep = float(high.iloc[i])
                af = af_start
            elif float(low.iloc[i]) < ep:
                ep = float(low.iloc[i])
                af = min(af + af_start, af_max)
    return sar


def calc_dpo(close, period=20) -> pd.Series:
    shifted_sma = close.rolling(period, min_periods=1).mean().shift(period // 2 + 1)
    return (close - shifted_sma).fillna(0.0)


def calc_realized_vol(close, period=14) -> pd.Series:
    return close.pct_change().rolling(period, min_periods=1).std().fillna(0.0)


def calc_chaikin_vol(high, low, period=10) -> pd.Series:
    hl_range = high - low
    ema_range = hl_range.ewm(span=period, adjust=False).mean()
    prev = ema_range.shift(period)
    return ((ema_range - prev) / prev.replace(0, np.nan) * 100).fillna(0.0)


def calc_garch_vol(close, period=20) -> pd.Series:
    """
    GARCH(1,1) variance recursion matching EA (omega=7e-6, alpha=0.08, beta=0.88).
    Returns sigma_t (not variance). Falls back to EWMA if series too short.
    """
    rets = close.pct_change().fillna(0.0).to_numpy(dtype=float)
    n = len(rets)
    if n < 5:
        return pd.Series(np.zeros(n), index=close.index)
    omega, alpha, beta = 7e-6, 0.08, 0.88
    var = np.empty(n, dtype=float)
    var[0] = max(float(np.var(rets[: min(period, n)])), 1e-8)
    for t in range(1, n):
        var[t] = omega + alpha * rets[t - 1] ** 2 + beta * var[t - 1]
        if var[t] < 1e-12:
            var[t] = 1e-12
    return pd.Series(np.sqrt(var), index=close.index)


SENTIMENT_MAP = {
    "EURUSD": "Bullish trend expected",
    "USDJPY": "Neutral market",
    "XAUUSD": "Bearish sentiment",
    "AUDUSD": "Positive outlook",
    "GBPUSD": "Strong buy signals",
    "USDCAD": "Sell pressure",
    "BTCUSD": "Volatile bullish momentum",
    "ETHUSD": "Crypto risk-on",
    "XRPUSD": "Speculative neutral",
}


def get_sentiment(symbol: str) -> float:
    base = symbol.replace(".r", "").replace(".R", "").upper()
    text = SENTIMENT_MAP.get(base, "Neutral")
    try:
        from textblob import TextBlob

        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def ohlcv_to_features(df: pd.DataFrame, symbol: str, spread: float = 0.0) -> pd.DataFrame:
    """Convert OHLCV dataframe to FXJEFE feature rows."""
    if df is None or df.empty:
        return pd.DataFrame()

    # normalize columns
    colmap = {c.lower().strip("<>"): c for c in df.columns}
    def col(*names):
        for n in names:
            for k, orig in colmap.items():
                if k == n or k.replace(" ", "") == n:
                    return orig
        return None

    c_open = col("open")
    c_high = col("high")
    c_low = col("low")
    c_close = col("close")
    c_vol = col("tickvol", "tick_volume", "volume", "vol")
    c_date = col("date")
    c_time = col("time")
    c_datetime = col("datetime", "time")

    if not c_close:
        # maybe already features
        if "price" in df.columns or "rsi" in df.columns:
            out = df.copy()
            if "symbol" not in out.columns:
                out["symbol"] = symbol
            return out
        return pd.DataFrame()

    work = pd.DataFrame()
    if c_date and c_time and c_date != c_time:
        work["time"] = (
            df[c_date].astype(str).str.strip() + " " + df[c_time].astype(str).str.strip()
        )
    elif c_datetime:
        work["time"] = df[c_datetime].astype(str)
    elif c_time:
        work["time"] = df[c_time].astype(str)
    else:
        work["time"] = pd.RangeIndex(len(df)).astype(str)

    work["symbol"] = symbol.replace(".r", "").replace(".R", "").upper()
    h = pd.to_numeric(df[c_high] if c_high else df[c_close], errors="coerce")
    l = pd.to_numeric(df[c_low] if c_low else df[c_close], errors="coerce")
    c = pd.to_numeric(df[c_close], errors="coerce")
    v = pd.to_numeric(df[c_vol] if c_vol else 1.0, errors="coerce").fillna(1.0).astype(float)
    h, l, c = h.ffill(), l.ffill(), c.ffill()

    atr = calc_atr(h, l, c, 14)
    ema_fast = calc_ema(c, 12)
    ema_slow = calc_ema(c, 26)
    macd = ema_fast - ema_slow
    macd_signal = calc_ema(macd, 9)

    work["price"] = c
    try:
        import talib as _ta
        work["atr"] = pd.Series(_ta.ATR(h.values, l.values, c.values, 14), index=c.index).fillna(atr)
        work["rsi"] = pd.Series(_ta.RSI(c.values, 14), index=c.index).fillna(50.0)
        macd_t, macds_t, _ = _ta.MACD(c.values, 12, 26, 9)
        work["macd_diff"] = pd.Series(macd_t - macds_t, index=c.index).fillna(macd - macd_signal)
        work["ema_diff"] = pd.Series(_ta.EMA(c.values, 12) - _ta.EMA(c.values, 26), index=c.index).fillna(ema_fast - ema_slow)
        log.info("talib wheel used for ATR/RSI/MACD/EMA on %s", symbol)
    except Exception:
        work["atr"] = atr
        work["ema_diff"] = ema_fast - ema_slow
        work["rsi"] = calc_rsi(c, 14)
        work["macd_diff"] = macd - macd_signal
    if "macd_diff" not in work.columns:
        work["macd_diff"] = macd - macd_signal
    work["garch_vol"] = calc_garch_vol(c, 20)
    work["macd_diff"] = macd - macd_signal
    vwap_cum = v.cumsum().replace(0, np.nan)
    work["vwap"] = ((c * v).cumsum() / vwap_cum).fillna(c)
    work["price_vwap_diff"] = c - work["vwap"]
    work["bb_position"] = calc_bb_position(c, 20)
    work["roc"] = calc_roc(c, 10)
    work["stochastic"] = calc_stochastic(h, l, c, 14)
    work["cci"] = calc_cci(h, l, c, 20)
    work["williams"] = calc_williams(h, l, c, 14)
    work["momentum"] = c.diff(10).fillna(0.0)
    work["realized_vol"] = calc_realized_vol(c, 14)
    work["chaikin_vol"] = calc_chaikin_vol(h, l, 10)
    work["adx"] = calc_adx(h, l, c, 14)
    work["rvi"] = calc_rvi(c, 10)
    work["obv"] = calc_obv(c, v)
    work["volume_delta"] = v.diff().fillna(0.0)
    work["ad_line"] = calc_ad_line(h, l, c, v)
    work["vol_osc"] = calc_vol_osc(v, 5, 20)
    work["supertrend"] = calc_supertrend(h, l, c, 10, 3.0)
    work["hma"] = calc_hma(c, 9)
    work["ichimoku_tenkan"] = calc_ichimoku_tenkan(h, l, 9)
    work["sar"] = calc_sar(h, l, c)
    work["dpo"] = calc_dpo(c, 20)
    work["spread"] = float(spread or 0.0)
    work["sentiment"] = get_sentiment(symbol)
    work["signal"] = ""

    # ── extra M15 math (live + historic) so specialist models get real values ──
    o = pd.to_numeric(df[c_open], errors="coerce").ffill() if c_open else c
    rng = (h - l).replace(0, np.nan)
    work["sma20"] = c.rolling(20, min_periods=1).mean()
    work["sma50"] = c.rolling(50, min_periods=1).mean()
    work["sma200"] = c.rolling(200, min_periods=1).mean()
    work["close_sma_5"] = c.rolling(5, min_periods=1).mean()
    work["close_sma_10"] = c.rolling(10, min_periods=1).mean()
    work["price_lag1"] = c.shift(1)
    work["rsi_lag1"] = work["rsi"].shift(1)
    work["macd_diff_lag1"] = work["macd_diff"].shift(1)
    work["atr_lag1"] = atr.shift(1)
    work["price_lag2"] = c.shift(2)
    work["rsi_lag2"] = work["rsi"].shift(2)
    work["macd_diff_lag2"] = work["macd_diff"].shift(2)
    work["atr_lag2"] = atr.shift(2)
    work["price_lag3"] = c.shift(3)
    work["rsi_lag3"] = work["rsi"].shift(3)
    work["macd_diff_lag3"] = work["macd_diff"].shift(3)
    work["atr_lag3"] = atr.shift(3)
    ts = pd.to_datetime(work["time"], errors="coerce")
    work["hour_of_day"] = ts.dt.hour.fillna(0).astype(float)
    work["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(float)
    vol_ma = v.rolling(20, min_periods=1).mean().replace(0, np.nan)
    work["volume_ratio"] = (v / vol_ma).fillna(1.0)
    work["candle_body_ratio"] = ((c - o).abs() / rng).fillna(0.0)
    work["candle_upper_shadow"] = ((h - pd.concat([c, o], axis=1).max(axis=1)) / rng).fillna(0.0)
    work["candle_lower_shadow"] = ((pd.concat([c, o], axis=1).min(axis=1) - l) / rng).fillna(0.0)
    std20 = c.rolling(20, min_periods=5).std().replace(0, np.nan)
    std50 = c.rolling(50, min_periods=5).std().replace(0, np.nan)
    work["zscore_20"] = ((c - work["sma20"]) / std20).fillna(0.0)
    work["zscore_50"] = ((c - work["sma50"]) / std50).fillna(0.0)
    work["price_above_sma50"] = (c > work["sma50"]).astype(float)
    work["price_above_sma200"] = (c > work["sma200"]).astype(float)
    work["sma_spread"] = (work["sma50"] - work["sma200"]).fillna(0.0)
    work["trend_filter"] = np.sign(work["sma20"] - work["sma50"]).fillna(0.0)
    work = work.fillna(0.0)

    # keep last HISTORY_BARS
    if len(work) > HISTORY_BARS:
        work = work.iloc[-HISTORY_BARS:].reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    return work[[c for c in FEATURE_COLS if c in work.columns]]


# ── source 1: live MT5 ────────────────────────────────────────────────────────
def _mt5_tf(mt5, name: str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return mapping.get(name.upper(), mt5.TIMEFRAME_H1)


def _resolve_symbol(mt5, symbol: str) -> Optional[str]:
    base = symbol.replace(".r", "").replace(".R", "").upper()
    candidates = [
        base,
        base + ".r",
        base + ".R",
        base + "m",
        base + ".a",
        base + "#",
        "XAUUSD" if base == "GOLD" else base,
    ]
    # also scan market watch for startswith
    for cand in candidates:
        info = mt5.symbol_info(cand)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(cand, True)
            return cand
    # fuzzy
    all_syms = mt5.symbols_get()
    if all_syms:
        for s in all_syms:
            name = s.name.upper()
            if name == base or name.startswith(base) or base in name:
                if not s.visible:
                    mt5.symbol_select(s.name, True)
                return s.name
    return None


def pull_live_mt5() -> Optional[str]:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.warning("MetaTrader5 package not installed — skip live pull")
        return None

    # Prefer configured terminal path
    term_path = None
    if MT5_MQL5:
        # ...\Terminal\ID\MQL5 → terminal root is two levels up from MQL5? 
        # Actually origin is Program Files; initialize() attaches to running terminal.
        pass
    terminal_exe = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    if os.path.isfile(terminal_exe):
        ok = mt5.initialize(path=terminal_exe)
    else:
        ok = mt5.initialize()
    if not ok:
        log.warning("MT5 initialize failed: %s", mt5.last_error())
        return None

    try:
        info = mt5.account_info()
        term = mt5.terminal_info()
        log.info(
            "MT5 live: login=%s server=%s terminal=%s",
            getattr(info, "login", None),
            getattr(info, "server", None),
            getattr(term, "path", None),
        )
        tf = _mt5_tf(mt5, TF_NAME)
        frames = []
        for sym in SYMBOLS:
            resolved = _resolve_symbol(mt5, sym)
            if not resolved:
                log.warning("Symbol not in MT5: %s", sym)
                continue
            rates = mt5.copy_rates_from_pos(resolved, tf, 0, HISTORY_BARS)
            if rates is None or len(rates) == 0:
                log.warning("No rates for %s (%s): %s", sym, resolved, mt5.last_error())
                continue
            rdf = pd.DataFrame(rates)
            rdf["time"] = pd.to_datetime(rdf["time"], unit="s")
            # rename tick_volume
            if "tick_volume" in rdf.columns and "volume" not in rdf.columns:
                rdf["volume"] = rdf["tick_volume"]
            si = mt5.symbol_info(resolved)
            spread = 0.0
            if si is not None:
                spread = float(si.spread) * float(si.point or 0.0)
            feat = ohlcv_to_features(rdf, sym, spread=spread)
            if feat.empty:
                continue
            # format time like MT5 exports
            feat["time"] = pd.to_datetime(feat["time"]).dt.strftime("%Y.%m.%d %H:%M")
            log.info(
                "  LIVE %s→%s: %s rows, price %.5f–%.5f",
                sym,
                resolved,
                len(feat),
                feat["price"].min(),
                feat["price"].max(),
            )
            frames.append(feat)

        if not frames:
            log.warning("Live MT5 returned no symbol frames")
            return None

        result = pd.concat(frames, ignore_index=True)
        result.to_csv(DEST_PATH, index=False, encoding="utf-8")
        log.info("Live features written: %s (%s rows)", DEST_PATH, len(result))
        _publish(DEST_PATH)
        return DEST_PATH
    finally:
        mt5.shutdown()


# ── source 2: existing MT5 CSV files ──────────────────────────────────────────
def find_mt5_csv_sources() -> List[str]:
    paths = []
    # configured
    for p in (
        os.path.join(MT5_FILES, FILENAME) if MT5_FILES else "",
        os.path.join(MT5_COMMON, FILENAME) if MT5_COMMON else "",
    ):
        if p:
            paths.append(p)
    # all terminals under Roaming
    term_root = os.path.join(
        os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal"
    )
    if os.path.isdir(term_root):
        paths.extend(glob.glob(os.path.join(term_root, "*", "MQL5", "Files", FILENAME)))
        paths.extend(glob.glob(os.path.join(term_root, "Common", "Files", FILENAME)))
    # project MQL5 copy
    paths.append(os.path.join(os.path.dirname(CONFIG_PATH), "MQL5", "Files", FILENAME))
    return [p for p in paths if p]


def pull_from_mt5_files() -> Optional[str]:
    candidates = [(p, os.path.getmtime(p)) for p in find_mt5_csv_sources() if _file_ok(p)]
    if not candidates:
        log.info("No valid FXJEFE_Features.csv in MT5 Files folders")
        return None
    best = max(candidates, key=lambda x: x[1])[0]
    # OG: always re-publish to all destinations (no mtime skip)
    _copy(best, DEST_PATH)
    _publish(DEST_PATH)
    return DEST_PATH


# ── source 3: HISTORIC--DATA OHLCV ────────────────────────────────────────────
def _find_historic_file(symbol: str, tf: str = "H1") -> Optional[str]:
    if not os.path.isdir(HISTORIC_DIR):
        return None
    patterns = [
        f"{symbol}_{tf}_*.csv",
        f"{symbol}_{tf}*.csv",
        f"{symbol}*{tf}*.csv",
    ]
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(HISTORIC_DIR, pat)))
    # prefer largest (most history)
    hits = [h for h in hits if _file_ok(h, min_bytes=500)]
    if not hits:
        return None
    return max(hits, key=os.path.getsize)


def _read_mt5_export(path: str) -> pd.DataFrame:
    # MT5 exports are often tab-separated with <DATE> headers
    for sep in ("\t", ",", ";"):
        try:
            df = pd.read_csv(path, sep=sep, engine="python", encoding='utf-8')
            if df.shape[1] >= 5:
                # strip <> from headers
                df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]
                return df
        except Exception:
            continue
    return pd.DataFrame()


def pull_from_historic() -> Optional[str]:
    if not os.path.isdir(HISTORIC_DIR):
        log.warning("Historic dir missing: %s", HISTORIC_DIR)
        return None
    log.info("Building features from historic OHLCV: %s", HISTORIC_DIR)
    frames = []
    tf = TF_NAME if TF_NAME in ("M1", "M5", "M15", "M30", "H1", "H4", "D1") else "H1"
    for sym in SYMBOLS:
        path = _find_historic_file(sym, tf)
        if not path:
            # try H1 fallback
            path = _find_historic_file(sym, "H1")
        if not path:
            log.warning("No historic file for %s", sym)
            continue
        raw = _read_mt5_export(path)
        if raw.empty:
            log.warning("Could not parse historic %s", path)
            continue
        feat = ohlcv_to_features(raw, sym, spread=0.0)
        if feat.empty:
            continue
        # normalize time string
        try:
            feat["time"] = pd.to_datetime(feat["time"], errors="coerce").dt.strftime(
                "%Y.%m.%d %H:%M"
            )
        except Exception:
            pass
        log.info("  HIST %s: %s rows from %s", sym, len(feat), os.path.basename(path))
        frames.append(feat)

    if not frames:
        return None
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(DEST_PATH, index=False, encoding="utf-8")
    log.info("Historic features written: %s (%s rows)", DEST_PATH, len(result))
    _publish(DEST_PATH)
    return DEST_PATH


# ── source 4: salvage existing project CSVs ───────────────────────────────────
def pull_from_project_cache() -> Optional[str]:
    roots = [
        os.path.dirname(CONFIG_PATH),
        os.path.join(os.path.dirname(CONFIG_PATH), "Data_small"),
        PROJECT_ROOT,
        os.path.join(os.path.dirname(CONFIG_PATH), "FXJEFE_ProjectWin11pro"),
    ]
    candidates = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat in ("*FXJEFE*Features*.csv", "*fxjefe_features*.csv"):
            for p in glob.glob(os.path.join(root, pat)):
                if _file_ok(p, min_bytes=1000):
                    candidates.append(p)
    if not candidates:
        return None
    # prefer larger recent
    best = max(candidates, key=lambda p: (os.path.getsize(p), os.path.getmtime(p)))
    log.info("Using cached feature CSV: %s", best)
    # normalize if header has 'hold' first col
    try:
        df = pd.read_csv(best, encoding="utf-8", low_memory=False)
        if df.columns[0].lower() in ("hold", "signal") and "time" in [c.lower() for c in df.columns]:
            # Data_small format: hold,time,symbol,...
            cols = {c.lower(): c for c in df.columns}
            if "signal" not in cols and df.columns[0].lower() == "hold":
                df = df.rename(columns={df.columns[0]: "signal"})
        # ensure symbol stripped
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.replace(r"\.r$", "", regex=True)
        df.to_csv(DEST_PATH, index=False, encoding="utf-8")
        _publish(DEST_PATH)
        return DEST_PATH
    except Exception as e:
        log.warning("Cache read failed %s: %s — raw copy", best, e)
        _copy(best, DEST_PATH)
        _publish(DEST_PATH)
        return DEST_PATH


# ── orchestrator ──────────────────────────────────────────────────────────────
def _read_dest_df(path: str) -> pd.DataFrame:
    if not path or not _file_ok(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.DataFrame()


def sync_once() -> bool:
    log.info(
        "mt5_data_sync start | data=%s | historic=%s | mt5_files=%s | tf=%s bars=%s",
        DATA_DIR,
        HISTORIC_DIR,
        MT5_FILES,
        TF_NAME,
        HISTORY_BARS,
    )

    frames: List[pd.DataFrame] = []
    sources: List[str] = []

    # Historic M15 first — depth for model math
    try:
        path = pull_from_historic()
        df = _read_dest_df(path or DEST_PATH if path else "")
        if not df.empty:
            df["_src"] = "historic"
            frames.append(df)
            sources.append(f"historic:{len(df)}")
    except Exception as e:
        log.exception("Historic rebuild failed: %s", e)

    # Live MT5 overlay (latest bars)
    try:
        path = pull_live_mt5()
        df = _read_dest_df(path or "")
        if not df.empty:
            df["_src"] = "live"
            frames.append(df)
            sources.append(f"live:{len(df)}")
    except Exception as e:
        log.exception("Live MT5 pull failed: %s", e)

    # EA / GenerateFeatures files
    try:
        path = pull_from_mt5_files()
        df = _read_dest_df(path or "")
        if not df.empty:
            df["_src"] = "mt5files"
            frames.append(df)
            sources.append(f"mt5files:{len(df)}")
    except Exception as e:
        log.exception("MT5 Files sync failed: %s", e)

    if not frames:
        try:
            path = pull_from_project_cache()
            df = _read_dest_df(path or DEST_PATH if path else "")
            if not df.empty:
                df["_src"] = "cache"
                frames.append(df)
                sources.append(f"cache:{len(df)}")
        except Exception as e:
            log.exception("Cache fallback failed: %s", e)

    if not frames:
        log.error(
            "FAILED: no live MT5 data, no valid Files CSV, no historic rebuild, no cache. "
            "Open MT5 terminal, ensure symbols visible, re-run."
        )
        return False

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "time" in merged.columns and "symbol" in merged.columns:
        merged["symbol"] = merged["symbol"].astype(str).str.replace(r"\.r$", "", regex=True)
        # later sources (live) win on duplicate bars
        merged = merged.drop_duplicates(subset=["time", "symbol"], keep="last")
        try:
            merged["_t"] = pd.to_datetime(merged["time"], errors="coerce")
            merged = merged.sort_values(["symbol", "_t"]).drop(columns=["_t"])
        except Exception:
            pass
    if "_src" in merged.columns:
        merged = merged.drop(columns=["_src"])
    merged.to_csv(DEST_PATH, index=False, encoding="utf-8")
    log.info("MERGED feature CSV %s rows from %s → %s", len(merged), sources, DEST_PATH)
    _publish(DEST_PATH)
    return True


def main_daemon():
    log.info("mt5_data_sync daemon started")
    while True:
        try:
            sync_once()
        except Exception as e:
            log.error("Sync error: %s", e)
        time.sleep(60)


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        main_daemon()
    else:
        ok = sync_once()
        # pipeline-friendly: exit 0 if we produced a usable file
        sys.exit(0 if ok else 1)
