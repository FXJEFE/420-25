"""
mt5_generate_features.py
Connects to MT5, pulls M1 bars for all trading symbols,
computes all 27 model features, and writes FXJEFE_Features.csv.

This replaces GenerateFeatures.mq5 for the data-generation step,
using the MetaTrader5 Python API instead of running inside the terminal.

Output: data/FXJEFE_Features.csv  (also copied to Common/Files for the EA)
"""
import os
import sys
import json
import logging
import shutil
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
from textblob import TextBlob

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'], exist_ok=True)
os.makedirs(config['data_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'mt5_generate_features.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Trading pairs — FTMO uses plain names (no .r suffix)
SYMBOLS = ['EURUSD', 'USDJPY', 'XAUUSD', 'AUDUSD', 'GBPUSD', 'USDCAD']
HISTORY_BARS = 2000   # bars per symbol

SENTIMENT_MAP = {
    "EURUSD": "Bullish trend expected",
    "USDJPY": "Neutral market",
    "XAUUSD": "Bearish sentiment",
    "AUDUSD": "Positive outlook",
    "GBPUSD": "Strong buy signals",
    "USDCAD": "Sell pressure",
}


def get_sentiment(symbol: str) -> float:
    base = symbol.replace('.r', '').replace('.R', '')
    text = SENTIMENT_MAP.get(base, "Neutral")
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def calc_bb_position(close: pd.Series, period: int = 20) -> pd.Series:
    sma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std().replace(0, np.nan)
    return ((close - sma) / (2 * std) + 0.5).fillna(0.5)


def calc_roc(close: pd.Series, period: int = 10) -> pd.Series:
    prev = close.shift(period)
    return ((close - prev) / prev.replace(0, np.nan)).fillna(0.0)


def calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    lowest  = low.rolling(period, min_periods=1).min()
    highest = high.rolling(period, min_periods=1).max()
    denom = (highest - lowest).replace(0, np.nan)
    return (100 * (close - lowest) / denom).fillna(50.0)


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period, min_periods=1).mean()
    mad = tp.rolling(period, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return ((tp - sma) / (0.015 * mad.replace(0, np.nan))).fillna(0.0)


def calc_williams(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=1).max()
    lowest  = low.rolling(period, min_periods=1).min()
    denom = (highest - lowest).replace(0, np.nan)
    return (-100 * (highest - close) / denom).fillna(-50.0)


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = calc_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period, min_periods=1).mean().fillna(25.0)


def calc_rvi(close: pd.Series, period: int = 10) -> pd.Series:
    returns = close.pct_change()
    pos = returns.where(returns > 0, 0.0).rolling(period, min_periods=1).std()
    neg = returns.where(returns < 0, 0.0).rolling(period, min_periods=1).std()
    denom = (pos + neg).replace(0, np.nan)
    return (pos / denom).fillna(0.0)


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def calc_ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    mfm_denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / mfm_denom
    return (mfm.fillna(0.0) * volume).cumsum()


def calc_vol_osc(volume: pd.Series, short_p: int = 5, long_p: int = 20) -> pd.Series:
    short_ema = volume.ewm(span=short_p, adjust=False).mean()
    long_ema  = volume.ewm(span=long_p, adjust=False).mean()
    return ((short_ema - long_ema) / long_ema.replace(0, np.nan) * 100).fillna(0.0)


def calc_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 10, mult: float = 3.0) -> pd.Series:
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st = pd.Series(0.0, index=close.index)
    direction = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return st


def calc_hma(close: pd.Series, period: int = 9) -> pd.Series:
    half = close.rolling(period // 2, min_periods=1).mean()
    full = close.rolling(period, min_periods=1).mean()
    diff = 2 * half - full
    return diff.rolling(int(period ** 0.5), min_periods=1).mean()


def calc_ichimoku_tenkan(high: pd.Series, low: pd.Series, period: int = 9) -> pd.Series:
    return (high.rolling(period, min_periods=1).max() + low.rolling(period, min_periods=1).min()) / 2


def calc_sar(high: pd.Series, low: pd.Series, close: pd.Series,
             af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
    n = len(close)
    sar = pd.Series(0.0, index=close.index)
    if n == 0:
        return sar
    bull = True
    af = af_start
    ep = high.iloc[0]
    sar.iloc[0] = low.iloc[0]
    for i in range(1, n):
        sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])
        if bull:
            if low.iloc[i] < sar.iloc[i]:
                bull = False
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
        else:
            if high.iloc[i] > sar.iloc[i]:
                bull = True
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)
    return sar


def calc_dpo(close: pd.Series, period: int = 20) -> pd.Series:
    shifted_sma = close.rolling(period, min_periods=1).mean().shift(period // 2 + 1)
    return (close - shifted_sma).fillna(0.0)


def calc_realized_vol(close: pd.Series, period: int = 14) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(period, min_periods=1).std().fillna(0.0)


def calc_chaikin_vol(high: pd.Series, low: pd.Series, period: int = 10) -> pd.Series:
    hl_range = high - low
    ema_range = hl_range.ewm(span=period, adjust=False).mean()
    prev = ema_range.shift(period)
    return ((ema_range - prev) / prev.replace(0, np.nan) * 100).fillna(0.0)


def compute_features_for_symbol(symbol: str) -> pd.DataFrame:
    """Pull M1 bars and compute all 27 model features."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, HISTORY_BARS)
    if rates is None or len(rates) == 0:
        logging.warning(f"No data for {symbol}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y.%m.%d %H:%M')
    df['symbol'] = symbol

    h, l, c, v = df['high'], df['low'], df['close'], df['tick_volume'].astype(float)

    # Compute all 27 indicators
    atr_series = calc_atr(h, l, c, 14)
    ema_fast = calc_ema(c, 12)
    ema_slow = calc_ema(c, 26)

    df['price']          = c
    df['atr']            = atr_series
    df['ema_diff']       = ema_fast - ema_slow
    df['rsi']            = calc_rsi(c, 14)
    macd_fast            = calc_ema(c, 12)
    macd_slow            = calc_ema(c, 26)
    macd_signal          = calc_ema(macd_fast - macd_slow, 9)
    df['macd_diff']      = (macd_fast - macd_slow) - macd_signal
    vwap_cum_vol         = v.cumsum()
    df['vwap']           = (c * v).cumsum() / vwap_cum_vol.replace(0, np.nan)
    df['vwap']           = df['vwap'].fillna(c)
    df['price_vwap_diff']= c - df['vwap']
    df['bb_position']    = calc_bb_position(c, 20)
    df['roc']            = calc_roc(c, 10)
    df['stochastic']     = calc_stochastic(h, l, c, 14)
    df['cci']            = calc_cci(h, l, c, 20)
    df['williams']       = calc_williams(h, l, c, 14)
    df['momentum']       = c.diff(10).fillna(0.0)
    df['realized_vol']   = calc_realized_vol(c, 14)
    df['chaikin_vol']    = calc_chaikin_vol(h, l, 10)
    df['adx']            = calc_adx(h, l, c, 14)
    df['rvi']            = calc_rvi(c, 10)
    df['obv']            = calc_obv(c, v)
    df['volume_delta']   = v.diff().fillna(0.0)
    df['ad_line']        = calc_ad_line(h, l, c, v)
    df['vol_osc']        = calc_vol_osc(v, 5, 20)
    df['supertrend']     = calc_supertrend(h, l, c, 10, 3.0)
    df['hma']            = calc_hma(c, 9)
    df['ichimoku_tenkan']= calc_ichimoku_tenkan(h, l, 9)
    df['sar']            = calc_sar(h, l, c, 0.02, 0.2)
    df['dpo']            = calc_dpo(c, 20)

    # Spread from MT5
    sym_info = mt5.symbol_info(symbol)
    spread_pts = sym_info.spread if sym_info else 0
    point = sym_info.point if sym_info and sym_info.point > 0 else 0.00001
    df['spread'] = spread_pts * point

    # Sentiment
    df['sentiment'] = get_sentiment(symbol)

    # Signal column (placeholder — the EA/server produces real signals)
    df['signal'] = ''

    # Select final columns
    output_cols = ['time', 'symbol', 'price',
                   'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
                   'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
                   'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
                   'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
                   'dpo', 'spread', 'sentiment', 'signal']

    return df[output_cols]


def main():
    logging.info("mt5_generate_features.py started")

    if not mt5.initialize():
        logging.error(f"MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)

    info = mt5.account_info()
    if info:
        logging.info(f"Connected: account {info.login}, server {info.server}, balance {info.balance}")
    else:
        logging.error("Not logged in to MT5.")
        mt5.shutdown()
        sys.exit(1)

    # Build features for all symbols
    all_frames = []
    for sym in SYMBOLS:
        sym_info = mt5.symbol_info(sym)
        if sym_info is None:
            logging.warning(f"Symbol {sym} not found — skipping.")
            continue
        if not sym_info.visible:
            mt5.symbol_select(sym, True)

        logging.info(f"Fetching {HISTORY_BARS} M1 bars for {sym}...")
        df = compute_features_for_symbol(sym)
        if df.empty:
            logging.warning(f"No data returned for {sym}.")
            continue
        logging.info(f"  {sym}: {len(df)} rows, price range {df['price'].min():.5f}–{df['price'].max():.5f}")
        all_frames.append(df)

    mt5.shutdown()

    if not all_frames:
        logging.error("No data for any symbol. Is market open?")
        sys.exit(1)

    result = pd.concat(all_frames, ignore_index=True)
    logging.info(f"Total: {len(result)} rows across {len(all_frames)} symbols.")

    # Write to project data folder
    output_path = os.path.join(config['data_path'], 'FXJEFE_Features.csv')
    result.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"Saved → {output_path}")

    # Also copy to MT5 Common\Files and terminal Files (EA may have the file locked)
    for label, folder_key in [('Common', 'mt5_common_path'), ('Terminal', 'mt5_files_path')]:
        folder = config.get(folder_key, '')
        if not folder:
            continue
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, 'FXJEFE_Features.csv')
        try:
            shutil.copy2(output_path, dest)
            logging.info(f"Copied → {dest}")
        except PermissionError:
            logging.warning(f"{label} copy skipped — file locked by EA (will use existing copy).")

    logging.info("mt5_generate_features.py completed.")


if __name__ == '__main__':
    main()
