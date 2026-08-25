import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json'

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find config file at {CONFIG_PATH}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Config file has invalid format - {e}")
    exit(1)

# Set up logging
log_file = os.path.join(config['log_path'], 'script.log')  # Change 'script.log' to match the script’s name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

import json
import os
with open('C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json', 'r') as f:
    config = json.load(f)
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime
import logging
import uuid
import filelock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\realtime_data.csv"
OUTPUT_PATH_CANDLES = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\candle_data.csv"
LOCK_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\data.lock"

def initialize_mt5():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not mt5.initialize():
                logging.error("MT5 initialization failed")
                return False
            account = 7141842  # Update with your account number
            password = "x3B!HFJH5"  # Update with your password
            server = "FXPIG-Server"  # Update with your broker's server name
            if not mt5.login(account, password=password, server=server):
                logging.error("MT5 login failed")
                return False
            logging.info("MT5 initialized and logged in successfully")
            return True
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return False
            time.sleep(5)

def get_realtime_tick_data(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to retrieve tick for {symbol}")
        return None
    return pd.DataFrame([{
        'time': datetime.fromtimestamp(tick.time),
        'symbol': symbol,
        'bid': tick.bid,
        'ask': tick.ask,
        'last': tick.last,
        'volume': tick.volume
    }])

def get_realtime_candle_data(symbol, timeframe, num_bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None:
        logging.error(f"Failed to retrieve candle data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def compute_technical_indicators(df, symbol):
    # Ensure sufficient data
    if len(df) < 50:
        logging.warning(f"Not enough data for {symbol} to compute indicators")
        return None
    
    # Compute indicators using pandas_ta
    df_ta = df.copy()
    df_ta['atr'] = ta.atr(df['high'], df['low'], df['close'], length=10)
    df_ta['ema_fast'] = ta.ema(df['close'], length=12)
    df_ta['ema_slow'] = ta.ema(df['close'], length=26)
    df_ta['ema_diff'] = df_ta['ema_fast'] - df_ta['ema_slow']
    df_ta['rsi'] = ta.rsi(df['close'], length=10)
    df_ta['macd'] = ta.macd(df['close'], fast=9, slow=21, signal=7)['MACD_9_21_7']
    df_ta['macd_signal'] = ta.macd(df['close'], fast=9, slow=21, signal=7)['MACDs_9_21_7']
    df_ta['macd_diff'] = df_ta['macd'] - df_ta['macd_signal']
    df_ta['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'], anchor='D')
    df_ta['price_vwap_diff'] = df['close'] - df_ta['vwap']
    bb = ta.bbands(df['close'], length=20, std=2)
    df_ta['bb_upper'] = bb[f'BBU_20_2.0']
    df_ta['bb_lower'] = bb[f'BBL_20_2.0']
    df_ta['bb_position'] = (df['close'] - df_ta['bb_lower']) / (df_ta['bb_upper'] - df_ta['bb_lower'])
    df_ta['roc'] = ta.roc(df['close'], length=14)
    df_ta['stochastic'] = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)['STOCHk_14_3_3']
    df_ta['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
    df_ta['williams'] = ta.willr(df['high'], df['low'], df['close'], length=14)
    df_ta['momentum'] = ta.mom(df['close'], length=14)
    df_ta['realized_vol'] = df['close'].pct_change().rolling(14).std() * (252 * 96) ** 0.5
    df_ta['chaikin_vol'] = (df['high'] - df['low']).rolling(14).mean()
    df_ta['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df_ta['rvi'] = ta.rvi(df['close'], df['open'], length=10)
    df_ta['obv'] = ta.obv(df['close'], df['volume'])
    df_ta['volume_delta'] = df['volume'].where(df['close'] > df['open'], -df['volume'])
    df_ta['ad_line'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
    df_ta['vol_osc'] = (ta.sma(df['volume'], length=5) - ta.sma(df['volume'], length=14)) / ta.sma(df['volume'], length=14) * 100
    df_ta['supertrend'] = ta.supertrend(df['high'], df['low'], df['close'], length=14, multiplier=3)['SUPERT_14_3.0']
    df_ta['hma'] = ta.hma(df['close'], length=9)
    df_ta['ichimoku_tenkan'] = ta.ichimoku(df['high'], df['low'], tenkan=9, kijun=26, senkou=52)[0]['ITS_9']
    df_ta['sar'] = ta.psar(df['high'], df['low'], step=0.02, max_step=0.2)['PSARl_0.02_0.2'].fillna(ta.psar(df['high'], df['low'], step=0.02, max_step=0.2)['PSARs_0.02_0.2'])
    df_ta['dpo'] = ta.dpo(df['close'], length=21)
    df_ta['spread'] = (df['high'] - df['low']) / df['close'] * 100
    df_ta['sentiment'] = 0.5  # Placeholder for external sentiment data

    # Select the latest row
    latest = df_ta.iloc[-1]
    return pd.DataFrame([{
        'time': latest['time'],
        'symbol': symbol,
        'price': latest['close'],
        'atr': latest['atr'],
        'ema_diff': latest['ema_diff'],
        'rsi': latest['rsi'],
        'macd_diff': latest['macd_diff'],
        'vwap': latest['vwap'],
        'price_vwap_diff': latest['price_vwap_diff'],
        'bb_position': latest['bb_position'],
        'roc': latest['roc'],
        'stochastic': latest['stochastic'],
        'cci': latest['cci'],
        'williams': latest['williams'],
        'momentum': latest['momentum'],
        'realized_vol': latest['realized_vol'],
        'chaikin_vol': latest['chaikin_vol'],
        'adx': latest['adx'],
        'rvi': latest['rvi'],
        'obv': latest['obv'],
        'volume_delta': latest['volume_delta'],
        'ad_line': latest['ad_line'],
        'vol_osc': latest['vol_osc'],
        'supertrend': latest['supertrend'],
        'hma': latest['hma'],
        'ichimoku_tenkan': latest['ichimoku_tenkan'],
        'sar': latest['sar'],
        'dpo': latest['dpo'],
        'spread': latest['spread'],
        'sentiment': latest['sentiment']
    }])

def main(symbols=["EURUSD", "USDJPY", "XAUUSD"], timeframe=mt5.TIMEFRAME_M1, tick_interval=10, candle_interval=60, num_bars=100):
    if not initialize_mt5():
        logging.error("Exiting due to MT5 initialization failure")
        return

    lock = filelock.FileLock(LOCK_PATH)
    while True:
        try:
            for symbol in symbols:
                # Fetch and save tick data
                tick_data = get_realtime_tick_data(symbol)
                if tick_data is not None:
                    with lock:
                        tick_data.to_csv(OUTPUT_PATH, mode='a', header=not pd.io.common.file_exists(OUTPUT_PATH), index=False, encoding='utf-8')
                    logging.info(f"Tick data synced for {symbol} at {datetime.now()}")

                # Fetch candle data and compute indicators
                candle_data = get_realtime_candle_data(symbol, timeframe, num_bars)
                if candle_data is not None:
                    indicator_data = compute_technical_indicators(candle_data, symbol)
                    if indicator_data is not None:
                        with lock:
                            indicator_data.to_csv(OUTPUT_PATH_CANDLES, mode='a', header=not pd.io.common.file_exists(OUTPUT_PATH_CANDLES), index=False, encoding='utf-8')
                        logging.info(f"Indicator data synced for {symbol} at {datetime.now()}")

            time.sleep(tick_interval)

        except Exception as e:
            logging.error(f"Error: {e}")
            mt5.shutdown()
            break

    mt5.shutdown()

if __name__ == "__main__":
    main(
        symbols=["EURUSD", "USDJPY", "XAUUSD", "AUDUSD", "GBPUSD", "USDCAD"],
        timeframe=mt5.TIMEFRAME_M1,
        tick_interval=10,
        candle_interval=60,
        num_bars=100
    )