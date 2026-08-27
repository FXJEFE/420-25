# -*- coding: utf-8 -*-
import os as _os_utf8, sys as _sys_utf8
_os_utf8.environ.setdefault('PYTHONUTF8','1')
_os_utf8.environ.setdefault('PYTHONIOENCODING','utf-8')
for _s in (getattr(_sys_utf8,'stdout',None), getattr(_sys_utf8,'stderr',None)):
    try:
        if _s is not None and hasattr(_s,'reconfigure'):
            _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\locallarry\\Documents\\FXJEFE_Project\\config.json'

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8', errors='replace') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find config file at {CONFIG_PATH}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Config file has invalid format - {e}")
    exit(1)

# Set up logging
log_file = os.path.join(config['log_path'], 'script.log')  # Change 'script.log' to match the script's name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

import json
import os
with open('C:\\Users\\locallarry\\Documents\\FXJEFE_Project\\config.json', 'r', encoding='utf-8', errors='replace') as f:
    config = json.load(f)
import pandas as pd 
import logging 
from textblob import TextBlob 
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
 
INPUT_PATH = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
OUTPUT_PATH = os.path.join(config['data_output_path'], 'FXJEFE_Features_fixed.csv')
 
def fix_csv(): 
    try: 
        df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig', low_memory=False) 
        logging.info(f"Read CSV with {len(df)} rows") 
 
        expected_columns = ['time', 'symbol', 'price', 'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 
                           'price_vwap_diff', 'bb_position', 'roc', 'stochastic', 'cci', 'williams', 
                           'momentum', 'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta', 
                           'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar', 'dpo', 
                           'spread', 'sentiment', 'signal'] 
        for col in expected_columns: 
            if col not in df.columns: 
                df[col] = '' if col in ['time', 'symbol', 'signal'] else 0.0 
 
        defaults = { 
            'price': df['price'].ffill(), 
            'atr': df['atr'].mean() if df['atr'].notna().any() else 0.0001, 
            'ema_diff': 0.0, 
            'rsi': 50.0, 
            'macd_diff': 0.0, 
            'vwap': df['price'], 
            'price_vwap_diff': 0.0, 
            'bb_position': 0.5, 
            'roc': 0.0, 
            'stochastic': 50.0, 
            'cci': 0.0, 
            'williams': -50.0, 
            'momentum': 0.0, 
            'realized_vol': 0.0, 
            'chaikin_vol': 0.0, 
            'adx': 25.0, 
            'rvi': 0.0, 
            'obv': 0.0, 
            'volume_delta': 0.0, 
            'ad_line': 0.0, 
            'vol_osc': 0.0, 
            'supertrend': 0.0, 
            'hma': df['price'], 
            'ichimoku_tenkan': df['price'], 
            'sar': df['price'], 
            'dpo': 0.0, 
            'spread': 2.0, 
            'sentiment': 0.0, 
            'signal': 'hold' 
        } 
        for col, default in defaults.items(): 
            df[col] = df[col].fillna(default) 
 
        def get_sentiment(symbol): 
            posts = { 
                "EURUSD.r": "Bullish trend expected", 
                "USDJPY.r": "Neutral market", 
                "XAUUSD.r": "Bearish sentiment", 
                "AUDUSD.r": "Positive outlook", 
                "GBPUSD.r": "Strong buy signals", 
                "USDCAD.r": "Sell pressure" 
            } 
            text = posts.get(symbol, "Neutral") 
            return TextBlob(text).sentiment.polarity 
 
        df['sentiment'] = df['symbol'].apply(get_sentiment) 
 
        numeric_cols = [col for col in expected_columns if col not in ['time', 'symbol', 'signal']] 
        for col in numeric_cols: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(defaults.get(col, 0.0)) 
 
        corr_matrix = df[numeric_cols].corr() 
        high_corr = corr_matrix[corr_matrix.abs() > 0.8] 
        logging.info(f"High correlations (>0.8):\n{high_corr[high_corr != 1.0].dropna(how='all')}") 
 
        if df[numeric_cols].isna().any().any(): 
            logging.warning("NaNs detected after filling; check data source") 
        if len(df) < 1000: 
            logging.warning(f"Low row count ({len(df)}); run GenerateFeatures.mq5 again") 
 
        # OG: always write fixed features to ALL destinations (never skip)
        from fxjefe_paths import write_feature_csv
        write_feature_csv(df, config, 'FXJEFE_Features_fixed.csv')
        write_feature_csv(df, config, 'FXJEFE_Features.csv')
        logging.info(f"Saved fixed CSV to {OUTPUT_PATH} (+ all OG mirrors)") 
    except Exception as e: 
        logging.error(f"Error: {e}") 
        raise 
 
if __name__ == "__main__": 
    fix_csv() 