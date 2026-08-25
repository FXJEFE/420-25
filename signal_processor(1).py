"""
signal_processor.py
Reads FXJEFE_Features.csv, applies SMA crossover logic to generate
buy/hold/sell signals per symbol, and writes signals_output.csv.
Supports --daemon for continuous operation; default is single-shot for pipeline use.
"""
import os
import sys
import json
import logging
import time
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'signal_processor.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

INPUT_PATH  = os.path.join(config['data_path'],        'FXJEFE_Features.csv')
OUTPUT_PATH = os.path.join(config['data_output_path'], 'signals_output.csv')


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA_short'] = df.groupby('symbol')['price'].transform(
        lambda x: x.rolling(5,  min_periods=1).mean()
    )
    df['SMA_long'] = df.groupby('symbol')['price'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df['signal'] = 'hold'
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 'buy'
    df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = 'sell'
    return df


def process_once() -> bool:
    if not os.path.exists(INPUT_PATH):
        logging.warning(f"Input file not found: {INPUT_PATH} — run mt5_data_sync.py first.")
        return False

    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig', low_memory=False)
    if df.empty:
        logging.warning("Input CSV is empty.")
        return False

    df['price'] = pd.to_numeric(df['price'], errors='coerce').ffill()
    df = generate_signals(df)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

    latest = df.groupby('symbol').last()[['price', 'signal']]
    logging.info(f"Signals written → {OUTPUT_PATH}\n{latest.to_string()}")
    return True


def main():
    logging.info("signal_processor started.")
    if '--daemon' in sys.argv:
        while True:
            try:
                process_once()
            except Exception as e:
                logging.error(f"Error: {e}")
            time.sleep(60)
    else:
        try:
            ok = process_once()
            sys.exit(0 if ok else 1)
        except Exception as e:
            logging.error(f"Error: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
