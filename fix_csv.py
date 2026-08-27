# -*- coding: utf-8 -*-
"""Normalize FXJEFE_Features.csv into FXJEFE_Features_fixed.csv using config paths."""
from fxjefe_paths import load_config, setup_logging, features_path, write_feature_csv
import logging
import os
import pandas as pd

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

config = load_config()
setup_logging(config, "fix_csv")

INPUT_PATH = features_path(config, "FXJEFE_Features.csv")
OUTPUT_PATH = features_path(config, config.get("features_fixed_csv", "FXJEFE_Features_fixed.csv"))
# also check MT5 files
if not os.path.isfile(INPUT_PATH):
    alt = os.path.join(config["mt5_files_path"], "FXJEFE_Features.csv")
    if os.path.isfile(alt):
        INPUT_PATH = alt

EXPECTED = [
    "time", "symbol", "price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff", "vwap",
    "price_vwap_diff", "bb_position", "roc", "stochastic", "cci", "williams", "momentum",
    "realized_vol", "chaikin_vol", "adx", "rvi", "obv", "volume_delta", "ad_line", "vol_osc",
    "supertrend", "hma", "ichimoku_tenkan", "sar", "dpo", "spread", "sentiment", "signal",
    "confidence",
]


def main() -> None:
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(INPUT_PATH, encoding=enc, low_memory=False)
            logging.info("Read %s with %s (%s rows)", INPUT_PATH, enc, len(df))
            break
        except Exception as e:
            logging.warning("read failed %s: %s", enc, e)
    if df is None or df.empty:
        raise ValueError(f"Could not read usable CSV: {INPUT_PATH}")

    df.columns = [str(c).strip() for c in df.columns]
    for col in EXPECTED:
        if col not in df.columns:
            df[col] = "" if col in ("time", "symbol", "signal") else 0.0

    numeric = [c for c in EXPECTED if c not in ("time", "symbol", "signal")]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "price" in df.columns:
        df["price"] = df["price"].ffill()
    if "garch_vol" in df.columns:
        df["garch_vol"] = df["garch_vol"].fillna(0.0)
    if "rsi" in df.columns:
        df["rsi"] = df["rsi"].fillna(50.0)
    if "signal" in df.columns:
        df["signal"] = df["signal"].fillna("hold").replace("", "hold")
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].fillna(0.5)

    if TextBlob is not None and "symbol" in df.columns:
        posts = {
            "EURUSD": "Bullish trend expected",
            "USDJPY": "Neutral market",
            "XAUUSD": "Bearish sentiment",
            "BTCUSD": "Volatile bullish momentum",
        }

        def sent(sym):
            base = str(sym).replace(".r", "").upper()
            try:
                return float(TextBlob(posts.get(base, "Neutral")).sentiment.polarity)
            except Exception:
                return 0.0

        df["sentiment"] = df["symbol"].apply(sent)

    # OG: always write fixed features to ALL destinations (never skip)
    written = write_feature_csv(df, config, "FXJEFE_Features_fixed.csv")
    # also keep primary Features.csv in sync (fixed content is the working set)
    write_feature_csv(df, config, "FXJEFE_Features.csv")
    logging.info("Saved fixed CSV to %s locations; primary=%s rows=%s cols=%s",
                 len(written), OUTPUT_PATH, len(df), len(df.columns))


if __name__ == "__main__":
    main()
