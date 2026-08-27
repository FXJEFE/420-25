# -*- coding: utf-8 -*-
"""
Write latest feature row snapshot to MT5 Files for EA consumption.
Pipeline-safe: single-shot (use --daemon for loop).
Uses currently logged-in MT5 terminal (no hardcoded credentials).
"""
from fxjefe_paths import load_config, setup_logging, features_path, mt5_file
import logging
import os
import sys
import time
import pandas as pd

config = load_config()
setup_logging(config, "mt5_signal_script")

OUTPUT_PATH = mt5_file(config, "realtime_data.csv")
OUTPUT_CANDLES = mt5_file(config, "candle_data.csv")
OUTPUT_SIGNAL = mt5_file(config, "FXJEFE_signal_snapshot.csv")


def ensure_dirs():
    os.makedirs(config["mt5_files_path"], exist_ok=True)
    os.makedirs(config["data_path"], exist_ok=True)


def snapshot_from_features() -> pd.DataFrame:
    src = features_path(config, "FXJEFE_Features_with_signals.csv")
    if not os.path.isfile(src):
        src = features_path(config, "FXJEFE_Features_fixed.csv")
    if not os.path.isfile(src):
        src = features_path(config)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"No features for snapshot: {src}")
    df = pd.read_csv(src, encoding="utf-8", low_memory=False)
    if df.empty:
        raise ValueError("Features CSV empty")
    # last bar per symbol
    if "symbol" in df.columns:
        snap = df.groupby("symbol", as_index=False).tail(1)
    else:
        snap = df.tail(1)
    return snap


def snapshot_from_mt5() -> pd.DataFrame:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    try:
        symbols = list(config.get("forex_symbols") or []) + list(config.get("crypto_symbols") or [])
        symbols = list(dict.fromkeys(symbols + ["XAUUSD", "EURUSD"]))
        rows = []
        for sym in symbols:
            info = mt5.symbol_info(sym)
            if info is None:
                # try .crp for gold etc
                for alt in (sym + ".crp", sym + ".r"):
                    info = mt5.symbol_info(alt)
                    if info is not None:
                        sym = alt
                        break
            if info is None:
                continue
            if not info.visible:
                mt5.symbol_select(sym, True)
            tick = mt5.symbol_info_tick(sym)
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 2)
            if tick is None:
                continue
            price = float(tick.bid or tick.last or 0)
            atr = 0.0
            if rates is not None and len(rates) >= 1:
                atr = float(rates[-1]["high"] - rates[-1]["low"])
            rows.append(
                {
                    "time": time.strftime("%Y.%m.%d %H:%M"),
                    "symbol": sym.split(".")[0].upper(),
                    "price": price,
                    "atr": atr,
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "spread": float(info.spread) * float(info.point or 0),
                }
            )
        return pd.DataFrame(rows)
    finally:
        mt5.shutdown()


def write_snapshot(df: pd.DataFrame) -> None:
    ensure_dirs()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    df.to_csv(OUTPUT_SIGNAL, index=False, encoding="utf-8")
    # candles-ish copy
    df.to_csv(OUTPUT_CANDLES, index=False, encoding="utf-8")
    # data folder copy
    df.to_csv(os.path.join(config["data_path"], "realtime_data.csv"), index=False, encoding="utf-8")
    logging.info("Wrote snapshot %s rows → %s", len(df), OUTPUT_PATH)


def main() -> None:
    ensure_dirs()
    df = None
    try:
        df = snapshot_from_mt5()
        logging.info("Live MT5 snapshot: %s symbols", len(df))
    except Exception as e:
        logging.warning("Live MT5 snapshot failed: %s — using features CSV", e)
    if df is None or df.empty:
        df = snapshot_from_features()
        logging.info("Features snapshot: %s rows", len(df))
    write_snapshot(df)

    if "--daemon" in sys.argv:
        logging.info("Daemon mode every 60s")
        while True:
            time.sleep(60)
            try:
                try:
                    d = snapshot_from_mt5()
                except Exception:
                    d = snapshot_from_features()
                write_snapshot(d)
            except Exception as e:
                logging.error("daemon tick failed: %s", e)


if __name__ == "__main__":
    main()
