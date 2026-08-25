# -*- coding: utf-8 -*-
"""Add simple pip-based signals → FXJEFE_Features_with_signals.csv."""
from fxjefe_paths import load_config, setup_logging, features_path, write_feature_csv
import logging
import os
import pandas as pd
import numpy as np

config = load_config()
setup_logging(config, "generate_new_csv")

INPUT = features_path(config, "FXJEFE_Features_fixed.csv")
if not os.path.isfile(INPUT):
    INPUT = features_path(config, "FXJEFE_Features.csv")
OUTPUT = features_path(config, config.get("features_signals_csv", "FXJEFE_Features_with_signals.csv"))


def main() -> None:
    if not os.path.isfile(INPUT):
        raise FileNotFoundError(f"Features not found: {INPUT}")
    df = pd.read_csv(INPUT, encoding="utf-8", low_memory=False)
    if "symbol" not in df.columns or "price" not in df.columns:
        raise ValueError(f"Need symbol+price columns, got {list(df.columns)}")

    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["signal"] = 0
    for sym, g in df.groupby("symbol", sort=False):
        idx = g.index
        px = g["price"]
        # ~10 pips for forex; scale for crypto/metals by ATR if present
        if "atr" in g.columns and g["atr"].notna().any():
            thr = pd.to_numeric(g["atr"], errors="coerce").fillna(px * 0.001) * 0.5
        else:
            thr = pd.Series(0.0010, index=idx)
        delta = px.diff()
        buy = delta > thr
        sell = delta < -thr
        df.loc[idx[buy.fillna(False)], "signal"] = 1
        df.loc[idx[sell.fillna(False)], "signal"] = -1

    if "direction" not in df.columns:
        df["direction"] = df["signal"]

    keep = [c for c in [
        "time", "symbol", "price", "direction", "atr", "ema_diff", "rsi", "garch_vol",
        "macd_diff", "vwap", "price_vwap_diff", "bb_position", "signal",
    ] if c in df.columns]
    out = df[keep] if keep else df
    # OG: always write signals CSV to ALL destinations (never skip)
    written = write_feature_csv(out, config, "FXJEFE_Features_with_signals.csv")
    logging.info("Wrote FXJEFE_Features_with_signals.csv to %s locations (%s rows)", len(written), len(out))


if __name__ == "__main__":
    main()
