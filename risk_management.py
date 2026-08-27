# -*- coding: utf-8 -*-
"""Risk checks + position sizing (uses MetaTrader5 package correctly)."""
from fxjefe_paths import load_config, setup_logging, features_path
import logging
import os
import json
import pandas as pd

config = load_config()
setup_logging(config, "risk_management")

# defaults if config section missing
RM = config.get("risk_management") or {
    "risk_percent": 1.0,
    "max_position_size": 1.0,
    "max_drawdown_percent": 20.0,
    "default_sl_atr_mult": 2.0,
}
config["risk_management"] = RM


def calculate_position_size(balance: float, risk_percent: float, sl_pips: float, pip_value: float) -> float:
    if sl_pips <= 0 or pip_value <= 0:
        return 0.01
    risk_amount = balance * (risk_percent / 100.0)
    lot = risk_amount / (sl_pips * pip_value)
    return float(min(max(lot, 0.01), float(RM.get("max_position_size", 1.0))))


def check_drawdown(balance: float, initial: float) -> bool:
    if initial <= 0:
        return True
    dd = ((initial - balance) / initial) * 100.0
    max_dd = float(RM.get("max_drawdown_percent", 20.0))
    if dd > max_dd:
        logging.warning("Max drawdown exceeded: %.2f%% > %.2f%%", dd, max_dd)
        return False
    return True


def apply_risk_management(symbol: str, signal: str, price: float, atr: float, balance: float = 10000.0):
    mult = float(RM.get("default_sl_atr_mult", 2.0))
    atr = atr if atr and atr > 0 else price * 0.001
    if signal == "buy":
        sl = price - mult * atr
        tp = price + mult * atr * 1.5
    elif signal == "sell":
        sl = price + mult * atr
        tp = price - mult * atr * 1.5
    else:
        return {"symbol": symbol, "signal": "hold", "lot": 0.0, "sl": None, "tp": None}

    # rough pip value
    pip = 0.0001 if price < 50 else (0.01 if price < 500 else 1.0)
    sl_pips = abs(price - sl) / pip
    pip_value = 10.0  # approx per standard lot for many FX pairs
    lot = calculate_position_size(balance, float(RM.get("risk_percent", 1.0)), sl_pips, pip_value)
    return {
        "symbol": symbol,
        "signal": signal,
        "price": price,
        "atr": atr,
        "lot": round(lot, 2),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "balance_ok": check_drawdown(balance, balance),  # placeholder true at start
    }


def main() -> None:
    balance = 10000.0
    equity = 10000.0
    try:
        import MetaTrader5 as mt5

        if mt5.initialize():
            info = mt5.account_info()
            if info:
                balance = float(info.balance)
                equity = float(info.equity)
                logging.info("MT5 account balance=%.2f equity=%.2f server=%s", balance, equity, info.server)
            mt5.shutdown()
        else:
            logging.warning("MT5 init failed — using default balance 10000")
    except Exception as e:
        logging.warning("MT5 unavailable: %s — using default balance", e)

    if not check_drawdown(equity, balance if balance > 0 else equity):
        logging.error("Drawdown gate failed")
    else:
        logging.info("Drawdown check OK")

    # sample from latest features
    src = features_path(config, "FXJEFE_Features_with_signals.csv")
    if not os.path.isfile(src):
        src = features_path(config)
    results = []
    if os.path.isfile(src):
        df = pd.read_csv(src, encoding="utf-8", low_memory=False)
        if not df.empty:
            last = df.groupby(df["symbol"] if "symbol" in df.columns else df.columns[0]).tail(1)
            for _, row in last.iterrows():
                sym = str(row.get("symbol", "EURUSD"))
                price = float(row.get("price", 0) or 0)
                atr = float(row.get("atr", 0) or 0)
                sig = str(row.get("signal", "hold")).lower()
                if sig in ("1", "buy"):
                    sig = "buy"
                elif sig in ("-1", "sell"):
                    sig = "sell"
                else:
                    sig = "hold"
                results.append(apply_risk_management(sym, sig, price, atr, balance=balance))
    else:
        results.append(apply_risk_management("EURUSD", "hold", 1.1, 0.0005, balance=balance))

    out = os.path.join(config["data_path"], "risk_management_output.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"balance": balance, "equity": equity, "positions": results}, f, indent=2)
    logging.info("Wrote %s (%s symbols)", out, len(results))
    for r in results:
        logging.info("  %s", r)


if __name__ == "__main__":
    main()
