# -*- coding: utf-8 -*-
"""Analyze trade outcomes if present; non-fatal empty report otherwise."""
from fxjefe_paths import load_config, setup_logging
import logging
import os
import json
import pandas as pd

config = load_config()
setup_logging(config, "analyze_outcomes")

CANDIDATES = [
    os.path.join(config["data_path"], "FXJEFE_trades_outcomes.csv"),
    os.path.join(config["mt5_files_path"], "FXJEFE_trades_outcomes.csv"),
    os.path.join(config["project_root"], "FXJEFE_trades_outcomes.csv"),
    os.path.join(config.get("scripts_path", ""), "FXJEFE_trades_outcomes.csv"),
]
OUT = os.path.join(config["data_path"], "outcomes_summary.json")


def main() -> None:
    path = next((p for p in CANDIDATES if os.path.isfile(p) and os.path.getsize(p) > 20), None)
    if not path:
        summary = {
            "status": "no_outcomes_file",
            "message": "No FXJEFE_trades_outcomes.csv found — run EA trades first",
            "searched": CANDIDATES,
        }
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.warning("No outcomes file; wrote empty summary %s", OUT)
        return

    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    summary = {
        "status": "ok",
        "path": path,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    # try common PnL columns
    for col in ("profit", "pnl", "Profit", "net_profit", "outcome"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            summary["sum_" + col] = float(s.sum(skipna=True))
            summary["mean_" + col] = float(s.mean(skipna=True)) if s.notna().any() else None
            break
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Outcomes summary → %s : %s", OUT, summary)


if __name__ == "__main__":
    main()
