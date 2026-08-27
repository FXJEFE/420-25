# -*- coding: utf-8 -*-
"""Parse MT5 / project logs for API signals → CSV (non-fatal if log missing)."""
from fxjefe_paths import load_config, setup_logging
import logging
import os
import re
import pandas as pd

config = load_config()
setup_logging(config, "parse_log_to_csv")

CANDIDATE_LOGS = [
    os.path.join(config["mt5_files_path"], "log.txt"),
    os.path.join(config["data_path"], "log.txt"),
    os.path.join(config["log_path"], "ai_server.log"),
    os.path.join(config["log_path"], "pipelineOG333.log"),
    os.path.join(config["project_root"], "log.txt"),
    os.path.join(config["project_root"], "FXJEFE_log.txt"),
    os.path.join(config.get("scripts_path", ""), "FXJEFE_log.txt"),
]
OUTPUT = os.path.join(config["data_path"], "parsed_log_signals.csv")

api_pattern = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?).*?(?:API|predict|signal).*?(\w+).*?(buy|sell|hold)",
    re.I,
)
signal_pattern = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?).*?\"signal\"\s*:\s*\"(buy|sell|hold)\"",
    re.I,
)


def read_log(path: str):
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252", "utf-16"):
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.readlines(), enc
        except Exception:
            continue
    return None, None


def main() -> None:
    rows = []
    used = None
    for path in CANDIDATE_LOGS:
        if not path or not os.path.isfile(path) or os.path.getsize(path) < 10:
            continue
        lines, enc = read_log(path)
        if not lines:
            continue
        used = path
        logging.info("Parsing %s (%s, %s lines)", path, enc, len(lines))
        for line in lines:
            m = signal_pattern.search(line)
            if m:
                rows.append({"time": m.group(1), "signal": m.group(2).lower(), "source": path})
                continue
            m2 = api_pattern.search(line)
            if m2:
                rows.append(
                    {
                        "time": m2.group(1),
                        "symbol": m2.group(2),
                        "signal": m2.group(3).lower(),
                        "source": path,
                    }
                )
        if rows:
            break

    if not rows:
        logging.warning("No parseable signal lines found — writing empty schema CSV")
        df = pd.DataFrame(columns=["time", "symbol", "signal", "source"])
    else:
        df = pd.DataFrame(rows)
        logging.info("Parsed %s signal rows from %s", len(df), used)

    os.makedirs(config["data_path"], exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    logging.info("Wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
