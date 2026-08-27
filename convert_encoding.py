# -*- coding: utf-8 -*-
"""Convert MT5 Files CSVs to UTF-8 (chardet optional)."""
from fxjefe_paths import load_config, setup_logging, mt5_file, features_path
import logging
import os

config = load_config()
setup_logging(config, "convert_encoding")

try:
    import chardet
except ImportError:
    chardet = None

FILES = [
    "FXJEFE_Features.csv",
    "FXJEFE_Features_fixed.csv",
    "FXJEFE_trades.csv",
    "FXJEFE_trades_outcomes.csv",
    "log.txt",
]


def detect_encoding(raw: bytes) -> str:
    if chardet is not None:
        try:
            guess = chardet.detect(raw) or {}
            if guess.get("encoding"):
                return guess["encoding"]
        except Exception:
            pass
    return "utf-8"


def convert_to_utf8(path: str) -> bool:
    if not os.path.isfile(path):
        logging.warning("not found: %s", path)
        return False
    with open(path, "rb") as f:
        raw = f.read()
    if not raw or all(b == 0 for b in raw[:64]):
        logging.error("corrupt/empty: %s", path)
        return False
    enc = detect_encoding(raw)
    text = None
    for e in (enc, "utf-8-sig", "utf-8", "latin1", "cp1252"):
        if not e:
            continue
        try:
            text = raw.decode(e)
            enc = e
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
        enc = "utf-8/replace"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    logging.info("converted %s (from %s)", path, enc)
    return True


def main() -> None:
    base = config["mt5_files_path"]
    data = config["data_path"]
    ok = 0
    for name in FILES:
        for folder in (base, data):
            p = os.path.join(folder, name)
            if convert_to_utf8(p):
                ok += 1
    # ensure data has features
    src = mt5_file(config, "FXJEFE_Features.csv")
    dst = features_path(config)
    if os.path.isfile(src) and (not os.path.isfile(dst) or os.path.getsize(dst) < 100):
        import shutil
        shutil.copy2(src, dst)
        logging.info("copied features to data: %s", dst)
    logging.info("convert_encoding done ok=%s", ok)


if __name__ == "__main__":
    main()
