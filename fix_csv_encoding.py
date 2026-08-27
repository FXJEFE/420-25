# -*- coding: utf-8 -*-
"""Ensure feature CSVs are valid UTF-8 (re-encode safely)."""
from fxjefe_paths import load_config, setup_logging, features_path, mt5_file
import logging
import os
import shutil

config = load_config()
setup_logging(config, "fix_csv_encoding")

TARGETS = [
    features_path(config, "FXJEFE_Features.csv"),
    features_path(config, "FXJEFE_Features_fixed.csv"),
    mt5_file(config, "FXJEFE_Features.csv"),
]


def reencode(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 10:
        logging.warning("skip missing/empty: %s", path)
        return False
    with open(path, "rb") as f:
        raw = f.read()
    if all(b == 0 for b in raw[:64]):
        logging.error("zero-filled file: %s", path)
        return False
    text = None
    used = None
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252", "utf-16"):
        try:
            text = raw.decode(enc)
            used = enc
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
        used = "utf-8/replace"
    bak = path + ".bak_enc"
    try:
        shutil.copy2(path, bak)
    except OSError:
        pass
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    logging.info("Re-encoded %s (%s → utf-8)", path, used)
    return True


def main() -> None:
    ok = 0
    for p in TARGETS:
        if reencode(p):
            ok += 1
    # also copy primary features to data if only MT5 has it
    data_feat = features_path(config)
    mt5_feat = mt5_file(config, "FXJEFE_Features.csv")
    if (not os.path.isfile(data_feat) or os.path.getsize(data_feat) < 100) and os.path.isfile(mt5_feat):
        shutil.copy2(mt5_feat, data_feat)
        logging.info("Copied MT5 features → %s", data_feat)
        ok += 1
    if ok == 0:
        logging.warning("No files re-encoded (nothing present yet is OK if sync runs first)")
    logging.info("fix_csv_encoding done (ok=%s)", ok)


if __name__ == "__main__":
    main()
