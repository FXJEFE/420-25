# -*- coding: utf-8 -*-
"""Insert garch_vol after rsi on live/historic FXJEFE_Features CSVs that lack it."""
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\locallarry\Documents\FXJEFE_Project")
TARGETS = [
    ROOT / "data" / "FXJEFE_Features.csv",
    ROOT / "data" / "FXJEFE_Features_fixed.csv",
    ROOT / "data" / "FXJEFE_Features_with_labels.csv",
]


def fix(path: Path) -> str:
    if not path.is_file():
        return f"MISS {path}"
    header = path.read_bytes().split(b"\n", 1)[0].decode("utf-8-sig", "replace").strip()
    cols = [c.strip() for c in header.split(",")]
    if "garch_vol" in cols:
        return f"OK  {path.name} already has garch_vol n={len(cols)}"
    df = pd.read_csv(path)
    src = df["realized_vol"] if "realized_vol" in df.columns else 0.0
    idx = list(df.columns).index("rsi") + 1 if "rsi" in df.columns else min(6, len(df.columns))
    df.insert(idx, "garch_vol", src)
    df.to_csv(path, index=False)
    return f"FIX {path.name} inserted garch_vol at {idx} rows={len(df)} n={len(df.columns)}"


if __name__ == "__main__":
    for p in TARGETS:
        print(fix(p), flush=True)
