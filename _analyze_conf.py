#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import collections
import json
import re
import statistics
from pathlib import Path

LOG = Path(r"C:\Users\locallarry\Documents\FXJEFE_Project\20260811.log")
AUDITS = [
    Path(r"C:\Users\locallarry\Documents\logs\audit_golden_20260811.csv"),
    Path(r"C:\Users\locallarry\Documents\logs\audit_golden_20260812.csv"),
]


def decode_log(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-16-le", "utf-16", "utf-8", "cp1252"):
        try:
            t = raw.decode(enc)
            if "confidence" in t or "Golden" in t or "API response" in t:
                print(f"log decoded as {enc}, chars={len(t)}")
                return t
        except Exception as e:
            print(f"fail {enc}: {e}")
    return raw.decode("utf-8", errors="replace")


def load_audit(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["conf"] = float(row["conf"])
                row["prob"] = float(row["prob"])
            except Exception:
                continue
            rows.append(row)
    return rows


def summarize(rows, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if not rows:
        print("no rows")
        return
    confs = [r["conf"] for r in rows]
    print(
        f"n={len(confs)} min={min(confs):.4f} max={max(confs):.4f} "
        f"mean={statistics.mean(confs):.4f} median={statistics.median(confs):.4f}"
    )
    for thr in (0.77, 0.70, 0.65, 0.60):
        n = sum(1 for c in confs if c >= thr)
        pct = 100.0 * n / len(confs)
        print(f"  >= {thr:.2f}: {n} ({pct:.1f}%)")
    sigs = collections.Counter(r.get("signal") for r in rows)
    print("signals:", dict(sigs))

    cons_ok = 0
    stat_ok = 0
    for r in rows:
        g = r.get("gates") or ""
        if "consensus_ok\": true" in g or '"consensus_ok": true' in g:
            cons_ok += 1
        if "stat_ok\": true" in g or '"stat_ok": true' in g:
            stat_ok += 1
    print(f"stat_ok true: {stat_ok}/{len(rows)}")
    print(f"consensus_ok true: {cons_ok}/{len(rows)}")

    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("symbol", "?")].append(r["conf"])
    print("\nby symbol:")
    for s, cs in sorted(by.items()):
        n77 = sum(1 for c in cs if c >= 0.77)
        print(
            f"  {s:10} n={len(cs):4} min={min(cs):.3f} max={max(cs):.3f} "
            f"mean={statistics.mean(cs):.3f} >=0.77={n77}"
        )

    print("\nhighest 8:")
    for r in sorted(rows, key=lambda x: -x["conf"])[:8]:
        print(
            f"  {r.get('ts','')} {r.get('symbol')} conf={r['conf']:.4f} "
            f"prob={r['prob']:.4f} sig={r.get('signal')} groups={r.get('groups')}"
        )


def main():
    all_rows = []
    for p in AUDITS:
        rows = load_audit(p)
        summarize(rows, f"AUDIT {p.name}")
        all_rows.extend(rows)
    summarize(all_rows, "AUDIT BOTH DAYS")

    text = decode_log(LOG)
    blobs = re.findall(r"\{[^{}]*\"confidence\"[^{}]*\}", text)
    recs = []
    for b in blobs:
        try:
            recs.append(json.loads(b))
        except Exception:
            pass
    print("\nEA log json blobs", len(blobs), "parsed", len(recs))
    if recs:
        rows = []
        mp_all = collections.defaultdict(list)
        pats = collections.Counter()
        for d in recs:
            rows.append(
                {
                    "conf": float(d.get("confidence") or 0),
                    "prob": float(d.get("probability") or 0.5),
                    "signal": d.get("signal"),
                    "symbol": d.get("symbol"),
                    "ts": "",
                    "groups": str(d.get("group_signals")),
                    "gates": json.dumps(d.get("gate_info") or {}),
                }
            )
            pats[str(d.get("group_signals"))] += 1
            for k, v in (d.get("model_probs") or {}).items():
                if v is not None:
                    mp_all[k].append(float(v))
        summarize(rows, "EA LOG 20260811 parsed JSON")
        print("\nmodel prob ranges:")
        for k, vs in sorted(mp_all.items()):
            print(
                f"  {k:16} n={len(vs):4} min={min(vs):.3f} max={max(vs):.3f} "
                f"mean={statistics.mean(vs):.3f} nunique={len(set(round(v,4) for v in vs))}"
            )
        print("\ngroup patterns:")
        for p, n in pats.most_common(12):
            print(f"  {n:4} {p}")

    rej = re.findall(
        r"Signal (\w+) rejected for (\w+): conf ([0-9.]+) outside",
        text,
    )
    print("\nEA reject lines", len(rej), collections.Counter((a, b) for a, b, _ in rej))
    passed = len(re.findall(r'"gate_passed":true', text))
    print("gate_passed true in log:", passed)


if __name__ == "__main__":
    main()
