#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latency benchmarks: symbol_info_tick + copy_ticks_range."""
from __future__ import annotations
import statistics, time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

def bench_symbol_info_tick(mt5: Any, symbol: str, n: int = 200) -> Dict[str, float]:
    mt5.symbol_select(symbol, True)
    samples, none = [], 0
    for _ in range(n):
        t0 = time.perf_counter()
        tick = mt5.symbol_info_tick(symbol)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if tick is None:
            none += 1
    samples.sort()
    return {"p50_ms": statistics.median(samples), "p95_ms": samples[int(0.95 * len(samples)) - 1],
            "max_ms": max(samples), "none": float(none), "n": float(n)}

def copy_ticks_range_safe(mt5: Any, symbol: str, minutes: float = 5.0, flags: Optional[int] = None) -> Dict[str, Any]:
    flags = flags if flags is not None else mt5.COPY_TICKS_ALL
    mt5.symbol_select(symbol, True)
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(minutes=minutes)
    t0 = time.perf_counter()
    ticks = mt5.copy_ticks_range(symbol, date_from, date_to, flags)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    n = 0 if ticks is None else len(ticks)
    return {"ok": ticks is not None, "count": n, "latency_ms": dt_ms, "last_error": mt5.last_error(), "minutes": minutes}

def bench_copy_ticks(mt5: Any, symbol: str, minutes: float = 5.0, repeats: int = 5) -> Dict[str, float]:
    latencies, counts = [], []
    for _ in range(repeats):
        r = copy_ticks_range_safe(mt5, symbol, minutes=minutes)
        latencies.append(r["latency_ms"])
        counts.append(float(r["count"]))
    return {"p50_ms": statistics.median(latencies), "max_ms": max(latencies),
            "avg_count": statistics.mean(counts), "minutes": minutes, "repeats": float(repeats)}
