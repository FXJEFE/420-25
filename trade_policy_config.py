#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frequency/accuracy/cost policy for high-probability cheap candidates."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class TradePolicy:
    min_confidence: float = 0.77
    require_consensus: bool = True
    min_n_models: int = 2
    min_seconds_between_trades: int = 300
    max_trades_per_day: int = 10
    max_open_positions: int = 1
    one_trade_per_symbol: bool = True
    max_spread_points: float = 25.0
    max_retries: int = 2
    max_margin_frac: float = 0.25
    prefer_min_volume: bool = True
    use_tick_for_entry: bool = True
    copy_ticks_minutes: float = 0.0
    rates_bars: int = 200
    only_bar_close: bool = True
    skip_if_vector_mismatches: bool = False
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

POLICY_ACCURACY = TradePolicy(min_confidence=0.80, min_seconds_between_trades=600, max_trades_per_day=6, max_retries=2, max_spread_points=20.0)
POLICY_BALANCED = TradePolicy()
POLICY_RESEARCH_HF = TradePolicy(min_seconds_between_trades=60, max_trades_per_day=30, copy_ticks_minutes=2.0, only_bar_close=False)

def select_candidate(signal: str, confidence: float, spread_points: float, policy: TradePolicy) -> Dict[str, Any]:
    if signal not in ("buy", "sell"):
        return {"take": False, "reason": "not_directional"}
    if confidence < policy.min_confidence:
        return {"take": False, "reason": "low_confidence"}
    if spread_points > policy.max_spread_points:
        return {"take": False, "reason": "spread_cost"}
    return {"take": True, "reason": "ok", "confidence": confidence, "spread": spread_points}
