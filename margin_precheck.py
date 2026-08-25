#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local margin/volume/stops pre-checks before order_check/order_send."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class PrecheckResult:
    ok: bool
    reason: str = ""
    volume: float = 0.0
    margin_est: float = 0.0
    free_margin: float = 0.0

def precheck_local(mt5: Any, symbol: str, side: str, volume: float,
                   sl: float = 0.0, price: Optional[float] = None,
                   max_margin_frac: float = 0.3) -> PrecheckResult:
    info = mt5.symbol_info(symbol)
    if info is None:
        return PrecheckResult(False, "no_symbol")
    if not mt5.symbol_select(symbol, True):
        return PrecheckResult(False, "symbol_select_failed")
    if int(getattr(info, "trade_mode", 4)) == 0:
        return PrecheckResult(False, "trade_disabled")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return PrecheckResult(False, "no_tick")
    side = side.lower()
    if side not in ("buy", "sell"):
        return PrecheckResult(False, "bad_side")
    px = price if price is not None else (tick.ask if side == "buy" else tick.bid)
    step = float(info.volume_step or 0.01) or 0.01
    vmin = float(info.volume_min or step)
    vmax = float(info.volume_max or volume)
    vol = max(vmin, min(vmax, round(volume / step) * step))
    if vol < vmin:
        return PrecheckResult(False, "volume_below_min", volume=vol)
    point = float(info.point or 1e-5)
    stops = int(getattr(info, "trade_stops_level", 0) or 0)
    if sl and stops > 0 and abs(px - sl) < stops * point:
        return PrecheckResult(False, "stops_level", volume=vol)
    acc = mt5.account_info()
    if acc is None:
        return PrecheckResult(False, "no_account", volume=vol)
    free = float(getattr(acc, "margin_free", 0) or 0)
    leverage = float(getattr(acc, "leverage", 100) or 100)
    contract = float(getattr(info, "trade_contract_size", 100000) or 100000)
    margin_est = (contract * vol * px) / max(leverage, 1.0)
    try:
        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        calc = mt5.order_calc_margin(order_type, symbol, vol, px)
        if calc is not None and calc > 0:
            margin_est = float(calc)
    except Exception:
        pass
    if free <= 0:
        return PrecheckResult(False, "no_free_margin", volume=vol, margin_est=margin_est, free_margin=free)
    if margin_est > free * max_margin_frac:
        return PrecheckResult(False, "margin_cap", volume=vol, margin_est=margin_est, free_margin=free)
    return PrecheckResult(True, "ok", volume=vol, margin_est=margin_est, free_margin=free)
