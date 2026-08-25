# FXJEFE Full Pipeline Alignment Plan — v2

## Intent
Zero-drift pipeline: same 29-feature vector across Predict333.mq5, GenerateFeatures, golden server, EA.

## Gates (needle-threading EDGE)
1. **Consensus** — XGB group + 9-feat group + 28-feat group must all agree (buy or sell)
2. **Confidence** — combined probability ≥ **0.77**

Any dissent or conf < 0.77 → **hold**

## Features (exact order, 29)
price, atr, ema_diff, rsi, garch_vol, macd_diff, vwap, price_vwap_diff, bb_position, roc, stochastic, cci, williams, momentum, realized_vol, chaikin_vol, adx, rvi, obv, volume_delta, ad_line, vol_osc, supertrend, hma, ichimoku_tenkan, sar, dpo, spread, sentiment

## Server
```bash
python ai_server_golden.py
curl http://127.0.0.1:8080/health
```

## MQL5 one-line edits
Predict333.mq5: `MinConfidence = 0.77`
EA: `MinAIConfidence = 0.77` and `Kelly_RecentWinRate = 0.77`

## Policy
NO feature filtering / labeling blocks / refuse. Missing keys → 0.0. Models from project root only (no retrain).
