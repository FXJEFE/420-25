# MATRIX.md — FXJEFE / SaturnMatrix pipeline

Owner: `locallarry` only. Do not overwrite OG models or OG scripts.

---

## 1. What this stack is

A live MT5 expert + Python OG333 pipeline + local golden AI server.

- **Trade** on **M15** only (entries, lots, exits).
- **Predict** price / regime / trend / volume on **M5** (fast) and **H1** (context).
- **Gate:** trade only if `confidence` is in **[0.77, 0.9888]**.  
  `1.0` and `≥ 0.9999` are invalid → **hold**.
- **GARCH(1,1)** on both sides: ω=`7e-6`, α=`0.08`, β=`0.88`.
- **Encoding:** Python UTF-8. EA live CSV is `FILE_ANSI`; `convert_encoding.py` turns it into UTF-8.
- **Broker:** Vantage Raw ECN (and any suffix `.r` `.s` …). Leverage auto-snaps 30/100/200/300/400/500/1000. True leverage = notional / equity. Never flatten on max DD.

---

## 2. Folders (IO)

| Role | Path |
|---|---|
| Config | `C:\Users\locallarry\Documents\config.json` |
| Live scripts (OG333 `scripts_path`) | `C:\Users\locallarry\Documents\` |
| Project | `C:\Users\locallarry\Documents\FXJEFE_Project\` |
| Venv | `FXJEFE_Project\venv\Scripts\python.exe` |
| Runner | `FXJEFE_Project\run_pipelineOG333.py` |
| Amen launcher | `Documents\SaturnSyncAmen\run_amen.ps1` |
| Data | `FXJEFE_Project\data\` |
| Historic OHLCV | `FXJEFE_Project\HISTORIC--DATA\` |
| Logs | `Documents\logs\` |
| OG models (read-only) | `Documents\models\` |
| New training writes | `Documents\models\og333_runs\` |
| Low-latency runtime | `Documents\fxjefe_runtime.py` → `http://127.0.0.1:8080` |
| Easy MQ5 copy | `Documents\SaturnMatrix\` |
| Live MT5 (Vantage) | `%APPDATA%\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\` |

**OG vaults — never overwrite:**  
`Documents\OG_SCRIPTS`, `FXJEFE_Project\OG_SCRIPTS`, `Documents\SaturnSyncMentor`, `FXJEFE_Project\SaturnSyncMentor`.

**CSV fan-out (same names):**  
`data\`, project root, Documents, `MQL5\Files`, Common Files, `HISTORIC--DATA\`, `HISTORIC--DATA\pipeline\`.

Primary feature file: `FXJEFE_Project\data\FXJEFE_Features.csv`.

---

## 3. Timeframe contract

| TF | Job |
|---|---|
| **M15** | Trading bar. EA `PositionOpen`, lots, SL/TP, feature row for `/predict` `timeframe=M15`. |
| **M5** | Live prediction of price impulse, volume ratio/surge/trend, micro RSI/EMA. Vetoes M15 if against. |
| **H1** | Slower prediction: regime, trend, volume, plus `SaturnMatrixPredictH1` / Predict.mq5 bridge. |

Pipeline `preferred_timeframe` / `mt5_timeframe` = **M15** (not H1).  
The 2026-08-11 log that says `tf=H1 bars=2000` is **stale** — current config is M15.

---

## 4. JSON schema (`fxjefe.saturn.v1`)

**Request** (EA / Predict / GenerateFeatures → server):

- `schema`, `symbol` (suffix stripped), `timeframe` (`M15` / `H1` / `M5`)
- Core 6: `price`, `atr`, `ema_diff`, `rsi`, `garch_vol`, `macd_diff`
- Then 9/28 extras: vwap, bb, roc, stoch, cci, williams, momentum, realized_vol, … sentiment
- Always: `m5_rsi`, `m5_ema_diff`, `m5_mom`, `m5_vol_ratio`, `m5_impulse`, `m5_confirm`
- Extra: `h1_signal`, `h1_conf`, `h1_age`, `h1_source`
- Lags (pipeline): `*_lag1/2/3`, `price_mom_lagN`, `price_roc_lagN`

**Response** — parse **top-level only** (never `gate_info.signal`):

```json
{ "schema": "fxjefe.saturn.v1", "signal": "buy|sell|hold", "confidence": 0.82, "gate_passed": true, "timeframe": "M15", "symbol": "AUDUSD" }
```

MQL5: `Include\FXJEFE\FXJEFE_SaturnSchema.mqh` (`Saturn_ParsePredictReply`).  
Python: `Documents\fxjefe_schema.py`.  
`ai_server.py` tags responses in `after_request`.

---

## 5. Models

### 5.1 OG (do not overwrite)

Under `Documents\models\`: xgb_6, ensemble 9-feat, rf_28, per-symbol `*_M15_binary_xgb.json`.  
Golden server: `Documents\ai_server.py` → `ai_server_golden789.py`.  
Groups: xgb-6 / 9-feat / full / symbol specialist. Consensus + stat gate.

### 5.2 Zoo (new runs only)

`train_model_zoo.py` writes **only** `models\og333_runs\`:

XGBoost, LightGBM, HistGB, GB, MLP, LSTM (torch), HMM, LTDM.  
Needs `garch_vol` + lagged columns when present. GridSearchCV on boosters.

### 5.3 Feature math

1. TA-Lib (if installed)  
2. `pandas_ta` (not on Python 3.14 — skipped)  
3. **pandas TA fallback** — `Documents\fxjefe_lags.py`

GARCH always computed in `mt5_data_sync.py` and in the EA.

---

## 6. Pipeline (OG333)

Launcher: `powershell -File C:\Users\locallarry\Documents\SaturnSyncAmen\run_amen.ps1`

1. Ensures `127.0.0.1:8080` runtime (Waitress, 8 threads, watchdog).  
2. Runs **all** scripts. `skip_training_in_pipeline` must stay **false**.  
3. Optional: `adjust_headers.py`, `analyze_outcomes.py`.

Order (38 with optional):

`fill_all_csvs` → `create_structure` → `mt5_data_sync` → `fix_csv` → `fix_csv_encoding` → `convert_encoding` → `generate_new_csv` → `process_trades` → `merge_datasets` → `generate_labels` → `feature_engineering` → `clean_training_data` → `generate_training_data` → `Load_and_Process` → `validate_data` → `train_models` → `train_model_zoo` → `ensemble_predictions` → `generate_signals_with_xgboost` → `get_lstm_prediction` → `fxjefe_xgboost_api` → `check_integrity` → `check_labels` → `log_summary` → `parse_log_to_csv` → `risk_management` → `signal_processor` → `mt5_signal_script` → `update_database` → `update_scripts` → `test_encoding` → `test_regex` → `test_server` → `waitress server.py` (health only) → `logging_utils` → `verify_m15_align` → optionals.

Missing script or empty required output = **FAIL**. Do not skip silently.

---

## 7. MT5 Experts (SaturnMatrix)

Compile **this terminal’s** `MQL5` folder (Vantage id `D0E8209F77C8CF37AD8BF550E51FF075`). Same files are copied to Pepperstone and `Documents\SaturnMatrix\`.

| File | Role |
|---|---|
| `Include\FXJEFE\FXJEFE_HFTKit.mqh` | Math 1-arg stdev, position size, ALGLIB light helpers |
| `Include\FXJEFE\FXJEFE_SaturnSchema.mqh` | Top-level JSON parse |
| `Include\FXJEFE\FXJEFE_SaturnAlign.mqh` | Safe copies, M5/H1 packs, bridges |
| `Include\FXJEFE\FXJEFE_Volume.mqh` / `FXJEFE_Regime.mqh` | Volume + regime |
| `Experts\SaturnMatrixEA.mq5` | Trade M15. M5+H1 veto. Holy min-lot. No DD flatten. Dynamic pair list. |
| `Experts\SaturnMatrixPredict.mq5` | 37-field Predict v1.20 + M5 live + schema |
| `Experts\SaturnMatrixPredictH1.mq5` | H1 `/predict` → GlobalVariable bridge |
| `Experts\SaturnMatixGenerateFeatures.mq5` | 37-col CSV writer, dynamic handles |

Compile order: HFTKit → Schema → Align → EA → Predict → GenerateFeatures.  
Allow WebRequest: `http://127.0.0.1:8080`. Error **4006 / HTTP 1001** = URL not allowed.

Attach: EA on **M15**, Predict on M5 or M15, PredictH1 on H1, GenerateFeatures on M15.

**Holy trade policy:** only 0.77+ after M5/H1; min lot; keep ~55% free margin; never discretionary-close a loser except regime/volume flip; bank pennies. `$0` equity → no lots (`CustomAccountSize`).

**OG always singular:** early-2025 OG models may **always** place a buy/sell without multi-group consensus. Ensemble still runs; if both pass `[0.77, 0.9888]`, the higher confidence wins. Never uses `og333_runs`, `my_model.pkl` (Aug 2026 rewrite), stacking, or `*_binary_*` specialists. Config: `og_always_singular=true`. Response: `og_fallback`, `og_model`, `vote_mode=og_early2025_singular`.

---

## 8. Runtime (lowest latency)

Not a cloud host. **Loopback**:

```
http://127.0.0.1:8080/predict     FastAPI (MT5 WebRequest)
http://127.0.0.1:8080/health
tcp://127.0.0.1:8081             ZeroMQ REP fallback (no WebRequest)
```

**Clocks:** `fxjefe_time.py` auto-detects device TZ (Windows → IANA, here `Europe/Oslo`) and broker TZ (Vantage tick.time vs UTC, typically UTC+3). Logs print `device | srv | utc`. Predict/health JSON includes `time`. MQ5 header sends `time_server` / `time_local` / `time_gmt` / offsets. Feature CSV `time` stays broker-native.

`fxjefe_runtime.py --ensure` / `--watchdog`. If 8080 dies mid-pipeline, `verify_m15_align` restarts it.  
The 2026-08-12 fail (`WinError 10061` at verify) was the old Flask process exiting during training.

### 5-minute Experts health ping

`Saturn_HealthPing()` in `FXJEFE_SaturnAlign.mqh` is called from EA, Predict, PredictH1, and GenerateFeatures:

- **OnInit** — immediate GET `/health`
- **OnTimer** (30s) — throttled to `HealthPingSec` default **300** (5 min) via `GetTickCount` (does **not** freeze when the market is closed)

Experts tab line:

```
HEALTH OK | expert=SaturnMatrixEA | python_alive=yes | models_loaded=18 | symbol_models=42 | server=ai_server_golden_comprehensive | status=running | gate=0.7700 | tf=M15 | ...
HEALTH OK | AI server connected, Python alive, models loaded
```

`HEALTH FAIL` **1001 / 1003** = `SOCKET_NO_CONNECTION` — Python is not listening on `127.0.0.1:8080` (URL list is already OK if you ever saw `HEALTH OK`). Start `python Documents\fxjefe_runtime.py --watchdog`.  
`4006` is `INVALID_ARRAY`, not “URL blocked”. Real allow-list errors are **4060** / **4014**.  
`HEALTH WARN` = HTTP 200 but `models_loaded < 1` (server up, models still loading).

`/health` JSON fields: `python_alive`, `models_loaded` (alias of `loaded_models`), `symbol_models`, `server`, `status`, `gate`, `preferred_timeframe`, `pid`, `python`, `uptime_sec`.

---

## 9. Python libs

**Required:** numpy, pandas, scikit-learn, joblib, scipy, xgboost, lightgbm, torch, requests, waitress, MetaTrader5.

**Optional (venv):** catboost, hmmlearn, arch (GARCH), ta, statsmodels, optuna, imbalanced-learn, onnxruntime, shap.

**Not on 3.14:** `pandas_ta` (numba), often TA-Lib wheel. Fallback is pandas TA in `fxjefe_lags.py`.

Probe: `python Documents\fxjefe_ml_stack.py`.

---

## 10. Fixes vs old logs (2026-08-11 04:41)

| Old log | Now |
|---|---|
| `skip_training_in_pipeline=True` (30 scripts, no train) | **false** — 38 scripts including train + zoo |
| `tf=H1 bars=2000` | **M15**, `history_bars=4000` |
| Server died before verify | Runtime + watchdog + verify `--ensure` |
| `MathStandardDeviation` 2-arg | 1-arg in HFTKit |
| Static `dynamicPairList[] = {…}` | `SeedPairList()` dynamic array |
| Predict parsed nested `signal` | Top-level schema parse |
| M5 omitted if pack not ok | M5 fields always sent |
| PYC only in Mentor vault | Fresh lock under SaturnSyncAmen after this run |

---

## 11. Live check (2026-08-12 night)

Python `/predict` on last historic+live rows:

- **AUDUSD sell 0.8285** — trade  
- **USDCAD sell 0.8169** — trade  
- Others hold (0.64–0.75)

Chart EA showed HOLD + HTTP 4006 until WebRequest is allowed; equity $0 blocks lots.

---

## 12. PYC lock

After a green OG333 run:

```
python Documents\lock_pyc.py
```

Writes bytecode to `Documents\SaturnSyncAmen\pyc_lock\` and `FXJEFE_Project\SaturnSyncAmen\pyc_lock\`.  
Does **not** replace OG `.py` and does **not** write into `OG_SCRIPTS` / `SaturnSyncMentor` sources.

---

## 13. How to run (locallarry)

```powershell
# 1) Runtime (leave window open)
& C:\Users\locallarry\Documents\FXJEFE_Project\venv\Scripts\python.exe -X utf8 -u C:\Users\locallarry\Documents\fxjefe_runtime.py --ensure

# 2) Full pipeline
powershell -File C:\Users\locallarry\Documents\SaturnSyncAmen\run_amen.ps1

# 3) PYC lock (after OK)
& C:\Users\locallarry\Documents\FXJEFE_Project\venv\Scripts\python.exe -X utf8 C:\Users\locallarry\Documents\lock_pyc.py
```

MT5: compile Saturn includes/experts, allow `127.0.0.1:8080`, set CustomAccountSize, attach EA on M15.
