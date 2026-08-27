# SESSION — 2026-08-13 locallarry / Saturn / OG333

Saved for the next Grok turn and for Mac remote use.  
Canonical live scripts: `C:\Users\locallarry\Documents\`  
Orchestrator: `Documents\FXJEFE_Project\run_pipelineOG333.py`  
Do **not** overwrite `OG_SCRIPTS`, `OG_pipeline222`, `SaturnSyncMentor`, or `models\` except `models\og333_runs\`.

## People / devices

- User: **locallarry** (Windows host)
- Remote: MacBook Pro via **Windows App** over **NordVPN Meshnet**
- Setup steps: `Documents\FXJEFE_Project\SETUP.md`

## Brokers / terminals

| Role | Broker | Login | Server | Data folder |
|---|---|---|---|---|
| Primary | Vantage Raw ECN 1:400 Live 14 | 34163119 | VantageMarkets-Live 14 | `D0E8209F77C8CF37AD8BF550E51FF075` |
| Secondary | FTMO demo | 1514276072 | FTMO-Demo | `521D02ECFBE1452167237D73BD8AC5A6` |

`pipeline_activations`: `user=locallarry`, `vantage=true`, `ftmo=true`, `pattern_recognition=true`.

## AI / gates

- FastAPI `127.0.0.1:8080` + ZMQ `8081`; HTTP fallbacks 8082/8084/…
- Band **[0.77, 0.9888]**; **1.0 / ≥0.9999 invalid**
- `golden_require_consensus=false`; `og_always_singular=true`
- On multi disagreement, pick highest **in-band** single vote (OG, then M15 specialist)
- Feature/label **blocks off**: strip/filter/block/refuse/label_block false; `features_forbidden=[]`
- Training still omits look-ahead: `future_price`, `future_return`, `price_change`, `regime`

## Pipeline queue (next test)

Activate both MT5 → sync → labels/Matrix_price → **pattern_recognition** → trainers → deploy to **both** Files folders → lock_pyc.

New scripts: `mt5_activate.py`, `pattern_recognition.py`, `Matrix_price.py`, `train_xgboost.py`, `train_ensemble.py`.

## Last OG333 that already ran

38-script old list: **35 OK / 3 FAIL** (`generate_labels` missing lags, `feature_engineering` timeout, `train_models` missing lags). Those three are fixed on disk. Zoo models wrote only to `og333_runs` (00:56 13 Aug). OG `stacking_model.pkl` / `my_model.pkl` / `ensemble_model.pkl` untouched (11 Aug / 11 Apr).

## Live EA notes

- OG `FXJEFE_ALGO_AIOG12.04.133334.mq5` — **do not overwrite**
- Saturn copies on Vantage + FTMO; recompile after Align / Pattern mqh
- Set `CustomAccountSize` if equity is $0
- Allow WebRequest `http://127.0.0.1:8080`

## Pattern recognition

Python: TA-Lib CDL* or pandas engulfing/doji/hammer/star/harami.  
MQ5: `Include\FXJEFE\FXJEFE_PatternRecognition.mqh`  
Outputs: `FXJEFE_patterns.csv`, `FXJEFE_patterns_last.csv`.

## Holy / risk (standing)

Min lot / margin reserve; no max-DD flatten; winners/pennies over loser flatten unless regime/volume change; M15 trades; M5+H1 assist; MT5 only.

## Next test command

```
C:\Users\locallarry\Documents\FXJEFE_Project\venv\Scripts\python.exe -X utf8 -u C:\Users\locallarry\Documents\FXJEFE_Project\run_pipelineOG333.py --config C:\Users\locallarry\Documents\config.json
```
