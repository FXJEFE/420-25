# SETUP — locallarry PC + MacBook Pro (Windows App over NordVPN Meshnet)

This PC is the **trading host**. The MacBook is the **remote keyboard**.  
Vantage is the **primary** live MT5. FTMO is the **secondary demo** MT5.  
Python / models / OG333 stay in `C:\Users\locallarry\Documents`.

---

## 1. What is already on this PC

| Piece | Where |
|---|---|
| Pipeline + AI server | `Documents\` + `Documents\FXJEFE_Project\` |
| Vantage MT5 (primary, login 34163119, Live 14) | `...\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5` |
| FTMO MT5 (secondary demo, login 1514276072) | App: `AppData\Roaming\FTMO Global Markets MT5 Terminal\terminal64.exe` · data: `...\521D02ECFBE1452167237D73BD8AC5A6\MQL5` |
| Saturn EA / Predict / H1 / GenerateFeatures | Both terminals’ `Experts\` |
| Includes | Both `Include\FXJEFE\` |
| Activations | `config.json` → `pipeline_activations.user=locallarry`, `vantage=true`, `ftmo=true` |
| Pattern recognition | `Documents\pattern_recognition.py` + `FXJEFE_PatternRecognition.mqh` |

AI server: `http://127.0.0.1:8080` (falls back to 8082/8084 if busy).  
ZMQ: `tcp://127.0.0.1:8081`.

---

## 2. NordVPN Meshnet — this Windows PC (do once)

1. Install **NordVPN** from nordvpn.com if it is not already installed.
2. Sign in with the **same Nord account** you use on the Mac.
3. Open NordVPN → **Meshnet** → turn **Meshnet On**.
4. Note this PC’s Meshnet name and IP (looks like `100.x.x.x`).  
   Earlier this LAN saw the Mac as `100.100.19.141` — the Windows box will have its **own** `100.` address.
5. In Meshnet → **Linked devices** → find the MacBook → set:
   - **Remote access**: On  
   - **File sharing**: On (optional)  
   - Routing: Off unless you know you need it
6. Windows Firewall: Nord Meshnet usually punches itself. If Windows App fails, allow inbound **TCP 3389** from the Meshnet adapter only (not the public internet).
7. Enable **Remote Desktop** on this PC:
   - Settings → System → Remote Desktop → **On**
   - Keep “Require devices to use Network Level Authentication”
   - User must be `locallarry` (already an admin)

Do **not** expose 3389 on the home router. Meshnet is the tunnel.

---

## 3. MacBook Pro — Windows App (Microsoft Remote Desktop)

1. App Store: install **Windows App** (formerly “Microsoft Remote Desktop”).
2. On the Mac, NordVPN → Meshnet **On**, same account, accept this PC if it shows as a pending device.
3. Windows App → **+** → **Add PC**
   - **PC name**: the Windows Meshnet IP (`100.x.x.x`) **or** the Meshnet device name Nord shows (often `computer-name.nord`).
   - **User account**: `locallarry` + this Windows password  
     (or Microsoft account if this PC signs in that way)
   - Display: start with “Optimise for Retina” off if it feels slow; 1920×1080 is enough for MT5.
   - Folders: you can redirect a Mac folder later; not required for trading.
4. Connect. You should see this desktop and can open both MT5 apps + a terminal.

If it cannot connect:

- Ping the Windows `100.` address from the Mac Terminal: `ping 100.x.x.x`
- Both devices Meshnet **On** and not on a guest Wi‑Fi that blocks UDP
- Remote Desktop **On** on Windows
- You accepted the device pair in Nord on **both** sides

Folder send in Meshnet is GUI-only on Windows (no CLI). For files, use Windows App disk redirect or Meshnet’s “Send” in the Nord app.

---

## 4. Both MT5 apps — first attach

**Vantage (live) and FTMO (demo) — same steps in each:**

1. Start the terminal (leave **both** running if you want both in the pipeline).
2. Tools → Options → Expert Advisors  
   - Allow algorithmic trading  
   - Allow WebRequest: `http://127.0.0.1:8080` and `http://127.0.0.1:8082`
3. MetaEditor → compile  
   `SaturnMatrixEA.mq5`, `SaturnMatrixPredict.mq5`, `SaturnMatrixPredictH1.mq5`, `SaturnMatixGenerateFeatures.mq5`
4. Charts: EA on **M15**, Predict on **M15**, PredictH1 on **H1**, GenerateFeatures on **M15**
5. If equity reads `$0`, set `CustomAccountSize` (e.g. 50 on a micro / challenge account)

Python does **not** live inside the FTMO folder. One `fxjefe_runtime.py` on this PC serves both terminals.

---

## 5. Start the stack (on this PC, or via Windows App)

```text
1. Start Vantage MT5 and/or FTMO MT5
2. Documents\FXJEFE_Project\venv\Scripts\python.exe -X utf8 Documents\fxjefe_runtime.py --watchdog
3. Wait until http://127.0.0.1:8080/health returns models_loaded >= 1
4. Full pipeline:
   python -X utf8 Documents\FXJEFE_Project\run_pipelineOG333.py --config Documents\config.json
```

`mt5_activate.py` (step 3 in the queue) probes **user + Vantage + FTMO** and writes `FXJEFE_activations.csv`.  
`pattern_recognition.py` runs after advanced features, before labels.

---

## 6. What not to do

- Do not overwrite `OG_SCRIPTS`, `OG_pipeline222`, or `models\*.pkl` outside `og333_runs`
- Do not put 3389 on the router
- Do not run OG `server-Copy.py` on port 8080
- Do not run `generate_synthetic_features.py` against live CSVs
