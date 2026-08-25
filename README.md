# FXJEFE

Production ML + MetaTrader 5 trading runtime (mentor line).

**Policy:** UTF-8 only · all features permitted · no filtering / labeling / blocking of feature names · signals only when model is loaded and featureset matches EA + server + Predict.mq5 + GenerateFeatures.mq5.

Repo: https://github.com/FXJEFE/user_documents_fxjefe_project

---

## Project root (all OS)

```text
~/Documents/FXJEFE_Project
Windows: %USERPROFILE%\\Documents\\FXJEFE_Project
```

---

## Quick setup

### 1. Clone

```bash
mkdir -p ~/Documents
cd ~/Documents
git clone https://github.com/FXJEFE/user_documents_fxjefe_project.git FXJEFE_Project
cd FXJEFE_Project
```

### 2. Python venv

**macOS**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements_mac.txt
```

**Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements_linux.txt
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv venv
.\\venv\\Scripts\\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements_win.txt
```

### 3. Environment

```bash
cp .env.example .env
```

Edit `.env` with MT5 account values locally. Never commit `.env`.

### 4. Lock runtime (expect 200)

```bash
python runtime_lock.py
```

### 5. Feature hash + signal gate smoke test

```bash
python feature_hash.py
python signal_gate.py
```

### 6. Production pipeline (optional)

```bash
python pipelinerun_production.py
```

After first green FINAL:

```bash
python secure_strap.py
```

---

## Feature policy (ALL OK)

| Rule | Value |
|------|--------|
| `feature_policy` | `ACCEPT_ALL_FEATURES` |
| Filter / block / refuse feature names | **Never** |
| Strip feature arrays | **Never** |
| Preferred 17 / 28 lists | Defaults for wiring only |
| Signal emit | Model loaded **and** featureset hash matches EA + server + Predict.mq5 + GenerateFeatures.mq5 |

---

## Encoding

UTF-8 only for `.py`, `.csv`, `.mq5`, `.mqh`, `.json`, configs.

```bash
python encoding_utf8.py --fix
python encoding_utf8.py --scan
```

---

## MT5

Live terminals (Pepperstone / Vantage / FTMO) run on **Windows** (native or VM).  
Mac/Linux: training, feature engineering, AI server.  
EA: allow WebRequest for `http://127.0.0.1:8080` (and LAN IP if split hosts).

---

## Do not commit

- `.env` (secrets)
- `venv/`
- `__pycache__/`
- `*.pkl` model binaries unless intentional
- live broker passwords

See `.gitignore`.
