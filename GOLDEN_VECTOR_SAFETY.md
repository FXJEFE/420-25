# Golden server — vector length + config safety

## Feature vector
- Missing keys → `0.0` (ACCEPT_ALL)
- Extras ignored
- If model `n_features_in_` differs: **pad or truncate** to expected length
- Never index past `proba_arr` bounds
- Mismatch count returned as `vector_mismatches` in `/predict`

## config.json discovery order
1. `$FXJEFE_CONFIG`
2. `~/Documents/FXJEFE_Project/config.json`
3. `./config.json`
4. Alongside `ai_server_golden.py`

## Validation
- Schema: requires `feature_policy`, `min_confidence_threshold`
- SHA256: if `checksums.json` contains `config.json`, must match
- `/health` exposes `config_path`, `config_schema_ok`, `config_sha_ok`

## Copy
```bash
curl -fsSL -o ai_server_golden.py https://raw.githubusercontent.com/FXJEFE/user_documents_fxjefe_project/main/ai_server_golden.py
```
(If raw not updated yet, use local download from mentor artifacts.)
