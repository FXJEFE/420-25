# Feature + encoding policy

- **UTF-8 ONLY** for `.py`, `.csv`, `.mq5`, `.mqh`, `.json`, configs
- **ACCEPT_ALL_FEATURES** — nothing forbidden/refused/stripped
- Preferred 17/28 lists are defaults for wiring only
- **Signal emit** only if model loaded AND featureset hash matches EA + server + Predict.mq5 + GenerateFeatures.mq5
- Demo = comment/label only
- Feature identity: SHA-256 of UTF-8 canonical name list (`set` or `ordered` mode)
