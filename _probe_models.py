from pathlib import Path
names = [
    "xgboost_model.json", "xgboost_model (1).json", "xgboost_model - Copy.json",
    "ensamble_model.pkl.json", "xgboost_best_sharpe.json",
    "ensemble_model.pkl", "ensemble_model_new.pkl", "11_feature_rf.pkl",
    "my_model.pkl", "my_model (1).pkl", "forex_model_2025.pkl",
    "EURUSD_M15_binary_xgb.json", "XAUUSD_H4_binary_xgb.json",
]
roots = [
    Path(r"C:\Users\locallarry\Documents\models"),
    Path(r"C:\Users\locallarry\Documents"),
    Path(r"C:\Users\locallarry\Documents\FXJEFE_Project"),
    Path(r"C:\Users\locallarry\Documents\FXJEFE_Project\models"),
    Path(r"C:\Users\locallarry\Documents\FXJEFE_Project\FOR_GROK_APRIL_2026_GOLDEN_BUNDLE"),
    Path(r"C:\Users\locallarry\Documents\FXJEFE_Project\FOR_GROK_APRIL_2026_GOLDEN_BUNDLE\models"),
    Path(r"C:\Users\locallarry\Documents\FXJEFE_Project\OG_SCRIPTS"),
]
print("=== valid (non-zero head) copies ===")
for name in names:
    hits = []
    for root in roots:
        p = root / name
        if not p.is_file():
            continue
        b = p.read_bytes()[:8]
        zero = (not b) or all(x == 0 for x in b)
        kind = "ZERO" if zero else ("JSON" if b[:1]==b"{" else ("PKL" if b[:1] in (b"\x80", b"c", b"(") else "OTHER"))
        hits.append(f"{kind}:{p} size={p.stat().st_size}")
    print(name)
    if not hits:
        print("  NONE")
    else:
        for h in hits:
            print(" ", h)

print("=== M15 valid xgb in Documents/models ===")
md = Path(r"C:\Users\locallarry\Documents\models")
n_ok = n_zero = 0
for p in sorted(md.glob("*_binary_xgb.json")):
    b = p.read_bytes()[:4]
    ok = b[:1]==b"{"
    if ok:
        n_ok += 1
        if "M15" in p.name:
            print(" OK", p.name, p.stat().st_size)
    else:
        n_zero += 1
        if "M15" in p.name:
            print(" ZERO", p.name, p.stat().st_size)
print(f"xgb json OK={n_ok} ZERO={n_zero}")
