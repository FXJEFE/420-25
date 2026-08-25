# ai_server_golden — no onnxruntime required

If your local file does `import onnxruntime`, either:

1. Replace with repo version (Flask + joblib/xgb only), or
2. Soft-import:

```python
try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    ort = None
    HAS_ORT = False
```

Install optional: `pip install onnxruntime`

Primary path does not need ONNX.
