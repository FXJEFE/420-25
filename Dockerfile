FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FXJEFE_FEATURE_POLICY=ACCEPT_ALL_FEATURES FXJEFE_ALLOW_ALL_FEATURES=1 FXJEFE_MIN_CONFIDENCE=0.77
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN python -m venv /app/fxjefe
ENV PATH="/app/fxjefe/bin:$PATH"
COPY requirements_linux.txt /app/requirements_linux.txt
RUN pip install -U pip setuptools wheel && pip install -r /app/requirements_linux.txt || pip install numpy pandas flask joblib scikit-learn xgboost pyzmq
COPY . /app/
RUN python verify_checksums.py || true
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8080/health || exit 1
CMD ["bash", "-lc", "if [ -f ai_server_golden.py ]; then python ai_server_golden.py; else python runtime_lock.py; fi"]
