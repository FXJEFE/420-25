# FXJEFE Makefile — venv name ALWAYS fxjefe
# Never overwrites OG scripts; versioned pyc only under production/pyc-v*/

.PHONY: help venv install-mac install-linux install-win verify lock compile-pyc ci-check pipeline strap

ROOT := $(shell pwd)
VENV := fxjefe
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
ifeq ($(OS),Windows_NT)
  PYTHON := $(VENV)/Scripts/python.exe
  PIP := $(VENV)/Scripts/pip.exe
endif

help:
	@echo "FXJEFE targets:"
	@echo "  make venv          create fxjefe venv"
	@echo "  make install-mac   pip install requirements_mac.txt"
	@echo "  make install-linux pip install requirements_linux.txt"
	@echo "  make install-win   pip install requirements_win.txt"
	@echo "  make verify        runtime_lock + path_resolver + feature_hash"
	@echo "  make lock          runtime_lock.py expect 200"
	@echo "  make pipeline      run full 32-stage production pipeline"
	@echo "  make compile-pyc   versioned pyc under production/pyc-v*/ (NEVER replace OG .py)"
	@echo "  make strap         secure_strap after first green FINAL"
	@echo "  make ci-check      non-destructive CI verification"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "venv ready: $(ROOT)/$(VENV)"

install-mac: venv
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PIP) install -r requirements_mac.txt

install-linux: venv
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PIP) install -r requirements_linux.txt

install-win: venv
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PIP) install -r requirements_win.txt

verify: lock
	$(PYTHON) path_resolver.py
	$(PYTHON) feature_hash.py
	$(PYTHON) signal_gate.py
	@echo "VERIFY OK"

lock:
	$(PYTHON) runtime_lock.py

pipeline:
	$(PYTHON) pipelinerun_production.py

compile-pyc:
	$(PYTHON) compile_versioned_pyc.py

strap:
	$(PYTHON) secure_strap.py

ci-check:
	$(PYTHON) -c "import json; m=json.load(open('pipeline_manifest.json')); assert m.get('skip_none') is True; assert len(m['stages'])>=30; print('stages', len(m['stages']))"
	$(PYTHON) path_resolver.py
	@echo "CI-CHECK OK"
