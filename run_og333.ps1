# Activate project venv and run full OG333 pipeline (no skips).
$ErrorActionPreference = "Continue"
$Root = "C:\Users\locallarry\Documents\FXJEFE_Project"
$VenvPy = Join-Path $Root "venv\Scripts\python.exe"
$Pipe = Join-Path $Root "run_pipelineOG333.py"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:FXJEFE_CONFIG = "C:\Users\locallarry\Documents\config.json"
if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating venv..."
    python -m venv (Join-Path $Root "venv")
}
& $VenvPy -m pip install -U pip
& $VenvPy -m pip install -r (Join-Path $Root "requirements-og333.txt")
& $VenvPy -X utf8 -u $Pipe --config $env:FXJEFE_CONFIG --include-optional
exit $LASTEXITCODE
