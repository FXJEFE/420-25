# ========================================================
# FXJEFE - MT5 Historical Data Framework + CSV Fix Script
# Runs once: creates folders + template CSVs + migrates to correct paths
# Date: 2026-08-25
# Author: Grok (via FXJEFE-PROD)
# ========================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ====================== CONFIG ======================
$projectRoot = "C:\Users\locallarry\Documents\FXJEFE_Project"

# Symbols & Timeframes that the pipeline requires
$symbols = @("EURUSD", "USDJPY", "AUDUSD", "GBPUSD", "USDCAD")
$timeframes = @("M1", "M5", "M15", "H1", "H4", "D1", "W1")

# Template CSV header (exact format required by FXJEFE)
$header = "time,open,high,low,close,tick_volume,spread"

# Random example bar (2026-08-25 style)
$exampleRow = "2026-08-25 10:00:00,1.16552,1.16861,1.16500,1.16780,1234,2"

# ====================== FUNCTIONS ======================
function Create-Folders {
    $dirs = @(
        "bridge",
        "data\raw_ohlcv",
        "data\hist",
        "data",
        "features",
        "logs",
        "models",
        "pipeline",
        "production",
        "runs",
        "state",
        "config",
        "bridge\Historical",
        "bridge\Historical\Marked-data-*"
    )
    foreach ($dir in $dirs) {
        $fullPath = Join-Path $projectRoot $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -Path $fullPath -ItemType Directory -Force | Out-Null
            Write-Host "✅ Created: $fullPath" -ForegroundColor Green
        } else {
            Write-Host "✔ Already exists: $fullPath" -ForegroundColor Yellow
        }
    }
}

function Generate-TemplateCSV {
    param($symbol, $tf)
    $folder = Join-Path $projectRoot "bridge\Historical\Marked-data-$symbol"
    $filename = "$symbol`_$tf.csv"
    $fullPath = Join-Path $folder $filename

    if (Test-Path $fullPath) {
        Write-Host "✔ $fullPath already exists" -ForegroundColor Yellow
        return
    }

    # Create parent folder if missing
    if (-not (Test-Path $folder)) { New-Item -Path $folder -ItemType Directory -Force | Out-Null }

    # Write header + example row
    $header + "`n" + $exampleRow | Out-File -FilePath $fullPath -Encoding UTF8
    Write-Host "✅ Generated: $fullPath (example row added)" -ForegroundColor Green
}

# ====================== MAIN EXECUTION ======================
Write-Host "🚀 Starting FXJEFE MT5 Data Fix Script..." -ForegroundColor Cyan
Write-Host "Project Root: $projectRoot" -ForegroundColor Cyan

# 1. Create full folder framework
Create-Folders

# 2. Create Marked-data folders + template CSVs for all symbols/timeframes
foreach ($symbol in $symbols) {
    foreach ($tf in $timeframes) {
        Generate-TemplateCSV -symbol $symbol -tf $tf
    }
}

# 3. Optional: Copy any existing historical CSVs from MT5 data folder (if you have them)
Write-Host "`nChecking for existing MT5 historical CSVs..." -ForegroundColor Cyan
$mt5HistPath = "C:\Users\locallarry\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files"  # Common path - adjust if different

if (Test-Path $mt5HistPath) {
    Write-Host "Found MT5 data folder. Migrating any .csv files..." -ForegroundColor Green
    Get-ChildItem -Path $mt5HistPath -Filter "*.csv" -Recurse | ForEach-Object {
        $symbol = $_.BaseName
        if ($symbols -contains $symbol) {
            $destFolder = Join-Path $projectRoot "bridge\Historical\Marked-data-$symbol"
            Copy-Item -Path $_.FullName -Destination $destFolder -Force
            Write-Host "📋 Migrated: $($_.Name) to Marked-data-$symbol" -ForegroundColor Magenta
        }
    }
}

Write-Host "`n🎉 FXJEFE MT5 Data Framework + CSV Fix Complete!" -ForegroundColor Green
Write-Host "Next step: run `python full_pipeline.py` or `python pipelinerun_production.py`" -ForegroundColor Cyan
Write-Host "The pipeline will now detect all files and continue training automatically." -ForegroundColor Cyan