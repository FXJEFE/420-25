#!/bin/bash
# Wine MT5 Synchronization Script
# Syncs organized expert advisors to MT5 terminal via Wine

set -e

EXPERTS_SRC="/Users/localhugo/Desktop/FXJEFE_Project/data/mt5_experts"
DATA_SRC="/Users/localhugo/Desktop/FXJEFE_Project/data"

# Wine terminal paths (Z: maps to macOS root)
WINE_PRIMARY_EXPERTS="${WINEPREFIX:-$HOME/.wine}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Experts"
WINE_SECONDARY_EXPERTS="${WINEPREFIX:-$HOME/.wine}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/AE2CC2E013FDE1E3CDF010AA51C60400/MQL5/Experts"

WINE_PRIMARY_INCLUDE="${WINEPREFIX:-$HOME/.wine}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Include"
WINE_SECONDARY_INCLUDE="${WINEPREFIX:-$HOME/.wine}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/AE2CC2E013FDE1E3CDF010AA51C60400/MQL5/Include"

WINE_PRIMARY_INDICATORS="${WINEPREFIX:-$HOME/.wine}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Indicators"
WINE_SECONDARY_INDICATORS="${WINEPREFIX:-$HOME/.wine}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/AE2CC2E013FDE1E3CDF010AA51C60400/MQL5/Indicators"

echo "🔄 Syncing MT5 Expert Advisors to Wine MT5 Terminal..."
echo "Platform: macOS M1 (Tahoe 26) → Wine → MT5"
echo ""

# Sync Advisors
echo "📁 Syncing Expert Advisors..."
if [ -d "$EXPERTS_SRC/advisors" ]; then
    mkdir -p "$WINE_PRIMARY_EXPERTS"
    cp -v "$EXPERTS_SRC"/advisors/*.ex5 "$WINE_PRIMARY_EXPERTS/" 2>/dev/null || true
    cp -v "$EXPERTS_SRC"/advisors/*.mq5 "$WINE_PRIMARY_EXPERTS/" 2>/dev/null || true
    
    if [ -d "$WINE_SECONDARY_EXPERTS" ]; then
        mkdir -p "$WINE_SECONDARY_EXPERTS"
        cp -v "$EXPERTS_SRC"/advisors/*.ex5 "$WINE_SECONDARY_EXPERTS/" 2>/dev/null || true
        cp -v "$EXPERTS_SRC"/advisors/*.mq5 "$WINE_SECONDARY_EXPERTS/" 2>/dev/null || true
    fi
    echo "✅ Expert Advisors synced"
fi

# Sync Include files
echo "📁 Syncing Include files..."
if [ -d "$EXPERTS_SRC/includes" ]; then
    mkdir -p "$WINE_PRIMARY_INCLUDE"
    cp -v "$EXPERTS_SRC"/includes/*.mqh "$WINE_PRIMARY_INCLUDE/" 2>/dev/null || true
    
    if [ -d "$WINE_SECONDARY_INCLUDE" ]; then
        mkdir -p "$WINE_SECONDARY_INCLUDE"
        cp -v "$EXPERTS_SRC"/includes/*.mqh "$WINE_SECONDARY_INCLUDE/" 2>/dev/null || true
    fi
    echo "✅ Include files synced"
fi

# Sync Indicators
echo "📁 Syncing Indicators..."
if [ -d "$EXPERTS_SRC/indicators" ]; then
    mkdir -p "$WINE_PRIMARY_INDICATORS"
    cp -v "$EXPERTS_SRC"/indicators/*.ex5 "$WINE_PRIMARY_INDICATORS/" 2>/dev/null || true
    cp -v "$EXPERTS_SRC"/indicators/*.mq5 "$WINE_PRIMARY_INDICATORS/" 2>/dev/null || true
    
    if [ -d "$WINE_SECONDARY_INDICATORS" ]; then
        mkdir -p "$WINE_SECONDARY_INDICATORS"
        cp -v "$EXPERTS_SRC"/indicators/*.ex5 "$WINE_SECONDARY_INDICATORS/" 2>/dev/null || true
        cp -v "$EXPERTS_SRC"/indicators/*.mq5 "$WINE_SECONDARY_INDICATORS/" 2>/dev/null || true
    fi
    echo "✅ Indicators synced"
fi

echo ""
echo "🎉 Sync complete! Expert advisors are ready in MT5."
echo ""
echo "📋 Next steps:"
echo "  1. Reload MT5 (Terminal → Restart)"
echo "  2. Check Expert Advisors in Navigator"
echo "  3. Compile any new MQ5 files in MetaEditor"
echo "  4. Attach EAs to charts to test"
