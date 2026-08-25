#!/usr/bin/env python3
"""
mt5_deployment_status.py
Check and verify MT5 Expert Advisor deployment status
"""

import os
import json
from pathlib import Path

EXPERTS_ROOT = "/Users/localhugo/Desktop/FXJEFE_Project/data/mt5_experts"

def print_status():
    print("\n" + "="*80)
    print("🎯 MT5 EXPERT ADVISORS DEPLOYMENT STATUS - macOS M1 Tahoe 26 via Wine")
    print("="*80)

    # Check folders
    folders = {
        "advisors": os.path.join(EXPERTS_ROOT, "advisors"),
        "includes": os.path.join(EXPERTS_ROOT, "includes"),
        "indicators": os.path.join(EXPERTS_ROOT, "indicators"),
        "models": os.path.join(EXPERTS_ROOT, "models"),
    }

    print("\n✅ FOLDER STRUCTURE:")
    for name, path in folders.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"   ✓ {name:15} → {count:3} files")
        else:
            print(f"   ✗ {name:15} → MISSING!")

    # Check manifest
    manifest_path = os.path.join(EXPERTS_ROOT, "DEPLOYMENT_MANIFEST.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"\n📋 DEPLOYMENT MANIFEST:")
        print(f"   ✓ Location: {manifest_path}")
        print(f"   ✓ Timestamp: {manifest['timestamp']}")
        print(f"   ✓ Platform: {manifest['platform']}")

    # Check README
    readme_path = os.path.join(EXPERTS_ROOT, "README.md")
    if os.path.exists(readme_path):
        print(f"\n📖 DOCUMENTATION:")
        print(f"   ✓ README: {readme_path}")

    # Check sync script
    sync_script = "/Users/localhugo/Desktop/FXJEFE_Project/sync_to_wine_mt5.sh"
    if os.path.exists(sync_script):
        print(f"\n🔄 SYNC SCRIPT:")
        print(f"   ✓ Location: {sync_script}")
        print(f"   ✓ Executable: {os.access(sync_script, os.X_OK)}")

    print("\n" + "="*80)
    print("📋 NEXT STEPS:")
    print("="*80)
    print("""
1. VERIFY WINE PATHS:
   Check if your Wine MT5 terminals exist at:
   ~/.wine/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/
   ~/.wine/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/AE2CC2E013FDE1E3CDF010AA51C60400/

2. RUN SYNC SCRIPT:
   cd /Users/localhugo/Desktop/FXJEFE_Project
   ./sync_to_wine_mt5.sh

3. RELOAD MT5:
   - Launch MT5 terminal
   - Press F5 or go to Terminal → Restart

4. CHECK NAVIGATOR:
   - Press Ctrl+N to open Navigator
   - Expert Advisors should show your .ex5 files
   - Indicators should show your indicator files

5. COMPILE NEW MQ5 FILES (Optional):
   - Open MetaEditor in MT5
   - Open .mq5 source files from:
     {advisors}
   - Compile (F5 or Compile button)
   - Compiled .ex5 files will be placed in MT5 folders

6. ATTACH EA TO CHART:
   - Right-click on chart → Expert Advisors
   - Select your EA from list
   - Configure parameters
   - Click OK to start
""".format(advisors=os.path.join(EXPERTS_ROOT, "advisors")))

    print("="*80)
    print("📚 DETAILED DOCUMENTATION:")
    print("="*80)
    print(f"""
READ THE README FOR DETAILS:
   {readme_path}

It includes:
  • Full folder structure explanation
  • Deployment methods (automated, manual, Wine direct)
  • Troubleshooting guide
  • File statistics
  • Workflow instructions
""")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    print_status()
