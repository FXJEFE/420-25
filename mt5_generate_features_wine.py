#!/usr/bin/env python3
"""
mt5_generate_features_wine.py
Wine-compatible MT5 feature generation for macOS.

Instead of using the MetaTrader5 Python package (which only works on Windows),
this script runs the MQL5 GenerateFeatures.mq5 script inside Wine MT5 terminal,
then copies the generated CSV back to the project data folder.

Usage:
    python mt5_generate_features_wine.py
"""
import os
import sys
import json
import logging
import shutil
import subprocess
from pathlib import Path

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'], exist_ok=True)
os.makedirs(config['data_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'mt5_generate_features_wine.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Wine paths
WINE_PREFIX = config.get('wine_prefix', '/Users/localhugo/.wine')
MT5_TERMINAL_ID = config['mt5_terminals'][0]['terminal_id']
MT5_EXE = f"{WINE_PREFIX}/drive_c/Program Files/MetaTrader 5/terminal64.exe"
MT5_MQL5_FILES = config['mt5_terminals'][0]['files_path']
MT5_COMMON_FILES = config['mt5_common_path']

# MQL5 script path (in Wine MT5)
MQL5_SCRIPT = "GenerateFeatures.ex5"
MQL5_SCRIPT_PATH_WINE = f"C:\\Users\\LarryLocal\\AppData\\Roaming\\MetaQuotes\\Terminal\\{MT5_TERMINAL_ID}\\MQL5\\Scripts\\{MQL5_SCRIPT}"

# Output paths
OUTPUT_CSV = os.path.join(config['data_path'], 'FXJEFE_Features.csv')


def check_wine():
    """Check if Wine is installed and accessible."""
    try:
        result = subprocess.run(['wine', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Wine detected: {result.stdout.strip()}")
            return True
        else:
            logging.error("Wine not found or not working")
            return False
    except FileNotFoundError:
        logging.error("Wine not installed. Install with: brew install wine-stable")
        return False


def check_mt5_terminal():
    """Check if MT5 terminal exists in Wine."""
    if os.path.exists(MT5_EXE):
        logging.info(f"MT5 terminal found: {MT5_EXE}")
        return True
    else:
        logging.error(f"MT5 terminal not found: {MT5_EXE}")
        logging.error("Make sure MT5 is installed in Wine")
        return False


def sync_mql5_script():
    """Copy GenerateFeatures.ex5 to MT5 Scripts folder in Wine."""
    source = os.path.join(config['mt5_experts_path'], 'GenerateFeatures.ex5')
    target_dir = f"{WINE_PREFIX}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/{MT5_TERMINAL_ID}/MQL5/Scripts"
    target = os.path.join(target_dir, MQL5_SCRIPT)
    
    os.makedirs(target_dir, exist_ok=True)
    
    if os.path.exists(source):
        shutil.copy2(source, target)
        logging.info(f"Copied {source} to {target}")
        return True
    else:
        logging.error(f"Source script not found: {source}")
        return False


def run_mql5_script():
    """Run the MQL5 script in MT5 via Wine."""
    # Note: This requires MT5 to be running with the script enabled
    # For automation, we'd need to use MT5's command-line options or a different approach
    
    logging.warning("MT5 script execution requires manual steps:")
    logging.warning("1. Start MT5 terminal via Wine")
    logging.warning("2. Open Navigator (Ctrl+N)")
    logging.warning("3. Navigate to Scripts folder")
    logging.warning("4. Double-click GenerateFeatures")
    logging.warning("5. Script will generate FXJEFE_Features.csv in Common\\Files")
    
    # Alternative: Try to start MT5 with script (if supported)
    # This is experimental and may not work with all MT5 versions
    try:
        cmd = [
            'wine',
            MT5_EXE,
            '/config:' + f"{WINE_PREFIX}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/{MT5_TERMINAL_ID}",
        ]
        logging.info(f"Starting MT5: {' '.join(cmd)}")
        # subprocess.Popen(cmd)  # Uncomment to auto-start MT5
        logging.info("MT5 start command prepared (commented out - requires manual start)")
        return True
    except Exception as e:
        logging.error(f"Failed to start MT5: {e}")
        return False


def copy_csv_from_mt5():
    """Copy generated CSV from MT5 Common/Files to project data folder."""
    sources = [
        os.path.join(MT5_COMMON_FILES, 'FXJEFE_Features.csv'),
        os.path.join(MT5_MQL5_FILES, 'FXJEFE_Features.csv'),
    ]
    
    for source in sources:
        if os.path.exists(source):
            shutil.copy2(source, OUTPUT_CSV)
            logging.info(f"Copied CSV from {source} to {OUTPUT_CSV}")
            return True
    
    logging.warning("FXJEFE_Features.csv not found in MT5 folders")
    logging.info("Run GenerateFeatures script in MT5 first")
    return False


def main():
    logging.info("mt5_generate_features_wine.py started")
    
    # Check prerequisites
    if not check_wine():
        sys.exit(1)
    
    if not check_mt5_terminal():
        logging.warning("MT5 terminal not found - will try to copy from existing CSV")
    
    # Sync MQL5 script to MT5
    if not sync_mql5_script():
        logging.warning("Could not sync MQL5 script - using existing CSV if available")
    
    # Try to copy existing CSV from MT5
    if copy_csv_from_mt5():
        logging.info("Successfully copied CSV from MT5")
        logging.info(f"Output: {OUTPUT_CSV}")
        sys.exit(0)
    
    # If CSV doesn't exist, provide instructions
    logging.info("=" * 80)
    logging.info("MANUAL STEPS REQUIRED:")
    logging.info("=" * 80)
    logging.info("1. Start MT5 terminal via Wine:")
    logging.info(f"   wine '{MT5_EXE}'")
    logging.info("")
    logging.info("2. In MT5, press Ctrl+N to open Navigator")
    logging.info("3. Navigate to Scripts → GenerateFeatures")
    logging.info("4. Double-click GenerateFeatures to run it")
    logging.info("5. Script will create FXJEFE_Features.csv in Common\\Files")
    logging.info("")
    logging.info("6. After script completes, run this script again to copy the CSV")
    logging.info("")
    logging.info("Or manually copy from:")
    logging.info(f"   {MT5_COMMON_FILES}/FXJEFE_Features.csv")
    logging.info("To:")
    logging.info(f"   {OUTPUT_CSV}")
    logging.info("=" * 80)
    
    sys.exit(1)


if __name__ == '__main__':
    main()
