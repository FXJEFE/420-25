"""
mt5_data_sync.py
Copies FXJEFE_Features.csv from the MT5 Files folder into the project data folder.
MT5 writes the file via FILE_COMMON → Common/Files, and also to the terminal-specific
MQL5/Files folder.  This script checks both locations and uses whichever is newer.
"""
import os
import shutil
import time
import json
import logging
import sys

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()

os.makedirs(config['log_path'],  exist_ok=True)
os.makedirs(config['data_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'mt5_data_sync.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

FILENAME  = 'FXJEFE_Features.csv'
DEST_PATH = os.path.join(config['data_path'], FILENAME)

# MT5 writes to either terminal-specific MQL5\Files or the Common\Files folder
SOURCE_CANDIDATES = [
    os.path.join(config.get('mt5_common_path', ''), FILENAME),
    os.path.join(config.get('mt5_files_path',  ''), FILENAME),
]

def best_source():
    candidates = [(p, os.path.getmtime(p)) for p in SOURCE_CANDIDATES if os.path.exists(p)]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]

def sync_once():
    src = best_source()
    if src is None:
        logging.warning("FXJEFE_Features.csv not found in any MT5 folder. "
                        "Run GenerateFeatures.mq5 first.")
        return False
    src_mtime  = os.path.getmtime(src)
    dest_mtime = os.path.getmtime(DEST_PATH) if os.path.exists(DEST_PATH) else 0
    if src_mtime > dest_mtime:
        shutil.copy2(src, DEST_PATH)
        logging.info(f"Synced  {src}  →  {DEST_PATH}")
    else:
        logging.debug("No update needed (destination is current).")
    return True

def main():
    logging.info("mt5_data_sync daemon started.")
    while True:
        try:
            sync_once()
        except Exception as e:
            logging.error(f"Sync error: {e}")
        time.sleep(60)

if __name__ == '__main__':
    # --daemon keeps running; default is single-shot for pipeline use
    if '--daemon' in sys.argv:
        main()
    else:
        ok = sync_once()
        sys.exit(0)   # pipeline continues even if nothing to sync
