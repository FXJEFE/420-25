import os
import shutil
import time
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\Logs\\mt5_data_sync.log'),
        logging.StreamHandler()
    ]
)

def load_config():
    config_path = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config.json: {e}")
        exit(1)

def sync_data(config):
    src_path = os.path.join(config['mt5_data_path'], 'FXJEFE_Features.csv')
    dst_path = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    
    try:
        if not os.path.exists(src_path):
            logging.error(f"Source file {src_path} does not exist")
            return False
        
        src_mtime = os.path.getmtime(src_path)
        dst_mtime = os.path.getmtime(dst_path) if os.path.exists(dst_path) else 0
        
        if src_mtime > dst_mtime:
            shutil.copy2(src_path, dst_path)
            logging.info(f"Synced {src_path} to {dst_path}")
        else:
            logging.debug(f"No update needed for {src_path}")
        return True
    except Exception as e:
        logging.error(f"Sync failed: {e}")
        return False

def main():
    config = load_config()
    while True:
        sync_data(config)
        time.sleep(60)

if __name__ == "__main__":
    main()