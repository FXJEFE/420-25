import json
import os
with open('C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json', 'r') as f:
    config = json.load(f)
import json
import logging
import pandas as pd
import os

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set up logging
logging.basicConfig(filename=os.path.join(config['log_path'], 'pipeline.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def merge_datasets():
    """Merge features and trades datasets."""
    features_csv = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    trades_csv = os.path.join(config['data_output_path'], 'FXJEFE_trades.csv')
    output_csv = os.path.join(config['data_output_path'], 'FXJEFE_merged.csv')
    
    if not os.path.exists(features_csv) or not os.path.exists(trades_csv):
        logging.error(f"Required files missing: {features_csv}, {trades_csv}")
        return
    
    features_df = pd.read_csv(features_csv)
    trades_df = pd.read_csv(trades_csv)
    merged_df = pd.merge(features_df, trades_df, on=['time', 'symbol'], how='outer')
    merged_df.to_csv(output_csv, index=False)
    logging.info(f"Merged datasets and saved to {output_csv}")

if __name__ == "__main__":
    merge_datasets()