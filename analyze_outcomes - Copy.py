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

def analyze_outcomes():
    """Analyze trade outcomes and log summary statistics."""
    csv_path = os.path.join(config['data_output_path'], 'FXJEFE_trades_outcomes.csv')
    if not os.path.exists(csv_path):
        logging.error(f"Trade outcomes file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    total_trades = len(df)
    total_profit = df['profit'].sum()
    win_rate = (df['profit'] > 0).mean() * 100 if total_trades > 0 else 0
    
    logging.info(f"Trade Analysis: Total Trades = {total_trades}, Total Profit = {total_profit:.2f}, Win Rate = {win_rate:.2f}%")

if __name__ == "__main__":
    analyze_outcomes()