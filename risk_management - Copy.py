import pandas as pd
import numpy as np
import logging
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\Logs\\risk_management.log'),
        logging.StreamHandler()
    ]
)

# Load configuration
def load_config():
    config_path = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config.json: {e}")
        exit(1)

def calculate_trade_parameters(data, account_balance=10000, risk_percent=0.3):
    """Calculate lot size, SL, and TP based on ATR and risk parameters."""
    try:
        if 'atr' not in data or 'close' not in data:
            logging.error("Required columns 'atr' and 'close' not found in data")
            return None, None, None

        atr = data['atr'].iloc[-1]
        price = data['close'].iloc[-1]
        
        # Risk parameters
        risk_amount = account_balance * (risk_percent / 100)
        pip_value = 10  # Simplified pip value (adjust for actual symbol)
        sl_pips = 2 * atr / 0.0001  # Convert ATR to pips
        lot_size = risk_amount / (sl_pips * pip_value)
        
        # Round lot size to nearest valid step (e.g., 0.01)
        lot_size = round(lot_size, 2)
        lot_size = max(0.01, min(100, lot_size))  # Ensure within broker limits
        
        sl = price - (2 * atr) if data.get('signal', 'buy') == 'buy' else price + (2 * atr)
        tp = price + (4 * atr) if data.get('signal', 'buy') == 'buy' else price - (4 * atr)
        
        logging.info(f"Trade parameters: Lot Size={lot_size}, SL={sl}, TP={tp}")
        return lot_size, sl, tp
    except Exception as e:
        logging.error(f"Error calculating trade parameters: {e}")
        return None, None, None

def main():
    config = load_config()
    data_path = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    
    try:
        data = pd.read_csv(data_path)
        lot_size, sl, tp = calculate_trade_parameters(data)
        if lot_size:
            logging.info(f"Calculated: Lot Size={lot_size}, SL={sl}, TP={tp}")
    except Exception as e:
        logging.error(f"Error processing data: {e}")

if __name__ == "__main__":
    main()