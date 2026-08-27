# -*- coding: utf-8 -*-
import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\LarryLocal\\Documents\\FXJEFE_Project\\config.json'

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find config file at {CONFIG_PATH}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Config file has invalid format - {e}")
    exit(1)

# Set up logging
log_file = os.path.join(config['log_path'], 'script.log')  # Change 'script.log' to match the script's name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

import json
import logging
import os
from MetaTrader5 import MT5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:\\Users\\LarryLocal\\Documents\\FXJEFE_Project\\Logs\\risk_management.log'),
        logging.StreamHandler()
    ]
)

def load_config():
    config_path = 'C:\\Users\\LarryLocal\\Documents\\FXJEFE_Project\\config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config.json: {e}")
        raise

config = load_config()

def calculate_position_size(account_balance, risk_percent, stop_loss_pips, pip_value):
    risk_amount = account_balance * (risk_percent / 100)
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    max_lot = config['risk_management']['max_position_size']
    return min(lot_size, max_lot)

def check_drawdown(account_balance, initial_balance):
    drawdown_percent = ((initial_balance - account_balance) / initial_balance) * 100
    max_drawdown = config['risk_management']['max_drawdown_percent']
    if drawdown_percent > max_drawdown:
        logging.warning(f"Max drawdown exceeded: {drawdown_percent}% > {max_drawdown}%")
        return False
    return True

def apply_risk_management(symbol, signal, price, atr):
    try:
        if not MT5.initialize():
            logging.error("MT5 initialization failed")
            return False

        account_info = MT5.account_info()
        if not account_info:
            logging.error("Failed to get account info")
            return False

        balance = account_info.balance
        initial_balance = balance  # Adjust based on your tracking method

        if not check_drawdown(balance, initial_balance):
            logging.error("Trading halted due to excessive drawdown")
            return False

        stop_loss = price - (config['risk_management']['stop_loss_multiplier'] * atr) if signal == 'buy' else price + (config['risk_management']['stop_loss_multiplier'] * atr)
        stop_loss_pips = abs(price - stop_loss) * 10000  # For 5-digit brokers
        pip_value = 10  # Adjust based on symbol and broker

        lot_size = calculate_position_size(balance, 1.0, stop_loss_pips, pip_value)
        logging.info(f"Calculated lot size: {lot_size} for {symbol}")

        request = {
            "action": MT5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": MT5.ORDER_TYPE_BUY if signal == 'buy' else MT5.ORDER_TYPE_SELL,
            "price": price,
            "sl": stop_loss,
            "type_time": MT5.ORDER_TIME_GTC,
            "type_filling": MT5.ORDER_FILLING_IOC
        }

        result = MT5.order_send(request)
        if result.retcode != MT5.TRADE_RETCODE_DONE:
            logging.error(f"Trade failed: {result.comment}")
            return False

        logging.info(f"Trade placed: {symbol}, {signal}, Lot: {lot_size}, SL: {stop_loss}")
        return True
    except Exception as e:
        logging.error(f"Risk management error: {str(e)}")
        return False

def main():
    logging.info("Risk management started")
    # Example usage: integrate with QuantumAlgoAI.mq5 or ai_server.py
    # apply_risk_management("EURUSD.r", "buy", 1.1000, 0.0005)

if __name__ == "__main__":
    main()