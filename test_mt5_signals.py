# -*- coding: utf-8 -*-
"""
MT5 Signal Pipeline Test - Flawless Signal Processing
Tests all servers with realistic forex data and validates responses
"""

import requests
import json
import time
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

# Server endpoints
MAIN_SERVER = "http://127.0.0.1:8080"
ENSEMBLE_SERVER = "http://127.0.0.1:5561"
SENTIMENT_SERVER = "http://127.0.0.1:8081"

# Test symbols with realistic data
TEST_SYMBOLS = [
    {
        "symbol": "EURUSD",
        "price": 1.0950,
        "atr": 0.0012,
        "ema_diff": 0.0003,
        "rsi": 52.5,
        "garch_vol": 0.015,
        "macd_diff": 0.0001,
        "vwap": 1.0948,
        "price_vwap_diff": 0.0002,
        "bb_position": 0.55,
        "roc": 0.0015,
        "stochastic": 48.0,
        "cci": 25.0,
        "williams": -45.0,
        "momentum": 0.0002,
        "realized_vol": 0.015,
        "chaikin_vol": 0.012,
        "adx": 22.0,
        "rvi": 0.3,
        "obv": 150000,
        "volume_delta": 1500,
        "ad_line": 0.002,
        "vol_osc": 0.001,
        "supertrend": 1.0945,
        "hma": 1.0949,
        "ichimoku_tenkan": 1.0947,
        "sar": 1.0940,
        "dpo": 0.0002,
        "spread": 0.0001,
        "sentiment": 0.75,
        "rsi_m5": 51.0,
        "rsi_h1": 54.0,
        "macd_diff_m5": 0.00009,
        "macd_diff_h1": 0.00012,
        "atr_m5": 0.0008,
        "atr_h1": 0.0018,
        "vwap_m5": 1.0949,
        "vwap_h1": 1.0946,
        "roc_m5": 0.0012,
        "roc_h1": 0.0020,
        "stochastic_m5": 49.0,
        "stochastic_h1": 52.0,
        "cci_m5": 20.0,
        "cci_h1": 28.0
    },
    {
        "symbol": "GBPUSD",
        "price": 1.2750,
        "atr": 0.0018,
        "ema_diff": -0.0005,
        "rsi": 48.0,
        "garch_vol": 0.020,
        "macd_diff": -0.0002,
        "vwap": 1.2755,
        "price_vwap_diff": -0.0005,
        "bb_position": 0.45,
        "roc": -0.0010,
        "stochastic": 42.0,
        "cci": -15.0,
        "williams": -55.0,
        "momentum": -0.0003,
        "realized_vol": 0.018,
        "chaikin_vol": 0.015,
        "adx": 28.0,
        "rvi": -0.2,
        "obv": -120000,
        "volume_delta": -800,
        "ad_line": -0.003,
        "vol_osc": -0.002,
        "supertrend": 1.2760,
        "hma": 1.2752,
        "ichimoku_tenkan": 1.2758,
        "sar": 1.2765,
        "dpo": -0.0003,
        "spread": 0.00015,
        "sentiment": 0.65,
        "rsi_m5": 47.0,
        "rsi_h1": 49.0,
        "macd_diff_m5": -0.00015,
        "macd_diff_h1": -0.00025,
        "atr_m5": 0.0012,
        "atr_h1": 0.0025,
        "vwap_m5": 1.2753,
        "vwap_h1": 1.2760,
        "roc_m5": -0.0008,
        "roc_h1": -0.0015,
        "stochastic_m5": 44.0,
        "stochastic_h1": 40.0,
        "cci_m5": -12.0,
        "cci_h1": -18.0
    },
    {
        "symbol": "USDJPY",
        "price": 145.50,
        "atr": 0.35,
        "ema_diff": 0.08,
        "rsi": 58.0,
        "garch_vol": 0.012,
        "macd_diff": 0.05,
        "vwap": 145.40,
        "price_vwap_diff": 0.10,
        "bb_position": 0.65,
        "roc": 0.0025,
        "stochastic": 62.0,
        "cci": 35.0,
        "williams": -30.0,
        "momentum": 0.0008,
        "realized_vol": 0.011,
        "chaikin_vol": 0.010,
        "adx": 32.0,
        "rvi": 0.5,
        "obv": 200000,
        "volume_delta": 2500,
        "ad_line": 0.005,
        "vol_osc": 0.003,
        "supertrend": 145.30,
        "hma": 145.45,
        "ichimoku_tenkan": 145.35,
        "sar": 145.20,
        "dpo": 0.0005,
        "spread": 0.02,
        "sentiment": 0.82,
        "rsi_m5": 56.0,
        "rsi_h1": 60.0,
        "macd_diff_m5": 0.04,
        "macd_diff_h1": 0.06,
        "atr_m5": 0.25,
        "atr_h1": 0.50,
        "vwap_m5": 145.42,
        "vwap_h1": 145.35,
        "roc_m5": 0.0020,
        "roc_h1": 0.0030,
        "stochastic_m5": 60.0,
        "stochastic_h1": 65.0,
        "cci_m5": 32.0,
        "cci_h1": 38.0
    }
]


def print_header(text):
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}{text:^70}")
    print(f"{Fore.CYAN}{'='*70}\n")


def print_success(text):
    print(f"{Fore.GREEN}✓ {text}")


def print_error(text):
    print(f"{Fore.RED}✗ {text}")


def print_warning(text):
    print(f"{Fore.YELLOW}⚠ {text}")


def print_info(text):
    print(f"{Fore.WHITE}  {text}")


def test_server_health():
    """Test all server health endpoints"""
    print_header("SERVER HEALTH CHECK")
    
    servers = [
        ("Main Server (8080)", f"{MAIN_SERVER}/health"),
        ("Ensemble Server (5561)", f"{ENSEMBLE_SERVER}/health"),
        ("Sentiment Server (8081)", f"{SENTIMENT_SERVER}/health")
    ]
    
    all_healthy = True
    for name, url in servers:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                print_success(f"{name} — {data.get('status', 'ONLINE')}")
                if 'legacy_model' in data:
                    print_info(f"Legacy Model: {data['legacy_model']}")
                if 'xgb_api' in data:
                    print_info(f"XGB API: {data['xgb_api']}")
            else:
                print_error(f"{name} — HTTP {response.status_code}")
                all_healthy = False
        except Exception as e:
            print_error(f"{name} — {str(e)}")
            all_healthy = False
    
    return all_healthy


def test_prediction(server_url, server_name, data):
    """Test prediction endpoint"""
    try:
        response = requests.post(
            f"{server_url}/predict",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            signal = result.get('signal', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)

            # Validate response structure
            required_fields = ['signal', 'confidence']
            missing = [f for f in required_fields if f not in result]
            if missing:
                print_warning(f"{server_name} — Missing fields: {missing}")
                return None

            # Signal must be buy/sell/hold
            if signal not in ['buy', 'sell', 'hold']:
                print_error(f"{server_name} — Invalid signal: {signal}")
                return None

            # Confidence must be 0-1
            if not (0.0 <= confidence <= 1.0):
                print_error(f"{server_name} — Invalid confidence: {confidence}")
                return None

            print_success(f"{server_name} — {signal.upper()} @ {confidence:.4f}")

            # Additional info
            if 'model_used' in result:
                print_info(f"Model: {result['model_used']}")
            if 'stop_loss' in result and result['stop_loss'] > 0:
                print_info(f"Stop Loss: {result['stop_loss']:.5f}")
            if 'gate_passed' in result:
                gate_status = "PASSED" if result['gate_passed'] else "BLOCKED"
                color = Fore.GREEN if result['gate_passed'] else Fore.RED
                print(f"  {color}Gate Status: {gate_status}")

            return result
        else:
            print_error(f"{server_name} — HTTP {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
            return None

    except Exception as e:
        print_error(f"{server_name} — {str(e)}")
        return None


def test_sentiment(symbol):
    """Test sentiment endpoint"""
    try:
        response = requests.get(
            f"{SENTIMENT_SERVER}/sentiment",
            params={"symbol": symbol},
            timeout=3
        )
        
        if response.status_code == 200:
            result = response.json()
            sentiment = result.get('sentiment', 0.0)
            
            if 0.0 <= sentiment <= 1.0:
                print_success(f"Sentiment — {sentiment:.2f}")
                return sentiment
            else:
                print_error(f"Sentiment — Invalid value: {sentiment}")
                return None
        else:
            print_error(f"Sentiment — HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print_error(f"Sentiment — {str(e)}")
        return None


def validate_mt5_format(result):
    """Validate response matches MT5 EA expectations"""
    errors = []
    warnings = []
    
    # Required fields for MT5
    required = ['signal', 'confidence']
    for field in required:
        if field not in result:
            errors.append(f"Missing required field: {field}")
    
    # Optional but recommended
    recommended = ['stop_loss', 'model_used']
    for field in recommended:
        if field not in result:
            warnings.append(f"Missing recommended field: {field}")
    
    # Validate signal
    if 'signal' in result:
        if result['signal'] not in ['buy', 'sell', 'hold']:
            errors.append(f"Invalid signal value: {result['signal']}")
    
    # Validate confidence
    if 'confidence' in result:
        conf = result['confidence']
        if not isinstance(conf, (int, float)):
            errors.append(f"Confidence must be numeric: {type(conf)}")
        elif not (0.0 <= conf <= 1.0):
            errors.append(f"Confidence out of range: {conf}")
    
    return errors, warnings


def run_full_test():
    """Run comprehensive test suite"""
    print_header("FXJEFE MT5 SIGNAL PIPELINE TEST")
    print(f"{Fore.WHITE}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Health checks
    if not test_server_health():
        print_error("\nServers not healthy. Aborting tests.")
        return False
    
    # Step 2: Test each symbol
    all_passed = True
    for test_data in TEST_SYMBOLS:
        symbol = test_data['symbol']
        print_header(f"TESTING {symbol}")
        
        # Test main server
        print(f"\n{Fore.YELLOW}Main Server (XGBoost):")
        main_result = test_prediction(MAIN_SERVER, "Main", test_data)
        
        # Test ensemble server
        print(f"\n{Fore.YELLOW}Ensemble Server (Legacy + XGB):")
        ensemble_result = test_prediction(ENSEMBLE_SERVER, "Ensemble", test_data)
        
        # Test sentiment
        print(f"\n{Fore.YELLOW}Sentiment Analysis:")
        sentiment = test_sentiment(symbol)
        
        # Validate MT5 format
        if ensemble_result:
            print(f"\n{Fore.YELLOW}MT5 Format Validation:")
            errors, warnings = validate_mt5_format(ensemble_result)
            
            if errors:
                for err in errors:
                    print_error(err)
                all_passed = False
            else:
                print_success("All required fields present")
            
            if warnings:
                for warn in warnings:
                    print_warning(warn)
        
        time.sleep(0.5)  # Small delay between tests
    
    # Final summary
    print_header("TEST SUMMARY")
    if all_passed:
        print_success("ALL TESTS PASSED — Signal pipeline ready for MT5")
        print_info("You can now attach GenerateFeatures.mq5 to MT5 charts")
    else:
        print_error("SOME TESTS FAILED — Review errors above")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = run_full_test()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user")
        exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        exit(1)
