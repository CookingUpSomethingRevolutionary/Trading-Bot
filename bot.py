import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from stable_baselines3 import PPO
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
VERCEL_DASHBOARD_URL = os.environ.get('VERCEL_URL') # e.g., https://your-app.vercel.app
SYMBOL = "SPY"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
model = PPO.load("regime_flexible_ppo_model")

def fetch_live_observation():
    raw = yf.download(SYMBOL, period="1mo", interval="1d")
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    raw['SMA_Fast'] = raw['Close'].rolling(window=5).mean()
    raw['SMA_Slow'] = raw['Close'].rolling(window=15).mean()
    raw['SMA_Ratio'] = raw['SMA_Fast'] / raw['SMA_Slow']
    raw['Volatility'] = raw['Close'].pct_change().rolling(window=10).std()
    delta = raw['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    raw['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    latest = raw.iloc[-1]
    try:
        shares = float(trading_client.get_open_position(SYMBOL).qty)
    except Exception:
        shares = 0.0
    return np.array([latest['RSI'], latest['Volatility'], latest['SMA_Ratio'], shares], dtype=np.float32), latest['Close']

def post_trade_to_dashboard(action, qty, price):
    if not VERCEL_DASHBOARD_URL: return
    payload = {"action": action, "qty": qty, "price": round(float(price), 2), "symbol": SYMBOL}
    try:
        requests.post(f"{VERCEL_DASHBOARD_URL}/api/log", json=payload)
    except Exception as e:
        print(f"Telemetry logging error: {e}")

if __name__ == "__main__":
    print("Bot pipeline active in GitHub Codespaces container...")
    while True:
        try:
            obs, current_price = fetch_live_observation()
            action, _ = model.predict(obs)
            
            if action == 1:
                trading_client.submit_order(MarketOrderRequest(symbol=SYMBOL, qty=10, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                post_trade_to_dashboard("BUY", 10, current_price)
                print("Order Sent: BUY 10 Shares")
            elif action == 2:
                try:
                    pos = trading_client.get_open_position(SYMBOL)
                    if float(pos.qty) > 0:
                        trading_client.submit_order(MarketOrderRequest(symbol=SYMBOL, qty=int(pos.qty), side=OrderSide.SELL, time_in_force=TimeInForce.GTC))
                        post_trade_to_dashboard("SELL", pos.qty, current_price)
                        print("Order Sent: LIQUIDATE POSITION")
                except Exception:
                    print("Sell signal received but holding position is empty.")
            else:
                print("Action decision profile: HOLD")
        except Exception as err:
            print(f"Error in engine execution loop: {err}")
        
        time.sleep(86400) # Execute daily