import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

class MultiEnvironmentTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000.0):
        super(MultiEnvironmentTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 14
        self.portfolio_value = self.initial_balance
        return self._get_observation(), {}
        
    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([row['RSI'], row['Volatility'], row['SMA_Ratio'], float(self.shares_held)], dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        prev_portfolio_value = self.portfolio_value
        
        if action == 1 and self.balance >= current_price:
            allocated_shares = int(self.balance // current_price)
            if allocated_shares > 0:
                self.shares_held += allocated_shares
                self.balance -= (allocated_shares * current_price)
        elif action == 2 and self.shares_held > 0:
            self.balance += (self.shares_held * current_price)
            self.shares_held = 0
            
        self.portfolio_value = self.balance + (self.shares_held * current_price)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        reward = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        if action != 0: reward -= 0.0005
            
        return self._get_observation(), reward, done, False, {}

def engineer_features(df):
    df['SMA_Fast'] = df['Close'].rolling(window=5).mean()
    df['SMA_Slow'] = df['Close'].rolling(window=15).mean()
    df['SMA_Ratio'] = df['SMA_Fast'] / df['SMA_Slow']
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df.fillna(method='bfill', inplace=True)
    return df

if __name__ == "__main__":
    print("Downloading historical market training data...")
    raw_data = yf.download("SPY", start="2008-01-01", end="2026-05-01")
    if isinstance(raw_data.columns, pd.MultiIndex): raw_data.columns = raw_data.columns.get_level_values(0)
    df = engineer_features(raw_data)
    env = MultiEnvironmentTradingEnv(df)
    
    print("Optimizing Neural Network (PPO Policy)...")
    model = PPO("MlpPolicy", env, learning_rate=0.0003, verbose=1)
    model.learn(total_timesteps=30000)
    model.save("regime_flexible_ppo_model")
    print("Training complete. Asset 'regime_flexible_ppo_model.zip' saved.")