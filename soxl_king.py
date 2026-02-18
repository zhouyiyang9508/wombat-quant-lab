# ==========================================
# 小袋熊量化实验室 - SOXL King (Semiconductor 3x)
# Code Name: soxl_king.py
# Author: Little Wombat 🐨
#
# STRATEGY PHILOSOPHY: "Riding the AI Wave"
#
# 1. 🌊 THE ASSET:
#    - SOXL (Direxion Daily Semiconductor Bull 3X Shares).
#    - Semis are the "Oil" of the 21st century (AI, Crypto, Cloud).
#    - Volatility is extreme (beta ~4-5 vs SPY).
#
# 2. 🚦 TREND FILTER (The Guardrail):
#    - SOXL can drop 80-90% in a bear market (2022).
#    - Rule: If Price > EMA 50 (Exponential MA), we are BULL.
#    - Rule: If Price < EMA 50, we are BEAR (Cash).
#    - Why EMA 50? It captures medium-term trends faster than SMA 200.
#
# 3. 🚀 RSI TURBO (The Boost):
#    - In Bull Mode, if RSI(10) < 40 (Dip), we assume it's a buying opportunity.
#    - (Since we are already 100% invested, this is just a confirmation to HOLD).
#
# ==========================================

import pandas as pd
import numpy as np
import yfinance as yf

class SOXLKing:
    def __init__(self, start_date='2012-01-01', initial_capital=10000):
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.data = None
        self.results = None

    def fetch_data(self):
        print("🐨 Fetching SOXL Data...")
        # SOXL Inception: 2010
        try:
            soxl = yf.download('SOXL', start=self.start_date, progress=False)
            
            # Handle MultiIndex
            if isinstance(soxl.columns, pd.MultiIndex):
                soxl.columns = [c[0] for c in soxl.columns]
            
            # Use Adj Close preferably
            col = 'Adj Close'
            if 'Adj Close' not in soxl.columns:
                if 'Close' in soxl.columns:
                    col = 'Close'
                else:
                    print(f"❌ SOXL columns missing Close/Adj Close: {soxl.columns}")
                    return

            self.data = soxl[[col]].rename(columns={col: 'Close'}).dropna()
            print(f"✅ Data fetched: {len(self.data)} rows.")
        except Exception as e:
            print(f"Error fetching SOXL: {e}")
            return

    def calculate_indicators(self):
        if self.data is None:
            return

        df = self.data.copy()
        prices = df['Close']
        
        # SMA 200 (Slower, fewer whipsaws)
        df['EMA50'] = prices.rolling(window=200).mean()
        
        # RSI 14
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        self.data = df.dropna()

    def run_backtest(self):
        if self.data is None:
            return

        print("🐨 Running SOXL King Simulation...")
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        
        # Logic Loop
        for date, row in self.data.iterrows():
            price = row['Close']
            ema = row['EMA50']
            
            # Signal: Trend Following
            # Bull: Price > EMA 50
            # Bear: Price < EMA 50 -> Cash
            
            if price > ema:
                # BULL
                if cash > 0:
                    shares = cash / price
                    cash = 0
            else:
                # BEAR
                if shares > 0:
                    cash = shares * price
                    shares = 0
            
            # Value
            val = cash + shares * price
            portfolio_values.append(val)
            
        self.results = pd.Series(portfolio_values, index=self.data.index)
        return self.results

    def show_metrics(self):
        r = self.results
        start_val = r.iloc[0]
        final_val = r.iloc[-1]
        years = (r.index[-1] - r.index[0]).days / 365.25
        cagr = (final_val / start_val) ** (1 / years) - 1
        
        peak = r.cummax()
        dd = (r - peak) / peak
        max_dd = dd.min()

        print("\n👑 === SOXL KING REPORT === 👑")
        print(f"📅 Period: {years:.1f} years")
        print(f"💰 Initial: ${start_val:,.0f} -> Final: ${final_val:,.0f}")
        print(f"🚀 CAGR: {cagr*100:.2f}%")
        print(f"🩸 Max Drawdown: {max_dd*100:.2f}%")
        print("==========================================")

if __name__ == "__main__":
    bot = SOXLKing()
    bot.fetch_data()
    bot.calculate_indicators()
    bot.run_backtest()
    bot.show_metrics()
