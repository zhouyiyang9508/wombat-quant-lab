import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Little Wombat Pro Quant: TQQQ Strategy ---
# Code Name: tqqq_pro_wombat.py
# Author: Little Wombat 🐨 (Pro Mode)
#
# STRATEGY PHILOSOPHY:
# 1. Trend Following (Regime Filter): SMA 200.
#    - Bull (Price > SMA200): Risk On.
#    - Bear (Price < SMA200): Risk Off (Cash/Treasuries).
# 2. Volatility Targeting (Risk Management):
#    - Instead of fixed 80/20, we target a specific annual volatility (e.g., 40%).
#    - Position Size = Target Vol / Realized Vol (capped at 100% or 120% leverage).
# 3. Dynamic Dip Buying (Alpha):
#    - Bull Market: Aggressive dip buying (RSI < 50).
#    - Bear Market: Conservative dip buying (RSI < 30) or complete freeze.
# 4. Cash Yield:
#    - Idle cash earns Risk-Free Rate (currently ~4.5% annualized).

class ProWombatQuant:
    def __init__(self, ticker="TQQQ", start_date="2010-01-01", initial_capital=10000):
        self.ticker = ticker
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.data = None
        self.results = None

    def fetch_data(self):
        print(f"🐨 Fetching data for {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        # Standardize columns
        df = df.rename(columns={'Close': 'Adj Close', 'Adj Close': 'Adj Close'})
        self.data = df[['Adj Close']].dropna()
        print(f"✅ Data fetched: {len(self.data)} rows.")

    def calculate_signals(self):
        df = self.data.copy()
        prices = df['Adj Close']

        # 1. Trend Filter (SMA 200)
        df['SMA200'] = prices.rolling(window=200).mean()
        df['Trend'] = np.where(prices > df['SMA200'], 1, 0) # 1 = Bull, 0 = Bear

        # 2. Volatility (20-day annualized)
        df['Daily_Ret'] = prices.pct_change()
        df['Vol_20'] = df['Daily_Ret'].rolling(window=20).std() * np.sqrt(252)

        # 3. RSI (14)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        self.data = df.dropna()

    def run_backtest(self, target_vol=0.40, cash_yield_pct=0.04):
        if self.data is None:
            self.fetch_data()
            self.calculate_signals()

        df = self.data
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        positions = []
        
        # Convert annual yield to daily factor
        daily_cash_yield = (1 + cash_yield_pct) ** (1/252) - 1

        for date, row in df.iterrows():
            price = row['Adj Close']
            trend = row['Trend']
            vol = row['Vol_20']
            rsi = row['RSI']

            # --- 1. Determine Target Exposure ---
            target_exposure = 0.0
            
            if trend == 1: # Bull Market
                # Volatility Targeting
                # If Vol is low (20%), we can leverage up (cap at 1.0 for now, or 1.5 if using margin)
                # If Vol is high (80%), we scale down to 0.5
                if vol > 0:
                    vol_weight = target_vol / vol
                    target_exposure = min(vol_weight, 1.0) # Cap at 100% equity
                
                # Dynamic RSI Boost
                if rsi < 45: 
                    target_exposure = min(target_exposure * 1.2, 1.0) # Boost exposure by 20% in dips
                
            else: # Bear Market
                # Defensive Mode
                target_exposure = 0.0 # Strict cut to cash
                # Optional: Small position if Deep Oversold (RSI < 25) for mean reversion bounce?
                if rsi < 25:
                    target_exposure = 0.20 # Sniper entry

            # --- 2. Rebalance ---
            # Calculate current portfolio value (before rebalance)
            current_val = cash + shares * price
            
            # Target value in TQQQ
            target_equity_val = current_val * target_exposure
            
            # Current equity value
            current_equity_val = shares * price
            
            diff = target_equity_val - current_equity_val
            
            # Transaction (simplified, no fee)
            if diff > 0: # Buy
                if cash >= diff:
                    shares += diff / price
                    cash -= diff
                else: # Buy max possible
                    shares += cash / price
                    cash = 0
            elif diff < 0: # Sell
                sell_val = abs(diff)
                shares -= sell_val / price
                cash += sell_val

            # --- 3. Accrue Interest on Cash ---
            if cash > 0:
                cash *= (1 + daily_cash_yield)

            # Log
            total_val = cash + shares * price
            portfolio_values.append(total_val)
            positions.append(target_exposure)

        self.results = pd.DataFrame({
            'Portfolio': portfolio_values,
            'Exposure': positions
        }, index=df.index)
        
        return self.results

    def show_metrics(self):
        r = self.results['Portfolio']
        exposure = self.results['Exposure']
        
        start_val = r.iloc[0]
        final_val = r.iloc[-1]
        years = (r.index[-1] - r.index[0]).days / 365.25
        
        cagr = (final_val / start_val) ** (1 / years) - 1
        
        # Max Drawdown
        peak = r.cummax()
        dd = (r - peak) / peak
        max_dd = dd.min()
        
        # Sharpe
        daily_ret = r.pct_change().dropna()
        rf_daily = 0.04 / 252
        excess = daily_ret - rf_daily
        sharpe = (excess.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        print("\n🐨 === PRO WOMBAT STRATEGY REPORT ===")
        print(f"📅 Period: {years:.1f} years")
        print(f"💰 Initial: ${start_val:,.0f} -> Final: ${final_val:,.0f}")
        print(f"🚀 CAGR: {cagr*100:.2f}%")
        print(f"📉 Max Drawdown: {max_dd*100:.2f}%")
        print(f"⚖️ Sharpe Ratio: {sharpe:.2f}")
        print(f"🛡️ Calmar Ratio: {calmar:.2f}")
        print(f"📊 Avg Exposure: {exposure.mean()*100:.1f}%")
        print("=====================================")

if __name__ == "__main__":
    bot = ProWombatQuant(initial_capital=10000)
    # Target 40% annualized volatility (aggressive but safer than raw TQQQ's ~80%)
    bot.run_backtest(target_vol=0.40, cash_yield_pct=0.04)
    bot.show_metrics()
