import yfinance as yf
import pandas as pd
import numpy as np

# --- Little Wombat Quant: Beast Mode 3Q80 ---
# Code Name: tqqq_beast_3q80.py
#
# STRATEGY:
# 1. Core: 3Q80 Logic (80% TQQQ / 20% Cash).
# 2. Bull Market (Price > SMA200):
#    - Maintain 80/20 baseline.
#    - TRIGGER "3Q": If Weekly Return < -3%, go BEAST MODE (100% TQQQ) for the next week.
#      (Deploy cash into the dip).
# 3. Bear Market (Price < SMA200):
#    - Standard: 0% TQQQ (Cash Safety).
#    - TRIGGER "Beast Bounce": If RSI(10) < 25 (Oversold), go 40% TQQQ.
#    - TRIGGER "Panic Buy": If Weekly Return < -10% (Crash), go 40% TQQQ.
#      (Attempt to catch V-shape bounces in bear markets, accepting high risk).
#
# GOAL: Maximize CAGR while accepting MDD ~70-80%.

class Beast3Q80Wombat:
    def __init__(self, ticker="TQQQ", start_date="2010-01-01", initial_capital=10000):
        self.ticker = ticker
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.data = None
        self.results = None

    def fetch_data(self):
        print(f"🐻 Fetching data for {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={'Close': 'Adj Close', 'Adj Close': 'Adj Close'})
        self.data = df[['Adj Close']].dropna()

    def calculate_indicators(self):
        df = self.data.copy()
        prices = df['Adj Close']

        # SMA 200
        df['SMA200'] = prices.rolling(200).mean()

        # Weekly Returns (rolling 5 days approximation or resample?)
        # Let's use exact Friday-to-Friday logic in the loop, or rolling 5-day pct_change here
        # Rolling 5-day return is easier for daily signal check
        df['Weekly_Ret'] = prices.pct_change(5)

        # RSI 10
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        self.data = df.dropna()

    def run_backtest(self):
        if self.data is None:
            self.fetch_data()
            self.calculate_indicators()

        df = self.data
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        
        # State
        beast_mode_active = False

        for i, (date, row) in enumerate(df.iterrows()):
            price = row['Adj Close']
            sma = row['SMA200']
            weekly_ret = row['Weekly_Ret']
            rsi = row['RSI']

            # Determine Target Allocation
            target_pct = 0.0

            if price > sma:
                # BULL MARKET
                if weekly_ret < -0.03:
                    # 3Q Trigger: Deploy Cash!
                    target_pct = 1.0 # Go Full Beast
                else:
                    # Standard 3Q80 Baseline (Better for dip buying power)
                    target_pct = 0.80
            else:
                # BEAR MARKET
                # Aggressive Bounce Catching
                if weekly_ret < -0.10: 
                    # Panic Crash (-10% in a week): Catch knife hard
                    target_pct = 1.0 
                elif rsi < 20:
                    # Extreme Oversold
                    target_pct = 1.0 # All in for the bounce
                elif rsi < 30:
                    # Oversold Bounce
                    target_pct = 0.80
                else:
                    # Safety
                    target_pct = 0.0

            # Rebalance Logic
            # We rebalance DAILY to stick to the target? 
            # Or only on Fridays? 
            # 3Q80 usually implies weekly checks.
            # But "Weekly Return" signal is updated daily here (rolling 5 day).
            # Let's rebalance DAILY to capture the "Beast" moves instantly.
            
            curr_val = cash + shares * price
            target_equity = curr_val * target_pct
            curr_equity = shares * price
            diff = target_equity - curr_equity

            # Transaction with slight friction buffer to avoid churn? 
            # Let's assume friction free for theoretical max, but maybe 0.1% cost?
            # Keeping it simple for now.
            
            if diff > 0:
                if cash > diff:
                    shares += diff / price
                    cash -= diff
                else:
                    shares += cash / price
                    cash = 0
            elif diff < 0:
                sell_val = abs(diff)
                shares -= sell_val / price
                cash += sell_val

            portfolio_values.append(curr_val)

        self.results = pd.Series(portfolio_values, index=df.index)
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

        print("\n🦅 === BEAST 3Q80 STRATEGY REPORT === 🦅")
        print(f"📅 Period: {years:.1f} years")
        print(f"💰 Initial: ${start_val:,.0f} -> Final: ${final_val:,.0f}")
        print(f"🚀 CAGR: {cagr*100:.2f}%")
        print(f"🩸 Max Drawdown: {max_dd*100:.2f}%")
        print("==========================================")

if __name__ == "__main__":
    bot = Beast3Q80Wombat(start_date="2010-01-01")
    # bot = Beast3Q80Wombat(start_date="2019-01-01") # Test recent volatile period
    bot.run_backtest()
    bot.show_metrics()
