import yfinance as yf
import pandas as pd
import numpy as np

# --- Little Wombat Quant: Beast Mode Strategy ---
# Code Name: tqqq_beast_mode.py
# Philosophy: "Fortune Favors the Bold" (High Risk, Maximum Return)
# Constraint: Max Drawdown allowed up to 70-80%.
#
# STRATEGY LOGIC:
# 1. Base Engine: 100% TQQQ in Bull Market (Price > SMA200).
#    - No cash drag. No volatility scaling. Full exposure.
# 2. Bear Market Protocol (The "Beast" Difference):
#    - Standard strategies go 100% Cash when Price < SMA200. This is "too safe" (MDD ~15-20%) and misses the V-bottom.
#    - Beast Mode:
#      - Default: 100% Cash (Preserve capital from trend collapse).
#      - SNIPER ENTRY: If RSI(10) < 30 (Oversold in Bear), deploy 40% Capital.
#      - CAPITULATION ENTRY: If RSI(10) < 20 (Extreme Fear), deploy 80% Capital.
#      - EXIT SNIPER: When RSI(10) > 55 or Price > SMA200.
# 3. Reasoning:
#    - TQQQ's biggest drawdowns come from long, slow grinds (2000, 2008, 2022). SMA200 avoids the bulk of this.
#    - But the biggest gains come from the initial "snap back" (2020 March, 2018 Dec).
#    - By "catching the knife" deeply only at RSI extremes, we accept high volatility (MDD 50-70%) to compound returns faster than SMA followers.

class BeastWombatQuant:
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

    def calculate_signals(self):
        df = self.data.copy()
        prices = df['Adj Close']

        # Trend
        df['SMA200'] = prices.rolling(window=200).mean()
        
        # RSI (Sensitive 10-day for Sniper entries)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        self.data = df.dropna()

    def run_backtest(self):
        if self.data is None:
            self.fetch_data()
            self.calculate_signals()

        df = self.data
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        
        # Track "Sniper" state to know when to exit bear market trades
        sniper_mode = False 

        for date, row in df.iterrows():
            price = row['Adj Close']
            sma = row['SMA200']
            rsi = row['RSI']

            # --- Target Allocation Logic ---
            target_pct = 0.0

            if price > sma:
                # BULL MARKET: Full Send
                target_pct = 1.0
                sniper_mode = False
            else:
                # BEAR MARKET
                # Check for Sniper entries (Catching the knife)
                if rsi < 20:
                    # Capitulation: High conviction bounce play
                    target_pct = 0.80
                    sniper_mode = True
                elif rsi < 30:
                    # Oversold: Moderate bounce play
                    target_pct = 0.40
                    sniper_mode = True
                elif sniper_mode:
                    # We are in a trade. Check exit conditions.
                    if rsi > 55:
                        # Bounce completed, take profit, back to cash
                        target_pct = 0.0
                        sniper_mode = False
                    else:
                        # Hold the position (it might be 0.40 or 0.80 from previous days)
                        # We need to know previous allocation? 
                        # Simplified: If we are in sniper mode, we hold until RSI > 55.
                        # But what allocation? Let's stick to the current logic:
                        # If we were 80% yesterday and today RSI is 25, do we sell down to 40%?
                        # No, let's keep it simple: If RSI is between 20-30, target is 40%.
                        # If RSI < 20, target is 80%.
                        # If RSI 30-55 and sniper_mode is True, hold previous? 
                        # Implementation detail: Use a persistent 'current_target' state.
                        pass # Logic handled below via state persistence
                else:
                    # Standard Bear: Cash
                    target_pct = 0.0

            # State Persistence for Sniper Hold
            # If we are in sniper mode (meaning we bought dip), we don't sell until RSI > 55
            # So if RSI is 40, and we bought at 25, we want target_pct to remain what it was.
            # But here target_pct defaults to 0.0 or 0.4/0.8 based on CURRENT RSI.
            # Fix:
            if sniper_mode and target_pct == 0.0 and rsi <= 55:
                # We should hold the position. But how much?
                # Let's assume we hold 50% as a generic "Sniper Hold" if not explicitly specified?
                # Better: calculate based on current equity.
                # Actually, simpler logic:
                # If Bear:
                #   Entry: RSI < 30 -> 100% (Let's go harder for Beast Mode)
                #   Exit: RSI > 60
                pass 
            
            # --- REVISED BEAST LOGIC (Simpler) ---
            if price > sma:
                target_pct = 1.0
            else:
                # Bear Market
                if rsi < 30:
                    target_pct = 1.0 # Beast Mode: Buy the fear fully
                elif rsi > 60:
                    target_pct = 0.0 # Sell the greed
                else:
                    # In between 30 and 60: HOLD current position
                    # We need to calculate current exposure to know what "HOLD" means
                    curr_val = cash + shares * price
                    if curr_val > 0:
                        curr_exposure = (shares * price) / curr_val
                    else:
                        curr_exposure = 0
                    target_pct = curr_exposure # Maintain current state

            # --- Rebalance ---
            curr_val = cash + shares * price
            target_equity = curr_val * target_pct
            curr_equity = shares * price
            diff = target_equity - curr_equity

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

        print("\n👹 === BEAST MODE WOMBAT REPORT === 👹")
        print(f"📅 Period: {years:.1f} years")
        print(f"💰 Initial: ${start_val:,.0f} -> Final: ${final_val:,.0f}")
        print(f"🚀 CAGR: {cagr*100:.2f}%")
        print(f"🩸 Max Drawdown: {max_dd*100:.2f}% (Target: 70-80%)")
        print("==========================================")

if __name__ == "__main__":
    bot = BeastWombatQuant(start_date="2019-01-01")
    bot.run_backtest()
    bot.show_metrics()
