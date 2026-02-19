# ==========================================
# 小袋熊量化实验室 - BTC Ultimate Wombat Mode
# Code Name: btc_ultimate_wombat.py
# Author: Little Wombat 🐨
# Challenger to: Code Bear's btc_beast_3q80_mode.py
#
# STRATEGY PHILOSOPHY: "Cycle Surfing" (Sell the Top, Buy the Crash)
# Unlike Code Bear's DCA-only approach, this strategy ACTIVELY TRADES cycles.
#
# 1. 🟢 AGGRESSIVE ACCUMULATION (The Floor):
#    - When Price < 200-Week MA (WMA200), we don't just DCA. We deploy CASH RESERVES aggressively.
#    - Multiplier: 2x - 3x DCA amount.
#
# 2. 🔴 CYCLIC TOP SCALPING (The Ceiling):
#    - Code Bear only "pauses" buys at tops. We SELL.
#    - Mayer Multiple (Price / SMA200) Bands:
#      - Band 1 (> 2.4): "Overheated" -> Sell 10% of stack per week.
#      - Band 2 (> 3.5): "Bubble" -> Sell 20% of stack per week.
#      - Band 3 (> 5.0): "Euphoria" -> Sell 50% of stack per week.
#    - The goal is to exit 60-80% of the position near cycle tops (2017, 2021) to generate cash.
#
# 3. ♻️ RECYCLING CASH:
#    - Cash generated from top-selling is held in USD (risk-free yield 4%).
#    - It is redeployed ONLY when Price drops below Mayer 1.0 (SMA200) or into the WMA200 accumulation zone.
#
# ==========================================

import pandas as pd
import numpy as np
import yfinance as yf

class BTCUltimateStrategy:
    def __init__(self, start_date='2017-01-01', initial_capital=10000):
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.df = None
        self.results = None
        self.trade_log = []

    def load_data(self):
        print("🐨 Loading BTC Data for Ultimate Cycle Analysis...")
        # Get data starting earlier to build MA200
        raw = yf.download('BTC-USD', start='2014-01-01', progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        self.df = raw[['Close']].dropna()
        self.df.index = pd.to_datetime(self.df.index)
        self.df.sort_index(inplace=True)

    def calculate_indicators(self):
        # SMA 200 (Daily) -> Used for Mayer Multiple
        self.df['SMA200'] = self.df['Close'].rolling(200).mean()
        self.df['Mayer_Multiple'] = self.df['Close'] / self.df['SMA200']

        # WMA 200 (Weekly MA) -> The Cycle Floor
        # Approx 1400 days
        self.df['WMA200'] = self.df['Close'].rolling(1400).mean()

        # RSI 14 (Daily)
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Trim to start date
        self.df = self.df[self.df.index >= self.start_date].copy()

    def run_backtest(self):
        print("🐨 Running Ultimate Wombat Cycle Simulation...")
        cash = self.initial_capital
        btc_balance = 0.0
        portfolio_values = []
        
        # Weekly DCA Base Amount
        WEEKLY_DCA_BASE = 1000 
        # Total external capital added over time (to calc ROI)
        total_invested = self.initial_capital

        for date, row in self.df.iterrows():
            price = row['Close']
            mayer = row['Mayer_Multiple']
            wma200 = row['WMA200']
            
            # --- 1. WEEKLY DCA (Mondays) ---
            if date.weekday() == 0:
                # Add fresh capital
                cash += WEEKLY_DCA_BASE 
                total_invested += WEEKLY_DCA_BASE
                
                # Dynamic Accumulation Logic
                # If Price < WMA200 (Cycle Floor): 3x Buy (Deplete Cash)
                if not pd.isna(wma200) and price < wma200:
                    buy_amount = min(cash, WEEKLY_DCA_BASE * 3)
                    btc_balance += buy_amount / price
                    cash -= buy_amount
                    self.trade_log.append(f"{date.date()} 🟢 FLOOR BUY: ${buy_amount:.0f} @ ${price:.0f}")
                
                # If Price < SMA200 (Fair Value): 1.5x Buy
                elif not pd.isna(mayer) and mayer < 1.0:
                    buy_amount = min(cash, WEEKLY_DCA_BASE * 1.5)
                    btc_balance += buy_amount / price
                    cash -= buy_amount
                
                # Normal Zone (Mayer 1.0 - 2.4): Normal DCA
                elif mayer < 2.4:
                    buy_amount = min(cash, WEEKLY_DCA_BASE)
                    btc_balance += buy_amount / price
                    cash -= buy_amount
                
                # Bubble Zone (Mayer > 2.4): STOP BUYING
                else:
                    pass # Hoard cash

            # --- 2. CYCLE TOP SELLING (Every Day Check) ---
            # If we are in a bubble, we trim the position to lock in profits
            # We don't want to sell everything instantly, so we scale out
            if not pd.isna(mayer):
                sell_pct = 0.0
                
                if mayer > 5.0:       # Euphoria (2013, 2017 peak)
                    sell_pct = 0.10   # Sell 10% of holdings DAILY
                elif mayer > 3.5:     # Bubble (2021 peak approx)
                    sell_pct = 0.05   # Sell 5% of holdings DAILY
                elif mayer > 2.4:     # Overheated
                    sell_pct = 0.01   # Sell 1% of holdings DAILY
                
                if sell_pct > 0 and btc_balance > 0:
                    btc_amt_to_sell = btc_balance * sell_pct
                    proceeds = btc_amt_to_sell * price
                    btc_balance -= btc_amt_to_sell
                    cash += proceeds
                    # self.trade_log.append(f"{date.date()} 🔴 TOP SELL: {sell_pct*100}% @ ${price:.0f} (Mayer: {mayer:.2f})")

            # --- 3. RE-ENTRY (Buy the Dip with Cash Pile) ---
            # If we sold at top, we have lots of cash. When do we buy back?
            # When Mayer drops back below 1.0 (Fair Value Reset)
            if not pd.isna(mayer) and mayer < 0.8:
                # We have a cash pile? Deploy it slowly (10% of cash per week)
                # To simulate "per week", we do 2% per day
                if cash > 10000: # Only if we have significant reserves
                    buy_amt = cash * 0.02
                    btc_balance += buy_amt / price
                    cash -= buy_amt
                    # self.trade_log.append(f"{date.date()} ♻️ RECYCLE: ${buy_amt:.0f} @ ${price:.0f}")

            # Risk Free Yield on Cash (4% annualized)
            if cash > 0:
                cash *= (1 + 0.04/365)

            # Record
            total_val = cash + btc_balance * price
            portfolio_values.append({
                'Date': date,
                'Value': total_val,
                'Invested': total_invested,
                'Mayer': mayer
            })

        self.results = pd.DataFrame(portfolio_values).set_index('Date')
        return self.results, total_invested

    def generate_report(self):
        res, total_invested = self.run_backtest()
        final_val = res.iloc[-1]['Value']
        roi = (final_val - total_invested) / total_invested
        
        # CAGR
        days = (res.index[-1] - res.index[0]).days
        cagr = (final_val / 10000) ** (365/days) - 1 # Approx based on initial capital perspective? 
        # Actually for DCA, Internal Rate of Return (IRR) is better, but let's stick to simple ROI multiplier
        
        # Max Drawdown
        peak = res['Value'].cummax()
        dd = (res['Value'] - peak) / peak
        mdd = dd.min()

        print("\n" + "=" * 50)
        print("🐨 BTC ULTIMATE WOMBAT (Cycle Surfer) — Results")
        print("=" * 50)
        print(f"Start Date:          {res.index[0].date()}")
        print(f"End Date:            {res.index[-1].date()}")
        print(f"Total Invested:      ${total_invested:>12,.2f}")
        print(f"Final Value:         ${final_val:>12,.2f}")
        print(f"Net Profit:          ${(final_val-total_invested):>12,.2f}")
        print(f"Total Return (ROI):  {roi*100:>11.2f}%")
        print(f"Max Drawdown:        {mdd*100:>11.2f}%")
        print("=" * 50)
        
        return res

if __name__ == "__main__":
    bot = BTCUltimateStrategy()
    bot.load_data()
    bot.calculate_indicators()
    bot.generate_report()
