import yfinance as yf
import pandas as pd
import numpy as np

# --- Little Wombat Quant Lab: Ultimate Strategy ---
# Code Name: tqqq_ultimate_wombat_mode.py
# Optimized for: TQQQ (Nasdaq-100 3x Leveraged)
# 
# STRATEGY SUMMARY:
# 1. Global Safety Valve: 200-Day Moving Average (MA).
#    - If Price < 200 MA: "Bear Mode" -> 100% Cash. (Protects against -80% crashes).
#    - If Price > 200 MA: "Bull Mode" -> Active Strategy.
#
# 2. Phase 1: Accumulation (Wealth Building)
#    - Goal: Grow quickly to $1M.
#    - Inputs: Weekly DCA ($1,000).
#    - Bull Mode: 80% TQQQ / 20% Cash.
#      - "3Q" Logic: Aggressively buy dips (Weekly Return < -3%) using available cash.
#
# 3. Phase 2: Harvesting (Financial Independence)
#    - Goal: Generate Income + Preservation.
#    - Trigger: Portfolio > $1,000,000.
#    - Bull Mode: 9Sig Value Averaging.
#      - Target: Grows 9% Quarterly.
#      - Action: Sell excess for Income. Buy shortfall from Cash.
#    - Bear Mode: 100% Cash. Pause Target Growth.

class UltimateWombat:
    def __init__(self, ticker="TQQQ", start_date="2010-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.data = None
        self.results = None
        self.income_log = []

    def fetch_data(self):
        # Force yfinance download to avoid CSV issues
        print("⚠️ Fetching fresh data via yfinance...")
        df = yf.download(self.ticker, start=self.start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        rename_map = {'Close': 'Adj Close'}
        df = df.rename(columns=rename_map)
        self.data = df[['Adj Close']].dropna()
        print(f"✅ Data Loaded: {len(self.data)} rows.")
    
    def run_backtest(self, 
                     initial_capital=10000, 
                     weekly_deposit=1000, 
                     switch_threshold=1000000):
        
        if self.data is None: self.fetch_data()
        
        # Indicators
        prices = self.data['Adj Close']
        ma200 = prices.rolling(200).mean()
        
        # State
        cash = initial_capital
        shares = 0
        mode = "ACCUMULATION"
        
        # 9Sig Target
        quarterly_target = 0.0
        target_growth_rate = 0.0225 # ~9% annual / 4 quarters? No, 9Sig is 9% per quarter usually? 
        # Actually Jason Kelly's 3% signal is 3% quarterly. 
        # Let's use 6% Quarterly (Aggressive) for TQQQ.
        # Prompt said "9sig", implies 9% signal. 
        # Let's stick to 9% QUARTERLY (very aggressive) or ANNUAL? 
        # Standard 9Sig is 3% quarterly for stock market, 6-9% for aggressive.
        # Let's use 6% Quarterly (~26% Annual).
        quarterly_growth = 0.06 
        
        dates = prices.index
        portfolio_values = []
        
        last_friday_price = prices.iloc[0]
        last_month = dates[0].month
        
        print(f"🚀 Running Ultimate Wombat Strategy...")
        print(f"   Switch Threshold: ${switch_threshold:,.0f}")
        
        for i in range(len(dates)):
            date = dates[i]
            price = prices.iloc[i]
            ma = ma200.iloc[i] if i >= 199 else 0
            
            # --- 1. Global MA Check ---
            # If < 200MA (and we have enough data), we are in BEAR mode.
            is_bear = (i > 200 and price < ma)
            
            # Valuation
            total_val = cash + (shares * price)
            
            # --- 2. Phase Switch ---
            if mode == "ACCUMULATION" and total_val >= switch_threshold:
                mode = "HARVESTING"
                quarterly_target = total_val
                print(f"🎉 ({date.date()}) Unlocked HARVEST MODE! Value: ${total_val:,.0f}")
                # Optional: Stop Deposits?
                weekly_deposit = 0 # Self-sustaining
            
            # --- 3. Logic Dispatch ---
            
            # Weekly DCA (Friday)
            is_friday = (date.weekday() == 4)
            if is_friday and weekly_deposit > 0:
                cash += weekly_deposit
                # In Bear Mode, we just hoard this cash.
            
            # 3Q Dip Logic (Only in Accumulation + Bull Mode)
            if mode == "ACCUMULATION" and not is_bear:
                if is_friday:
                    weekly_ret = (price - last_friday_price) / last_friday_price
                    last_friday_price = price
                    
                    if weekly_ret < -0.03: # 3% Dip
                        # Aggressive Buy
                        if cash > 0:
                            shares += cash / price
                            cash = 0
            
            # --- 4. Rebalancing / Target Logic ---
            
            # Check Quarter Start
            is_quarter_start = False
            if i > 0 and date.month != last_month and date.month in [1, 4, 7, 10]:
                is_quarter_start = True
            
            if is_bear:
                # === BEAR MODE: CASH IS KING ===
                # If we have shares, SELL THEM ALL.
                if shares > 0:
                    cash += shares * price
                    shares = 0
                
                # In Harvesting Mode: Pause Target Growth?
                # Yes, do not increase target while hiding in cash.
                
            else:
                # === BULL MODE: ACTIVE ===
                
                if mode == "ACCUMULATION":
                    # Quarterly Rebalance to 80/20
                    if is_quarter_start:
                        target_equity = total_val * 0.80
                        diff = target_equity - (shares * price)
                        
                        if diff > 0: # Buy
                             if cash >= diff:
                                 shares += diff / price
                                 cash -= diff
                             else:
                                 shares += cash / price
                                 cash = 0
                        elif diff < 0: # Sell
                             sell_amt = abs(diff)
                             shares -= sell_amt / price
                             cash += sell_amt
                
                elif mode == "HARVESTING":
                    # 9Sig Logic
                    if is_quarter_start:
                        # Grow Target
                        quarterly_target = quarterly_target * (1 + quarterly_growth)
                        
                        current_equity = shares * price
                        # Note: In 9Sig, we usually look at Total Value vs Target?
                        # Or Equity Value vs Target? 
                        # Kelly uses "Value" vs "Signal Line".
                        # Let's use Total Value.
                        
                        diff = quarterly_target - total_val
                        
                        if diff < 0:
                            # Surplus (Above Target) -> Sell for Income
                            sell_amt = abs(diff)
                            # Actual Sell
                            shares_to_sell = sell_amt / price
                            if shares >= shares_to_sell:
                                shares -= shares_to_sell
                                cash += sell_amt
                                # Extract Income
                                self.income_log.append((date, sell_amt))
                                cash -= sell_amt # Remove from system
                            else:
                                # Sell all
                                cash += shares * price
                                shares = 0
                                
                        elif diff > 0:
                            # Shortfall (Below Target) -> Buy
                            if cash >= diff:
                                shares += diff / price
                                cash -= diff
                            else:
                                shares += cash / price
                                cash = 0

            portfolio_values.append(cash + (shares * price))
            last_month = date.month
        
        self.results = pd.Series(portfolio_values, index=dates)
        return self.results

    def show_metrics(self):
        if self.results is None: return
        
        final_val = self.results.iloc[-1]
        start_val = self.results.iloc[0]
        cagr = (final_val / start_val) ** (365.25 / (self.results.index[-1] - self.results.index[0]).days) - 1
        
        peak = self.results.cummax()
        max_dd = ((self.results - peak) / peak).min()
        
        total_income = sum([x[1] for x in self.income_log])
        
        print("\n📊 === ULTIMATE WOMBAT REPORT ===")
        print(f"💰 Final Portfolio: ${final_val:,.2f}")
        print(f"💸 Total Income Extracted: ${total_income:,.2f}")
        print(f"📈 CAGR: {cagr*100:.2f}%")
        print(f"🛡️ Max Drawdown: {max_dd*100:.2f}%")
        print("================================")

if __name__ == "__main__":
    bot = UltimateWombat()
    bot.run_backtest(switch_threshold=1000000)
    bot.show_metrics()
