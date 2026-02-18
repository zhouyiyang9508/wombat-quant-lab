# ==========================================
# 小袋熊量化实验室 - BTC Beast 3Q80 Mode
# ==========================================
# 策略逻辑：
# 1. 3Q80 核心：每周五择时定投 + 季度再平衡
# 2. Ahr999 增强：根据估值调整持仓比例 (0.45/1.2/5.0)
# 3. 熔断机制：季度跌幅 > 30% 暂停加仓
# ==========================================

import pandas as pd
import numpy as np
import datetime
import sys

class BTCBeastStrategy:
    def __init__(self, csv_path='BTC-USD.csv'):
        self.csv_path = csv_path
        self.df = None
        self.results = []
        
        # Parameters
        self.WEEKLY_DCA = 1000      # Weekly Investment
        self.WEEKLY_THRESHOLD = 0.05 # 5% Pump Limit
        self.AHR_BOTTOM = 0.45      # Accumulation Zone
        self.AHR_MID = 1.2          # Markup Zone
        self.AHR_HIGH = 3.0         # Fomo Zone
        self.CIRCUIT_BREAKER = 0.30 # 30% Quarterly Drop
        
    def load_data(self):
        print("🐨 Loading Market Data...")
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
            self.df.sort_index(inplace=True)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def calculate_indicators(self):
        print("🐨 Calculating Ahr999 Index...")
        # 1. Geometric Mean 200d
        self.df['log_price'] = np.log(self.df['Close'])
        self.df['geom_mean_200'] = np.exp(self.df['log_price'].rolling(window=200).mean())
        
        # 2. Exponential Growth Valuation
        # Formula: 10^(2.68 + 0.00057 * Days)
        genesis_date = pd.Timestamp('2009-01-03')
        self.df['days_since_genesis'] = (self.df.index - genesis_date).days
        self.df['exp_growth_val'] = 10 ** (2.68 + 0.00057 * self.df['days_since_genesis'])
        
        # 3. Ahr999 Index
        self.df['ahr999'] = (self.df['Close'] / self.df['geom_mean_200']) * \
                            (self.df['Close'] / self.df['exp_growth_val'])
        
        # Filter for backtest period
        self.df = self.df[self.df.index >= '2017-01-01'].copy()

    def run_backtest(self):
        print("🐨 Running Beast Mode Simulation...")
        cash = 0
        btc_balance = 0
        total_invested = 0
        missed_weeks = 0
        last_quarter_val = 0
        
        history = []

        for date, row in self.df.iterrows():
            price = row['Close']
            ahr = row['ahr999']
            
            # Daily Portfolio Value
            val = cash + (btc_balance * price)
            history.append({
                'Date': date, 
                'Value': val, 
                'Invested': total_invested, 
                'Ahr': ahr,
                'Cash': cash,
                'BTC': btc_balance
            })
            
            # Init tracker
            if last_quarter_val == 0: last_quarter_val = val

            # ==========================
            # 1. Weekly DCA Logic (Friday)
            # ==========================
            if date.weekday() == 4:
                # Calculate Weekly Change
                prev_date = date - datetime.timedelta(days=7)
                weekly_change = 0
                try:
                    # Find closest previous date if exact match missing
                    if prev_date in self.df.index:
                        prev_price = self.df.loc[prev_date]['Close']
                        weekly_change = (price - prev_price) / prev_price
                except: pass

                # Decision Logic
                invest_amount = self.WEEKLY_DCA
                should_buy = False
                
                # Rule A: Don't chase pumps (> 5%)
                if weekly_change < self.WEEKLY_THRESHOLD:
                    should_buy = True
                    missed_weeks = 0
                else:
                    missed_weeks += 1
                    # Patch: Force buy after 3 missed weeks
                    if missed_weeks >= 4:
                        should_buy = True
                        missed_weeks = 0
                
                # Rule B: Ahr999 Bottom Fishing Override
                if ahr < self.AHR_BOTTOM:
                    should_buy = True
                    invest_amount *= 2 # Beast Mode: Double Down
                    
                # Execution
                if should_buy:
                    total_spend = invest_amount + cash
                    if total_spend > 0:
                        btc_bought = total_spend / price
                        btc_balance += btc_bought
                        cash = 0
                    total_invested += invest_amount
                else:
                    cash += invest_amount
                    total_invested += invest_amount

            # ==========================
            # 2. Quarterly Rebalance
            # ==========================
            # Dates: 3/20, 6/18, 9/21, 12/18
            do_rebal = False
            if date.month == 3 and date.day == 20: do_rebal = True
            elif date.month == 6 and date.day == 18: do_rebal = True
            elif date.month == 9 and date.day == 21: do_rebal = True
            elif date.month == 12 and date.day == 18: do_rebal = True
            
            if do_rebal:
                # Circuit Breaker Check
                is_crash = False
                if last_quarter_val > 0:
                    if (val - last_quarter_val) / last_quarter_val < -self.CIRCUIT_BREAKER:
                        is_crash = True
                
                # Determine Target Ratio (Dynamic 9sig-style allocation)
                if ahr < self.AHR_BOTTOM:      target_ratio = 1.0  # 100% BTC
                elif ahr < self.AHR_MID:       target_ratio = 0.80 # 80% BTC
                elif ahr < self.AHR_HIGH:      target_ratio = 0.50 # 50% BTC (De-risk)
                else:                          target_ratio = 0.10 # 10% BTC (Almost out)
                
                # Execute
                total_val = val
                target_btc = total_val * target_ratio
                current_btc = btc_balance * price
                
                if current_btc > target_btc:
                    # Sell (Take Profit) - Allowed even in crash
                    sell_amt = current_btc - target_btc
                    btc_balance -= sell_amt / price
                    cash += sell_amt
                else:
                    # Buy (Rebalance in) - Blocked if Crash
                    if not is_crash:
                        buy_amt = target_btc - current_btc
                        if cash >= buy_amt:
                            cash -= buy_amt
                            btc_balance += buy_amt / price
                        else:
                            btc_balance += cash / price
                            cash = 0
                
                # Update Quarter Reference
                last_quarter_val = total_val

        self.results = pd.DataFrame(history)
        self.results.set_index('Date', inplace=True)
        return self.results, total_invested

    def generate_report(self):
        res, total_invested = self.run_backtest()
        final_val = res.iloc[-1]['Value']
        roi = (final_val - total_invested) / total_invested
        
        # Max Drawdown
        peak = res['Value'].cummax()
        dd = (res['Value'] - peak) / peak
        mdd = dd.min()
        
        # Annualized Return (CAGR) estimation
        days = (res.index[-1] - res.index[0]).days
        years = days / 365.25
        # CAGR for DCA is tricky, using simple ROI/Year approx or IRR
        # Simple CAGR of total value vs total invested is misleading for DCA
        # Let's use End Value / Total Invested
        
        print("\n" + "="*40)
        print("🐨 BTC Beast Mode (3Q80 Hybrid) Results")
        print("="*40)
        print(f"Start Date:      {res.index[0].date()}")
        print(f"End Date:        {res.index[-1].date()}")
        print(f"Total Invested:  ${total_invested:,.2f}")
        print(f"Final Value:     ${final_val:,.2f}")
        print(f"Total Return:    {roi*100:.2f}%")
        print(f"Max Drawdown:    {mdd*100:.2f}%")
        print("="*40)
        print("Key Logic Applied:")
        print(f"- Weekly Threshold: {self.WEEKLY_THRESHOLD*100}%")
        print(f"- Ahr999 Bottom:    {self.AHR_BOTTOM}")
        print(f"- Circuit Breaker:  {self.CIRCUIT_BREAKER*100}% Drop")
        print("="*40)

if __name__ == "__main__":
    bot = BTCBeastStrategy()
    bot.load_data()
    bot.calculate_indicators()
    bot.generate_report()
