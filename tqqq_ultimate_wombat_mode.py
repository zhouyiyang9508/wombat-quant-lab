import yfinance as yf
import pandas as pd
import numpy as np

# --- Little Wombat Quant Lab: Ultimate Strategy ---
# Code Name: tqqq_ultimate_wombat_mode.py
# Optimized by: 代码熊 🐻
# Version: 2.0
#
# STRATEGY SUMMARY:
# 1. Global Safety Valve: 200-Day Moving Average (MA).
#    - If Price < 200 MA: "Bear Mode" -> 100% Cash.
# 2. Phase 1: Accumulation (Wealth Building)
#    - Bull Mode: 80% TQQQ / 20% Cash reserve.
#    - "3Q" Logic: Buy dips (Weekly Return < -3%) using cash.
#    - RSI Oversold Boost: Extra buy when RSI < 30.
#    - Volatility Scaling: Reduce equity % in high-vol regimes.
# 3. Phase 2: Harvesting (Financial Independence)
#    - 9Sig Value Averaging with bear-mode target pause.
#    - Drawdown Guard: Halt harvesting if portfolio drops >15% from peak.
#
# KEY IMPROVEMENTS IN V2:
# - Fixed weekly return bug (now always tracks last Friday price)
# - RSI(14) factor for oversold signal
# - Volatility regime scaling (ATR-based)
# - 9Sig target paused during bear markets
# - Sharpe & Sortino ratio in metrics
# - Calmar ratio in metrics
# - Better max drawdown calculation

class UltimateWombat:
    def __init__(self, ticker="TQQQ", start_date="2010-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.data = None
        self.results = None
        self.income_log = []

    def fetch_data(self):
        print("⚠️ Fetching fresh data via yfinance...")
        df = yf.download(self.ticker, start=self.start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        rename_map = {'Close': 'Adj Close'}
        df = df.rename(columns=rename_map)
        self.data = df[['Adj Close']].dropna()
        print(f"✅ Data Loaded: {len(self.data)} rows.")

    def _compute_indicators(self, prices):
        """Compute MA200, RSI(14), and ATR-based volatility regime."""
        ma200 = prices.rolling(200).mean()

        # RSI(14)
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Volatility: 20-day rolling std of daily returns (annualized)
        daily_ret = prices.pct_change()
        vol20 = daily_ret.rolling(20).std() * np.sqrt(252)

        return ma200, rsi, vol20

    def _vol_adjusted_equity_ratio(self, base_ratio, vol, vol_low=0.50, vol_high=1.20):
        """
        Scale equity ratio down in high-volatility regimes.
        Below vol_low  -> full base_ratio
        Above vol_high -> reduce to 60% of base_ratio
        Linear interpolation in between.
        """
        if pd.isna(vol) or vol <= vol_low:
            return base_ratio
        elif vol >= vol_high:
            return base_ratio * 0.60
        else:
            t = (vol - vol_low) / (vol_high - vol_low)
            return base_ratio * (1 - 0.40 * t)

    def run_backtest(self,
                     initial_capital=10000,
                     weekly_deposit=1000,
                     switch_threshold=1_000_000,
                     base_equity_ratio=0.80,
                     quarterly_growth=0.06):

        if self.data is None:
            self.fetch_data()

        prices = self.data['Adj Close']
        ma200, rsi, vol20 = self._compute_indicators(prices)

        cash = initial_capital
        shares = 0
        mode = "ACCUMULATION"
        quarterly_target = 0.0

        dates = prices.index
        portfolio_values = []
        daily_returns_list = []

        last_friday_price = prices.iloc[0]
        last_month = dates[0].month
        all_time_high = initial_capital

        for i in range(len(dates)):
            date = dates[i]
            price = prices.iloc[i]
            ma = ma200.iloc[i]
            r = rsi.iloc[i]
            v = vol20.iloc[i]

            is_bear = (i >= 200 and not pd.isna(ma) and price < ma)
            total_val = cash + shares * price

            # Update ATH
            if total_val > all_time_high:
                all_time_high = total_val

            # --- Phase Switch ---
            if mode == "ACCUMULATION" and total_val >= switch_threshold:
                mode = "HARVESTING"
                quarterly_target = total_val
                weekly_deposit = 0
                print(f"🎉 ({date.date()}) HARVEST MODE! Value: ${total_val:,.0f}")

            is_friday = (date.weekday() == 4)

            # --- Weekly DCA ---
            if is_friday and weekly_deposit > 0:
                cash += weekly_deposit

            # --- Quarter boundary ---
            is_quarter_start = (
                i > 0
                and date.month != last_month
                and date.month in [1, 4, 7, 10]
            )

            if is_bear:
                # ==== BEAR MODE ====
                if shares > 0:
                    cash += shares * price
                    shares = 0
                # Harvesting: do NOT advance quarterly target

            else:
                # ==== BULL MODE ====
                eq_ratio = self._vol_adjusted_equity_ratio(base_equity_ratio, v)

                if mode == "ACCUMULATION":
                    # 3Q Dip Logic
                    if is_friday:
                        weekly_ret = (price - last_friday_price) / last_friday_price

                        if weekly_ret < -0.03:
                            # Standard dip buy: deploy all available cash
                            if cash > 0:
                                shares += cash / price
                                cash = 0
                        elif not pd.isna(r) and r < 30:
                            # RSI Oversold Boost: deploy 50% of cash
                            deploy = cash * 0.50
                            if deploy > 0:
                                shares += deploy / price
                                cash -= deploy

                    # Quarterly Rebalance to target ratio
                    if is_quarter_start:
                        target_equity = total_val * eq_ratio
                        diff = target_equity - shares * price
                        if diff > 0:
                            buy = min(diff, cash)
                            shares += buy / price
                            cash -= buy
                        elif diff < 0:
                            sell_val = abs(diff)
                            shares -= sell_val / price
                            cash += sell_val

                elif mode == "HARVESTING":
                    if is_quarter_start:
                        # Advance target
                        quarterly_target *= (1 + quarterly_growth)

                        # Drawdown guard: skip harvesting if >15% below ATH
                        dd_from_ath = (total_val - all_time_high) / all_time_high
                        if dd_from_ath > -0.15:
                            diff = quarterly_target - total_val
                            if diff < 0:
                                # Surplus -> sell for income
                                sell_val = abs(diff)
                                shares_to_sell = sell_val / price
                                if shares >= shares_to_sell:
                                    shares -= shares_to_sell
                                    self.income_log.append((date, sell_val))
                                    # Income extracted from portfolio
                                else:
                                    income = shares * price
                                    self.income_log.append((date, income))
                                    shares = 0
                            elif diff > 0:
                                # Shortfall -> buy
                                buy = min(diff, cash)
                                shares += buy / price
                                cash -= buy

            pv = cash + shares * price
            portfolio_values.append(pv)

            # Daily return for Sharpe/Sortino
            if i > 0:
                prev_pv = portfolio_values[-2]
                daily_returns_list.append((pv - prev_pv) / prev_pv if prev_pv > 0 else 0)

            # Always update last friday price on Fridays (BUG FIX)
            if is_friday:
                last_friday_price = price

            last_month = date.month

        self.results = pd.Series(portfolio_values, index=dates)
        self._daily_returns = pd.Series(daily_returns_list)
        return self.results

    def show_metrics(self):
        if self.results is None:
            return

        r = self.results
        final_val = r.iloc[-1]
        start_val = r.iloc[0]
        years = (r.index[-1] - r.index[0]).days / 365.25
        cagr = (final_val / start_val) ** (1 / years) - 1

        peak = r.cummax()
        dd = (r - peak) / peak
        max_dd = dd.min()

        total_income = sum(x[1] for x in self.income_log)

        dr = self._daily_returns
        rf_daily = 0.045 / 252  # 4.5% risk-free rate
        excess = dr - rf_daily
        sharpe = (excess.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
        downside = dr[dr < 0].std()
        sortino = (excess.mean() / downside) * np.sqrt(252) if downside > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        print("\n📊 === ULTIMATE WOMBAT REPORT v2.0 🐻 ===")
        print(f"💰 Final Portfolio:     ${final_val:>15,.2f}")
        print(f"💸 Total Income:        ${total_income:>15,.2f}")
        print(f"📈 CAGR:               {cagr*100:>14.2f}%")
        print(f"🛡️  Max Drawdown:       {max_dd*100:>14.2f}%")
        print(f"⚡ Sharpe Ratio:        {sharpe:>14.2f}")
        print(f"🎯 Sortino Ratio:       {sortino:>14.2f}")
        print(f"🏔️  Calmar Ratio:        {calmar:>14.2f}")
        print(f"💵 Income Events:       {len(self.income_log):>14d}")
        print("==========================================")

if __name__ == "__main__":
    bot = UltimateWombat()
    bot.run_backtest(switch_threshold=1_000_000)
    bot.show_metrics()
