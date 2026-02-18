import pandas as pd
import numpy as np

# --- Little Wombat Quant Lab: Ultimate Strategy ---
# Code Name: tqqq_ultimate_wombat_mode.py
# Optimized by: 代码熊 🐻
# Version: 3.0
#
# STRATEGY SUMMARY:
# 1. THREE-REGIME Detection (MA50 + MA200):
#    - BEAR:             Price < MA200              → 0% equity
#    - TRANSITION:       Price > MA200, MA50 < MA200 → 50% equity (cautious)
#    - BULL_CONFIRMED:   Price > MA200, MA50 > MA200 → 80–88% equity
# 2. Momentum Score: Weighted composite of 1M (21d) + 3M (63d) returns
#    - Strong momentum in bull: boost equity ratio up to 88%
#    - Deep momentum collapse: reduce dip-buy fraction (avoid falling knives)
# 3. Tiered Dip-Buy Logic (replaces binary 3Q trigger):
#    - Weekly < -3%:  deploy 70% cash
#    - Weekly < -7%:  deploy 90% cash
#    - Weekly < -12%: deploy 100% cash (crash)
#    - RSI < 30:      deploy 60%+
#    - Momentum filter: halve deploy if deep structural downtrend
# 4. Harvesting Phase: Same 9Sig Value Averaging + smart surplus retention
#    - In strong bull momentum: keep 20% of surplus instead of selling all
#
# KEY IMPROVEMENTS IN V3 vs V2:
# - Added MA50 for three-regime detection (BEAR/TRANSITION/BULL)
# - Added momentum score (1M+3M composite) for regime boost and dip quality
# - Tiered dip-buy fractions (vs binary all-or-nothing in v2)
# - Momentum-aware profit taking in harvesting mode
# - Built-in synthetic data generator (no yfinance needed for testing)
# - Better position sizing discipline: no over-concentration in transition zone

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class UltimateWombat:
    def __init__(self, ticker="TQQQ", start_date="2010-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.data = None
        self.results = None
        self.income_log = []

    # ------------------------------------------------------------------
    # Synthetic Data (for offline testing / CI)
    # ------------------------------------------------------------------
    @staticmethod
    def generate_synthetic_data(n_years=14, seed=42):
        """Generate synthetic TQQQ-like price series for offline testing.
        Mimics 3x leveraged ETF: high drift (~35%/yr), high vol (~80%/yr),
        with occasional large drawdowns.
        """
        np.random.seed(seed)
        n_days = int(n_years * 252)
        # TQQQ realistic parameters:
        # Geometric annual return ≈ drift_log - vol²/2 = 0.52 - 0.65²/2 ≈ 30%/yr
        daily_mu    = 0.52 / 252          # log drift: ~52%/yr
        daily_sigma = 0.65 / np.sqrt(252) # log vol:   ~65%/yr (realistic 3x ETF)

        prices = [100.0]
        for i in range(n_days):
            # Simulate occasional crash days (3–4 per year)
            if np.random.random() < 0.012:
                shock = np.random.uniform(-0.25, -0.08)
            else:
                shock = np.random.normal(daily_mu, daily_sigma)
            prices.append(max(prices[-1] * np.exp(shock), 0.01))

        dates = pd.bdate_range(start="2010-01-01", periods=n_days + 1)
        df = pd.DataFrame({'Adj Close': prices}, index=dates)
        return df

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def fetch_data(self):
        if HAS_YFINANCE:
            try:
                print("⚠️ Fetching fresh data via yfinance...")
                df = yf.download(self.ticker, start=self.start_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.rename(columns={'Close': 'Adj Close'})
                self.data = df[['Adj Close']].dropna()
                print(f"✅ Data Loaded: {len(self.data)} rows.")
                return
            except Exception as e:
                print(f"⚠️ yfinance failed ({e}), falling back to synthetic data.")

        print("📊 Using synthetic TQQQ-like data...")
        self.data = self.generate_synthetic_data()
        print(f"✅ Synthetic Data: {len(self.data)} rows.")

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def _compute_indicators(self, prices):
        """Compute MA50, MA200, RSI(14), volatility, and momentum score."""
        ma200 = prices.rolling(200).mean()
        ma50  = prices.rolling(50).mean()

        # RSI(14)
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))

        # Volatility: 20-day rolling std of daily returns (annualized)
        daily_ret = prices.pct_change()
        vol20 = daily_ret.rolling(20).std() * np.sqrt(252)

        # Momentum: 1-month (21d) + 3-month (63d) weighted composite
        mom_1m = prices.pct_change(21)
        mom_3m = prices.pct_change(63)
        mom_score = 0.35 * mom_1m + 0.65 * mom_3m  # 3M weighted higher

        return ma200, ma50, rsi, vol20, mom_score

    # ------------------------------------------------------------------
    # Regime-Aware Equity Ratio
    # ------------------------------------------------------------------
    def _regime_equity_ratio(self, price, ma50, ma200, base_ratio, vol, mom,
                              vol_low=0.50, vol_high=1.20):
        """
        Three-regime system:
        - BEAR        (price < MA200)             → 0% (full cash)
        - TRANSITION  (price > MA200, MA50 < MA200) → 55% of base (cautious)
        - BULL_CONFIRMED (price > MA200, MA50 > MA200) → base ± momentum boost

        Then apply volatility scaling on top.
        """
        if pd.isna(ma200) or pd.isna(ma50):
            return base_ratio

        if price < ma200:
            return 0.0  # BEAR

        if ma50 < ma200:
            # TRANSITION: scale back sharply
            ratio = base_ratio * 0.55
        else:
            # BULL_CONFIRMED
            if not pd.isna(mom) and mom > 0.15:
                # Strong momentum → slight overweight (cap at 90%)
                ratio = min(base_ratio * 1.10, 0.90)
            else:
                ratio = base_ratio

        # Volatility scaling (applies on top of regime ratio)
        if not pd.isna(vol):
            if vol >= vol_high:
                ratio *= 0.60
            elif vol > vol_low:
                t = (vol - vol_low) / (vol_high - vol_low)
                ratio *= (1.0 - 0.40 * t)

        return ratio

    # ------------------------------------------------------------------
    # Tiered Dip-Buy Fraction
    # ------------------------------------------------------------------
    @staticmethod
    def _dip_buy_fraction(weekly_ret, rsi, mom_score):
        """
        Tiered cash deployment based on dip quality.
        Returns fraction [0, 1] of available cash to deploy.
        """
        fraction = 0.0

        # Tier 1: Weekly return dip
        if weekly_ret < -0.03:
            fraction = max(fraction, 0.70)
        if weekly_ret < -0.07:
            fraction = max(fraction, 0.90)
        if weekly_ret < -0.12:
            fraction = max(fraction, 1.00)  # Crash: all-in

        # Tier 2: RSI oversold
        if not pd.isna(rsi):
            if rsi < 30:
                fraction = max(fraction, 0.60)
            if rsi < 20:
                fraction = max(fraction, 0.85)

        # Momentum quality filter: avoid catching knives in deep structural downtrend
        if fraction > 0 and not pd.isna(mom_score) and mom_score < -0.30:
            fraction *= 0.50

        return min(fraction, 1.0)

    # ------------------------------------------------------------------
    # Backtest Engine
    # ------------------------------------------------------------------
    def run_backtest(self,
                     initial_capital=10000,
                     weekly_deposit=1000,
                     switch_threshold=1_000_000,
                     base_equity_ratio=0.80,
                     quarterly_growth=0.06):

        if self.data is None:
            self.fetch_data()

        prices = self.data['Adj Close']
        ma200, ma50, rsi, vol20, mom_score = self._compute_indicators(prices)

        cash   = initial_capital
        shares = 0
        mode   = "ACCUMULATION"
        quarterly_target = 0.0

        dates = prices.index
        portfolio_values    = []
        daily_returns_list  = []

        last_friday_price = prices.iloc[0]
        last_month        = dates[0].month
        all_time_high     = initial_capital

        for i in range(len(dates)):
            date  = dates[i]
            price = prices.iloc[i]
            ma200_v = ma200.iloc[i]
            ma50_v  = ma50.iloc[i]
            r    = rsi.iloc[i]
            v    = vol20.iloc[i]
            mom  = mom_score.iloc[i]

            is_bear  = (i >= 200 and not pd.isna(ma200_v) and price < ma200_v)
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

            # --- Weekly DCA (deposit even in bear, accumulates as cash) ---
            if is_friday and weekly_deposit > 0:
                cash += weekly_deposit

            # --- Quarter boundary (first trading day of Jan/Apr/Jul/Oct) ---
            is_quarter_start = (
                i > 0
                and date.month != last_month
                and date.month in [1, 4, 7, 10]
            )

            if is_bear:
                # ==== BEAR MODE: exit to 100% cash ====
                if shares > 0:
                    cash += shares * price
                    shares = 0

            else:
                # ==== BULL / TRANSITION MODE ====
                eq_ratio = self._regime_equity_ratio(
                    price, ma50_v, ma200_v, base_equity_ratio, v, mom
                )

                if mode == "ACCUMULATION":
                    # --- Weekly dip buy ---
                    if is_friday:
                        weekly_ret = (price - last_friday_price) / last_friday_price
                        frac = self._dip_buy_fraction(weekly_ret, r, mom)
                        if frac > 0 and cash > 0:
                            deploy = cash * frac
                            shares += deploy / price
                            cash   -= deploy

                    # --- Quarterly rebalance ---
                    if is_quarter_start:
                        total_val = cash + shares * price
                        target_equity = total_val * eq_ratio
                        diff = target_equity - shares * price
                        if diff > 0:
                            buy = min(diff, cash)
                            shares += buy / price
                            cash   -= buy
                        elif diff < 0:
                            sell_val = abs(diff)
                            shares  -= sell_val / price
                            cash    += sell_val

                elif mode == "HARVESTING":
                    if is_quarter_start:
                        quarterly_target *= (1 + quarterly_growth)
                        total_val = cash + shares * price

                        dd_from_ath = (total_val - all_time_high) / all_time_high
                        if dd_from_ath > -0.15:
                            diff = quarterly_target - total_val
                            if diff < 0:
                                # Surplus → extract income
                                sell_val = abs(diff)
                                # Momentum-aware: in strong bull, keep 20% extra
                                if not pd.isna(mom) and mom > 0.20:
                                    sell_val *= 0.80
                                shares_to_sell = sell_val / price
                                if shares >= shares_to_sell:
                                    shares -= shares_to_sell
                                    cash   += sell_val
                                    self.income_log.append((date, sell_val))
                                else:
                                    income = shares * price
                                    cash += income
                                    self.income_log.append((date, income))
                                    shares = 0
                            elif diff > 0:
                                # Shortfall → buy in
                                buy = min(diff, cash)
                                shares += buy / price
                                cash   -= buy

            pv = cash + shares * price
            portfolio_values.append(pv)

            if i > 0:
                prev_pv = portfolio_values[-2]
                daily_returns_list.append((pv - prev_pv) / prev_pv if prev_pv > 0 else 0)

            if is_friday:
                last_friday_price = price

            last_month = date.month

        self.results       = pd.Series(portfolio_values, index=dates)
        self._daily_returns = pd.Series(daily_returns_list)
        return self.results

    # ------------------------------------------------------------------
    # Metrics Report
    # ------------------------------------------------------------------
    def show_metrics(self):
        if self.results is None:
            return

        r = self.results
        final_val = r.iloc[-1]
        start_val = r.iloc[0]
        years = (r.index[-1] - r.index[0]).days / 365.25
        cagr  = (final_val / start_val) ** (1 / years) - 1

        peak  = r.cummax()
        dd    = (r - peak) / peak
        max_dd = dd.min()

        total_income = sum(x[1] for x in self.income_log)

        dr = self._daily_returns
        rf_daily = 0.045 / 252
        excess  = dr - rf_daily
        sharpe  = (excess.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
        downside = dr[dr < 0].std()
        sortino = (excess.mean() / downside) * np.sqrt(252) if downside > 0 else 0
        calmar  = cagr / abs(max_dd) if max_dd != 0 else 0

        print("\n📊 === ULTIMATE WOMBAT REPORT v3.0 🐻 ===")
        print(f"💰 Final Portfolio:     ${final_val:>15,.2f}")
        print(f"💸 Total Income:        ${total_income:>15,.2f}")
        print(f"📈 CAGR:               {cagr*100:>14.2f}%")
        print(f"🛡️  Max Drawdown:       {max_dd*100:>14.2f}%")
        print(f"⚡ Sharpe Ratio:        {sharpe:>14.2f}")
        print(f"🎯 Sortino Ratio:       {sortino:>14.2f}")
        print(f"🏔️  Calmar Ratio:        {calmar:>14.2f}")
        print(f"💵 Income Events:       {len(self.income_log):>14d}")
        print("==========================================")

        return {
            'cagr': cagr, 'max_dd': max_dd,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        }


if __name__ == "__main__":
    bot = UltimateWombat()
    bot.fetch_data()
    bot.run_backtest(switch_threshold=1_000_000)
    bot.show_metrics()
