# ==========================================
# 小袋熊量化实验室 - BTC Beast 3Q80 Mode
# Optimized by: 代码熊 🐻 v3.0
# ==========================================
# 策略逻辑：
# 1. 3Q80 核心：每周五择时定投 + 季度再平衡
# 2. Ahr999 + Mayer Multiple：双重估值信号动态调整持仓比例
# 3. 熔断机制：季度跌幅 > 30% 暂停加仓
# 4. 减半周期感知：平滑 sigmoid 式乘数（取代硬分段）
# 5. ATH追踪止损：从历史高点回撤 > 60% 时保护性减仓
# 6. Fear & Greed 代理：波动率分位 + 价格动量综合判断
# 7. 智能跳过：极度泡沫（Mayer > 3.0 且 FG > 0.85）时跳过定投
#
# KEY IMPROVEMENTS IN V3 vs V2:
# - 修复季度再平衡多次触发 Bug（现在每季度只触发一次）
# - 新增 Mayer Multiple（价格/MA200）双重估值过滤
# - 减半乘数改为平滑分段线性（更平滑的过渡）
# - 新增 Fear & Greed 代理指标（波动率百分位 + 动量百分位）
# - 智能 DCA：极度泡沫时跳过买入
# - 季度目标持仓比：综合 Ahr999 + Mayer Multiple
# - 内置合成 BTC 数据生成器（离线测试 / CI 友好）
# ==========================================

import pandas as pd
import numpy as np
import sys

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# Known Bitcoin halving dates
HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]


def months_since_halving(date):
    """Return months since the most recent past halving; None if before first halving."""
    past = [h for h in HALVING_DATES if h <= date]
    if not past:
        return None
    return (date - past[-1]).days / 30.44


class BTCBeastStrategy:
    def __init__(self, csv_path=None, use_yfinance=True, start_date='2017-01-01'):
        self.csv_path    = csv_path
        self.use_yfinance = use_yfinance and HAS_YFINANCE
        self.start_date  = start_date
        self.df          = None
        self.results     = []

        # --- Core parameters ---
        self.WEEKLY_DCA          = 1000
        self.WEEKLY_PUMP_LIMIT   = 0.07   # Skip buy if weekly gain > 7%
        self.MISSED_WEEK_FORCE   = 3      # Force buy after N skipped weeks
        self.AHR_BOTTOM          = 0.45   # Ahr999 extreme undervalue
        self.AHR_MID             = 1.2    # Ahr999 fair value ceiling
        self.AHR_HIGH            = 5.0    # Ahr999 bubble zone
        self.MAYER_HIGH          = 2.4    # Mayer Multiple: historically overheated
        self.MAYER_EXTREME       = 3.5    # Mayer: extreme bubble (historical peak ~10)
        self.CIRCUIT_BREAKER     = 0.30   # Quarterly drop > 30% → pause accumulation
        self.ATH_DRAWDOWN_GUARD  = 0.60   # Price ATH drawdown > 60% → protect positions
        self.FG_BUBBLE_THRESHOLD = 0.85   # Fear & Greed "extreme greed" cutoff

    # ------------------------------------------------------------------
    # Synthetic Data (offline testing)
    # ------------------------------------------------------------------
    @staticmethod
    def generate_synthetic_data(n_years=8, seed=99):
        """Synthetic BTC-like price series: 4-year halving cycles, high vol."""
        np.random.seed(seed)
        n_days = int(n_years * 365)

        # BTC realistic parameters (2017–2024 average cycle):
        # Geometric annual return ≈ drift_log - vol²/2 = 1.00 - 0.85²/2 ≈ 64%/yr
        daily_mu    = 1.00 / 365           # log drift: ~100%/yr
        daily_sigma = 0.85 / np.sqrt(365)  # log vol:   ~85%/yr

        prices = [10000.0]
        for i in range(n_days):
            # Crypto crash days (fat tails)
            if np.random.random() < 0.008:
                shock = np.random.uniform(-0.30, -0.10)
            elif np.random.random() < 0.005:
                shock = np.random.uniform(0.10, 0.25)  # moon days
            else:
                shock = np.random.normal(daily_mu, daily_sigma)
            prices.append(max(prices[-1] * np.exp(shock), 1.0))

        dates = pd.date_range(start='2017-01-01', periods=n_days + 1, freq='D')
        df = pd.DataFrame({'Close': prices}, index=dates)
        return df

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def load_data(self):
        print("🐻 Loading Market Data...")
        if self.use_yfinance:
            try:
                self._load_from_yfinance()
                return
            except Exception as e:
                print(f"⚠️ yfinance failed ({e}), falling back to synthetic.")

        if self.csv_path:
            self._load_from_csv()
        else:
            print("📊 Using synthetic BTC-like data...")
            self.df = self.generate_synthetic_data()
            print(f"✅ Synthetic Data: {len(self.df)} rows.")

    def _load_from_yfinance(self):
        print("   Source: yfinance (BTC-USD)")
        raw = yf.download('BTC-USD', start='2014-09-17', progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        self.df = raw[['Close']].dropna()
        self.df.index = pd.to_datetime(self.df.index)
        self.df.sort_index(inplace=True)
        print(f"   ✅ Loaded {len(self.df)} rows via yfinance.")

    def _load_from_csv(self):
        self.df = pd.read_csv(self.csv_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self.df.sort_index(inplace=True)
        print(f"   ✅ Loaded {len(self.df)} rows from CSV.")

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def calculate_indicators(self):
        print("🐻 Calculating Indicators...")

        # --- Ahr999 ---
        self.df['log_price']      = np.log(self.df['Close'])
        self.df['geom_mean_200']  = np.exp(self.df['log_price'].rolling(200).mean())
        genesis = pd.Timestamp('2009-01-03')
        self.df['days_since_genesis'] = (self.df.index - genesis).days
        self.df['exp_growth_val'] = 10 ** (2.68 + 0.00057 * self.df['days_since_genesis'])
        self.df['ahr999'] = (
            (self.df['Close'] / self.df['geom_mean_200']) *
            (self.df['Close'] / self.df['exp_growth_val'])
        )

        # --- Mayer Multiple: price / MA200 ---
        self.df['ma200']          = self.df['Close'].rolling(200).mean()
        self.df['mayer_multiple'] = self.df['Close'] / self.df['ma200']

        # --- Weekly return (5 trading days) ---
        self.df['weekly_ret'] = self.df['Close'].pct_change(5)

        # --- ATH tracking ---
        self.df['ath']          = self.df['Close'].cummax()
        self.df['ath_drawdown'] = (self.df['Close'] - self.df['ath']) / self.df['ath']

        # --- Halving months ---
        self.df['halving_months'] = self.df.index.map(months_since_halving)

        # --- Fear & Greed Proxy ---
        # Component 1: Volatility percentile rank (low vol = greed, high vol = fear)
        daily_ret = self.df['Close'].pct_change()
        vol20     = daily_ret.rolling(20).std() * np.sqrt(252)
        self.df['vol_pct_rank'] = vol20.rolling(252).rank(pct=True)

        # Component 2: Momentum rank (60-day return percentile)
        mom60 = self.df['Close'].pct_change(60)
        self.df['mom_pct_rank'] = mom60.rolling(252).rank(pct=True)

        # FG proxy: high score = greed (high momentum, low vol)
        self.df['fg_score'] = (
            0.50 * self.df['mom_pct_rank'].fillna(0.5) +
            0.50 * (1 - self.df['vol_pct_rank'].fillna(0.5))
        )

        # --- Filter backtest window ---
        self.df = self.df[self.df.index >= self.start_date].copy()
        print(f"   Backtest period: {self.df.index[0].date()} → {self.df.index[-1].date()}")

    # ------------------------------------------------------------------
    # Smooth Halving Multiplier (piecewise linear, no hard steps)
    # ------------------------------------------------------------------
    @staticmethod
    def _halving_multiplier(halving_months):
        """
        Smooth halving-cycle multiplier:
          0–6  months → ramp 1.0 → 1.4  (early accumulation)
          6–18 months → hold 1.4         (peak accumulation phase)
          18–30 months → ramp 1.4 → 1.0 (bull run, normalizing)
          30–42 months → ramp 1.0 → 0.7  (late cycle, cautious)
          42+  months → hold 0.7          (cycle end / bear)
        """
        if halving_months is None:
            return 1.0
        m = halving_months
        if m <= 0:
            return 1.0
        elif m <= 6:
            return 1.0 + 0.4 * (m / 6)
        elif m <= 18:
            return 1.4
        elif m <= 30:
            return 1.4 - 0.4 * ((m - 18) / 12)
        elif m <= 42:
            return 1.0 - 0.3 * ((m - 30) / 12)
        else:
            return 0.7

    # ------------------------------------------------------------------
    # Quarter ID helper (fire rebalance exactly once per quarter)
    # ------------------------------------------------------------------
    @staticmethod
    def _quarter_id(date):
        return (date.year, (date.month - 1) // 3)

    # ------------------------------------------------------------------
    # Target allocation from valuation signals
    # ------------------------------------------------------------------
    def _target_btc_ratio(self, ahr, mayer):
        """
        Combined Ahr999 × Mayer Multiple → target BTC allocation ratio.

        Ahr999 primary signal + Mayer Multiple secondary overlay:
        - If Mayer extremely high (>3.5): cap ratio at 10%
        - If Mayer high (>2.4): reduce ratio by ~30%
        - Otherwise: Ahr999-driven allocation
        """
        # Ahr999 base
        if pd.isna(ahr) or ahr <= 0:
            base = 0.80
        elif ahr < self.AHR_BOTTOM:
            base = 1.00
        elif ahr < self.AHR_MID:
            base = 0.80
        elif ahr < self.AHR_HIGH:
            base = 0.50
        else:
            base = 0.10  # Near cycle top

        # Mayer Multiple overlay
        if not pd.isna(mayer):
            if mayer >= self.MAYER_EXTREME:
                base = min(base, 0.10)
            elif mayer >= self.MAYER_HIGH:
                t = (mayer - self.MAYER_HIGH) / (self.MAYER_EXTREME - self.MAYER_HIGH)
                base *= (1.0 - 0.70 * t)  # Scale down up to 70%

        return base

    # ------------------------------------------------------------------
    # Backtest Engine
    # ------------------------------------------------------------------
    def run_backtest(self):
        print("🐻 Running Beast Mode v3.0 Simulation...")
        cash            = 0.0
        btc_balance     = 0.0
        total_invested  = 0.0
        missed_weeks    = 0
        last_quarter_val= 0.0
        all_time_high_val = 0.0
        last_rebalanced_quarter = None   # FIX: prevent multiple rebalances per quarter

        history = []

        for date, row in self.df.iterrows():
            price       = row['Close']
            ahr         = row['ahr999']     if not pd.isna(row['ahr999'])     else 1.0
            mayer       = row['mayer_multiple'] if not pd.isna(row['mayer_multiple']) else 1.0
            weekly_chg  = row['weekly_ret'] if not pd.isna(row['weekly_ret']) else 0.0
            ath_dd      = row['ath_drawdown']
            halving_mo  = row['halving_months']
            fg          = row['fg_score']   if not pd.isna(row['fg_score'])   else 0.5

            val = cash + btc_balance * price

            if last_quarter_val == 0:
                last_quarter_val = val
            if val > all_time_high_val:
                all_time_high_val = val

            history.append({
                'Date': date, 'Value': val,
                'Invested': total_invested,
                'Ahr': ahr, 'Mayer': mayer,
                'Cash': cash, 'BTC': btc_balance,
            })

            # ============================================================
            # 1. ATH Drawdown Protection
            # ============================================================
            if ath_dd < -self.ATH_DRAWDOWN_GUARD and btc_balance > 0:
                # Reduce to 20% BTC allocation
                target_btc_val  = val * 0.20
                current_btc_val = btc_balance * price
                if current_btc_val > target_btc_val:
                    sell = current_btc_val - target_btc_val
                    btc_balance -= sell / price
                    cash        += sell

            # ============================================================
            # 2. Weekly DCA Logic (every weekday — BTC trades 24/7)
            #    Use day-of-week 0 (Monday) as the weekly DCA trigger
            # ============================================================
            if date.weekday() == 0:
                invest_amount = self.WEEKLY_DCA
                should_buy    = False

                # Skip if weekly gain is too high (avoid buying tops)
                if weekly_chg < self.WEEKLY_PUMP_LIMIT:
                    should_buy  = True
                    missed_weeks = 0
                else:
                    missed_weeks += 1
                    if missed_weeks >= self.MISSED_WEEK_FORCE:
                        should_buy  = True
                        missed_weeks = 0

                # Extreme undervalue: double down
                if ahr < self.AHR_BOTTOM:
                    should_buy    = True
                    invest_amount *= 2.0

                # Smart skip: extreme bubble (Mayer > MAYER_EXTREME and FG > threshold)
                if mayer >= self.MAYER_EXTREME and fg >= self.FG_BUBBLE_THRESHOLD:
                    should_buy = False
                    missed_weeks += 1  # counts toward force-buy counter

                # Halving-cycle multiplier (smooth)
                hm = self._halving_multiplier(halving_mo)
                invest_amount *= hm

                if should_buy:
                    total_spend = invest_amount + cash
                    if total_spend > 0:
                        btc_balance    += total_spend / price
                        cash            = 0
                    total_invested += self.WEEKLY_DCA
                else:
                    cash           += invest_amount
                    total_invested += self.WEEKLY_DCA

            # ============================================================
            # 3. Quarterly Rebalance (ONCE per quarter — BUG FIX)
            # ============================================================
            cur_quarter = self._quarter_id(date)
            if cur_quarter != last_rebalanced_quarter:
                # Check if we're in the last 5 calendar days of a quarter
                quarter_end_months = {3: 31, 6: 30, 9: 30, 12: 31}
                m, d = date.month, date.day
                if m in quarter_end_months and d >= (quarter_end_months[m] - 4):
                    val = cash + btc_balance * price  # fresh value

                    # Circuit breaker
                    is_crash = (
                        last_quarter_val > 0 and
                        (val - last_quarter_val) / last_quarter_val < -self.CIRCUIT_BREAKER
                    )

                    target_ratio = self._target_btc_ratio(ahr, mayer)
                    target_btc_val   = val * target_ratio
                    current_btc_val  = btc_balance * price

                    if current_btc_val > target_btc_val:
                        # Take profit (always allowed)
                        sell = current_btc_val - target_btc_val
                        btc_balance -= sell / price
                        cash        += sell
                    elif not is_crash:
                        # Buy in to reach target (blocked if circuit breaker)
                        buy = min(target_btc_val - current_btc_val, cash)
                        btc_balance += buy / price
                        cash        -= buy

                    last_quarter_val       = val
                    last_rebalanced_quarter = cur_quarter

        self.results      = pd.DataFrame(history)
        self.results.set_index('Date', inplace=True)
        self._total_invested = total_invested
        return self.results, total_invested

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(self):
        res, total_invested = self.run_backtest()
        final_val = res.iloc[-1]['Value']
        roi = (final_val - total_invested) / total_invested if total_invested > 0 else 0

        # Max Drawdown
        peak = res['Value'].cummax()
        dd   = (res['Value'] - peak) / peak
        mdd  = dd.min()

        # CAGR: computed from first date with non-zero portfolio value
        # (DCA strategies start from $0, so initial days are excluded)
        nonzero = res[res['Value'] > 0]
        if len(nonzero) > 1:
            days  = (nonzero.index[-1] - nonzero.index[0]).days
            years = days / 365.25
            cagr  = (final_val / nonzero.iloc[0]['Value']) ** (1 / years) - 1 if years > 0 else 0
        else:
            cagr = 0

        # Sharpe / Sortino (exclude early zero-value days for DCA strategies)
        daily_ret = res['Value'].pct_change().dropna()
        daily_ret = daily_ret[res['Value'].shift(1) > 0]  # exclude days starting from $0
        rf_daily  = 0.045 / 365
        sharpe    = ((daily_ret - rf_daily).mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() > 0 else 0

        # Sortino
        downside = daily_ret[daily_ret < 0].std()
        sortino  = ((daily_ret - rf_daily).mean() / downside) * np.sqrt(365) if downside > 0 else 0

        # Calmar
        calmar = cagr / abs(mdd) if mdd != 0 else 0

        print("\n" + "=" * 50)
        print("🐻 BTC Beast Mode v3.0 — Results")
        print("=" * 50)
        print(f"Start Date:          {res.index[0].date()}")
        print(f"End Date:            {res.index[-1].date()}")
        print(f"Total Invested:      ${total_invested:>12,.2f}")
        print(f"Final Value:         ${final_val:>12,.2f}")
        print(f"Total Return:        {roi*100:>11.2f}%")
        print(f"CAGR:                {cagr*100:>11.2f}%")
        print(f"Max Drawdown:        {mdd*100:>11.2f}%")
        print(f"Sharpe Ratio:        {sharpe:>11.2f}")
        print(f"Sortino Ratio:       {sortino:>11.2f}")
        print(f"Calmar Ratio:        {calmar:>11.2f}")
        print("=" * 50)
        print("Key Parameters:")
        print(f"  Weekly DCA:          ${self.WEEKLY_DCA:,.0f}")
        print(f"  Pump Limit:          {self.WEEKLY_PUMP_LIMIT*100:.0f}%")
        print(f"  Ahr999 Bottom/Mid/High: {self.AHR_BOTTOM}/{self.AHR_MID}/{self.AHR_HIGH}")
        print(f"  Mayer High/Extreme:  {self.MAYER_HIGH}/{self.MAYER_EXTREME}")
        print(f"  Circuit Breaker:     {self.CIRCUIT_BREAKER*100:.0f}% quarterly drop")
        print(f"  ATH Guard:           {self.ATH_DRAWDOWN_GUARD*100:.0f}% drawdown")
        print(f"  FG Bubble Threshold: {self.FG_BUBBLE_THRESHOLD}")
        print("=" * 50)

        return {
            'cagr': cagr, 'max_dd': mdd,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        }


if __name__ == "__main__":
    bot = BTCBeastStrategy(use_yfinance=True)
    bot.load_data()
    bot.calculate_indicators()
    bot.generate_report()
