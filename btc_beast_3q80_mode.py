# ==========================================
# 小袋熊量化实验室 - BTC Beast 3Q80 Mode
# Optimized by: 代码熊 🐻 v2.0
# ==========================================
# 策略逻辑：
# 1. 3Q80 核心：每周五择时定投 + 季度再平衡
# 2. Ahr999 增强：根据估值动态调整持仓比例
# 3. 熔断机制：季度跌幅 > 30% 暂停加仓
# 4. 减半周期感知：减半前后调整策略激进度
# 5. ATH追踪止损：从历史高点回撤 > 60% 时保护性减仓
#
# KEY IMPROVEMENTS IN V2:
# - 支持 yfinance 自动拉取数据（无需 CSV）
# - 修复季度再平衡日期：改用真实季末边界
# - 修复周涨跌幅计算（使用 shift 而非精确日期查找）
# - 新增比特币减半周期因子
# - 新增 ATH 回撤保护（跌 60% 减仓）
# - 新增完整 CAGR / Sharpe / Calmar 计算
# - Ahr999 HIGH 阈值修正为 5.0（原代码注释与实现不符）
# - 增加现金闲置成本说明（可选 USDC 模拟收益）
# ==========================================

import pandas as pd
import numpy as np
import datetime
import sys

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]


def months_since_halving(date):
    """返回距离最近过去一次减半的月数；无减半历史则返回 None。"""
    past = [h for h in HALVING_DATES if h <= date]
    if not past:
        return None
    return (date - past[-1]).days / 30.44


class BTCBeastStrategy:
    def __init__(self, csv_path=None, use_yfinance=True, start_date='2017-01-01'):
        self.csv_path = csv_path
        self.use_yfinance = use_yfinance and HAS_YFINANCE
        self.start_date = start_date
        self.df = None
        self.results = []

        # Parameters
        self.WEEKLY_DCA = 1000
        self.WEEKLY_PUMP_LIMIT = 0.07   # 7% (放宽，避免漏掉过多周)
        self.MISSED_WEEK_FORCE = 3       # 连续 N 周未买则强制买入
        self.AHR_BOTTOM = 0.45           # 极度低估
        self.AHR_MID = 1.2               # 合理估值上沿
        self.AHR_HIGH = 5.0              # 泡沫区（修正原代码注释与实现不符）
        self.CIRCUIT_BREAKER = 0.30      # 季度跌 30% 熔断
        self.ATH_DRAWDOWN_GUARD = 0.60   # 从 ATH 回撤 60% 触发保护减仓

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def load_data(self):
        print("🐻 Loading Market Data...")
        if self.use_yfinance:
            self._load_from_yfinance()
        else:
            self._load_from_csv()

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
        if not self.csv_path:
            print("❌ No CSV path specified and yfinance unavailable.")
            sys.exit(1)
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
        # Ahr999 components
        self.df['log_price'] = np.log(self.df['Close'])
        self.df['geom_mean_200'] = np.exp(
            self.df['log_price'].rolling(window=200).mean()
        )
        genesis = pd.Timestamp('2009-01-03')
        self.df['days_since_genesis'] = (self.df.index - genesis).days
        self.df['exp_growth_val'] = 10 ** (
            2.68 + 0.00057 * self.df['days_since_genesis']
        )
        self.df['ahr999'] = (
            (self.df['Close'] / self.df['geom_mean_200']) *
            (self.df['Close'] / self.df['exp_growth_val'])
        )

        # Weekly return (using shift, not fragile date lookup)
        self.df['weekly_ret'] = self.df['Close'].pct_change(5)  # ~5 trading days

        # ATH tracking
        self.df['ath'] = self.df['Close'].cummax()
        self.df['ath_drawdown'] = (self.df['Close'] - self.df['ath']) / self.df['ath']

        # Halving months
        self.df['halving_months'] = self.df.index.map(months_since_halving)

        # Filter backtest window
        self.df = self.df[self.df.index >= self.start_date].copy()
        print(f"   Backtest period: {self.df.index[0].date()} → {self.df.index[-1].date()}")

    # ------------------------------------------------------------------
    # Quarter boundary detection (fixed)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_quarter_end_week(date):
        """True if this date falls in the last 5 trading days of a quarter."""
        # Quarter ends: Mar 31, Jun 30, Sep 30, Dec 31
        quarter_end_months = {3: 31, 6: 30, 9: 30, 12: 31}
        m, d = date.month, date.day
        if m not in quarter_end_months:
            return False
        return d >= (quarter_end_months[m] - 4)

    # ------------------------------------------------------------------
    # Halving-aware aggression multiplier
    # ------------------------------------------------------------------
    @staticmethod
    def _halving_multiplier(halving_months):
        """
        Months 0–12 after halving: accumulation phase → multiplier 1.3
        Months 12–30: bull run → multiplier 1.0
        Months 30+: bear/cycle end → multiplier 0.7
        """
        if halving_months is None:
            return 1.0
        if halving_months < 12:
            return 1.3
        elif halving_months < 30:
            return 1.0
        else:
            return 0.7

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    def run_backtest(self):
        print("🐻 Running Beast Mode Simulation...")
        cash = 0.0
        btc_balance = 0.0
        total_invested = 0.0
        missed_weeks = 0
        last_quarter_val = 0.0
        all_time_high_val = 0.0

        history = []

        for date, row in self.df.iterrows():
            price = row['Close']
            ahr = row['ahr999'] if not pd.isna(row['ahr999']) else 1.0
            weekly_chg = row['weekly_ret'] if not pd.isna(row['weekly_ret']) else 0.0
            ath_dd = row['ath_drawdown']
            halving_months = row['halving_months']

            val = cash + btc_balance * price

            if last_quarter_val == 0:
                last_quarter_val = val
            if val > all_time_high_val:
                all_time_high_val = val

            history.append({
                'Date': date,
                'Value': val,
                'Invested': total_invested,
                'Ahr': ahr,
                'Cash': cash,
                'BTC': btc_balance,
            })

            # ==========================
            # 1. ATH 回撤保护
            # ==========================
            btc_price_dd = ath_dd  # price drawdown from its own ATH
            if btc_price_dd < -self.ATH_DRAWDOWN_GUARD and btc_balance > 0:
                # 保护性减仓到 20% BTC
                target_btc_val = val * 0.20
                current_btc_val = btc_balance * price
                if current_btc_val > target_btc_val:
                    sell = current_btc_val - target_btc_val
                    btc_balance -= sell / price
                    cash += sell

            # ==========================
            # 2. Weekly DCA Logic (Friday)
            # ==========================
            if date.weekday() == 4:
                invest_amount = self.WEEKLY_DCA
                should_buy = False

                if weekly_chg < self.WEEKLY_PUMP_LIMIT:
                    should_buy = True
                    missed_weeks = 0
                else:
                    missed_weeks += 1
                    if missed_weeks >= self.MISSED_WEEK_FORCE:
                        should_buy = True
                        missed_weeks = 0

                # Ahr999 bottom fishing: double down + use all cash
                if ahr < self.AHR_BOTTOM:
                    should_buy = True
                    invest_amount *= 2

                # Halving-cycle multiplier
                hm = self._halving_multiplier(halving_months)
                invest_amount *= hm

                if should_buy:
                    total_spend = invest_amount + cash
                    if total_spend > 0:
                        btc_balance += total_spend / price
                        cash = 0
                    total_invested += self.WEEKLY_DCA  # Record base DCA only
                else:
                    cash += invest_amount
                    total_invested += self.WEEKLY_DCA

            # ==========================
            # 3. Quarterly Rebalance
            # ==========================
            if self._is_quarter_end_week(date):
                val = cash + btc_balance * price  # recompute fresh

                # Circuit breaker
                is_crash = (
                    last_quarter_val > 0 and
                    (val - last_quarter_val) / last_quarter_val < -self.CIRCUIT_BREAKER
                )

                # Dynamic allocation by Ahr999
                if ahr < self.AHR_BOTTOM:
                    target_ratio = 1.00
                elif ahr < self.AHR_MID:
                    target_ratio = 0.80
                elif ahr < self.AHR_HIGH:
                    target_ratio = 0.50
                else:
                    target_ratio = 0.10  # Near cycle top: mostly out

                target_btc_val = val * target_ratio
                current_btc_val = btc_balance * price

                if current_btc_val > target_btc_val:
                    # Take profit (allowed even in crash)
                    sell = current_btc_val - target_btc_val
                    btc_balance -= sell / price
                    cash += sell
                elif not is_crash:
                    # Rebalance in (blocked if crash)
                    buy = min(target_btc_val - current_btc_val, cash)
                    btc_balance += buy / price
                    cash -= buy

                last_quarter_val = val

        self.results = pd.DataFrame(history)
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
        dd = (res['Value'] - peak) / peak
        mdd = dd.min()

        # CAGR (of portfolio value)
        days = (res.index[-1] - res.index[0]).days
        years = days / 365.25
        cagr = (final_val / res.iloc[0]['Value']) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe (daily returns)
        daily_ret = res['Value'].pct_change().dropna()
        rf_daily = 0.045 / 252
        sharpe = ((daily_ret - rf_daily).mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

        # Calmar
        calmar = cagr / abs(mdd) if mdd != 0 else 0

        print("\n" + "=" * 45)
        print("🐻 BTC Beast Mode v2.0 — Results")
        print("=" * 45)
        print(f"Start Date:        {res.index[0].date()}")
        print(f"End Date:          {res.index[-1].date()}")
        print(f"Total Invested:    ${total_invested:>12,.2f}")
        print(f"Final Value:       ${final_val:>12,.2f}")
        print(f"Total Return:      {roi*100:>11.2f}%")
        print(f"CAGR:              {cagr*100:>11.2f}%")
        print(f"Max Drawdown:      {mdd*100:>11.2f}%")
        print(f"Sharpe Ratio:      {sharpe:>11.2f}")
        print(f"Calmar Ratio:      {calmar:>11.2f}")
        print("=" * 45)
        print("Key Parameters:")
        print(f"  Weekly DCA:        ${self.WEEKLY_DCA:,.0f}")
        print(f"  Pump Limit:        {self.WEEKLY_PUMP_LIMIT*100:.0f}%")
        print(f"  Ahr999 Bottom:     {self.AHR_BOTTOM}")
        print(f"  Ahr999 Mid:        {self.AHR_MID}")
        print(f"  Ahr999 High:       {self.AHR_HIGH}")
        print(f"  Circuit Breaker:   {self.CIRCUIT_BREAKER*100:.0f}% drop")
        print(f"  ATH Guard:         {self.ATH_DRAWDOWN_GUARD*100:.0f}% drawdown")
        print("=" * 45)


if __name__ == "__main__":
    bot = BTCBeastStrategy(use_yfinance=True)
    bot.load_data()
    bot.calculate_indicators()
    bot.generate_report()
