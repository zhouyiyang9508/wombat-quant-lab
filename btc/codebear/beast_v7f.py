"""
BTC Beast v7f — Dual Momentum Rotation (BTC vs GLD)
代码熊 🐻

核心改进：最简洁的双资产动量轮动
- 比较 BTC 和 GLD 的相对动量，配置更强的那个
- 绝对动量过滤：如果两个都是负动量，降低总仓位
- 减半周期作为唯一的 "macro overlay"

设计哲学：
- 简单的策略更稳健，更容易通过 Walk-Forward
- 双动量是最经典的因子之一，在所有资产类别上都有效
- BTC 和 GLD 相关性很低（甚至有时负相关）
- 减少参数 → 减少过拟合

策略：
1. 计算 BTC 6M return 和 GLD 6M return
2. 如果 BTC > GLD > 0: 80% BTC + 15% GLD（两个都涨，BTC更强）
3. 如果 BTC > 0 > GLD: 85% BTC + 5% GLD（只有BTC涨）
4. 如果 GLD > BTC > 0: 50% BTC + 40% GLD（两个涨，GLD更强）
5. 如果 GLD > 0 > BTC: 25% BTC + 50% GLD（只有GLD涨）
6. 如果 both < 0: 20% BTC + 30% GLD + 50% cash（都跌，防御）
7. 减半周期后 0-18 月：BTC 最低 50%
"""

import pandas as pd
import numpy as np

HALVING_DATES = [
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]


def halving_info(date, price, prices_series):
    past = [h for h in HALVING_DATES if h <= date]
    if not past:
        return None, None
    last_h = past[-1]
    months = (date - last_h).days / 30.44
    mask = prices_series.index >= last_h
    if mask.any():
        h_price = prices_series.loc[mask].iloc[0]
        gain = (price / h_price) - 1.0
    else:
        gain = 0.0
    return months, gain


class BTCBeastV7f:
    """Dual Momentum Rotation — simple BTC vs GLD relative strength."""

    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.data = None
        self.results = None

    def load_data(self, btc_path, gld_path, start='2017-01-01', end='2026-02-20'):
        btc = pd.read_csv(btc_path, parse_dates=['Date'], index_col='Date')
        btc = btc[['Close']].dropna().sort_index().loc[start:end]
        btc.columns = ['BTC']
        gld = pd.read_csv(gld_path, parse_dates=['Date'], index_col='Date')
        gld = gld[['Close']].dropna().sort_index()
        gld.columns = ['GLD']
        combined = btc.join(gld, how='left')
        combined['GLD'] = combined['GLD'].ffill()
        combined = combined.dropna()
        self.data = combined
        return self

    def run_backtest(self):
        btc_prices = self.data['BTC']
        gld_prices = self.data['GLD']

        # 6-month momentum for both
        btc_mom6 = btc_prices.pct_change(180)
        gld_mom6 = gld_prices.pct_change(180)
        # Also 3-month for blending
        btc_mom3 = btc_prices.pct_change(90)
        gld_mom3 = gld_prices.pct_change(90)

        sma200 = btc_prices.rolling(200).mean()
        mayer = btc_prices / sma200

        cash = self.initial_capital
        btc = 0.0
        gld = 0.0
        portfolio_values = []

        for i in range(len(btc_prices)):
            price_btc = btc_prices.iloc[i]
            price_gld = gld_prices.iloc[i]
            mm = mayer.iloc[i]
            b6 = btc_mom6.iloc[i]
            g6 = gld_mom6.iloc[i]
            b3 = btc_mom3.iloc[i]
            g3 = gld_mom3.iloc[i]
            hm, h_gain = halving_info(btc_prices.index[i], price_btc, btc_prices)

            # Blended momentum (50% 6M + 50% 3M)
            if not pd.isna(b6) and not pd.isna(b3):
                btc_mom = 0.5 * b6 + 0.5 * b3
            elif not pd.isna(b6):
                btc_mom = b6
            elif not pd.isna(b3):
                btc_mom = b3
            else:
                btc_mom = 0.05  # default slightly positive

            if not pd.isna(g6) and not pd.isna(g3):
                gld_mom = 0.5 * g6 + 0.5 * g3
            elif not pd.isna(g6):
                gld_mom = g6
            elif not pd.isna(g3):
                gld_mom = g3
            else:
                gld_mom = 0.02

            # Dual momentum allocation
            if btc_mom > 0 and gld_mom > 0:
                if btc_mom > gld_mom:
                    # Both up, BTC stronger
                    target_btc = 0.80
                    target_gld = 0.15
                else:
                    # Both up, GLD stronger
                    target_btc = 0.50
                    target_gld = 0.40
            elif btc_mom > 0 and gld_mom <= 0:
                # Only BTC up
                target_btc = 0.85
                target_gld = 0.05
            elif btc_mom <= 0 and gld_mom > 0:
                # Only GLD up — shift to GLD
                target_btc = 0.25
                target_gld = 0.50
            else:
                # Both down — defensive
                target_btc = 0.20
                target_gld = 0.30

            # Halving cycle boost: early cycle always bullish BTC
            if hm is not None and hm <= 18:
                target_btc = max(target_btc, 0.50)
                target_gld = min(target_gld, 0.30)

            # Mayer bubble protection
            if not pd.isna(mm):
                if mm > 3.5:
                    target_btc = min(target_btc, 0.35)
                    target_gld = max(target_gld, 0.25)
                elif mm > 3.0:
                    target_btc = min(target_btc, 0.50)
                    target_gld = max(target_gld, 0.20)
                elif mm > 2.4:
                    target_btc = min(target_btc, 0.65)
                    target_gld = max(target_gld, 0.15)

            # Gain-based late cycle
            if h_gain is not None:
                if h_gain > 5.0:
                    target_btc = min(target_btc, 0.35)
                    target_gld = max(target_gld, 0.25)
                elif h_gain > 3.0:
                    target_btc = min(target_btc, 0.50)
                    target_gld = max(target_gld, 0.20)

            # Cap total
            total = target_btc + target_gld
            if total > 1.0:
                scale = 1.0 / total
                target_btc *= scale
                target_gld *= scale

            # Rebalance BTC
            cv = cash + btc * price_btc + gld * price_gld
            target_btc_val = cv * target_btc
            diff_btc = target_btc_val - btc * price_btc
            if diff_btc > 0:
                buy = min(diff_btc, cash)
                btc += buy / price_btc
                cash -= buy
            elif diff_btc < 0:
                sell = abs(diff_btc)
                btc -= sell / price_btc
                cash += sell

            # Rebalance GLD
            cv = cash + btc * price_btc + gld * price_gld
            target_gld_val = cv * target_gld
            diff_gld = target_gld_val - gld * price_gld
            if diff_gld > 0:
                buy = min(diff_gld, cash)
                gld += buy / price_gld
                cash -= buy
            elif diff_gld < 0:
                sell = abs(diff_gld)
                gld -= sell / price_gld
                cash += sell

            portfolio_values.append(cash + btc * price_btc + gld * price_gld)

        self.results = pd.Series(portfolio_values, index=btc_prices.index)
        self._dr = self.results.pct_change().dropna()
        return self.results

    def get_metrics(self):
        r = self.results
        years = (r.index[-1] - r.index[0]).days / 365.25
        cagr = (r.iloc[-1] / r.iloc[0]) ** (1 / years) - 1
        dd = (r - r.cummax()) / r.cummax()
        max_dd = dd.min()
        dr = self._dr
        rf = 0.045 / 365
        sharpe = ((dr - rf).mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        return {'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar}
