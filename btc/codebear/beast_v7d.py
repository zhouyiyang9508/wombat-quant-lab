"""
BTC Beast v7d — 绝对动量 + GLD 轮动
代码熊 🐻

核心改进：最激进的 Walk-Forward 优化设计
- 绝对动量信号：BTC 6 月涨幅 >0 → 持有 BTC，否则 → 转 GLD
- 减半周期例外：减半后 0-18 个月内，即使动量为负也保留 BTC
- 渐进式转换（非 0/1 切换）：
  - 6M > +20%: 85% BTC + 10% GLD + 5% cash
  - 6M 0-20%: 70% BTC + 20% GLD + 10% cash
  - 6M -20%-0%: 40% BTC + 40% GLD + 20% cash
  - 6M < -20%: 20% BTC + 50% GLD + 30% cash
- Mayer 和 RSI 叠加层保持

设计哲学：
- 绝对动量是最稳健的 alpha 因子（在股票/TQQQ/BTC 上都有效）
- 渐进式而非二元切换，减少 whipsaw
- GLD 作为 "safe haven" 替代现金
- 目标：缩小 IS/OOS 差距，以通过 Walk-Forward
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


class BTCBeastV7d:
    """Absolute Momentum + GLD Rotation — gradient-based allocation."""

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
        sma200 = btc_prices.rolling(200).mean()
        mayer = btc_prices / sma200
        delta = btc_prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi14 = 100 - (100 / (1 + rs))
        weekly_ret = btc_prices.pct_change(7)

        # Multi-period momentum: 3M, 6M, 9M blended
        mom_3m = btc_prices.pct_change(90)
        mom_6m = btc_prices.pct_change(180)
        mom_9m = btc_prices.pct_change(270)

        cash = self.initial_capital
        btc = 0.0
        gld = 0.0
        portfolio_values = []

        for i in range(len(btc_prices)):
            price_btc = btc_prices.iloc[i]
            price_gld = gld_prices.iloc[i]
            sma = sma200.iloc[i]
            mm = mayer.iloc[i]
            rsi = rsi14.iloc[i]
            wret = weekly_ret.iloc[i]
            m3 = mom_3m.iloc[i]
            m6 = mom_6m.iloc[i]
            m9 = mom_9m.iloc[i]
            hm, h_gain = halving_info(btc_prices.index[i], price_btc, btc_prices)

            # Blended momentum signal (0.2×3M + 0.5×6M + 0.3×9M)
            mom_vals = []
            if not pd.isna(m3): mom_vals.append(('3m', m3, 0.2))
            if not pd.isna(m6): mom_vals.append(('6m', m6, 0.5))
            if not pd.isna(m9): mom_vals.append(('9m', m9, 0.3))

            if mom_vals:
                total_w = sum(w for _, _, w in mom_vals)
                blended_mom = sum(v * w for _, v, w in mom_vals) / total_w
            else:
                blended_mom = 0.10  # default slight positive

            # Halving cycle override: early cycle is bullish regardless
            halving_boost = False
            if hm is not None and hm <= 18:
                halving_boost = True

            # Gradient-based allocation from momentum signal
            if blended_mom > 0.50:
                target_btc = 0.90
                target_gld = 0.05
            elif blended_mom > 0.20:
                target_btc = 0.85
                target_gld = 0.10
            elif blended_mom > 0.0:
                target_btc = 0.70
                target_gld = 0.20
            elif blended_mom > -0.10:
                target_btc = 0.50
                target_gld = 0.35
            elif blended_mom > -0.20:
                target_btc = 0.40
                target_gld = 0.40
            elif blended_mom > -0.35:
                target_btc = 0.25
                target_gld = 0.45
            else:
                target_btc = 0.20
                target_gld = 0.50

            # Halving cycle early → boost BTC allocation
            if halving_boost:
                target_btc = max(target_btc, 0.55)
                target_gld = min(target_gld, 0.25)

            # Mayer bubble protection (override)
            if not pd.isna(mm):
                if mm > 3.5:
                    target_btc = min(target_btc, 0.40)
                    target_gld = max(target_gld, 0.25)
                elif mm > 3.0:
                    target_btc = min(target_btc, 0.55)
                    target_gld = max(target_gld, 0.20)
                elif mm > 2.4:
                    target_btc = min(target_btc, 0.70)
                    target_gld = max(target_gld, 0.15)

            # Gain-based late cycle
            if h_gain is not None:
                if h_gain > 5.0:
                    target_btc = min(target_btc, 0.40)
                    target_gld = max(target_gld, 0.25)
                elif h_gain > 3.0:
                    target_btc = min(target_btc, 0.55)
                    target_gld = max(target_gld, 0.20)

            # RSI extreme: boost BTC on oversold
            if not pd.isna(rsi) and rsi < 20:
                target_btc = max(target_btc, 0.70)
                target_gld = min(target_gld, 0.15)
            elif not pd.isna(wret) and wret < -0.20:
                target_btc = max(target_btc, 0.60)
                target_gld = min(target_gld, 0.20)

            # Ensure total <= 1.0
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
