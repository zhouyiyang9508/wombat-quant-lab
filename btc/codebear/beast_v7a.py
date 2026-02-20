"""
BTC Beast v7a — Soft Bear (Higher Floor)
代码熊 🐻

核心改进：借鉴 Stock v2d 的 Soft Bear 思路
- v6b 熊市底仓 35% → v7a 提升至 45%
- 早期周期底仓 52% → 58%
- 绝对动量过滤：6 月涨幅 <-30% 时才降到最低仓位（25%）
- 牛市更保守（max 95%），降低 IS/OOS 差异

策略思路：
- BTC 不像股票可以选股，所以 "soft bear" = 提高仓位下限
- 但要结合绝对动量避免最深的熊市坑
- 降低牛市上限，让 IS 和 OOS 更平衡
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


class BTCBeastV7a:
    """Soft Bear — higher bear floor + absolute momentum filter."""

    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.data = None
        self.results = None

    def load_data(self, btc_path, start='2017-01-01', end='2026-02-20'):
        df = pd.read_csv(btc_path, parse_dates=['Date'], index_col='Date')
        df = df[['Close']].dropna().sort_index()
        df = df.loc[start:end]
        self.data = df
        return self

    def run_backtest(self):
        prices = self.data['Close']
        sma200 = prices.rolling(200).mean()
        mayer = prices / sma200
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi14 = 100 - (100 / (1 + rs))
        weekly_ret = prices.pct_change(7)
        # 6-month absolute momentum
        mom_6m = prices.pct_change(180)

        cash = self.initial_capital
        btc = 0.0
        in_bear = False
        portfolio_values = []

        for i in range(len(prices)):
            price = prices.iloc[i]
            sma = sma200.iloc[i]
            mm = mayer.iloc[i]
            rsi = rsi14.iloc[i]
            wret = weekly_ret.iloc[i]
            mom6 = mom_6m.iloc[i]
            hm, h_gain = halving_info(prices.index[i], price, prices)

            # Regime detection (same hysteresis as v6b)
            if i >= 200 and not pd.isna(sma):
                if not in_bear and price < sma * 0.92:
                    in_bear = True
                elif in_bear and price > sma * 1.03:
                    in_bear = False

            if not in_bear:
                # BULL — slightly more conservative than v6b (cap at 95%)
                target_pct = 0.95

                # Mayer bubble protection (same as v6b)
                if not pd.isna(mm) and mm > 3.5:
                    target_pct = 0.50
                elif not pd.isna(mm) and mm > 3.0:
                    target_pct = 0.65
                elif not pd.isna(mm) and mm > 2.4:
                    target_pct = 0.80

                # Gain-based late cycle caution
                if h_gain is not None and h_gain > 5.0:
                    target_pct = min(target_pct, 0.50)
                elif h_gain is not None and h_gain > 3.0:
                    target_pct = min(target_pct, 0.65)

            else:
                # BEAR — Soft Bear: higher floor than v6b
                if h_gain is not None:
                    if h_gain < 1.0:
                        # Early cycle: aggressive accumulation
                        floor = 0.58
                    elif h_gain < 3.0:
                        # Normal: SOFT BEAR floor (was 0.35, now 0.45)
                        floor = 0.45
                    else:
                        # Post-bubble: still keep some
                        floor = 0.30
                else:
                    floor = 0.45

                # Absolute momentum override: if 6M < -30%, go defensive
                if not pd.isna(mom6) and mom6 < -0.30:
                    floor = min(floor, 0.25)

                # RSI-based tactical (same structure, higher levels)
                if not pd.isna(rsi) and rsi < 20:
                    target_pct = 0.80
                elif not pd.isna(wret) and wret < -0.20:
                    target_pct = 0.70
                elif not pd.isna(rsi) and rsi < 30:
                    target_pct = 0.60
                elif not pd.isna(rsi) and rsi > 60:
                    target_pct = floor
                else:
                    cv = cash + btc * price
                    current = (btc * price) / cv if cv > 0 else floor
                    target_pct = max(current, floor)

            # Rebalance
            cv = cash + btc * price
            target_equity = cv * target_pct
            diff = target_equity - btc * price
            if diff > 0 and cash > 0:
                buy = min(diff, cash)
                btc += buy / price
                cash -= buy
            elif diff < 0:
                sell = abs(diff)
                btc -= sell / price
                cash += sell

            portfolio_values.append(cash + btc * price)

        self.results = pd.Series(portfolio_values, index=prices.index)
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
