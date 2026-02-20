"""
BTC Beast v6b — 改进减半周期逻辑（累计涨幅驱动）
代码熊 🐻

改进点：用减半后累计涨幅替代硬编码月数
- 涨幅 <100%: 早期牛市，激进底仓 52%
- 涨幅 100-300%: 中期牛市，100% 持仓
- 涨幅 >300%: 晚期牛市，上限 70%
"""

import pandas as pd
import numpy as np

HALVING_DATES = [
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]


def halving_info(date, price, prices_series):
    """Return (months_since, gain_since_halving) or (None, None)"""
    past = [h for h in HALVING_DATES if h <= date]
    if not past:
        return None, None
    last_h = past[-1]
    months = (date - last_h).days / 30.44
    # Get price at halving (or closest after)
    mask = prices_series.index >= last_h
    if mask.any():
        h_price = prices_series.loc[mask].iloc[0]
        gain = (price / h_price) - 1.0
    else:
        gain = 0.0
    return months, gain


class BTCBeastV6b:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.data = None
        self.results = None

    def load_csv(self, path, start='2017-01-01', end='2026-02-20'):
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        df = df[['Close']].dropna().sort_index()
        df = df.loc[start:end]
        self.data = df
        print(f"✅ v6b data: {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")

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
            hm, h_gain = halving_info(prices.index[i], price, prices)

            if i >= 200 and not pd.isna(sma):
                if not in_bear and price < sma * 0.92:
                    in_bear = True
                elif in_bear and price > sma * 1.03:
                    in_bear = False

            if not in_bear:
                # BULL — gain-based cycle position
                target_pct = 1.00

                # Mayer bubble
                if not pd.isna(mm) and mm > 3.5:
                    target_pct = 0.50
                elif not pd.isna(mm) and mm > 3.0:
                    target_pct = 0.70
                elif not pd.isna(mm) and mm > 2.4:
                    target_pct = 0.85

                # Gain-based late cycle caution (replaces month-based)
                if h_gain is not None and h_gain > 5.0:
                    target_pct = min(target_pct, 0.50)  # >500% gain: very late
                elif h_gain is not None and h_gain > 3.0:
                    target_pct = min(target_pct, 0.70)  # >300% gain: conservative

            else:
                # BEAR — gain-based floor
                if h_gain is not None:
                    if h_gain < 1.0:  # <100% gain: early, accumulate aggressively
                        floor = 0.52
                    elif h_gain < 3.0:  # 100-300%: normal
                        floor = 0.35
                    else:  # >300%: post-bubble, conservative
                        floor = 0.25
                else:
                    floor = 0.35

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
                    target_pct = max((btc * price) / cv if cv > 0 else floor, floor)

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
