"""
BTC Beast v6a — 多周期 SMA 组合
代码熊 🐻

改进点：用 SMA50/SMA100/SMA200 三重确认替代单一 SMA200
- price > SMA50 > SMA100 > SMA200: 全仓牛市
- price > SMA200 但 SMA 未对齐: 半牛（80%）
- price < SMA200: 熊市（保留 v5 熊市逻辑）
"""

import pandas as pd
import numpy as np

HALVING_DATES = [
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

def months_since_halving(date):
    past = [h for h in HALVING_DATES if h <= date]
    if not past:
        return None
    return (date - past[-1]).days / 30.44


class BTCBeastV6a:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.data = None
        self.results = None

    def load_csv(self, path, start='2017-01-01', end='2026-02-20'):
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        df = df[['Close']].dropna().sort_index()
        df = df.loc[start:end]
        self.data = df
        print(f"✅ v6a data: {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")

    def run_backtest(self):
        prices = self.data['Close']
        sma50 = prices.rolling(50).mean()
        sma100 = prices.rolling(100).mean()
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
            s50 = sma50.iloc[i]
            s100 = sma100.iloc[i]
            s200 = sma200.iloc[i]
            mm = mayer.iloc[i]
            rsi = rsi14.iloc[i]
            wret = weekly_ret.iloc[i]
            hm = months_since_halving(prices.index[i])

            if i >= 200 and not pd.isna(s200):
                if not in_bear and price < s200 * 0.92:
                    in_bear = True
                elif in_bear and price > s200 * 1.03:
                    in_bear = False

            if not in_bear:
                # Triple SMA alignment check
                if not pd.isna(s50) and not pd.isna(s100) and not pd.isna(s200):
                    if price > s50 > s100 > s200:
                        # Perfect alignment — full bull
                        target_pct = 1.00
                    elif price > s200:
                        # Above SMA200 but not aligned — cautious bull
                        target_pct = 0.80
                    else:
                        target_pct = 0.70
                else:
                    target_pct = 1.00

                # Mayer bubble protection
                if not pd.isna(mm) and mm > 3.5:
                    target_pct = min(target_pct, 0.50)
                elif not pd.isna(mm) and mm > 3.0:
                    target_pct = min(target_pct, 0.70)
                elif not pd.isna(mm) and mm > 2.4:
                    target_pct = min(target_pct, 0.85)

                if hm is not None and hm > 30:
                    target_pct = min(target_pct, 0.90)
            else:
                # BEAR — same as v5
                h_mult = 1.0
                if hm is not None:
                    if hm <= 6:
                        h_mult = 1.0 + 0.5 * (hm / 6)
                    elif hm <= 12:
                        h_mult = 1.5
                    elif hm <= 18:
                        h_mult = 1.5 - 0.5 * ((hm - 12) / 6)
                    elif hm <= 30:
                        h_mult = 1.0
                    elif hm <= 42:
                        h_mult = 1.0 - 0.15 * ((hm - 30) / 12)
                    else:
                        h_mult = 0.85

                floor = min(0.35 * h_mult, 0.50)
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
        dr = self.results.pct_change().dropna()
        self._dr = dr
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
