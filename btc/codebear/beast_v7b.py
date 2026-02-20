"""
BTC Beast v7b — GLD 对冲 (Bear Market Gold Allocation)
代码熊 🐻

核心改进：借鉴 TQQQ v9g 的 GLD 对冲思路
- 熊市时将闲置资金配置 GLD（而非全部持现金）
- 熊市配置：35% BTC + 35% GLD + 30% cash
- 牛市保持 v6b 逻辑（100% BTC）
- GLD 在危机期间提供正收益或至少保值

关键区别 vs TQQQ:
- BTC 24/7 交易，GLD 只在工作日交易 → 需要 forward-fill
- BTC 波动率 >> TQQQ，GLD 对冲效果可能更有限
- 但 2022-2024 GLD 表现优异（+40%），对 OOS 有帮助
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
    mask = prices_series.index >= last_h
    if mask.any():
        h_price = prices_series.loc[mask].iloc[0]
        gain = (price / h_price) - 1.0
    else:
        gain = 0.0
    return (date - last_h).days / 30.44, gain


class BTCBeastV7b:
    """GLD Hedge — allocate idle bear-market cash to gold."""

    GLD_BEAR_ALLOC = 0.35  # fraction of portfolio in GLD during bear

    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.data = None
        self.gld = None
        self.results = None

    def load_data(self, btc_path, gld_path, start='2017-01-01', end='2026-02-20'):
        btc = pd.read_csv(btc_path, parse_dates=['Date'], index_col='Date')
        btc = btc[['Close']].dropna().sort_index().loc[start:end]
        btc.columns = ['BTC']

        gld = pd.read_csv(gld_path, parse_dates=['Date'], index_col='Date')
        gld = gld[['Close']].dropna().sort_index()
        gld.columns = ['GLD']

        # Align GLD to BTC's daily index (forward-fill weekends/holidays)
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

        cash = self.initial_capital
        btc = 0.0
        gld = 0.0
        in_bear = False
        portfolio_values = []

        for i in range(len(btc_prices)):
            price_btc = btc_prices.iloc[i]
            price_gld = gld_prices.iloc[i]
            sma = sma200.iloc[i]
            mm = mayer.iloc[i]
            rsi = rsi14.iloc[i]
            wret = weekly_ret.iloc[i]
            hm, h_gain = halving_info(btc_prices.index[i], price_btc, btc_prices)

            # Regime detection
            if i >= 200 and not pd.isna(sma):
                if not in_bear and price_btc < sma * 0.92:
                    in_bear = True
                elif in_bear and price_btc > sma * 1.03:
                    in_bear = False

            if not in_bear:
                # BULL — same as v6b (100% BTC, no GLD)
                target_btc = 1.00
                target_gld = 0.00

                if not pd.isna(mm) and mm > 3.5:
                    target_btc = 0.50
                elif not pd.isna(mm) and mm > 3.0:
                    target_btc = 0.70
                elif not pd.isna(mm) and mm > 2.4:
                    target_btc = 0.85

                if h_gain is not None and h_gain > 5.0:
                    target_btc = min(target_btc, 0.50)
                elif h_gain is not None and h_gain > 3.0:
                    target_btc = min(target_btc, 0.70)

            else:
                # BEAR — v6b BTC logic + GLD hedge
                if h_gain is not None:
                    if h_gain < 1.0:
                        floor = 0.52
                    elif h_gain < 3.0:
                        floor = 0.35
                    else:
                        floor = 0.25
                else:
                    floor = 0.35

                if not pd.isna(rsi) and rsi < 20:
                    target_btc = 0.80
                elif not pd.isna(wret) and wret < -0.20:
                    target_btc = 0.70
                elif not pd.isna(rsi) and rsi < 30:
                    target_btc = 0.60
                elif not pd.isna(rsi) and rsi > 60:
                    target_btc = floor
                else:
                    cv = cash + btc * price_btc + gld * price_gld
                    target_btc = max((btc * price_btc) / cv if cv > 0 else floor, floor)

                # GLD gets a fixed allocation in bear
                target_gld = self.GLD_BEAR_ALLOC

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
