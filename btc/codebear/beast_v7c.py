"""
BTC Beast v7c — Soft Bear + GLD Combo (Best of Both Worlds)
代码熊 🐻

核心改进：结合 v7a 和 v7b 的优点
- 熊市：45% BTC + 30% GLD + 25% cash
- 早期周期：55% BTC + 25% GLD + 20% cash
- 牛市：90% BTC + 5% GLD + 5% cash（小部分 GLD 做常驻对冲）
- 绝对动量过滤：6M <-30% 时减少 BTC 增加 GLD

策略哲学：
- "永远不要 100% 看多或 100% 看空"
- 牛市保留少量 GLD 作保险
- 熊市保持较高 BTC 仓位 + GLD 缓冲
- 目标：降低 IS/OOS Sharpe 差异以通过 Walk-Forward
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


class BTCBeastV7c:
    """Soft Bear + GLD Combo — always-invested with gold buffer."""

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
        mom_6m = btc_prices.pct_change(180)

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
            mom6 = mom_6m.iloc[i]
            hm, h_gain = halving_info(btc_prices.index[i], price_btc, btc_prices)

            # Regime detection
            if i >= 200 and not pd.isna(sma):
                if not in_bear and price_btc < sma * 0.92:
                    in_bear = True
                elif in_bear and price_btc > sma * 1.03:
                    in_bear = False

            if not in_bear:
                # BULL — mostly BTC, small GLD hedge
                target_btc = 0.92
                target_gld = 0.05

                # Mayer bubble: reduce BTC, increase GLD
                if not pd.isna(mm) and mm > 3.5:
                    target_btc = 0.45
                    target_gld = 0.20
                elif not pd.isna(mm) and mm > 3.0:
                    target_btc = 0.60
                    target_gld = 0.15
                elif not pd.isna(mm) and mm > 2.4:
                    target_btc = 0.75
                    target_gld = 0.10

                # Gain-based late cycle
                if h_gain is not None and h_gain > 5.0:
                    target_btc = min(target_btc, 0.45)
                    target_gld = max(target_gld, 0.20)
                elif h_gain is not None and h_gain > 3.0:
                    target_btc = min(target_btc, 0.60)
                    target_gld = max(target_gld, 0.15)

            else:
                # BEAR — Soft Bear + GLD
                if h_gain is not None:
                    if h_gain < 1.0:
                        # Early cycle: aggressive BTC + some GLD
                        floor_btc = 0.55
                        target_gld = 0.25
                    elif h_gain < 3.0:
                        # Normal: soft bear
                        floor_btc = 0.45
                        target_gld = 0.30
                    else:
                        # Post-bubble: more GLD, less BTC
                        floor_btc = 0.30
                        target_gld = 0.35
                else:
                    floor_btc = 0.45
                    target_gld = 0.30

                # Absolute momentum filter: deep crash → reduce BTC, boost GLD
                if not pd.isna(mom6) and mom6 < -0.30:
                    floor_btc = max(floor_btc - 0.15, 0.20)
                    target_gld = min(target_gld + 0.10, 0.45)

                # RSI tactical
                if not pd.isna(rsi) and rsi < 20:
                    target_btc = 0.75
                    target_gld = 0.15
                elif not pd.isna(wret) and wret < -0.20:
                    target_btc = 0.65
                    target_gld = 0.20
                elif not pd.isna(rsi) and rsi < 30:
                    target_btc = 0.55
                    target_gld = 0.25
                elif not pd.isna(rsi) and rsi > 60:
                    target_btc = floor_btc
                else:
                    cv = cash + btc * price_btc + gld * price_gld
                    current = (btc * price_btc) / cv if cv > 0 else floor_btc
                    target_btc = max(current, floor_btc)

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
