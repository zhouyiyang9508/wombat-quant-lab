#!/usr/bin/env python3
"""
动量轮动 v3b — Sector Rotation + SMA50 + Blend30 + Tighter Bear ⭐⭐ CHAMPION
代码熊 🐻

v3 系列最优策略！Composite 1.173 (vs v2d 1.013, +15.8%)

核心创新（在v3a基础上）:
1. SMA50 趋势过滤: 只买价格在50日均线以上的股票（减少MaxDD 4.2pp）
2. 两阶段行业轮动: 先选Top行业, 再选行业内Top个股（提升Sharpe +0.12）
3. 混合加权: 70% inverse-vol + 30% momentum-proportional（更多资金给最强股）
4. 更紧的熊市处理: 3行业×2股 + 20%现金（熊市集中最强）

策略逻辑:
Bull (SPY > SMA200):
  - 选Top 4行业 × Top 3股票 = 12只
  - 100% invested
  - 70% 反波动率 + 30% 动量权重混合
Bear (SPY < SMA200):
  - 选Top 3行业 × Top 2股票 = 6只
  - 80% invested + 20% cash
  - 70% 反波动率 + 30% 动量权重混合

Results:
  Full (2015-2025): CAGR 25.8%, Sharpe 1.35, MaxDD -17.7%, Calmar 1.46
  IS (2015-2020): Sharpe 1.39
  OOS (2021-2025): Sharpe 1.18
  WF ratio: 0.85 ✅ (target ≥ 0.70)
  Turnover: 61.2%/month
  Composite: 1.173

vs v2d:
  MaxDD: -21.9% → -17.7% (+4.2pp !!!)
  Sharpe: 1.22 → 1.35 (+0.13)
  Calmar: 1.18 → 1.46 (+0.28)
  OOS Sharpe: 1.00 → 1.18 (+0.18)
  WF: 0.73 → 0.85 (+0.12)
  Composite: 1.013 → 1.173 (+0.160)

为什么v3b是冠军:
- SMA50 过滤在2022熊市中自动淘汰下跌股票，减少回撤
- 行业轮动让策略在能源(2022)和科技(2023-24)之间自然切换
- 混合加权给最强动量股更多资金，放大alpha
- 紧熊市处理(3×2=6只)在衰退中更集中
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


def load_csv(filepath):
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    for col in ['Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_all_data(tickers):
    close_dict = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                close_dict[t] = df['Close'].dropna()
        except:
            pass
    return pd.DataFrame(close_dict)


def precompute_signals(close_df):
    ret_1m = close_df / close_df.shift(22) - 1
    ret_3m = close_df / close_df.shift(63) - 1
    ret_6m = close_df / close_df.shift(126) - 1
    ret_12m = close_df / close_df.shift(252) - 1
    log_ret = np.log(close_df / close_df.shift(1))
    vol_30d = log_ret.rolling(30).std() * np.sqrt(252)
    spy_close = close_df['SPY'] if 'SPY' in close_df.columns else None
    spy_sma200 = spy_close.rolling(200).mean() if spy_close is not None else None
    sma_50 = close_df.rolling(50).mean()
    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'vol_30d': vol_30d, 'spy_sma200': spy_sma200, 'spy_close': spy_close,
        'sma_50': sma_50, 'close': close_df,
    }


def get_regime(signals, date):
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid = spy_sma.loc[:date].dropna()
    spy_close = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_close) == 0:
        return 'bull'
    return 'bull' if spy_close.iloc[-1] > valid.iloc[-1] else 'bear'


def select_stocks(signals, sectors, date, prev_holdings):
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime(signals, date)

    # Blended momentum
    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'momentum': mom, 'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx], 'price': close.loc[idx],
        'sma50': signals['sma_50'].loc[idx],
    }).dropna(subset=['momentum', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    df = df[df['price'] > df['sma50']]  # SMA50 trend filter

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03

    # Two-stage sector rotation
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)

    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3  # stocks per sector
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 2  # tighter bear
        cash = 0.20

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())

    if not selected:
        return {}

    # Blended weighting: 70% inverse-vol + 30% momentum-proportional
    mom_share = 0.30
    iv = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    iv_total = sum(iv.values())
    iv_w = {t: v / iv_total for t, v in iv.items()}

    mom_min = min(df.loc[t, 'momentum'] for t in selected)
    shift = max(-mom_min + 0.01, 0)
    mw = {t: df.loc[t, 'momentum'] + shift for t in selected}
    mw_total = sum(mw.values())
    mw_w = {t: v / mw_total for t, v in mw.items()}

    invested = 1.0 - cash
    return {t: ((1 - mom_share) * iv_w[t] + mom_share * mw_w[t]) * invested for t in selected}


def run_backtest(close_df, signals, sectors, start='2015-01-01', end='2025-12-31',
                 cost_per_trade=0.0015):
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    portfolio_values, portfolio_dates = [], []
    holdings_history = {}
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]
        new_weights = select_stocks(signals, sectors, date, prev_holdings)
        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = new_weights.copy()
        prev_holdings = set(new_weights.keys())
        holdings_history[date.strftime('%Y-%m')] = list(new_weights.keys())

        port_ret = sum(
            (close_df[t].loc[date:next_date].dropna().iloc[-1] /
             close_df[t].loc[date:next_date].dropna().iloc[0] - 1) * w
            for t, w in new_weights.items()
            if len(close_df[t].loc[date:next_date].dropna()) >= 2
        )
        port_ret -= turnover * cost_per_trade * 2
        current_value *= (1 + port_ret)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)

    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    return equity, holdings_history, np.mean(turnover_list)


def compute_metrics(equity, name="Strategy"):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    monthly = equity.pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    dd = (equity - equity.cummax()) / equity.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'name': name, 'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar}


def main():
    print("=" * 60)
    print("🐻 动量轮动 v3b — CHAMPION ⭐⭐")
    print("  Sector Rotation + SMA50 + Blend30 + Tighter Bear")
    print("=" * 60)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)

    eq_full, hold, to = run_backtest(close_df, signals, sectors)
    m = compute_metrics(eq_full)
    eq_is, _, _ = run_backtest(close_df, signals, sectors, '2015-01-01', '2020-12-31')
    eq_oos, _, _ = run_backtest(close_df, signals, sectors, '2021-01-01', '2025-12-31')
    m_is = compute_metrics(eq_is, "IS")
    m_oos = compute_metrics(eq_oos, "OOS")
    wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
    comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2

    print(f"\nFull (2015-2025):")
    print(f"  CAGR: {m['cagr']:.1%}  Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_dd']:.1%}  Calmar: {m['calmar']:.2f}")
    print(f"\nWalk-Forward:")
    print(f"  IS Sharpe:  {m_is['sharpe']:.2f}")
    print(f"  OOS Sharpe: {m_oos['sharpe']:.2f}")
    print(f"  WF ratio:   {wf:.2f} {'✅' if wf >= 0.70 else '❌'}")
    print(f"\nTurnover: {to:.1%}/month")
    print(f"Composite: {comp:.3f}")

    # Holdings
    hot = {'NVDA', 'TSLA', 'META', 'AVGO', 'AMD', 'SMCI', 'PLTR'}
    print(f"\n2023-2024 Holdings:")
    for ym in sorted(hold.keys()):
        if ym.startswith('2023') or ym.startswith('2024'):
            h = [s for s in hold[ym] if s in hot]
            print(f"  {ym}: {', '.join(hold[ym][:10])} {'🔥' + ','.join(h) if h else ''}")

    return m, m_is, m_oos, hold


if __name__ == '__main__':
    main()
