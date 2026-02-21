#!/usr/bin/env python3
"""
动量轮动 v5d — v4d + 波动率预警对冲 (Proactive Vol-Triggered Hedge)
代码熊 🐻

核心假设：
  v4d 的 DD-responsive 对冲是"亡羊补牢"——必须先亏8%才触发
  v5d 添加"预警"机制：SPY 短期波动率飙升时提前对冲

策略逻辑：
  1. 股票选择: 完全继承 v3b（不变）
  2. 对冲层一（预警）: SPY_20d_vol > threshold → 提前配置 GLD
     - SPY_vol > 22%: 15% GLD
     - SPY_vol > 30%: 30% GLD  
  3. 对冲层二（DD响应）: 同 v4d
     - DD < -8%: +15% GLD (on top of vol hedge, total capped 50%)
     - DD < -15%: +30% GLD
  4. 总对冲上限: 60%

为什么可能有效:
  - 2018 Q4: SPY vol飙到38%，提前触发对冲可早于DD响应
  - 2020 COVID: Vol先飙升再暴跌，提前对冲减少损失
  - 问题: 2022年高通胀期vol持续高位但GLD也下跌
  
附加变种:
  v5d_vol: 纯波动率触发（不含DD响应）
  v5d_combo: 波动率预警 + DD响应（主策略）
  v5d_early: 更激进预警 (vol > 18% → 10% GLD)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

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
    vol_20d = log_ret.rolling(20).std() * np.sqrt(252)
    spy_close = close_df['SPY'] if 'SPY' in close_df.columns else None
    spy_sma200 = spy_close.rolling(200).mean() if spy_close is not None else None
    spy_vol_20d = vol_20d['SPY'] if spy_close is not None and 'SPY' in vol_20d.columns else None
    sma_50 = close_df.rolling(50).mean()
    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'vol_30d': vol_30d, 'vol_20d': vol_20d,
        'spy_sma200': spy_sma200, 'spy_close': spy_close, 'spy_vol_20d': spy_vol_20d,
        'sma_50': sma_50, 'close': close_df,
    }


def get_regime(signals, date):
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid = spy_sma.loc[:date].dropna()
    spy_c = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_c) == 0:
        return 'bull'
    return 'bull' if spy_c.iloc[-1] > valid.iloc[-1] else 'bear'


def get_spy_vol(signals, date):
    """Get SPY realized 20-day volatility at date."""
    if signals['spy_vol_20d'] is None:
        return 0.15
    v = signals['spy_vol_20d'].loc[:date].dropna()
    if len(v) == 0:
        return 0.15
    return float(v.iloc[-1])


def select_stocks_v3b(signals, sectors, date, prev_holdings):
    """v3b stock selection — exactly as in the champion."""
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}, 'bull'
    idx = idx[-1]
    regime = get_regime(signals, date)

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
    df = df[df['price'] > df['sma50']]

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03

    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)

    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 2
        cash = 0.20

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())

    if not selected:
        return {}, regime

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
    weights = {t: ((1 - mom_share) * iv_w[t] + mom_share * mw_w[t]) * invested for t in selected}
    return weights, regime


def compute_gld_alloc(spy_vol, dd, variant):
    """
    Compute GLD allocation based on variant:
    - vol: vol-triggered only
    - dd: dd-responsive only (v4d)
    - combo: vol pre-warning + dd
    - early: more aggressive vol trigger
    """
    if variant == 'dd':
        # v4d original
        if dd < -0.15:
            return 0.50
        elif dd < -0.08:
            return 0.30
        return 0.0

    elif variant == 'vol':
        # Pure vol-triggered
        if spy_vol > 0.30:
            return 0.30
        elif spy_vol > 0.22:
            return 0.15
        return 0.0

    elif variant == 'combo':
        # Vol pre-warning + DD response
        vol_alloc = 0.0
        if spy_vol > 0.30:
            vol_alloc = 0.25
        elif spy_vol > 0.22:
            vol_alloc = 0.12

        dd_alloc = 0.0
        if dd < -0.15:
            dd_alloc = 0.25
        elif dd < -0.08:
            dd_alloc = 0.15

        return min(vol_alloc + dd_alloc, 0.50)

    elif variant == 'early':
        # Aggressive early warning
        if spy_vol > 0.30:
            return 0.35
        elif spy_vol > 0.22:
            return 0.20
        elif spy_vol > 0.18:
            return 0.10
        return 0.0

    elif variant == 'combo_early':
        # Early vol + DD
        vol_alloc = 0.0
        if spy_vol > 0.28:
            vol_alloc = 0.25
        elif spy_vol > 0.20:
            vol_alloc = 0.15
        elif spy_vol > 0.17:
            vol_alloc = 0.08

        dd_alloc = 0.0
        if dd < -0.12:
            dd_alloc = 0.25
        elif dd < -0.07:
            dd_alloc = 0.12

        return min(vol_alloc + dd_alloc, 0.50)

    return 0.0


def add_gld(weights, gld_frac):
    if gld_frac <= 0 or not weights:
        return weights
    total_w = sum(weights.values())
    if total_w <= 0:
        return weights
    stock_frac = 1.0 - gld_frac
    new = {t: (w / total_w) * stock_frac for t, w in weights.items()}
    new['GLD'] = gld_frac
    return new


def run_backtest(close_df, signals, sectors, gld_prices, variant='combo',
                 start='2015-01-01', end='2025-12-31', cost_per_trade=0.0015):
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index

    portfolio_values, portfolio_dates = [], []
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    peak_value = 1.0

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]

        dd = (current_value - peak_value) / peak_value if peak_value > 0 else 0
        spy_vol = get_spy_vol(signals, date)

        weights, regime = select_stocks_v3b(signals, sectors, date, prev_holdings)
        gld_alloc = compute_gld_alloc(spy_vol, dd, variant)
        new_weights = add_gld(weights, gld_alloc)

        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = new_weights.copy()
        prev_holdings = set(k for k in new_weights.keys() if k != 'GLD')

        port_ret = 0.0
        for t, w in new_weights.items():
            if t == 'GLD':
                s = gld_prices.loc[date:next_date].dropna()
                if len(s) >= 2:
                    port_ret += (s.iloc[-1] / s.iloc[0] - 1) * w
            elif t in close_df.columns:
                s = close_df[t].loc[date:next_date].dropna()
                if len(s) >= 2:
                    port_ret += (s.iloc[-1] / s.iloc[0] - 1) * w

        port_ret -= turnover * cost_per_trade * 2
        current_value *= (1 + port_ret)
        if current_value > peak_value:
            peak_value = current_value
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)

    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    return equity, np.mean(turnover_list) if turnover_list else 0


def compute_metrics(equity, name="Strategy"):
    if len(equity) < 3:
        return {'name': name, 'cagr': 0, 'max_dd': 0, 'sharpe': 0, 'calmar': 0}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years < 0.5:
        return {'name': name, 'cagr': 0, 'max_dd': 0, 'sharpe': 0, 'calmar': 0}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    monthly = equity.pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    dd = (equity - equity.cummax()) / equity.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'name': name, 'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar}


def main():
    print("=" * 70)
    print("🐻 动量轮动 v5d — v4d + 波动率预警对冲")
    print("=" * 70)

    print("\n📊 Loading data...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)
    gld_prices = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    print(f"  Loaded {len(close_df.columns)} tickers")

    variants = ['dd', 'vol', 'combo', 'early', 'combo_early']
    variant_names = {
        'dd': 'v4d (DD only)',
        'vol': 'vol-only',
        'combo': 'vol+DD combo',
        'early': 'early vol',
        'combo_early': 'early vol+DD',
    }

    results = {}
    for var in variants:
        print(f"\n🔄 Running {variant_names[var]}...")
        eq_full, to = run_backtest(close_df, signals, sectors, gld_prices, var)
        eq_is, _ = run_backtest(close_df, signals, sectors, gld_prices, var,
                                 start='2015-01-01', end='2020-12-31')
        eq_oos, _ = run_backtest(close_df, signals, sectors, gld_prices, var,
                                  start='2021-01-01', end='2025-12-31')

        m = compute_metrics(eq_full, var)
        m_is = compute_metrics(eq_is)
        m_oos = compute_metrics(eq_oos)
        wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
        comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2
        results[var] = {'m': m, 'is': m_is, 'oos': m_oos, 'wf': wf, 'comp': comp, 'to': to}

    print("\n" + "=" * 100)
    print(f"{'Variant':<20} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 100)
    for var in variants:
        r = results[var]
        m = r['m']
        wf_flag = "✅" if r['wf'] >= 0.70 else "❌"
        print(f"{variant_names[var]:<20} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
              f"{r['is']['sharpe']:>7.2f} {r['oos']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f}{wf_flag} {r['comp']:>8.3f}")

    best = max((r for r in results.items() if r[1]['wf'] >= 0.70),
               key=lambda x: x[1]['comp'], default=None)
    if best:
        var, r = best
        m = r['m']
        m_base = results['dd']['m']
        print(f"\n🏆 Best variant: {variant_names[var]} (Composite: {r['comp']:.3f})")
        print(f"  vs v4d DD: CAGR {m_base['cagr']:.1%}→{m['cagr']:.1%}, "
              f"MaxDD {m_base['max_dd']:.1%}→{m['max_dd']:.1%}, "
              f"Sharpe {m_base['sharpe']:.2f}→{m['sharpe']:.2f}, "
              f"Comp {results['dd']['comp']:.3f}→{r['comp']:.3f}")

        if r['comp'] > 1.8 or m['sharpe'] > 2.0:
            print("\n🚨🚨🚨 【重大突破】 Composite > 1.8 or Sharpe > 2.0! 🚨🚨🚨")
        elif r['comp'] > 1.356:
            print(f"\n✅ Improvement over v4d champion (Composite {r['comp']:.3f} > 1.356)")
        else:
            print(f"\n⚠️ No improvement over v4d (best Composite {r['comp']:.3f} < 1.356)")

    # Save
    results_file = Path(__file__).parent / "momentum_v5d_results.json"
    out = {}
    for var, r in results.items():
        out[var] = {
            'cagr': float(r['m']['cagr']), 'max_dd': float(r['m']['max_dd']),
            'sharpe': float(r['m']['sharpe']), 'calmar': float(r['m']['calmar']),
            'wf': float(r['wf']), 'composite': float(r['comp']),
        }
    with open(results_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    return results


if __name__ == '__main__':
    main()
