#!/usr/bin/env python3
"""
动量轮动 v5f — 条件式对冲：只在对冲资产有动量时才对冲
代码熊 🐻

核心假设:
  当GLD/TLT自身在下跌趋势中时，用它们对冲效果差甚至反向
  例如2022年：股票跌，GLD跌，TLT也跌 → 对冲失效
  
  解决方案: 只有当对冲资产有正动量时才触发对冲
  "有效对冲" = 遇到回撤 AND 对冲资产处于上升趋势

变种:
  v5f_a: DD响应 + 要求GLD 3m > 0才配置GLD
  v5f_b: DD响应 + 动态选GLD/TLT/SHY中动量最强正动量者
  v5f_c: 组合 (v5f_a) + 早期预警 (SPY 1m < -5% 时提前警戒)
  v5f_d: 三资产对冲池 (GLD/TLT/IEF)，只配置有正动量的那些

这个策略的理论优势:
  1. 避免2022年同向崩溃带来的对冲失效
  2. 在真正需要对冲时(对冲资产涨)，优先用效果最好的资产
  3. 在对冲资产下跌时，保持更高股票仓位获得正收益
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
    ret_1m  = close_df / close_df.shift(22) - 1
    ret_3m  = close_df / close_df.shift(63) - 1
    ret_6m  = close_df / close_df.shift(126) - 1
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
    spy_c = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_c) == 0:
        return 'bull'
    return 'bull' if spy_c.iloc[-1] > valid.iloc[-1] else 'bear'


def get_asset_momentum(prices, date, lookback=63):
    """Get n-month momentum of an asset at date."""
    avail = prices.loc[:date].dropna()
    if len(avail) < lookback + 1:
        return None
    return float(avail.iloc[-1] / avail.iloc[-lookback - 1] - 1)


def get_best_positive_hedge(safe_assets, date, min_momentum=0.0):
    """
    Find the safe asset with the highest positive 3-month momentum.
    Returns (ticker, momentum) or (None, None) if none qualify.
    """
    best_ticker, best_mom = None, -999
    for ticker, prices in safe_assets.items():
        mom = get_asset_momentum(prices, date)
        if mom is not None and mom > min_momentum and mom > best_mom:
            best_mom = mom
            best_ticker = ticker
    return best_ticker, best_mom if best_ticker else None


def select_stocks_v3b(signals, sectors, date, prev_holdings):
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


def add_hedge(weights, hedge_ticker, hedge_frac):
    if hedge_frac <= 0 or not weights or not hedge_ticker:
        return weights
    total_w = sum(weights.values())
    if total_w <= 0:
        return weights
    stock_frac = 1.0 - hedge_frac
    new = {t: (w / total_w) * stock_frac for t, w in weights.items()}
    new[hedge_ticker] = new.get(hedge_ticker, 0) + hedge_frac
    return new


def run_backtest(close_df, signals, sectors, safe_assets, variant='v5f_b',
                 start='2015-01-01', end='2025-12-31', cost_per_trade=0.0015):
    """
    Variants:
    v4d_base: original v4d (fixed GLD, no condition)
    v5f_a: GLD only when GLD momentum > 0
    v5f_b: best positive-momentum safe asset (GLD/TLT/IEF)
    v5f_c: v5f_b + early warning when SPY 1m < -5%
    v5f_d: split hedge across all positive-momentum safe assets
    """
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index

    portfolio_values, portfolio_dates = [], []
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    peak_value = 1.0
    gld_prices = safe_assets['GLD']

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]

        dd = (current_value - peak_value) / peak_value if peak_value > 0 else 0
        weights, regime = select_stocks_v3b(signals, sectors, date, prev_holdings)

        # Determine DD-responsive allocation
        dd_alloc = 0.0
        if dd < -0.15:
            dd_alloc = 0.50
        elif dd < -0.08:
            dd_alloc = 0.30

        if variant == 'v4d_base':
            # Original: always use GLD regardless of momentum
            hedge_ticker = 'GLD'
            hedge_frac = dd_alloc

        elif variant == 'v5f_a':
            # GLD only if GLD momentum > 0
            gld_mom = get_asset_momentum(safe_assets['GLD'], date)
            if dd_alloc > 0 and gld_mom is not None and gld_mom > 0:
                hedge_ticker = 'GLD'
                hedge_frac = dd_alloc
            elif dd_alloc > 0 and gld_mom is not None and gld_mom <= 0:
                # GLD trending down → use SHY instead (treasury bills, near-0 return)
                hedge_ticker = 'SHY'
                hedge_frac = dd_alloc * 0.5  # Half allocation when hedge trending down
            else:
                hedge_ticker = None
                hedge_frac = 0

        elif variant == 'v5f_b':
            # Best positive-momentum safe asset
            if dd_alloc > 0:
                best_t, best_m = get_best_positive_hedge(safe_assets, date)
                hedge_ticker = best_t
                hedge_frac = dd_alloc if best_t else 0
            else:
                hedge_ticker = None
                hedge_frac = 0

        elif variant == 'v5f_c':
            # v5f_b + early warning: if SPY 1-month < -5%, pre-hedge at 15%
            spy_1m = get_asset_momentum(signals['spy_close'], date, lookback=22)
            pre_hedge = spy_1m is not None and spy_1m < -0.05

            base_alloc = dd_alloc
            if pre_hedge and dd_alloc == 0:
                base_alloc = 0.15  # Early warning pre-hedge

            if base_alloc > 0:
                best_t, best_m = get_best_positive_hedge(safe_assets, date)
                hedge_ticker = best_t
                hedge_frac = base_alloc if best_t else 0
            else:
                hedge_ticker = None
                hedge_frac = 0

        elif variant == 'v5f_d':
            # Split hedge across all positive-momentum safe assets equally
            hedge_ticker = None
            hedge_frac = 0
            if dd_alloc > 0:
                positive_assets = {}
                for t, prices in safe_assets.items():
                    if t == 'SHY':
                        continue  # Exclude SHY from split (too low return)
                    mom = get_asset_momentum(prices, date)
                    if mom is not None and mom > 0:
                        positive_assets[t] = mom
                if positive_assets:
                    # Equal weight among positive-momentum hedges
                    n = len(positive_assets)
                    per_asset = dd_alloc / n
                    total_w = sum(weights.values())
                    stock_frac = 1.0 - dd_alloc
                    if total_w > 0:
                        weights = {t: (w / total_w) * stock_frac for t, w in weights.items()}
                    for t in positive_assets:
                        weights[t] = weights.get(t, 0) + per_asset
                elif dd_alloc > 0:
                    # All negative → use SHY as fallback
                    weights = add_hedge(weights, 'SHY', dd_alloc * 0.5)
                # Skip the add_hedge below since we handled it here
                hedge_ticker = None
                hedge_frac = 0
        else:
            hedge_ticker = None
            hedge_frac = 0

        if hedge_ticker and hedge_frac > 0:
            weights = add_hedge(weights, hedge_ticker, hedge_frac)

        all_t = set(list(weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = weights.copy()
        prev_holdings = set(k for k in weights.keys() if k not in safe_assets.keys())

        port_ret = 0.0
        for t, w in weights.items():
            if t in safe_assets:
                s = safe_assets[t].loc[date:next_date].dropna()
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
    print("🐻 动量轮动 v5f — 条件式对冲（只在对冲资产有动量时触发）")
    print("=" * 70)

    print("\n📊 Loading data...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)
    safe_assets = {
        'GLD': load_csv(CACHE / "GLD.csv")['Close'].dropna(),
        'TLT': load_csv(CACHE / "TLT.csv")['Close'].dropna(),
        'IEF': load_csv(CACHE / "IEF.csv")['Close'].dropna(),
        'SHY': load_csv(CACHE / "SHY.csv")['Close'].dropna(),
    }
    print(f"  Loaded {len(close_df.columns)} tickers")

    variants = ['v4d_base', 'v5f_a', 'v5f_b', 'v5f_c', 'v5f_d']
    variant_labels = {
        'v4d_base': 'v4d (GLD fixed)',
        'v5f_a': 'v5f_a GLD-conditional',
        'v5f_b': 'v5f_b Best safe asset',
        'v5f_c': 'v5f_c + early warn',
        'v5f_d': 'v5f_d Split hedge',
    }

    results = {}
    for var in variants:
        print(f"\n🔄 Running {variant_labels[var]}...")
        eq_full, to = run_backtest(close_df, signals, sectors, safe_assets, var)
        eq_is, _ = run_backtest(close_df, signals, sectors, safe_assets, var,
                                 start='2015-01-01', end='2020-12-31')
        eq_oos, _ = run_backtest(close_df, signals, sectors, safe_assets, var,
                                  start='2021-01-01', end='2025-12-31')

        m = compute_metrics(eq_full, var)
        m_is = compute_metrics(eq_is)
        m_oos = compute_metrics(eq_oos)
        wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
        comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2
        results[var] = {'m': m, 'is': m_is, 'oos': m_oos, 'wf': wf, 'comp': comp, 'to': to}
        print(f"  CAGR: {m['cagr']:.1%}  Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_dd']:.1%}  "
              f"Calmar: {m['calmar']:.2f}  Comp: {comp:.3f}  WF: {wf:.2f} {'✅' if wf >= 0.70 else '❌'}")

    print("\n" + "=" * 105)
    print(f"{'Variant':<24} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 105)
    for var in variants:
        r = results[var]
        m = r['m']
        wf_flag = "✅" if r['wf'] >= 0.70 else "❌"
        print(f"{variant_labels[var]:<24} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
              f"{r['is']['sharpe']:>7.2f} {r['oos']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f}{wf_flag} {r['comp']:>8.3f}")

    best = max((r for r in results.items() if r[1]['wf'] >= 0.70),
               key=lambda x: x[1]['comp'], default=None)
    if best:
        var, r = best
        m = r['m']
        v4d = results['v4d_base']
        print(f"\n🏆 Best variant: {variant_labels[var]} (Composite: {r['comp']:.3f})")
        print(f"  vs v4d_base: CAGR {v4d['m']['cagr']:.1%}→{m['cagr']:.1%}, "
              f"MaxDD {v4d['m']['max_dd']:.1%}→{m['max_dd']:.1%}, "
              f"Sharpe {v4d['m']['sharpe']:.2f}→{m['sharpe']:.2f}, "
              f"Comp {v4d['comp']:.3f}→{r['comp']:.3f}")
        if r['comp'] > 1.8 or m['sharpe'] > 2.0:
            print("\n🚨🚨🚨 【重大突破】 Composite > 1.8 or Sharpe > 2.0! 🚨🚨🚨")
        elif r['comp'] > 1.356:
            print(f"✅ Improvement over v4d! Composite {r['comp']:.3f} > 1.356")
        else:
            print(f"⚠️ No improvement over v4d (Composite {r['comp']:.3f} < 1.356)")

    results_file = Path(__file__).parent / "momentum_v5f_results.json"
    out = {var: {'cagr': float(r['m']['cagr']), 'max_dd': float(r['m']['max_dd']),
                  'sharpe': float(r['m']['sharpe']), 'calmar': float(r['m']['calmar']),
                  'wf': float(r['wf']), 'composite': float(r['comp'])}
           for var, r in results.items()}
    with open(results_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    return results


if __name__ == '__main__':
    main()
