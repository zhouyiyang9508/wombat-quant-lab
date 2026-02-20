#!/usr/bin/env python3
"""
动量轮动选股策略 v2 — 全面优化探索（向量化版本）
代码熊 🐻

v2a: Regime Filter + Absolute Momentum
v2b: Bimonthly + Skip-1M + Holdover bonus
v2c: Sector Diversified + Vol-Weighted + Regime
v2d: Adaptive Regime + Dynamic Size + Sector + Vol
v2e: Conservative Combo (bimonthly + all filters)
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

# ─── Data Loading ───────────────────────────────────────────────

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
    close_dict, volume_dict = {}, {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                close_dict[t] = df['Close'].dropna()
                if 'Volume' in df.columns:
                    volume_dict[t] = df['Volume'].dropna()
        except:
            pass
    return pd.DataFrame(close_dict), pd.DataFrame(volume_dict), len(close_dict)

def load_sectors():
    sf = CACHE / "sp500_sectors.json"
    if sf.exists():
        with open(sf) as f:
            return json.load(f)
    return {}

# ─── Precompute signals (vectorized) ───────────────────────────

def precompute_signals(close_df, volume_df):
    """Precompute all signals needed for strategies."""
    print("  Precomputing signals...", flush=True)
    
    # Returns at different lookbacks
    ret_1m = close_df / close_df.shift(22) - 1
    ret_3m = close_df / close_df.shift(63) - 1
    ret_6m = close_df / close_df.shift(126) - 1
    ret_12m = close_df / close_df.shift(252) - 1
    
    # Skip-1-month returns (measured from 22 days ago)
    shifted = close_df.shift(22)
    ret_s1m = shifted / shifted.shift(22) - 1  # 1M return, 1M ago
    ret_s3m = shifted / shifted.shift(63) - 1
    ret_s6m = shifted / shifted.shift(126) - 1
    ret_s12m = shifted / shifted.shift(252) - 1
    
    # 30-day rolling volatility (annualized)
    log_ret = np.log(close_df / close_df.shift(1))
    vol_30d = log_ret.rolling(30).std() * np.sqrt(252)
    
    # SPY SMA200 regime
    spy_sma200 = close_df['SPY'].rolling(200).mean() if 'SPY' in close_df.columns else None
    
    # Volume 20d average
    vol_avg_20 = volume_df.rolling(20).mean() if not volume_df.empty else None
    
    signals = {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'ret_s1m': ret_s1m, 'ret_s3m': ret_s3m, 'ret_s6m': ret_s6m, 'ret_s12m': ret_s12m,
        'vol_30d': vol_30d, 'spy_sma200': spy_sma200, 'vol_avg_20': vol_avg_20,
        'close': close_df,
    }
    print("  Signals precomputed.", flush=True)
    return signals


def get_scores_at_date(signals, date, weights=(0.25, 0.40, 0.35, 0.0), 
                       skip_recent=False, min_price=5.0):
    """Get momentum scores for all stocks at a given date, vectorized."""
    close = signals['close']
    
    # Get the date index closest to but not after `date`
    valid_idx = close.index[close.index <= date]
    if len(valid_idx) == 0:
        return pd.DataFrame()
    idx = valid_idx[-1]
    
    w1m, w3m, w6m, w12m = weights
    
    if skip_recent:
        r1 = signals['ret_s1m'].loc[idx] * w1m if w1m > 0 else 0
        r3 = signals['ret_s3m'].loc[idx] * w3m if w3m > 0 else 0
        r6 = signals['ret_s6m'].loc[idx] * w6m if w6m > 0 else 0
        r12 = signals['ret_s12m'].loc[idx] * w12m if w12m > 0 else 0
    else:
        r1 = signals['ret_1m'].loc[idx] * w1m if w1m > 0 else 0
        r3 = signals['ret_3m'].loc[idx] * w3m if w3m > 0 else 0
        r6 = signals['ret_6m'].loc[idx] * w6m if w6m > 0 else 0
        r12 = signals['ret_12m'].loc[idx] * w12m if w12m > 0 else 0
    
    momentum = r1 + r3 + r6 + r12
    abs_6m = signals['ret_6m'].loc[idx]
    vol = signals['vol_30d'].loc[idx]
    price = close.loc[idx]
    
    df = pd.DataFrame({
        'momentum': momentum,
        'abs_6m': abs_6m,
        'vol_30d': vol,
        'price': price,
    })
    
    # Basic filters: valid data, min price, min history
    df = df.dropna(subset=['momentum'])
    df = df[df['price'] >= min_price]
    df = df[df.index != 'SPY']
    
    return df


def get_regime(signals, date):
    """Check SPY regime at date."""
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid = spy_sma.loc[:date].dropna()
    if len(valid) == 0:
        return 'bull'
    spy_close = signals['close']['SPY'].loc[:date].dropna()
    if len(spy_close) == 0:
        return 'bull'
    return 'bull' if spy_close.iloc[-1] > valid.iloc[-1] else 'bear'


# ─── Strategy Functions ─────────────────────────────────────────

def strategy_v1(signals, sectors, date, prev_holdings):
    """v1 baseline: Pure momentum Top 10, equal weight."""
    df = get_scores_at_date(signals, date, weights=(0.25, 0.40, 0.35, 0.0))
    if df.empty:
        return {}
    top = df.nlargest(10, 'momentum')
    n = len(top)
    return {t: 1.0/n for t in top.index} if n > 0 else {}


def strategy_v2a(signals, sectors, date, prev_holdings):
    """v2a: Regime + Absolute Momentum Filter."""
    if get_regime(signals, date) == 'bear':
        return {}
    
    df = get_scores_at_date(signals, date, weights=(0.25, 0.40, 0.35, 0.0))
    if df.empty:
        return {}
    
    df = df[df['abs_6m'] > 0]  # Absolute momentum filter
    top = df.nlargest(10, 'momentum')
    n = len(top)
    return {t: 1.0/n for t in top.index} if n > 0 else {}


def strategy_v2b(signals, sectors, date, prev_holdings):
    """v2b: Skip-1M + Holdover bonus + Regime + Abs Momentum."""
    if get_regime(signals, date) == 'bear':
        return {}
    
    df = get_scores_at_date(signals, date, 
                            weights=(0.0, 0.40, 0.35, 0.25),
                            skip_recent=True)
    if df.empty:
        return {}
    
    df = df[df['abs_6m'] > 0]
    
    # Holdover bonus
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.05
    
    top = df.nlargest(10, 'momentum')
    n = len(top)
    return {t: 1.0/n for t in top.index} if n > 0 else {}


def strategy_v2c(signals, sectors, date, prev_holdings):
    """v2c: Sector-Diversified + Vol-Weighted + Regime + Abs Momentum."""
    if get_regime(signals, date) == 'bear':
        return {}
    
    df = get_scores_at_date(signals, date, weights=(0.25, 0.40, 0.35, 0.0))
    if df.empty:
        return {}
    
    df = df[df['abs_6m'] > 0]
    df = df.sort_values('momentum', ascending=False)
    
    # Sector-diversified selection (max 3 per sector)
    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < 3:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= 10:
            break
    
    if not selected:
        return {}
    
    # Inverse-vol weighting
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    return {t: v/total for t, v in inv_vols.items()}


def strategy_v2d(signals, sectors, date, prev_holdings):
    """v2d: Adaptive Regime + Dynamic Size + Sector + Vol + Holdover."""
    regime = get_regime(signals, date)
    
    df = get_scores_at_date(signals, date, weights=(0.20, 0.40, 0.30, 0.10))
    if df.empty:
        return {}
    
    # Absolute momentum + vol filter
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    
    # Holdover bonus
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    
    df = df.sort_values('momentum', ascending=False)
    
    if regime == 'bull':
        top_n, max_sec, cash = 12, 3, 0.0
    else:
        top_n, max_sec, cash = 5, 2, 0.50
    
    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < max_sec:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= top_n:
            break
    
    if not selected:
        return {}
    
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v/total) * invested for t, v in inv_vols.items()}


def strategy_v2e(signals, sectors, date, prev_holdings):
    """v2e: Conservative Combo (skip-1M + sector + vol + holdover + regime)."""
    if get_regime(signals, date) == 'bear':
        return {}
    
    df = get_scores_at_date(signals, date, 
                            weights=(0.0, 0.35, 0.35, 0.30),
                            skip_recent=True)
    if df.empty:
        return {}
    
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.60)]
    
    # Large holdover bonus (reduce turnover)
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.08
    
    df = df.sort_values('momentum', ascending=False)
    
    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < 3:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= 10:
            break
    
    if not selected:
        return {}
    
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    return {t: v/total for t, v in inv_vols.items()}


# ─── Backtest Engine ────────────────────────────────────────────

def run_backtest(close_df, signals, sectors, strategy_fn,
                 start='2015-01-01', end='2025-12-31',
                 rebalance_months=1, cost_per_trade=0.0015):
    """Generic monthly backtest."""
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    
    portfolio_values = []
    portfolio_dates = []
    holdings_history = {}
    turnover_list = []
    prev_weights = {}
    prev_holdings = set()
    current_value = 1.0
    rebal_counter = 0
    
    for i in range(len(month_ends) - 1):
        date = month_ends[i]
        next_date = month_ends[i + 1]
        rebal_counter += 1
        
        is_rebal = (rebal_counter >= rebalance_months) or (i == 0)
        if is_rebal:
            rebal_counter = 0
        
        if is_rebal:
            new_weights = strategy_fn(signals, sectors, date, prev_holdings)
            
            # Turnover
            all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
            turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
            turnover_list.append(turnover)
            
            current_weights = new_weights
            prev_weights = new_weights.copy()
            prev_holdings = set(new_weights.keys())
            holdings_history[date.strftime('%Y-%m')] = list(new_weights.keys())
        else:
            current_weights = prev_weights
        
        # Period return
        invested = sum(current_weights.values())
        port_ret = 0.0
        for t, w in current_weights.items():
            try:
                p = close_df[t].loc[date:next_date].dropna()
                if len(p) >= 2:
                    port_ret += (p.iloc[-1] / p.iloc[0] - 1) * w
            except:
                pass
        
        # Cash portion earns 0%
        if is_rebal:
            port_ret -= turnover * cost_per_trade * 2
        
        current_value *= (1 + port_ret)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)
    
    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    avg_to = np.mean(turnover_list) if turnover_list else 0
    return equity, holdings_history, avg_to


def compute_metrics(equity, name="Strategy"):
    if len(equity) < 2:
        return {'name': name, 'cagr': 0, 'total_return': 0, 'max_dd': 0,
                'sharpe': 0, 'calmar': 0, 'win_rate': 0, 'max_dd_date': 'N/A'}
    
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
    monthly = equity.pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    max_dd = dd.min()
    max_dd_date = dd.idxmin()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'name': name, 'cagr': cagr, 'max_dd': max_dd,
        'max_dd_date': str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else 'N/A',
        'sharpe': sharpe, 'calmar': calmar,
        'win_rate': (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0,
    }


# ─── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🐻 代码熊 — 动量轮动选股策略 v2 全面优化")
    print("=" * 70, flush=True)
    
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df, volume_df, loaded = load_all_data(tickers + ['SPY'])
    sectors = load_sectors()
    print(f"Loaded {loaded} stocks, {len(sectors)} sectors", flush=True)
    
    # Precompute all signals (vectorized)
    signals = precompute_signals(close_df, volume_df)
    
    strategies = [
        ('v1_baseline', strategy_v1, 1, 'v1: 纯动量Top10等权月度'),
        ('v2a_regime', strategy_v2a, 1, 'v2a: Regime+绝对动量'),
        ('v2b_lowturn', strategy_v2b, 2, 'v2b: 双月+Skip1M+惯性'),
        ('v2c_sector', strategy_v2c, 1, 'v2c: 行业分散+Vol权重+Regime'),
        ('v2d_adaptive', strategy_v2d, 1, 'v2d: 自适应持仓+行业+Vol'),
        ('v2e_conservative', strategy_v2e, 2, 'v2e: 保守组合(双月+全过滤)'),
    ]
    
    all_results = {}
    
    for key, fn, rebal, desc in strategies:
        print(f"\n{'─'*60}", flush=True)
        print(f"📊 {key}: {desc}", flush=True)
        
        eq_full, hold_full, to_full = run_backtest(close_df, signals, sectors, fn,
                                                    '2015-01-01', '2025-12-31', rebal)
        eq_is, _, to_is = run_backtest(close_df, signals, sectors, fn,
                                        '2015-01-01', '2020-12-31', rebal)
        eq_oos, _, to_oos = run_backtest(close_df, signals, sectors, fn,
                                          '2021-01-01', '2025-12-31', rebal)
        
        m_full = compute_metrics(eq_full, key)
        m_is = compute_metrics(eq_is, f"{key}_IS")
        m_oos = compute_metrics(eq_oos, f"{key}_OOS")
        
        wf_ratio = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
        composite = m_full['sharpe'] * 0.4 + m_full['calmar'] * 0.4 + m_full['cagr'] * 0.2
        
        all_results[key] = {
            'desc': desc, 'full': m_full, 'is': m_is, 'oos': m_oos,
            'wf_ratio': wf_ratio, 'wf_pass': wf_ratio >= 0.70,
            'avg_turnover': to_full, 'composite': composite,
            'holdings': hold_full, 'rebal_months': rebal,
        }
        
        wm = '✅' if wf_ratio >= 0.70 else '❌'
        print(f"  Full: CAGR {m_full['cagr']:.1%} MaxDD {m_full['max_dd']:.1%} "
              f"Sharpe {m_full['sharpe']:.2f} Calmar {m_full['calmar']:.2f}", flush=True)
        print(f"  IS {m_is['sharpe']:.2f} OOS {m_oos['sharpe']:.2f} WF {wf_ratio:.2f} {wm} "
              f"TO {to_full:.1%} Comp {composite:.3f}", flush=True)
    
    # SPY
    spy = close_df['SPY'].loc['2015-01-01':'2025-12-31'].dropna().resample('ME').last()
    spy_eq = spy / spy.iloc[0]
    m_spy = compute_metrics(spy_eq, "SPY")
    
    # ─── Summary Table ─────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"{'Version':<16} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'TO':>8} "
          f"{'IS':>7} {'OOS':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 110)
    
    for key, r in all_results.items():
        wm = '✅' if r['wf_pass'] else '❌'
        print(f"{key:<16} {r['full']['cagr']:>6.1%} {r['full']['max_dd']:>7.1%} "
              f"{r['full']['sharpe']:>7.2f} {r['avg_turnover']:>7.1%} "
              f"{r['is']['sharpe']:>7.2f} {r['oos']['sharpe']:>7.2f} "
              f"{wm:>4} {r['composite']:>8.3f}")
    
    print(f"{'SPY B&H':<16} {m_spy['cagr']:>6.1%} {m_spy['max_dd']:>7.1%} "
          f"{m_spy['sharpe']:>7.2f} {'—':>8} {'—':>7} {'—':>7} {'—':>4} {'—':>8}")
    
    # Best selection
    v2_only = {k: v for k, v in all_results.items() if k != 'v1_baseline'}
    wf_passed = {k: v for k, v in v2_only.items() if v['wf_pass']}
    
    if wf_passed:
        best_key = max(wf_passed, key=lambda k: wf_passed[k]['composite'])
        print(f"\n🏆 Best (WF passed): {best_key}")
    else:
        best_key = max(v2_only, key=lambda k: v2_only[k]['wf_ratio'])
        print(f"\n🏆 Best (highest WF ratio): {best_key}")
    
    best = all_results[best_key]
    print(f"   CAGR {best['full']['cagr']:.1%} | Sharpe {best['full']['sharpe']:.2f} | "
          f"MaxDD {best['full']['max_dd']:.1%} | Composite {best['composite']:.3f}")
    print(f"   IS {best['is']['sharpe']:.2f} | OOS {best['oos']['sharpe']:.2f} | "
          f"WF {best['wf_ratio']:.2f} | TO {best['avg_turnover']:.1%}")
    
    # Holdings analysis for best
    hold = best.get('holdings', {})
    if hold:
        all_h = [t for stocks in hold.values() for t in stocks]
        freq = Counter(all_h).most_common(15)
        total_p = len(hold)
        print(f"\n📋 {best_key} Top 15 stocks ({total_p} periods):")
        for t, c in freq:
            sec = sectors.get(t, '?')
            print(f"  {t:6s} [{sec:12s}] {c:3d}/{total_p} ({c/total_p:.0%})")
        
        print(f"\n🔍 2023-2024 Holdings:")
        hot = {'NVDA','TSLA','META','AVGO','AMD','SMCI','PLTR'}
        for ym in sorted(hold.keys()):
            if ym.startswith('2023') or ym.startswith('2024'):
                h = [s for s in hold[ym] if s in hot]
                stocks_str = ', '.join(hold[ym][:6])
                print(f"  {ym}: {stocks_str} {'🔥'+','.join(h) if h else ''}")
    
    # Save
    output = {'comparison': {}, 'best': best_key, 'spy': m_spy}
    for key, r in all_results.items():
        output['comparison'][key] = {
            'desc': r['desc'], 'full': r['full'], 'is': r['is'], 'oos': r['oos'],
            'wf_ratio': r['wf_ratio'], 'wf_pass': r['wf_pass'],
            'avg_turnover': r['avg_turnover'], 'composite': r['composite'],
            'holdings': r['holdings'],
        }
    
    out_file = BASE / "stocks" / "codebear" / "momentum_v2_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved to {out_file}")
    
    return all_results, best_key

if __name__ == '__main__':
    main()
