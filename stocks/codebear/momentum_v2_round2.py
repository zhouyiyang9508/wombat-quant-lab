#!/usr/bin/env python3
"""
动量轮动 v2 Round 2 — 精细化调优
代码熊 🐻

Round 1 发现:
- v2b (双月+skip1M+惯性) WF 1.17 ✅ 但 CAGR 22.8%, Sharpe 0.92
- v2d (自适应+行业+vol) Sharpe 1.23, WF 0.65 ❌ 差一点

Round 2 思路:
- v2f: v2d + 双月 (期望: 降低IS, 提升WF ratio)
- v2g: v2d softer bear (30% cash instead of 50%)  
- v2h: 不用regime, 用abs momentum + sector + vol + holdover (最小化过拟合)
- v2i: v2b + sector diversification (提升v2b的diversification)
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

# Import shared utilities from v2 compare
sys.path.insert(0, str(BASE / "stocks" / "codebear"))
from momentum_v2_compare import (
    load_all_data, load_sectors, precompute_signals,
    get_scores_at_date, get_regime, compute_metrics, run_backtest
)


# ─── Round 2 Strategies ────────────────────────────────────────

def strategy_v2f(signals, sectors, date, prev_holdings):
    """v2f: v2d + bimonthly (note: rebal handled externally, this is monthly logic)
    Same as v2d but run with rebalance_months=2."""
    regime = get_regime(signals, date)
    
    df = get_scores_at_date(signals, date, weights=(0.20, 0.40, 0.30, 0.10))
    if df.empty:
        return {}
    
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    
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
    return {t: (v/total)*invested for t, v in inv_vols.items()}


def strategy_v2g(signals, sectors, date, prev_holdings):
    """v2g: v2d with softer bear regime (20% cash instead of 50%)."""
    regime = get_regime(signals, date)
    
    df = get_scores_at_date(signals, date, weights=(0.20, 0.40, 0.30, 0.10))
    if df.empty:
        return {}
    
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    
    df = df.sort_values('momentum', ascending=False)
    
    if regime == 'bull':
        top_n, max_sec, cash = 12, 3, 0.0
    else:
        top_n, max_sec, cash = 8, 2, 0.20  # softer bear: 80% invested, Top 8
    
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
    return {t: (v/total)*invested for t, v in inv_vols.items()}


def strategy_v2h(signals, sectors, date, prev_holdings):
    """v2h: No regime filter, just abs momentum + sector + vol + holdover.
    Let absolute momentum be the only bear market defense."""
    df = get_scores_at_date(signals, date, weights=(0.20, 0.40, 0.30, 0.10))
    if df.empty:
        return {}
    
    # Absolute momentum filter (only stocks with positive 6M return)
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.60)]
    
    # Holdover bonus
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.04
    
    df = df.sort_values('momentum', ascending=False)
    
    # Sector diversified (max 3), Top 10
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


def strategy_v2i(signals, sectors, date, prev_holdings):
    """v2i: v2b enhanced with sector diversification + vol weighting.
    Bimonthly, skip-1M, holdover, regime, abs momentum + sector + vol."""
    if get_regime(signals, date) == 'bear':
        return {}
    
    df = get_scores_at_date(signals, date, 
                            weights=(0.0, 0.40, 0.35, 0.25),
                            skip_recent=True)
    if df.empty:
        return {}
    
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.60)]
    
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.06
    
    df = df.sort_values('momentum', ascending=False)
    
    # Sector diversified, Top 10
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


def strategy_v2j(signals, sectors, date, prev_holdings):
    """v2j: Minimal filters, maximum simplicity.
    Monthly, standard momentum, abs momentum filter only, equal weight Top 10.
    No regime, no sector, no vol weighting. Just absolute momentum."""
    df = get_scores_at_date(signals, date, weights=(0.25, 0.40, 0.35, 0.0))
    if df.empty:
        return {}
    
    # Only absolute momentum filter
    df = df[df['abs_6m'] > 0]
    
    top = df.nlargest(10, 'momentum')
    n = len(top)
    return {t: 1.0/n for t in top.index} if n > 0 else {}


# ─── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🐻 代码熊 — 动量轮动 v2 Round 2 精细调优")
    print("=" * 70, flush=True)
    
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df, volume_df, loaded = load_all_data(tickers + ['SPY'])
    sectors = load_sectors()
    print(f"Loaded {loaded} stocks, {len(sectors)} sectors", flush=True)
    
    signals = precompute_signals(close_df, volume_df)
    
    strategies = [
        ('v2f_bimon_adap', strategy_v2f, 2, 'v2f: v2d+双月'),
        ('v2g_soft_bear', strategy_v2g, 1, 'v2g: v2d+软熊(20%cash)'),
        ('v2h_no_regime', strategy_v2h, 1, 'v2h: 无regime+行业+Vol+惯性'),
        ('v2i_b_enhanced', strategy_v2i, 2, 'v2i: v2b+行业+Vol'),
        ('v2j_simple_abs', strategy_v2j, 1, 'v2j: 仅绝对动量过滤'),
    ]
    
    all_results = {}
    
    for key, fn, rebal, desc in strategies:
        print(f"\n{'─'*60}", flush=True)
        print(f"📊 {key}: {desc}", flush=True)
        
        eq_full, hold_full, to_full = run_backtest(close_df, signals, sectors, fn,
                                                    '2015-01-01', '2025-12-31', rebal)
        eq_is, _, _ = run_backtest(close_df, signals, sectors, fn,
                                    '2015-01-01', '2020-12-31', rebal)
        eq_oos, _, _ = run_backtest(close_df, signals, sectors, fn,
                                     '2021-01-01', '2025-12-31', rebal)
        
        m_full = compute_metrics(eq_full, key)
        m_is = compute_metrics(eq_is, f"{key}_IS")
        m_oos = compute_metrics(eq_oos, f"{key}_OOS")
        
        wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
        comp = m_full['sharpe'] * 0.4 + m_full['calmar'] * 0.4 + m_full['cagr'] * 0.2
        
        all_results[key] = {
            'desc': desc, 'full': m_full, 'is': m_is, 'oos': m_oos,
            'wf_ratio': wf, 'wf_pass': wf >= 0.70,
            'avg_turnover': to_full, 'composite': comp,
            'holdings': hold_full,
        }
        
        wm = '✅' if wf >= 0.70 else '❌'
        print(f"  Full: CAGR {m_full['cagr']:.1%} MaxDD {m_full['max_dd']:.1%} "
              f"Sharpe {m_full['sharpe']:.2f} Calmar {m_full['calmar']:.2f}", flush=True)
        print(f"  IS {m_is['sharpe']:.2f} OOS {m_oos['sharpe']:.2f} WF {wf:.2f} {wm} "
              f"TO {to_full:.1%} Comp {comp:.3f}", flush=True)
    
    # Summary
    print(f"\n{'='*100}")
    print(f"{'Version':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'TO':>8} "
          f"{'IS':>7} {'OOS':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 100)
    
    # Include Round 1 best for reference
    print(f"{'v2b (R1 best)':<18} {'22.8%':>7} {'-25.3%':>8} {'0.92':>7} {'44.1%':>8} "
          f"{'0.82':>7} {'0.96':>7} {'✅':>4} {'0.773':>8}")
    print(f"{'v2d (R1 #2)':<18} {'24.6%':>7} {'-22.0%':>8} {'1.23':>7} {'46.6%':>8} "
          f"{'1.44':>7} {'0.93':>7} {'❌':>4} {'0.987':>8}")
    
    for key, r in all_results.items():
        wm = '✅' if r['wf_pass'] else '❌'
        cagr_s = f"{r['full']['cagr']:.1%}" if not np.isnan(r['full']['cagr']) else 'nan'
        print(f"{key:<18} {cagr_s:>7} {r['full']['max_dd']:>7.1%} "
              f"{r['full']['sharpe']:>7.2f} {r['avg_turnover']:>7.1%} "
              f"{r['is']['sharpe']:>7.2f} {r['oos']['sharpe']:>7.2f} "
              f"{wm:>4} {r['composite']:>8.3f}")
    
    # Find best WF-passing
    wf_passed = {k: v for k, v in all_results.items() if v['wf_pass']}
    if wf_passed:
        best = max(wf_passed, key=lambda k: wf_passed[k]['composite'])
        print(f"\n🏆 Round 2 Best (WF passed): {best}")
        r = all_results[best]
        print(f"   CAGR {r['full']['cagr']:.1%} Sharpe {r['full']['sharpe']:.2f} "
              f"WF {r['wf_ratio']:.2f} Comp {r['composite']:.3f}")
    else:
        best = max(all_results, key=lambda k: all_results[k]['wf_ratio'])
        print(f"\n🏆 Round 2 Best (WF ratio): {best}")
    
    # Save
    out = BASE / "stocks" / "codebear" / "momentum_v2_round2_results.json"
    with open(out, 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'holdings'} 
                   for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\n💾 Saved to {out}")
    
    return all_results

if __name__ == '__main__':
    main()
