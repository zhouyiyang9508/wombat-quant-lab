#!/usr/bin/env python3
"""
周频行业 ETF 轮动策略 v1 — 代码熊 🐻
Weekly Sector ETF Rotation — Numpy-optimized

🚨 无前瞻偏差:
  - 所有信号 shift(1): 本周五收盘 → 下周执行
"""

import json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"

SECTOR_ETFS = ['XLK','XLE','XLV','XLF','XLY','XLI','XLP','XLU','XLB','XLRE','XLC']
ALL_TICKERS = SECTOR_ETFS + ['GLD','TLT','SHY','SPY']
TRADEABLE   = SECTOR_ETFS + ['GLD', 'TLT']  # 13 tradeable assets


def load_weekly():
    frames = {}
    for t in ALL_TICKERS:
        fp = CACHE / f"{t}.csv"
        if fp.exists():
            df = pd.read_csv(fp, parse_dates=['Date'], index_col='Date')
            frames[t] = pd.to_numeric(df['Close'], errors='coerce')
    prices = pd.DataFrame(frames).dropna(how='all')
    weekly = prices.resample('W-FRI').last()
    return weekly


def fast_backtest(weekly, mom_periods, n_top, sma_weeks, defensive_mode,
                  cost_bps=5, mom_thresh=0.0, start='2015-01-01', end='2025-12-31'):
    """Numpy-optimized backtest."""
    wp = weekly.loc[start:end].copy()
    n = len(wp)
    if n < 60:
        return None
    
    all_cols = list(wp.columns)
    trade_idx = [all_cols.index(t) for t in TRADEABLE if t in all_cols]
    spy_idx = all_cols.index('SPY')
    shy_idx = all_cols.index('SHY')
    tlt_idx = all_cols.index('TLT')
    gld_idx = all_cols.index('GLD')
    
    arr = wp.values  # (n, ncols)
    
    # Weekly returns
    ret = np.zeros_like(arr)
    ret[1:] = arr[1:] / arr[:-1] - 1
    ret = np.nan_to_num(ret, 0.0)
    
    # Blended momentum (shifted by 1 for no lookahead)
    total_w = sum(mom_periods.values())
    mom = np.zeros((n, len(trade_idx)))
    for lb, w in mom_periods.items():
        m = np.full((n, len(trade_idx)), np.nan)
        for j, col_i in enumerate(trade_idx):
            for i in range(lb + 1, n):  # +1 for shift
                prev = arr[i - 1 - lb, col_i]
                cur = arr[i - 1, col_i]  # shift(1): use previous week's close
                if prev > 0 and not np.isnan(prev) and not np.isnan(cur):
                    m[i, j] = (cur / prev - 1)
        mom += np.nan_to_num(m, 0.0) * (w / total_w)
    
    # SPY regime (shifted by 1)
    spy_close = arr[:, spy_idx]
    spy_sma = np.full(n, np.nan)
    for i in range(sma_weeks, n):
        vals = spy_close[i - sma_weeks:i]
        valid = vals[~np.isnan(vals)]
        if len(valid) == sma_weeks:
            spy_sma[i] = valid.mean()
    
    is_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(spy_sma[i]) and not np.isnan(spy_close[i - 1]):
            is_bull[i] = spy_close[i - 1] > spy_sma[i]  # shift(1)
        else:
            is_bull[i] = True
    
    # Build weights
    ncols = len(all_cols)
    weights = np.zeros((n, ncols))
    
    for i in range(1, n):
        if not is_bull[i]:
            # Bear mode
            if defensive_mode == 'shy':
                weights[i, shy_idx] = 1.0
            elif defensive_mode == 'tlt':
                weights[i, tlt_idx] = 1.0
            elif defensive_mode == 'gld':
                weights[i, gld_idx] = 1.0
            elif defensive_mode == 'blend':
                weights[i, shy_idx] = 0.40
                weights[i, tlt_idx] = 0.30
                weights[i, gld_idx] = 0.30
            continue
        
        # Bull: rank by momentum
        scores = mom[i]
        valid_mask = ~np.isnan(scores) & (scores > mom_thresh)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            weights[i, shy_idx] = 1.0
            continue
        
        valid_scores = scores[valid_indices]
        top_k = min(n_top, len(valid_indices))
        top_local = np.argsort(valid_scores)[-top_k:][::-1]
        top_global = [trade_idx[valid_indices[j]] for j in top_local]
        
        eq_w = 1.0 / top_k
        for gi in top_global:
            weights[i, gi] = eq_w
    
    # Portfolio returns
    port_ret = np.sum(weights * ret, axis=1)
    
    # Transaction costs
    turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
    turnover = np.concatenate([[0.0], turnover])
    port_ret -= turnover * (cost_bps / 10000)
    
    # Equity curve
    equity = np.cumprod(1 + port_ret)
    
    # Skip warmup
    start_i = max(30, sma_weeks + 1)
    if start_i >= n:
        return None
    
    eq = equity[start_i:]
    pr = port_ret[start_i:]
    nw = len(pr)
    if nw < 52:
        return None
    
    n_years = nw / 52
    total_ret = eq[-1] / eq[0] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1
    vol = np.std(pr) * np.sqrt(52)
    mean_ret = np.mean(pr) * 52
    sharpe = mean_ret / vol if vol > 0 else 0
    
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1
    max_dd = np.min(dd)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    composite = sharpe * 0.4 + calmar * 0.4 + cagr * 0.2
    
    return {
        'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd,
        'calmar': calmar, 'composite': composite,
        'annual_turnover': np.mean(turnover[start_i:]) * 52,
    }


def walk_forward(weekly, params):
    p = {k: v for k, v in params.items()}
    is_r = fast_backtest(weekly, start='2015-01-01', end='2020-12-31', **p)
    oos_r = fast_backtest(weekly, start='2021-01-01', end='2025-12-31', **p)
    full_r = fast_backtest(weekly, start='2015-01-01', end='2025-12-31', **p)
    if not all([is_r, oos_r, full_r]):
        return None
    wf = oos_r['sharpe'] / is_r['sharpe'] if is_r['sharpe'] > 0 else 0
    return {'is': is_r, 'oos': oos_r, 'full': full_r, 'wf_ratio': wf}


def print_table(results, title, top_n=20):
    sr = sorted(results, key=lambda x: x['full_composite'], reverse=True)
    print(f"\n{'='*115}")
    print(f"  {title}")
    print(f"{'='*115}")
    print(f"{'#':>3} {'Name':<42} {'CAGR':>7} {'Shrp':>6} {'MaxDD':>8} {'Calm':>6} {'Comp':>7} {'WF':>5} {'IS':>6} {'OOS':>6} {'Turn':>5}")
    print(f"{'-'*115}")
    for i, r in enumerate(sr[:top_n]):
        print(f"{i+1:>3} {r['name']:<42} {r['full_cagr']:>6.1%} {r['full_sharpe']:>6.2f} "
              f"{r['full_maxdd']:>7.1%} {r['full_calmar']:>6.2f} {r['full_composite']:>7.3f} "
              f"{r['wf_ratio']:>5.2f} {r.get('is_sharpe',0):>6.2f} {r.get('oos_sharpe',0):>6.2f} "
              f"{r['annual_turnover']:>4.1f}")
    return sr


if __name__ == '__main__':
    t0 = time.time()
    print("🐻 代码熊 Weekly ETF Rotation v1")
    print("=" * 60)
    
    weekly = load_weekly()
    print(f"  {len(weekly.columns)} tickers, {len(weekly)} weekly bars")
    print(f"  Range: {weekly.index[0].date()} to {weekly.index[-1].date()}")
    
    # ── Phase 1: Grid Search ──
    mom_configs = {
        'M1_4w13w':      {4: 0.5, 13: 0.5},
        'M2_4w13w26w':   {4: 0.33, 13: 0.34, 26: 0.33},
        'M3_8w13w':      {8: 0.5, 13: 0.5},
        'M4_4w8w13w':    {4: 0.3, 8: 0.4, 13: 0.3},
        'M5_4w26w':      {4: 0.5, 26: 0.5},
        'M6_8w26w':      {8: 0.5, 26: 0.5},
        'M7_4w8w13w26w': {4: 0.25, 8: 0.25, 13: 0.25, 26: 0.25},
        'M8_12w':        {12: 1.0},
        'M9_4w12w':      {4: 0.5, 12: 0.5},
        'M10_4w8w26w':   {4: 0.3, 8: 0.3, 26: 0.4},
    }
    
    print(f"\n📊 Phase 1: Grid Search...")
    results = []
    count = 0
    for mname, mper in mom_configs.items():
        for ntop in [3, 4, 5]:
            for smaw in [30, 40, 50]:
                for dmode in ['shy', 'blend', 'tlt']:
                    count += 1
                    params = dict(mom_periods=mper, n_top=ntop, sma_weeks=smaw,
                                  defensive_mode=dmode, cost_bps=5, mom_thresh=0.0)
                    wf = walk_forward(weekly, params)
                    if wf is None:
                        continue
                    results.append({
                        'name': f"{mname}_t{ntop}_s{smaw}_{dmode}",
                        'mom_name': mname, 'n_top': ntop, 'sma_weeks': smaw,
                        'defensive_mode': dmode,
                        'full_cagr': wf['full']['cagr'], 'full_sharpe': wf['full']['sharpe'],
                        'full_maxdd': wf['full']['max_dd'], 'full_calmar': wf['full']['calmar'],
                        'full_composite': wf['full']['composite'],
                        'is_sharpe': wf['is']['sharpe'], 'oos_sharpe': wf['oos']['sharpe'],
                        'wf_ratio': wf['wf_ratio'],
                        'annual_turnover': wf['full']['annual_turnover'],
                        'params': params,
                    })
                    if count % 90 == 0:
                        print(f"  ... {count}/270 ({time.time()-t0:.0f}s)")
    
    print(f"  Done: {len(results)} valid / {count} total ({time.time()-t0:.0f}s)")
    sr = print_table(results, "Phase 1: Grid Search")
    
    # ── Phase 2: threshold tuning on top params ──
    if sr:
        best = sr[0]
        print(f"\n📊 Phase 2: Threshold tuning on {best['name']}...")
        p2 = []
        bp = best['params'].copy()
        for thr in [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]:
            p = bp.copy()
            p['mom_thresh'] = thr
            wf = walk_forward(weekly, p)
            if wf:
                p2.append({
                    'name': f"P2_thr{thr:+.2f}",
                    'full_cagr': wf['full']['cagr'], 'full_sharpe': wf['full']['sharpe'],
                    'full_maxdd': wf['full']['max_dd'], 'full_calmar': wf['full']['calmar'],
                    'full_composite': wf['full']['composite'],
                    'is_sharpe': wf['is']['sharpe'], 'oos_sharpe': wf['oos']['sharpe'],
                    'wf_ratio': wf['wf_ratio'],
                    'annual_turnover': wf['full']['annual_turnover'],
                    'params': p, 'mom_name': best['mom_name'], 'n_top': best['n_top'],
                    'sma_weeks': best['sma_weeks'], 'defensive_mode': best['defensive_mode'],
                })
        if p2:
            print_table(p2, "Phase 2: Threshold Tuning", 10)
    
    # ── Champions ──
    all_r = results + (p2 if p2 else [])
    all_s = sorted(all_r, key=lambda x: x['full_composite'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"  🏆 TOP 5 STRATEGIES")
    print(f"{'='*80}")
    for i, r in enumerate(all_s[:5]):
        wf = walk_forward(weekly, r['params'])
        if not wf:
            continue
        f = wf['full']
        print(f"\n  #{i+1} {r['name']}")
        print(f"  Full:  CAGR={f['cagr']:.1%}  Sharpe={f['sharpe']:.2f}  MaxDD={f['max_dd']:.1%}  Calmar={f['calmar']:.2f}  Comp={f['composite']:.4f}")
        s = wf['is']
        print(f"  IS:    CAGR={s['cagr']:.1%}  Sharpe={s['sharpe']:.2f}  MaxDD={s['max_dd']:.1%}")
        o = wf['oos']
        print(f"  OOS:   CAGR={o['cagr']:.1%}  Sharpe={o['sharpe']:.2f}  MaxDD={o['max_dd']:.1%}")
        print(f"  WF={wf['wf_ratio']:.2f}  Turnover={f['annual_turnover']:.1f}x/yr")
    
    # ── vs Monthly v9b ──
    print(f"\n{'='*80}")
    print(f"  📊 WEEKLY ETF v1 vs MONTHLY STOCK v9b")
    print(f"{'='*80}")
    print(f"  Monthly v9b:  Comp=1.533  Sharpe=1.58  CAGR=31.2%  MaxDD=-14.9%  WF=0.79")
    if all_s:
        b = all_s[0]
        print(f"  Weekly v1:    Comp={b['full_composite']:.3f}  Sharpe={b['full_sharpe']:.2f}  CAGR={b['full_cagr']:.1%}  MaxDD={b['full_maxdd']:.1%}  WF={b['wf_ratio']:.2f}")
    
    # Save
    sv = [{k: v for k, v in r.items() if k != 'params'} for r in all_s[:30]]
    out = Path(__file__).parent / "weekly_v1_results.json"
    with open(out, 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print(f"  Saved to {out}")
