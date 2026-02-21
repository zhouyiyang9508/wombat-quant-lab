#!/usr/bin/env python3
"""
周频策略 v2 — 代码熊 🐻
增强版：动量加权 + 双动量 + 趋势跟踪 + 个股周频

v1 结论：纯 ETF 轮动 Composite 仅 0.53，远低于月频 1.533
v2 尝试：
  A) ETF 增强版（动量加权、双动量、绝对+相对）
  B) 个股周频（与月频类似逻辑，但每周评估）
"""

import json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

SECTOR_ETFS = ['XLK','XLE','XLV','XLF','XLY','XLI','XLP','XLU','XLB','XLRE','XLC']
ALL_ETF = SECTOR_ETFS + ['GLD','TLT','SHY','SPY']
TRADEABLE_ETF = SECTOR_ETFS + ['GLD', 'TLT']


def load_csv(fp):
    df = pd.read_csv(fp, parse_dates=['Date'], index_col='Date')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df


# ══════════════════════════════════════════════════════════════════
#  PART A: Enhanced ETF strategies
# ══════════════════════════════════════════════════════════════════
def run_etf_enhanced():
    """Test momentum-weighted, dual momentum, and trend ETF strategies."""
    print("\n📊 PART A: Enhanced ETF Strategies")
    print("=" * 60)
    
    frames = {}
    for t in ALL_ETF:
        fp = CACHE / f"{t}.csv"
        if fp.exists():
            frames[t] = load_csv(fp)['Close']
    daily = pd.DataFrame(frames).dropna(how='all')
    weekly = daily.resample('W-FRI').last()
    wp = weekly.loc['2015-01-01':'2025-12-31'].copy()
    
    spy = wp['SPY']
    weekly_ret = wp.pct_change()
    n = len(wp)
    all_cols = list(wp.columns)
    arr = wp.values
    
    results = []
    
    # Strategy A1: Dual Momentum (absolute + relative)
    # Only buy if asset has positive absolute momentum AND is top-ranked
    for abs_lb in [13, 26]:
        for rel_lbs in [{4:0.5, 13:0.5}, {8:0.5, 13:0.5}, {4:0.3, 8:0.3, 26:0.4}]:
            for ntop in [3, 4, 5]:
                for smaw in [40, 50]:
                    r = backtest_dual_mom(wp, weekly_ret, spy, rel_lbs, abs_lb, ntop, smaw)
                    if r:
                        lb_name = '_'.join([f"{k}w" for k in rel_lbs.keys()])
                        results.append({
                            'name': f"DualMom_abs{abs_lb}_{lb_name}_t{ntop}_s{smaw}",
                            **r
                        })
    
    # Strategy A2: Momentum-weighted (weight proportional to momentum score)
    for mom_lbs in [{4:0.5, 13:0.5}, {8:0.5, 13:0.5}, {4:0.3, 8:0.3, 26:0.4}]:
        for ntop in [3, 4, 5]:
            for smaw in [40, 50]:
                r = backtest_mom_weighted(wp, weekly_ret, spy, mom_lbs, ntop, smaw)
                if r:
                    lb_name = '_'.join([f"{k}w" for k in mom_lbs.keys()])
                    results.append({
                        'name': f"MomWt_{lb_name}_t{ntop}_s{smaw}",
                        **r
                    })
    
    # Strategy A3: Trend Following (SMA crossover on each ETF)
    for fast_w, slow_w in [(4, 13), (4, 26), (8, 26), (13, 26)]:
        for ntop in [3, 4, 5]:
            r = backtest_trend(wp, weekly_ret, spy, fast_w, slow_w, ntop)
            if r:
                results.append({
                    'name': f"Trend_{fast_w}x{slow_w}_t{ntop}",
                    **r
                })
    
    if results:
        sr = sorted(results, key=lambda x: x['full_composite'], reverse=True)
        print(f"\n  ETF Enhanced: {len(results)} variants tested")
        print(f"  {'#':>3} {'Name':<45} {'CAGR':>7} {'Shrp':>6} {'MDD':>7} {'Comp':>7} {'WF':>5}")
        print(f"  {'-'*85}")
        for i, r in enumerate(sr[:15]):
            print(f"  {i+1:>3} {r['name']:<45} {r['full_cagr']:>6.1%} {r['full_sharpe']:>6.2f} "
                  f"{r['full_maxdd']:>6.1%} {r['full_composite']:>7.3f} {r['wf_ratio']:>5.2f}")
        return sr
    return []


def backtest_dual_mom(wp, weekly_ret, spy, rel_lbs, abs_lb, ntop, smaw, cost_bps=5):
    """Dual momentum: relative ranking + absolute filter."""
    n = len(wp)
    trade_cols = [c for c in TRADEABLE_ETF if c in wp.columns]
    all_cols = list(wp.columns)
    
    # Compute signals
    rel_mom = pd.DataFrame(0.0, index=wp.index, columns=trade_cols)
    tw = sum(rel_lbs.values())
    for lb, w in rel_lbs.items():
        m = wp[trade_cols] / wp[trade_cols].shift(lb) - 1
        rel_mom += m * (w / tw)
    
    abs_mom = wp[trade_cols] / wp[trade_cols].shift(abs_lb) - 1
    
    spy_sma = spy.rolling(smaw).mean()
    
    # shift(1) for no lookahead
    rel_sig = rel_mom.shift(1)
    abs_sig = abs_mom.shift(1)
    bull_sig = (spy > spy_sma).shift(1)
    
    weights = pd.DataFrame(0.0, index=wp.index, columns=all_cols)
    shy_i = all_cols.index('SHY')
    
    for i in range(1, n):
        if not bull_sig.iloc[i]:
            weights.iloc[i, shy_i] = 0.40
            weights.iloc[i, all_cols.index('TLT')] = 0.30
            weights.iloc[i, all_cols.index('GLD')] = 0.30
            continue
        
        r = rel_sig.iloc[i]
        a = abs_sig.iloc[i]
        # Filter: positive absolute momentum
        valid = r[(~r.isna()) & (~a.isna()) & (a > 0)]
        
        if len(valid) == 0:
            weights.iloc[i, shy_i] = 1.0
            continue
        
        top = valid.nlargest(min(ntop, len(valid)))
        eq = 1.0 / len(top)
        for t in top.index:
            weights.iloc[i, all_cols.index(t)] = eq
    
    return calc_metrics(weights, weekly_ret, cost_bps)


def backtest_mom_weighted(wp, weekly_ret, spy, mom_lbs, ntop, smaw, cost_bps=5):
    """Momentum-weighted allocation."""
    n = len(wp)
    trade_cols = [c for c in TRADEABLE_ETF if c in wp.columns]
    all_cols = list(wp.columns)
    
    mom = pd.DataFrame(0.0, index=wp.index, columns=trade_cols)
    tw = sum(mom_lbs.values())
    for lb, w in mom_lbs.items():
        m = wp[trade_cols] / wp[trade_cols].shift(lb) - 1
        mom += m * (w / tw)
    
    spy_sma = spy.rolling(smaw).mean()
    mom_sig = mom.shift(1)
    bull_sig = (spy > spy_sma).shift(1)
    
    weights = pd.DataFrame(0.0, index=wp.index, columns=all_cols)
    shy_i = all_cols.index('SHY')
    
    for i in range(1, n):
        if not bull_sig.iloc[i]:
            weights.iloc[i, shy_i] = 0.40
            weights.iloc[i, all_cols.index('TLT')] = 0.30
            weights.iloc[i, all_cols.index('GLD')] = 0.30
            continue
        
        scores = mom_sig.iloc[i].dropna()
        scores = scores[scores > 0]
        
        if len(scores) == 0:
            weights.iloc[i, shy_i] = 1.0
            continue
        
        top = scores.nlargest(min(ntop, len(scores)))
        # Momentum-weighted
        w = top / top.sum()
        for t in w.index:
            weights.iloc[i, all_cols.index(t)] = w[t]
    
    return calc_metrics(weights, weekly_ret, cost_bps)


def backtest_trend(wp, weekly_ret, spy, fast_w, slow_w, ntop, cost_bps=5):
    """Trend following: buy ETFs where fast SMA > slow SMA, rank by momentum."""
    n = len(wp)
    trade_cols = [c for c in TRADEABLE_ETF if c in wp.columns]
    all_cols = list(wp.columns)
    
    fast_sma = wp[trade_cols].rolling(fast_w).mean()
    slow_sma = wp[trade_cols].rolling(slow_w).mean()
    trend_up = fast_sma > slow_sma
    
    mom = wp[trade_cols] / wp[trade_cols].shift(fast_w) - 1
    
    # shift(1)
    trend_sig = trend_up.shift(1)
    mom_sig = mom.shift(1)
    
    spy_sma50 = spy.rolling(50).mean()
    bull_sig = (spy > spy_sma50).shift(1)
    
    weights = pd.DataFrame(0.0, index=wp.index, columns=all_cols)
    shy_i = all_cols.index('SHY')
    
    for i in range(1, n):
        if not bull_sig.iloc[i]:
            weights.iloc[i, shy_i] = 0.40
            weights.iloc[i, all_cols.index('TLT')] = 0.30
            weights.iloc[i, all_cols.index('GLD')] = 0.30
            continue
        
        tr = trend_sig.iloc[i]
        m = mom_sig.iloc[i]
        
        # Only pick uptrending assets
        valid = m[tr == True].dropna()
        valid = valid[valid > 0]
        
        if len(valid) == 0:
            weights.iloc[i, shy_i] = 1.0
            continue
        
        top = valid.nlargest(min(ntop, len(valid)))
        eq = 1.0 / len(top)
        for t in top.index:
            weights.iloc[i, all_cols.index(t)] = eq
    
    return calc_metrics(weights, weekly_ret, cost_bps)


def calc_metrics(weights, weekly_ret, cost_bps, is_end='2020-12-31', oos_start='2021-01-01'):
    """Calculate full, IS, OOS metrics from weight matrix."""
    
    def _metrics(w, wr, label):
        pr = (w * wr).sum(axis=1)
        to = w.diff().abs().sum(axis=1)
        pr = pr - to * (cost_bps / 10000)
        eq = (1 + pr).cumprod()
        # warmup
        start_i = 52
        if start_i >= len(eq):
            return None
        eq = eq.iloc[start_i:]
        pr_v = pr.iloc[start_i:]
        nw = len(pr_v)
        if nw < 26:
            return None
        ny = nw / 52
        tr = eq.iloc[-1] / eq.iloc[0] - 1
        cagr = (1 + tr) ** (1/ny) - 1
        vol = pr_v.std() * np.sqrt(52)
        sharpe = (pr_v.mean() * 52) / vol if vol > 0 else 0
        cm = eq.cummax()
        mdd = (eq / cm - 1).min()
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        comp = sharpe * 0.4 + calmar * 0.4 + cagr * 0.2
        return {'cagr': cagr, 'sharpe': sharpe, 'max_dd': mdd, 'calmar': calmar,
                'composite': comp, 'annual_turnover': to.iloc[start_i:].mean() * 52}
    
    full = _metrics(weights, weekly_ret, 'full')
    is_w = weights.loc[:is_end]
    is_wr = weekly_ret.loc[:is_end]
    is_r = _metrics(is_w, is_wr, 'is')
    oos_w = weights.loc[oos_start:]
    oos_wr = weekly_ret.loc[oos_start:]
    oos_r = _metrics(oos_w, oos_wr, 'oos')
    
    if not all([full, is_r, oos_r]):
        return None
    
    wf = oos_r['sharpe'] / is_r['sharpe'] if is_r['sharpe'] > 0 else 0
    return {
        'full_cagr': full['cagr'], 'full_sharpe': full['sharpe'],
        'full_maxdd': full['max_dd'], 'full_calmar': full['calmar'],
        'full_composite': full['composite'],
        'is_sharpe': is_r['sharpe'], 'oos_sharpe': oos_r['sharpe'],
        'wf_ratio': wf, 'annual_turnover': full['annual_turnover'],
    }


# ══════════════════════════════════════════════════════════════════
#  PART B: Individual Stock Weekly Momentum
# ══════════════════════════════════════════════════════════════════
def load_stocks():
    """Load S&P500 stock data."""
    stocks = {}
    if STOCK_CACHE.exists():
        for fp in sorted(STOCK_CACHE.glob("*.csv")):
            ticker = fp.stem
            try:
                df = pd.read_csv(fp, parse_dates=['Date'], index_col='Date')
                s = pd.to_numeric(df['Close'], errors='coerce')
                if len(s.dropna()) > 200:
                    stocks[ticker] = s
            except:
                pass
    return stocks


def run_stock_weekly():
    """Weekly stock momentum with monthly-style logic but weekly evaluation."""
    print("\n📊 PART B: Individual Stock Weekly Momentum")
    print("=" * 60)
    
    # Load stocks
    stock_data = load_stocks()
    print(f"  Loaded {len(stock_data)} stocks")
    
    # Load SPY, GLD, SHY
    spy_d = load_csv(CACHE / "SPY.csv")['Close']
    gld_d = load_csv(CACHE / "GLD.csv")['Close']
    shy_d = load_csv(CACHE / "SHY.csv")['Close']
    
    # Build weekly stock prices
    all_frames = {k: v for k, v in stock_data.items()}
    all_frames['SPY'] = spy_d
    all_frames['GLD'] = gld_d
    all_frames['SHY'] = shy_d
    
    daily = pd.DataFrame(all_frames)
    weekly = daily.resample('W-FRI').last()
    
    tickers = [t for t in stock_data.keys()]
    wp = weekly.loc['2015-01-01':'2025-12-31']
    print(f"  Weekly bars: {len(wp)}, stocks: {len(tickers)}")
    
    spy_w = wp['SPY']
    
    results = []
    
    # Test various configurations
    # Key difference from monthly: we only rebalance when signals change significantly
    for mom_cfg_name, mom_lbs in [
        ('4w13w', {4: 0.5, 13: 0.5}),
        ('4w8w13w', {4: 0.3, 8: 0.4, 13: 0.3}),
        ('4w13w26w', {4: 0.3, 13: 0.4, 26: 0.3}),
    ]:
        for ntop in [10, 15, 20]:
            for smaw in [40, 50]:
                for cost in [10, 15]:  # bps per side
                    for rebal_mode in ['always', 'threshold']:
                        r = backtest_stock_weekly(
                            wp, tickers, spy_w, mom_lbs, ntop, smaw,
                            cost_bps=cost, rebal_mode=rebal_mode
                        )
                        if r:
                            results.append({
                                'name': f"Stock_{mom_cfg_name}_t{ntop}_s{smaw}_c{cost}_{rebal_mode}",
                                **r
                            })
    
    if results:
        sr = sorted(results, key=lambda x: x['full_composite'], reverse=True)
        print(f"\n  Stock Weekly: {len(results)} variants tested")
        print(f"  {'#':>3} {'Name':<55} {'CAGR':>7} {'Shrp':>6} {'MDD':>7} {'Comp':>7} {'WF':>5} {'Turn':>5}")
        print(f"  {'-'*100}")
        for i, r in enumerate(sr[:15]):
            print(f"  {i+1:>3} {r['name']:<55} {r['full_cagr']:>6.1%} {r['full_sharpe']:>6.2f} "
                  f"{r['full_maxdd']:>6.1%} {r['full_composite']:>7.3f} {r['wf_ratio']:>5.2f} "
                  f"{r.get('annual_turnover',0):>4.1f}")
        return sr
    return []


def backtest_stock_weekly(wp, tickers, spy_w, mom_lbs, ntop, smaw,
                          cost_bps=15, rebal_mode='always'):
    """
    Weekly stock momentum.
    rebal_mode:
      'always' = rebalance every week
      'threshold' = only rebalance if top set changes by >30%
    """
    n = len(wp)
    weekly_ret = wp.pct_change()
    
    # Compute blended momentum for stocks
    tw = sum(mom_lbs.values())
    stock_mom = pd.DataFrame(0.0, index=wp.index, columns=tickers)
    for lb, w in mom_lbs.items():
        m = wp[tickers] / wp[tickers].shift(lb) - 1
        stock_mom += m.fillna(0) * (w / tw)
    
    # Regime
    spy_sma = spy_w.rolling(smaw).mean()
    
    # Shift signals
    mom_sig = stock_mom.shift(1)
    bull_sig = (spy_w > spy_sma).shift(1)
    
    # Tracking current portfolio
    current_holdings = set()
    
    port_ret = np.zeros(n)
    turnover_arr = np.zeros(n)
    prev_weights = {}
    
    for i in range(1, n):
        if not bull_sig.iloc[i]:
            # Bear: hold SHY
            shy_ret = weekly_ret.iloc[i].get('SHY', 0)
            if np.isnan(shy_ret):
                shy_ret = 0
            port_ret[i] = shy_ret
            
            # Turnover from switching
            if prev_weights:
                turnover_arr[i] = sum(abs(v) for v in prev_weights.values()) + 1.0
                prev_weights = {'SHY': 1.0}
            else:
                prev_weights = {'SHY': 1.0}
            continue
        
        scores = mom_sig.iloc[i]
        valid = scores.dropna()
        valid = valid[valid > 0]
        
        if len(valid) == 0:
            shy_ret = weekly_ret.iloc[i].get('SHY', 0)
            if np.isnan(shy_ret):
                shy_ret = 0
            port_ret[i] = shy_ret
            if prev_weights:
                turnover_arr[i] = sum(abs(v) for v in prev_weights.values()) + 1.0
            prev_weights = {'SHY': 1.0}
            continue
        
        top = valid.nlargest(min(ntop, len(valid))).index.tolist()
        new_set = set(top)
        
        if rebal_mode == 'threshold' and current_holdings:
            # Only rebalance if overlap < 70%
            overlap = len(new_set & current_holdings) / max(len(current_holdings), 1)
            if overlap >= 0.7:
                # Keep current holdings
                eq = 1.0 / len(current_holdings)
                ret_sum = 0.0
                for t in current_holdings:
                    r = weekly_ret.iloc[i].get(t, 0)
                    if np.isnan(r):
                        r = 0
                    ret_sum += r * eq
                port_ret[i] = ret_sum
                continue
        
        # Rebalance to new top set
        eq = 1.0 / len(top)
        new_weights = {t: eq for t in top}
        
        # Calculate turnover
        all_t = set(list(prev_weights.keys()) + list(new_weights.keys()))
        turn = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t)
        turnover_arr[i] = turn
        
        # Portfolio return
        ret_sum = 0.0
        for t in top:
            r = weekly_ret.iloc[i].get(t, 0)
            if np.isnan(r):
                r = 0
            ret_sum += r * eq
        port_ret[i] = ret_sum
        
        prev_weights = new_weights
        current_holdings = new_set
    
    # Apply costs
    cost_rate = cost_bps / 10000
    port_ret -= turnover_arr * cost_rate
    
    # Metrics
    eq = np.cumprod(1 + port_ret)
    start_i = 52
    if start_i >= len(eq):
        return None
    
    def _m(eq_slice, pr_slice):
        nw = len(pr_slice)
        if nw < 26:
            return None
        ny = nw / 52
        tr = eq_slice[-1] / eq_slice[0] - 1
        cagr = (1 + tr) ** (1/ny) - 1
        vol = np.std(pr_slice) * np.sqrt(52)
        sharpe = (np.mean(pr_slice) * 52) / vol if vol > 0 else 0
        cm = np.maximum.accumulate(eq_slice)
        mdd = np.min(eq_slice / cm - 1)
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        comp = sharpe * 0.4 + calmar * 0.4 + cagr * 0.2
        return {'cagr': cagr, 'sharpe': sharpe, 'max_dd': mdd, 'calmar': calmar, 'composite': comp}
    
    dates = wp.index
    is_mask = dates <= '2020-12-31'
    oos_mask = dates >= '2021-01-01'
    
    full = _m(eq[start_i:], port_ret[start_i:])
    
    # IS: 2015-2020
    is_start = start_i
    is_end_i = np.where(is_mask)[0][-1] + 1 if is_mask.any() else n
    is_r = _m(eq[is_start:is_end_i], port_ret[is_start:is_end_i])
    
    # OOS: 2021-2025
    oos_start_i = np.where(oos_mask)[0][0] if oos_mask.any() else n
    oos_r = _m(eq[oos_start_i:], port_ret[oos_start_i:])
    
    if not all([full, is_r, oos_r]):
        return None
    
    wf = oos_r['sharpe'] / is_r['sharpe'] if is_r['sharpe'] > 0 else 0
    
    return {
        'full_cagr': full['cagr'], 'full_sharpe': full['sharpe'],
        'full_maxdd': full['max_dd'], 'full_calmar': full['calmar'],
        'full_composite': full['composite'],
        'is_sharpe': is_r['sharpe'], 'oos_sharpe': oos_r['sharpe'],
        'wf_ratio': wf,
        'annual_turnover': np.mean(turnover_arr[start_i:]) * 52,
    }


if __name__ == '__main__':
    t0 = time.time()
    print("🐻 代码熊 Weekly Strategy v2 — Enhanced")
    print("=" * 60)
    
    # Part A: ETF Enhanced
    etf_results = run_etf_enhanced()
    
    # Part B: Stock Weekly
    stock_results = run_stock_weekly()
    
    # ── Grand Comparison ──
    print(f"\n{'='*80}")
    print(f"  📊 GRAND COMPARISON")
    print(f"{'='*80}")
    print(f"  Monthly Stock v9b:  Comp=1.533  Sharpe=1.58  CAGR=31.2%  WF=0.79")
    
    if etf_results:
        b = etf_results[0]
        print(f"  Weekly ETF Best:    Comp={b['full_composite']:.3f}  Sharpe={b['full_sharpe']:.2f}  "
              f"CAGR={b['full_cagr']:.1%}  WF={b['wf_ratio']:.2f}")
    
    if stock_results:
        b = stock_results[0]
        print(f"  Weekly Stock Best:  Comp={b['full_composite']:.3f}  Sharpe={b['full_sharpe']:.2f}  "
              f"CAGR={b['full_cagr']:.1%}  WF={b['wf_ratio']:.2f}")
    
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    
    # Save all results
    all_r = []
    if etf_results:
        for r in etf_results[:15]:
            all_r.append({k: v for k, v in r.items() if not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))})
    if stock_results:
        for r in stock_results[:15]:
            all_r.append({k: v for k, v in r.items() if not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))})
    
    out = Path(__file__).parent / "weekly_v2_results.json"
    with open(out, 'w') as f:
        json.dump(all_r, f, indent=2, default=str)
    print(f"  Saved to {out}")
