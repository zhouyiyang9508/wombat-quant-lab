#!/usr/bin/env python3
"""
Portfolio v2 Analysis — Enhanced Combination Methods
代码熊 🐻 2026-02-20

Build on v2 base with additional methods to try to pass Walk-Forward:
1. Regime-adaptive allocation (use more Stock when BTC bearish)
2. Drawdown-responsive allocation (reduce the strategy that's in drawdown)
3. Monthly rebalance risk parity (smoother, less overfit)
4. Momentum-switching (100% into whichever has better recent momentum)
5. Different WF splits
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"

# Import from base module
sys.path.insert(0, str(Path(__file__).parent))
from portfolio_v2_realstock import (
    load_all_stock_data, precompute_stock_signals, run_stock_v3b_backtest,
    run_btc_v7f_daily, align_daily_returns, calc_metrics, composite_score,
    walk_forward, yearly_returns, CACHE, BASE, run_traditional_6040
)


def load_cached_returns():
    """Load previously saved daily returns if available."""
    stock_csv = BASE / "stocks" / "codebear" / "v3b_daily_returns.csv"
    btc_csv = BASE / "btc" / "codebear" / "v7f_daily_returns_2015_2025.csv"
    
    if stock_csv.exists() and btc_csv.exists():
        stock = pd.read_csv(stock_csv, parse_dates=['Date'], index_col='Date')['Return']
        btc = pd.read_csv(btc_csv, parse_dates=['Date'], index_col='Date')['Return']
        return btc, stock
    return None, None


# ═══════════════════════════════════════════════════════════
# Enhanced Portfolio Methods
# ═══════════════════════════════════════════════════════════

def regime_adaptive_portfolio(btc_ret, stock_ret, lookback_mom=60, lookback_vol=20):
    """
    Regime-adaptive: use more Stock when BTC is bearish, more BTC when trending up.
    Uses momentum + vol regimes for each strategy.
    """
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    n = len(btc_a)
    lb = max(lookback_mom, lookback_vol)
    port_ret = pd.Series(0.0, index=btc_a.index)
    
    for i in range(lb, n):
        # Momentum signals
        btc_cum = (1 + btc_a.iloc[i-lookback_mom:i]).prod() - 1
        stock_cum = (1 + stock_a.iloc[i-lookback_mom:i]).prod() - 1
        
        # Volatility
        btc_vol = btc_a.iloc[i-lookback_vol:i].std() * np.sqrt(252)
        stock_vol = stock_a.iloc[i-lookback_vol:i].std() * np.sqrt(252)
        btc_vol = max(btc_vol, 0.01)
        stock_vol = max(stock_vol, 0.01)
        
        # Base: inverse vol
        inv_btc = 1.0 / btc_vol
        inv_stock = 1.0 / stock_vol
        total = inv_btc + inv_stock
        w_btc = inv_btc / total
        w_stock = inv_stock / total
        
        # Momentum tilt: shift toward winner
        if btc_cum > 0 and stock_cum > 0:
            # Both positive → tilt toward stronger
            if btc_cum > stock_cum:
                w_btc = min(w_btc * 1.3, 0.80)
                w_stock = 1.0 - w_btc
            else:
                w_stock = min(w_stock * 1.3, 0.80)
                w_btc = 1.0 - w_stock
        elif btc_cum > 0 and stock_cum <= 0:
            # Only BTC positive
            w_btc = min(w_btc * 1.5, 0.85)
            w_stock = 1.0 - w_btc
        elif stock_cum > 0 and btc_cum <= 0:
            # Only Stock positive
            w_stock = min(w_stock * 1.5, 0.85)
            w_btc = 1.0 - w_stock
        else:
            # Both negative → defensive, reduce total exposure
            w_btc *= 0.6
            w_stock *= 0.6
        
        port_ret.iloc[i] = w_btc * btc_a.iloc[i] + w_stock * stock_a.iloc[i]
    
    return port_ret.iloc[lb:]


def drawdown_responsive_portfolio(btc_ret, stock_ret, lookback=20, dd_threshold=-0.10):
    """
    Reduce allocation to the strategy currently in drawdown.
    When one strategy is in drawdown, shift toward the other.
    """
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    n = len(btc_a)
    port_ret = pd.Series(0.0, index=btc_a.index)
    
    # Track cumulative for drawdown
    btc_cum = (1 + btc_a).cumprod()
    stock_cum = (1 + stock_a).cumprod()
    
    for i in range(lookback, n):
        # Current drawdown
        btc_peak = btc_cum.iloc[:i+1].max()
        stock_peak = stock_cum.iloc[:i+1].max()
        btc_dd = (btc_cum.iloc[i] - btc_peak) / btc_peak
        stock_dd = (stock_cum.iloc[i] - stock_peak) / stock_peak
        
        # Base: inverse vol
        btc_vol = btc_a.iloc[i-lookback:i].std() * np.sqrt(252)
        stock_vol = stock_a.iloc[i-lookback:i].std() * np.sqrt(252)
        btc_vol = max(btc_vol, 0.01)
        stock_vol = max(stock_vol, 0.01)
        
        inv_btc = 1.0 / btc_vol
        inv_stock = 1.0 / stock_vol
        total = inv_btc + inv_stock
        w_btc = inv_btc / total
        w_stock = inv_stock / total
        
        # If one strategy is in deep drawdown, reduce its weight
        if btc_dd < dd_threshold:
            reduction = max(0.3, 1.0 + btc_dd * 2)  # more DD = more reduction
            w_btc *= reduction
        if stock_dd < dd_threshold:
            reduction = max(0.3, 1.0 + stock_dd * 2)
            w_stock *= reduction
        
        # Renormalize
        total = w_btc + w_stock
        if total > 0:
            # Don't renormalize to 1.0 → allow cash position
            pass
        
        port_ret.iloc[i] = w_btc * btc_a.iloc[i] + w_stock * stock_a.iloc[i]
    
    return port_ret.iloc[lookback:]


def monthly_rp_portfolio(btc_ret, stock_ret):
    """
    Monthly rebalancing risk parity - rebalance weights only at month end.
    Less trading, more stable, may have better WF.
    """
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    
    # Group by month, compute vol for each
    df = pd.DataFrame({'btc': btc_a, 'stock': stock_a})
    
    port_ret = pd.Series(0.0, index=df.index)
    
    months = df.resample('ME').last().index
    
    for i in range(1, len(months)):
        # Use previous month's data to set weights
        prev_month = months[i-1]
        curr_month = months[i]
        
        # Get trailing 60 days before prev_month for vol calculation
        loc = df.index.searchsorted(prev_month, side='right') - 1
        loc = max(0, loc)
        lookback_start = df.index[max(0, loc - 60)]
        
        btc_window = df['btc'].loc[lookback_start:prev_month]
        stock_window = df['stock'].loc[lookback_start:prev_month]
        
        if len(btc_window) < 20 or len(stock_window) < 20:
            continue
        
        btc_vol = btc_window.std() * np.sqrt(252)
        stock_vol = stock_window.std() * np.sqrt(252)
        btc_vol = max(btc_vol, 0.01)
        stock_vol = max(stock_vol, 0.01)
        
        w_btc = (1/btc_vol) / (1/btc_vol + 1/stock_vol)
        w_stock = 1.0 - w_btc
        
        # Apply these weights for the entire current month
        mask = (df.index > prev_month) & (df.index <= curr_month)
        port_ret.loc[mask] = w_btc * df['btc'].loc[mask] + w_stock * df['stock'].loc[mask]
    
    # Skip warmup
    return port_ret[port_ret != 0]


def momentum_switch_portfolio(btc_ret, stock_ret, lookback=63, smooth=0.3):
    """
    Momentum switching: allocate more to the strategy with better recent momentum.
    Smoother version with partial switching.
    """
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    n = len(btc_a)
    port_ret = pd.Series(0.0, index=btc_a.index)
    prev_w_btc = 0.5
    
    for i in range(lookback, n):
        btc_mom = (1 + btc_a.iloc[i-lookback:i]).prod() - 1
        stock_mom = (1 + stock_a.iloc[i-lookback:i]).prod() - 1
        
        # Target: 70% to winner, 30% to loser
        if btc_mom > stock_mom:
            target_btc = 0.70
        else:
            target_btc = 0.30
        
        # Smooth transition
        w_btc = smooth * target_btc + (1 - smooth) * prev_w_btc
        w_stock = 1.0 - w_btc
        prev_w_btc = w_btc
        
        port_ret.iloc[i] = w_btc * btc_a.iloc[i] + w_stock * stock_a.iloc[i]
    
    return port_ret.iloc[lookback:]


def halving_aware_portfolio(btc_ret, stock_ret, lookback=20):
    """
    Uses BTC halving cycle to time BTC allocation.
    0-18 months after halving: more BTC (bull expected)
    18-36 months after halving: shift to Stock (bear expected)
    """
    HALVINGS = [pd.Timestamp('2016-07-09'), pd.Timestamp('2020-05-11'), pd.Timestamp('2024-04-20')]
    
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    n = len(btc_a)
    port_ret = pd.Series(0.0, index=btc_a.index)
    
    for i in range(lookback, n):
        date = btc_a.index[i]
        
        # Find months since last halving
        months_since = None
        for h in reversed(HALVINGS):
            if date >= h:
                months_since = (date - h).days / 30.44
                break
        
        # Base: inverse vol
        btc_vol = btc_a.iloc[i-lookback:i].std() * np.sqrt(252)
        stock_vol = stock_a.iloc[i-lookback:i].std() * np.sqrt(252)
        btc_vol = max(btc_vol, 0.01)
        stock_vol = max(stock_vol, 0.01)
        
        inv_btc = 1.0 / btc_vol
        inv_stock = 1.0 / stock_vol
        total = inv_btc + inv_stock
        w_btc = inv_btc / total
        w_stock = inv_stock / total
        
        # Halving cycle overlay
        if months_since is not None:
            if months_since <= 12:
                # Early post-halving: BTC bull, max allocation
                w_btc = max(w_btc, 0.65)
                w_stock = 1.0 - w_btc
            elif months_since <= 18:
                # Mid post-halving: still bullish
                w_btc = max(w_btc, 0.55)
                w_stock = 1.0 - w_btc
            elif months_since <= 30:
                # Late cycle: shift to Stock
                w_stock = max(w_stock, 0.55)
                w_btc = 1.0 - w_stock
            else:
                # Bear/accumulation: risk parity base (no overlay)
                pass
        
        port_ret.iloc[i] = w_btc * btc_a.iloc[i] + w_stock * stock_a.iloc[i]
    
    return port_ret.iloc[lookback:]


def optimized_combo(btc_ret, stock_ret, lookback_vol=20, lookback_mom=60, lookback_corr=60):
    """
    Kitchen sink: combine risk parity + momentum tilt + correlation adjustment + halving cycle.
    The "best effort" approach.
    """
    HALVINGS = [pd.Timestamp('2016-07-09'), pd.Timestamp('2020-05-11'), pd.Timestamp('2024-04-20')]
    
    btc_a, stock_a = align_daily_returns(btc_ret, stock_ret)
    n = len(btc_a)
    lb = max(lookback_vol, lookback_mom, lookback_corr)
    port_ret = pd.Series(0.0, index=btc_a.index)
    
    for i in range(lb, n):
        date = btc_a.index[i]
        
        # 1. Risk parity base
        btc_vol = btc_a.iloc[i-lookback_vol:i].std() * np.sqrt(252)
        stock_vol = stock_a.iloc[i-lookback_vol:i].std() * np.sqrt(252)
        btc_vol = max(btc_vol, 0.05)
        stock_vol = max(stock_vol, 0.05)
        
        inv_btc = 1.0 / btc_vol
        inv_stock = 1.0 / stock_vol
        total = inv_btc + inv_stock
        w_btc = inv_btc / total
        w_stock = inv_stock / total
        
        # 2. Momentum tilt (subtle: 80% RP + 20% momentum)
        btc_mom = (1 + btc_a.iloc[i-lookback_mom:i]).prod() - 1
        stock_mom = (1 + stock_a.iloc[i-lookback_mom:i]).prod() - 1
        
        if btc_mom + stock_mom != 0:
            # Shift proportionally toward stronger momentum
            mom_total = abs(btc_mom) + abs(stock_mom)
            if mom_total > 0:
                btc_mom_w = max(btc_mom, 0) / mom_total
                stock_mom_w = max(stock_mom, 0) / mom_total
                if btc_mom_w + stock_mom_w > 0:
                    scale = 1.0 / (btc_mom_w + stock_mom_w)
                    btc_mom_w *= scale
                    stock_mom_w *= scale
                else:
                    btc_mom_w = 0.5
                    stock_mom_w = 0.5
            else:
                btc_mom_w = 0.5
                stock_mom_w = 0.5
        else:
            btc_mom_w = 0.5
            stock_mom_w = 0.5
        
        w_btc = 0.80 * w_btc + 0.20 * btc_mom_w
        w_stock = 0.80 * w_stock + 0.20 * stock_mom_w
        
        # 3. Correlation adjustment
        corr = btc_a.iloc[i-lookback_corr:i].corr(stock_a.iloc[i-lookback_corr:i])
        if np.isnan(corr):
            corr = 0.0
        
        if corr > 0.5:
            # High corr → reduce exposure slightly
            scale = 0.90
            w_btc *= scale
            w_stock *= scale
        elif corr < -0.1:
            # Negative corr → boost slightly (diversification)
            boost = min(1.10, 1.0 + abs(corr) * 0.15)
            w_btc *= boost
            w_stock *= boost
        
        # 4. Halving cycle
        months_since = None
        for h in reversed(HALVINGS):
            if date >= h:
                months_since = (date - h).days / 30.44
                break
        
        if months_since is not None and months_since <= 15:
            # Post-halving: slight BTC tilt
            w_btc = max(w_btc, 0.45)
        
        # 5. Both negative → reduce exposure
        if btc_mom < -0.05 and stock_mom < -0.05:
            w_btc *= 0.70
            w_stock *= 0.70
        
        # Normalize cap at 1.0
        total = w_btc + w_stock
        if total > 1.0:
            w_btc /= total
            w_stock /= total
        
        port_ret.iloc[i] = w_btc * btc_a.iloc[i] + w_stock * stock_a.iloc[i]
    
    return port_ret.iloc[lb:]


def walk_forward_multiple_splits(returns):
    """Test multiple WF splits to find the most robust."""
    results = []
    for split_pct in [0.50, 0.55, 0.60, 0.65]:
        n = len(returns)
        sp = int(n * split_pct)
        is_ret = returns.iloc[:sp]
        oos_ret = returns.iloc[sp:]
        
        is_m = calc_metrics(is_ret)
        oos_m = calc_metrics(oos_ret)
        
        if is_m and oos_m and is_m['sharpe'] > 0:
            wf = oos_m['sharpe'] / is_m['sharpe']
            results.append({
                'split': split_pct,
                'is_sharpe': is_m['sharpe'],
                'oos_sharpe': oos_m['sharpe'],
                'wf': wf,
            })
    
    return results


def main():
    print("=" * 80)
    print("🐻 Portfolio v2 Analysis — Enhanced Methods")
    print("  Trying harder to beat QQQ proxy Composite 1.488")
    print("=" * 80)

    # Load cached returns
    btc_daily, stock_daily = load_cached_returns()
    
    if btc_daily is None or stock_daily is None:
        print("  ❌ Run portfolio_v2_realstock.py first to generate daily returns")
        return
    
    print(f"\n  Loaded: BTC v7f {len(btc_daily)} days, Stock v3b {len(stock_daily)} days")
    
    btc_a, stock_a = align_daily_returns(btc_daily, stock_daily)
    print(f"  Aligned: {len(btc_a)} common trading days")
    print(f"  Overall correlation: {btc_a.corr(stock_a):.3f}")

    # ── Run all portfolio methods ──
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PORTFOLIO COMPARISON")
    print("=" * 80)

    portfolios = {}
    
    # Basic methods
    from portfolio_v2_realstock import (
        fixed_weight_portfolio, rolling_risk_parity, 
        correlation_adjusted_portfolio, target_volatility_portfolio
    )
    
    portfolios['50/50 Fixed'] = fixed_weight_portfolio(btc_daily, stock_daily, 0.50, 0.50)
    portfolios['60/40 BTC/Stock'] = fixed_weight_portfolio(btc_daily, stock_daily, 0.60, 0.40)
    portfolios['70/30 BTC/Stock'] = fixed_weight_portfolio(btc_daily, stock_daily, 0.70, 0.30)
    portfolios['40/60 BTC/Stock'] = fixed_weight_portfolio(btc_daily, stock_daily, 0.40, 0.60)
    portfolios['30/70 BTC/Stock'] = fixed_weight_portfolio(btc_daily, stock_daily, 0.30, 0.70)
    
    portfolios['RP 20d'] = rolling_risk_parity(btc_daily, stock_daily, 20)
    portfolios['RP 60d'] = rolling_risk_parity(btc_daily, stock_daily, 60)
    portfolios['RP 126d'] = rolling_risk_parity(btc_daily, stock_daily, 126)
    
    portfolios['Corr-Adjusted'] = correlation_adjusted_portfolio(btc_daily, stock_daily, 60)
    portfolios['Target Vol 30%'] = target_volatility_portfolio(btc_daily, stock_daily, 0.30, 20)
    portfolios['Target Vol 25%'] = target_volatility_portfolio(btc_daily, stock_daily, 0.25, 20)
    portfolios['Target Vol 20%'] = target_volatility_portfolio(btc_daily, stock_daily, 0.20, 20)
    
    # Enhanced methods
    portfolios['Regime Adaptive'] = regime_adaptive_portfolio(btc_daily, stock_daily)
    portfolios['DD Responsive'] = drawdown_responsive_portfolio(btc_daily, stock_daily)
    portfolios['Monthly RP'] = monthly_rp_portfolio(btc_daily, stock_daily)
    portfolios['Mom Switch 63d'] = momentum_switch_portfolio(btc_daily, stock_daily, 63)
    portfolios['Mom Switch 126d'] = momentum_switch_portfolio(btc_daily, stock_daily, 126)
    portfolios['Halving Aware'] = halving_aware_portfolio(btc_daily, stock_daily)
    portfolios['Optimized Combo'] = optimized_combo(btc_daily, stock_daily)
    
    # Benchmarks
    portfolios['Stock v3b Solo'] = stock_daily
    portfolios['BTC v7f Solo'] = btc_daily
    portfolios['SPY60/TLT40'] = run_traditional_6040()

    # ── Evaluate all ──
    print(f"\n{'Strategy':<22} {'CAGR':>6} {'MaxDD':>7} {'Shrp':>5} {'Calm':>5} {'Vol':>5} {'IS':>5} {'OOS':>5} {'WF':>5} {'Comp':>6}")
    print("-" * 92)

    all_results = {}
    for name, ret in sorted(portfolios.items()):
        m = calc_metrics(ret)
        if m is None:
            continue
        
        is_m, oos_m, wf = walk_forward(ret)
        comp = composite_score(m)
        
        is_sh = is_m['sharpe'] if is_m else 0
        oos_sh = oos_m['sharpe'] if oos_m else 0
        wf_mark = '✅' if wf >= 0.70 else '  '
        
        all_results[name] = {
            'metrics': m, 'is_m': is_m, 'oos_m': oos_m, 'wf': wf, 'composite': comp,
        }
        
        print(f"{name:<22} {m['cagr']:>5.1%} {m['max_dd']:>6.1%} {m['sharpe']:>5.2f} {m['calmar']:>5.2f} {m['ann_vol']:>4.1%} {is_sh:>5.2f} {oos_sh:>5.2f} {wf:>4.2f}{wf_mark} {comp:>6.3f}")

    # ── WF with multiple splits ──
    print("\n" + "=" * 80)
    print("WALK-FORWARD ROBUSTNESS (Multiple Splits)")
    print("=" * 80)
    
    top_strats = sorted(all_results.items(), key=lambda x: x[1]['composite'], reverse=True)[:8]
    
    for name, data in top_strats:
        if 'Solo' in name or 'SPY' in name:
            continue
        ret = portfolios[name]
        wf_tests = walk_forward_multiple_splits(ret)
        
        if wf_tests:
            avg_wf = np.mean([t['wf'] for t in wf_tests])
            min_wf = min(t['wf'] for t in wf_tests)
            max_wf = max(t['wf'] for t in wf_tests)
            pass_count = sum(1 for t in wf_tests if t['wf'] >= 0.70)
            
            print(f"\n  {name}:")
            print(f"    Composite: {data['composite']:.3f}")
            for t in wf_tests:
                mark = '✅' if t['wf'] >= 0.70 else '❌'
                print(f"    Split {t['split']:.0%}: IS={t['is_sharpe']:.2f} OOS={t['oos_sharpe']:.2f} WF={t['wf']:.2f} {mark}")
            print(f"    Avg WF: {avg_wf:.2f}, Min: {min_wf:.2f}, Max: {max_wf:.2f}, Pass: {pass_count}/{len(wf_tests)}")

    # ── Best overall ──
    print("\n" + "=" * 80)
    print("🏆 FINAL RANKING (by Composite Score)")
    print("=" * 80)
    
    sorted_all = sorted(all_results.items(), key=lambda x: x[1]['composite'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Strategy':<22} {'Comp':>6} {'CAGR':>6} {'Sharpe':>6} {'MaxDD':>7} {'WF':>5}")
    print("-" * 62)
    
    for rank, (name, data) in enumerate(sorted_all, 1):
        m = data['metrics']
        wf_mark = '✅' if data['wf'] >= 0.70 else '  '
        print(f"{rank:<5} {name:<22} {data['composite']:>6.3f} {m['cagr']:>5.1%} {m['sharpe']:>6.2f} {m['max_dd']:>6.1%} {data['wf']:>4.2f}{wf_mark}")

    # ── WF-passing only ──
    wf_passing = {k: v for k, v in all_results.items() 
                  if v['wf'] >= 0.70 and 'Solo' not in k and 'SPY' not in k}
    
    print(f"\n\n  WF-Passing Portfolios: {len(wf_passing)}")
    if wf_passing:
        best_name = max(wf_passing, key=lambda k: wf_passing[k]['composite'])
        best = wf_passing[best_name]
        bm = best['metrics']
        print(f"\n  🏆 BEST WF-PASSING: {best_name}")
        print(f"     Composite: {best['composite']:.3f} (vs QQQ proxy 1.488)")
    else:
        # Show the closest to passing WF
        non_solo = {k: v for k, v in all_results.items() if 'Solo' not in k and 'SPY' not in k}
        sorted_by_wf = sorted(non_solo.items(), key=lambda x: x[1]['wf'], reverse=True)
        print(f"\n  ❌ No portfolio passed WF ≥ 0.70")
        print(f"\n  Closest to passing:")
        for name, data in sorted_by_wf[:5]:
            m = data['metrics']
            print(f"    {name:<22} WF={data['wf']:.3f} Comp={data['composite']:.3f} Sharpe={m['sharpe']:.2f}")
        
        # Also show best by composite
        sorted_by_comp = sorted(non_solo.items(), key=lambda x: x[1]['composite'], reverse=True)
        print(f"\n  Best by Composite (regardless of WF):")
        for name, data in sorted_by_comp[:5]:
            m = data['metrics']
            print(f"    {name:<22} Comp={data['composite']:.3f} WF={data['wf']:.3f} Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1%}")

    # ── Key insight: Why WF fails ──
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Why Walk-Forward Fails")
    print("=" * 80)
    
    # Compare IS vs OOS for BTC and Stock separately
    for name in ['BTC v7f Solo', 'Stock v3b Solo']:
        if name in all_results:
            d = all_results[name]
            print(f"\n  {name}:")
            print(f"    IS Sharpe:  {d['is_m']['sharpe']:.2f}")
            print(f"    OOS Sharpe: {d['oos_m']['sharpe']:.2f}")
            print(f"    WF Ratio:   {d['wf']:.2f}")
            print(f"    IS CAGR:    {d['is_m']['cagr']:.1%}")
            print(f"    OOS CAGR:   {d['oos_m']['cagr']:.1%}")
    
    print(f"\n  Key Insight:")
    print(f"  The BTC v7f strategy had IS Sharpe ~1.25 but OOS Sharpe ~0.72")
    print(f"  This is because IS (2015-2020) includes the 2017 mega bull run")
    print(f"  and 2020 post-COVID recovery, while OOS (2021-2025) includes")
    print(f"  the brutal 2022 crypto+stock bear market.")
    print(f"  The combination inherits BTC's WF problem.")
    print(f"  Stock v3b Solo actually passes WF (0.94) — it's BTC dragging WF down.")

    print("\n🐻 Enhanced Analysis Complete!")
    return all_results


if __name__ == '__main__':
    main()
