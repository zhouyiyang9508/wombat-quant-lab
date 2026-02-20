"""
Multi-Strategy Portfolio v2 — Rolling Risk Parity + Walk-Forward Focus
Fix WF failure by using adaptive (rolling) weights instead of fixed weights.

代码熊 🐻 2026-02-20
"""

import pandas as pd
import numpy as np
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from multi_strategy_v1 import (
    run_tqqq_v9g, run_btc_v7f, run_stock_momentum_v2d, run_hedge_v4,
    align_returns, calc_metrics, composite_score
)

def rolling_risk_parity(returns_list, lookback=126):
    """Rolling inverse-vol weighting, rebalanced daily"""
    aligned = align_returns(*returns_list)
    n = len(aligned[0])
    k = len(aligned)
    port_ret = np.zeros(n)
    
    for i in range(lookback, n):
        vols = []
        for j in range(k):
            window = aligned[j].values[i-lookback:i]
            vol = np.std(window) * np.sqrt(252)
            vols.append(max(vol, 0.01))
        inv = [1/v for v in vols]
        total = sum(inv)
        weights = [v/total for v in inv]
        port_ret[i] = sum(w * aligned[j].values[i] for j, w in enumerate(weights))
    
    return pd.Series(port_ret[lookback:], index=aligned[0].index[lookback:])


def rolling_momentum_tilt(returns_list, lookback=126, mom_lookback=63):
    """Rolling risk parity + momentum tilt: overweight recent winners"""
    aligned = align_returns(*returns_list)
    n = len(aligned[0])
    k = len(aligned)
    port_ret = np.zeros(n)
    
    start = max(lookback, mom_lookback)
    for i in range(start, n):
        # Base: inverse vol
        vols = []
        for j in range(k):
            window = aligned[j].values[i-lookback:i]
            vol = np.std(window) * np.sqrt(252)
            vols.append(max(vol, 0.01))
        inv = [1/v for v in vols]
        total_inv = sum(inv)
        base_w = [v/total_inv for v in inv]
        
        # Momentum tilt
        moms = []
        for j in range(k):
            cum = np.prod(1 + aligned[j].values[i-mom_lookback:i])
            moms.append(cum - 1)
        
        # Tilt: 70% risk parity + 30% momentum rank
        rank = np.argsort(np.argsort(moms))  # 0=worst, k-1=best
        rank_w = (rank + 1) / sum(range(1, k+1))
        
        final_w = [0.7 * base_w[j] + 0.3 * rank_w[j] for j in range(k)]
        total = sum(final_w)
        final_w = [w/total for w in final_w]
        
        port_ret[i] = sum(final_w[j] * aligned[j].values[i] for j in range(k))
    
    return pd.Series(port_ret[start:], index=aligned[0].index[start:])


def rolling_max_sharpe(returns_list, lookback=252):
    """Rolling max-Sharpe optimization (simplified mean-variance)"""
    aligned = align_returns(*returns_list)
    n = len(aligned[0])
    k = len(aligned)
    port_ret = np.zeros(n)
    prev_w = np.ones(k) / k
    
    for i in range(lookback, n):
        # Get window returns
        window = np.array([aligned[j].values[i-lookback:i] for j in range(k)])
        means = np.mean(window, axis=1) * 252
        cov = np.cov(window) * 252
        
        # Simple: try many random portfolios, pick best Sharpe
        best_sharpe = -999
        best_w = prev_w.copy()
        rf = 0.04
        
        # Include some structured candidates
        candidates = [np.ones(k)/k]  # equal
        for j in range(k):
            w = np.zeros(k)
            w[j] = 1.0
            candidates.append(w)
        
        # Random samples
        rng = np.random.RandomState(i)
        for _ in range(200):
            w = rng.dirichlet(np.ones(k))
            candidates.append(w)
        
        for w in candidates:
            port_mean = w @ means
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-6:
                continue
            sharpe = (port_mean - rf) / port_vol
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = w
        
        # Smooth with previous weights
        smooth_w = 0.8 * prev_w + 0.2 * best_w
        smooth_w /= smooth_w.sum()
        prev_w = smooth_w
        
        port_ret[i] = sum(smooth_w[j] * aligned[j].values[i] for j in range(k))
    
    return pd.Series(port_ret[lookback:], index=aligned[0].index[lookback:])


def walk_forward_detailed(port_ret, split=0.6):
    n = len(port_ret)
    sp = int(n * split)
    is_m = calc_metrics(port_ret.values[:sp])
    oos_m = calc_metrics(port_ret.values[sp:])
    return is_m, oos_m


def main():
    print("=" * 80)
    print("Multi-Strategy Portfolio v2 — Adaptive Weights")
    print("代码熊 🐻 2026-02-20")
    print("=" * 80)

    print("\n[1] Running strategies...")
    strats = {
        'TQQQ_v9g': run_tqqq_v9g(),
        'BTC_v7f': run_btc_v7f(),
        'StockMom': run_stock_momentum_v2d(),
        'Hedge_v4': run_hedge_v4(),
    }
    names = list(strats.keys())
    rets = [strats[n] for n in names]

    # Also test 2-strategy combo (BTC + StockMom only)
    btc_stock = [strats['BTC_v7f'], strats['StockMom']]
    
    configs = {}
    
    # 4-strategy adaptive
    print("\n[2] 4-Strategy Adaptive Portfolios:")
    print(f"{'Method':<30} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'Score':>7} {'IS Sh':>7} {'OOS Sh':>7} {'WF':>5}")
    print("-" * 95)

    methods = [
        ("4S RiskParity 6M", lambda: rolling_risk_parity(rets, 126)),
        ("4S RiskParity 12M", lambda: rolling_risk_parity(rets, 252)),
        ("4S MomTilt 6M", lambda: rolling_momentum_tilt(rets, 126, 63)),
        ("4S MomTilt 12M", lambda: rolling_momentum_tilt(rets, 252, 126)),
        ("4S MaxSharpe 12M", lambda: rolling_max_sharpe(rets, 252)),
        ("4S Fixed 0/30/65/5", lambda: pd.Series(
            sum(w * r.values for w, r in zip([0,0.30,0.65,0.05], align_returns(*rets))),
            index=align_returns(*rets)[0].index)),
    ]
    
    # 2-strategy (BTC + StockMom)
    methods += [
        ("2S BTC30/Stock70 Fixed", lambda: pd.Series(
            sum(w * r.values for w, r in zip([0.30, 0.70], align_returns(*btc_stock))),
            index=align_returns(*btc_stock)[0].index)),
        ("2S BTC40/Stock60 Fixed", lambda: pd.Series(
            sum(w * r.values for w, r in zip([0.40, 0.60], align_returns(*btc_stock))),
            index=align_returns(*btc_stock)[0].index)),
        ("2S RiskParity 6M", lambda: rolling_risk_parity(btc_stock, 126)),
        ("2S RiskParity 12M", lambda: rolling_risk_parity(btc_stock, 252)),
        ("2S MomTilt 6M", lambda: rolling_momentum_tilt(btc_stock, 126, 63)),
        ("2S MaxSharpe 12M", lambda: rolling_max_sharpe(btc_stock, 252)),
    ]

    for label, fn in methods:
        port = fn()
        m = calc_metrics(port)
        if not m:
            continue
        sc = composite_score(m)
        is_m, oos_m = walk_forward_detailed(port)
        if is_m and oos_m and is_m.get('Sharpe', 0) > 0:
            wf = oos_m['Sharpe'] / is_m['Sharpe']
            wf_s = "✅" if wf >= 0.70 else "❌"
            print(f"{label:<30} {m['CAGR']:>6.1%} {m['MaxDD']:>6.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {sc:>7.3f} {is_m['Sharpe']:>7.2f} {oos_m['Sharpe']:>7.2f} {wf:.2f}{wf_s}")
            configs[label] = {'metrics': m, 'score': sc, 'wf_ratio': wf, 'is': is_m, 'oos': oos_m}
        else:
            print(f"{label:<30} {m['CAGR']:>6.1%} {m['MaxDD']:>6.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {sc:>7.3f} {'N/A':>7} {'N/A':>7} {'N/A':>5}")

    # Find best WF-passing config
    print("\n[3] Best Walk-Forward Passing Configurations:")
    wf_pass = {k: v for k, v in configs.items() if v.get('wf_ratio', 0) >= 0.70}
    if wf_pass:
        sorted_pass = sorted(wf_pass.items(), key=lambda x: x[1]['score'], reverse=True)
        for name, data in sorted_pass[:3]:
            m = data['metrics']
            print(f"  🏆 {name}: Score={data['score']:.3f} Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']:.1%} MaxDD={m['MaxDD']:.1%} WF={data['wf_ratio']:.2f}")
    else:
        print("  ❌ No configuration passed Walk-Forward")
        # Show best overall anyway
        sorted_all = sorted(configs.items(), key=lambda x: x[1]['score'], reverse=True)
        for name, data in sorted_all[:3]:
            m = data['metrics']
            print(f"  📊 {name}: Score={data['score']:.3f} Sharpe={m['Sharpe']:.2f} WF={data.get('wf_ratio',0):.2f}")

    print("\n" + "=" * 80)
    print("Done! 🐻")


if __name__ == '__main__':
    main()
