"""
Portfolio v3 — Stock v4d (DD+GLD) + BTC v7f Dual Momentum
Combines the two best strategies with low correlation.

Key difference from v2: Stock v4d already has built-in GLD hedge,
so the portfolio-level DD responsive might be redundant or need recalibration.

Methods tested:
1. Fixed weights (40/60, 50/50, 60/40 BTC/Stock)
2. Risk Parity (inverse vol, 60d/126d lookback)
3. Momentum Tilt (tilt toward higher momentum)
4. DD Responsive (portfolio-level drawdown protection)
5. Correlation-adjusted (reduce BTC when corr spikes)

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[2]

def load_returns():
    """Load daily returns for both strategies."""
    stock_ret = pd.read_csv(BASE / "stocks/codebear/v3b_daily_returns.csv", 
                           parse_dates=["Date"]).set_index("Date")
    btc_ret = pd.read_csv(BASE / "btc/codebear/v7f_daily_returns_2015_2025.csv",
                          parse_dates=["Date"]).set_index("Date")
    
    # Need Stock v4d returns - generate from v3b + GLD DD hedge
    gld = pd.read_csv(BASE / "data_cache/GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    gld_ret = gld["Close"].pct_change()
    
    # Align all
    common = stock_ret.index.intersection(btc_ret.index).intersection(gld_ret.index)
    common = common.sort_values()
    
    stock_r = stock_ret.loc[common, "Return"].values
    btc_r = btc_ret.loc[common, "Return"].values
    gld_r = gld_ret.loc[common].values
    dates = common
    
    # Apply DD responsive GLD hedge to stock returns (simulate v4d from v3b)
    v4d_r = apply_dd_gld_hedge(stock_r, gld_r)
    
    return dates, btc_r, v4d_r, stock_r, gld_r

def apply_dd_gld_hedge(stock_ret, gld_ret, thresholds=(-0.08, -0.12, -0.18), 
                        gld_pcts=(0.30, 0.50, 0.60)):
    """Apply DD-responsive GLD hedge to stock returns."""
    n = len(stock_ret)
    equity = np.ones(n + 1)
    hedged_ret = np.zeros(n)
    
    for i in range(n):
        # Current drawdown
        peak = equity[:i+1].max()
        dd = (equity[i] - peak) / peak if peak > 0 else 0
        
        # Determine GLD allocation
        if dd < thresholds[2]:
            gld_pct = gld_pcts[2]
        elif dd < thresholds[1]:
            gld_pct = gld_pcts[1]
        elif dd < thresholds[0]:
            gld_pct = gld_pcts[0]
        else:
            gld_pct = 0.0
        
        stock_pct = 1.0 - gld_pct
        hedged_ret[i] = stock_pct * stock_ret[i] + gld_pct * gld_ret[i]
        equity[i+1] = equity[i] * (1 + hedged_ret[i])
    
    return hedged_ret

def calc_metrics(returns, rf=0.04):
    """Calculate metrics from daily returns array."""
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    
    equity = np.cumprod(1 + returns) * 10000
    equity = np.insert(equity, 0, 10000)
    
    years = n / 252
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = dd.min()
    
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    
    return {
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar,
        "final_value": equity[-1], "ann_vol": ann_vol, "years": years
    }

def composite_score(m):
    return m["sharpe"] * 0.4 + m["calmar"] * 0.4 + min(m["cagr"], 1.0) * 0.2

def walk_forward(returns, split=0.6):
    n = len(returns)
    s = int(n * split)
    m_is = calc_metrics(returns[:s])
    m_oos = calc_metrics(returns[s:])
    return m_is, m_oos

def portfolio_fixed(btc_r, stock_r, w_btc):
    return w_btc * btc_r + (1 - w_btc) * stock_r

def portfolio_risk_parity(btc_r, stock_r, lookback=126):
    n = len(btc_r)
    port_r = np.zeros(n)
    for i in range(lookback, n):
        vol_b = np.std(btc_r[i-lookback:i]) * np.sqrt(252)
        vol_s = np.std(stock_r[i-lookback:i]) * np.sqrt(252)
        if vol_b + vol_s == 0:
            w_b = 0.5
        else:
            inv_b = 1 / vol_b if vol_b > 0 else 10
            inv_s = 1 / vol_s if vol_s > 0 else 10
            w_b = inv_b / (inv_b + inv_s)
        port_r[i] = w_b * btc_r[i] + (1 - w_b) * stock_r[i]
    # Fill initial period with equal weight
    port_r[:lookback] = 0.5 * btc_r[:lookback] + 0.5 * stock_r[:lookback]
    return port_r

def portfolio_mom_tilt(btc_r, stock_r, lookback=126):
    """Tilt toward higher momentum strategy."""
    n = len(btc_r)
    port_r = np.zeros(n)
    for i in range(lookback, n):
        cum_b = np.prod(1 + btc_r[i-lookback:i]) - 1
        cum_s = np.prod(1 + stock_r[i-lookback:i]) - 1
        
        # Inverse vol for base, then tilt by momentum
        vol_b = np.std(btc_r[i-lookback:i]) * np.sqrt(252)
        vol_s = np.std(stock_r[i-lookback:i]) * np.sqrt(252)
        inv_b = 1 / max(vol_b, 0.01)
        inv_s = 1 / max(vol_s, 0.01)
        
        # Momentum score (sigmoid mapping)
        mom_diff = cum_b - cum_s
        tilt = 1 / (1 + np.exp(-5 * mom_diff))  # 0-1, >0.5 favors BTC
        
        # Base RP + tilt
        base_b = inv_b / (inv_b + inv_s)
        w_b = 0.5 * base_b + 0.5 * tilt
        w_b = np.clip(w_b, 0.15, 0.85)
        
        port_r[i] = w_b * btc_r[i] + (1 - w_b) * stock_r[i]
    port_r[:lookback] = 0.5 * btc_r[:lookback] + 0.5 * stock_r[:lookback]
    return port_r

def portfolio_dd_responsive(btc_r, stock_r, gld_r):
    """Portfolio-level DD protection with GLD."""
    n = len(btc_r)
    port_r = np.zeros(n)
    equity = np.ones(n + 1)
    
    for i in range(n):
        peak = equity[:i+1].max()
        dd = (equity[i] - peak) / peak if peak > 0 else 0
        
        if dd < -0.15:
            # Heavy protection
            port_r[i] = 0.20 * btc_r[i] + 0.30 * stock_r[i] + 0.40 * gld_r[i]
        elif dd < -0.10:
            port_r[i] = 0.25 * btc_r[i] + 0.40 * stock_r[i] + 0.25 * gld_r[i]
        elif dd < -0.05:
            port_r[i] = 0.30 * btc_r[i] + 0.50 * stock_r[i] + 0.15 * gld_r[i]
        else:
            port_r[i] = 0.40 * btc_r[i] + 0.60 * stock_r[i]
        
        equity[i+1] = equity[i] * (1 + port_r[i])
    
    return port_r

def portfolio_corr_adjusted(btc_r, stock_r, lookback=60):
    """Reduce BTC when correlation with stocks spikes."""
    n = len(btc_r)
    port_r = np.zeros(n)
    
    for i in range(lookback, n):
        corr = np.corrcoef(btc_r[i-lookback:i], stock_r[i-lookback:i])[0, 1]
        # High correlation = reduce diversification benefit, reduce BTC
        if corr > 0.5:
            w_b = 0.25
        elif corr > 0.3:
            w_b = 0.35
        else:
            w_b = 0.45
        port_r[i] = w_b * btc_r[i] + (1 - w_b) * stock_r[i]
    
    port_r[:lookback] = 0.40 * btc_r[:lookback] + 0.60 * stock_r[:lookback]
    return port_r

def main():
    dates, btc_r, v4d_r, v3b_r, gld_r = load_returns()
    n = len(dates)
    print(f"Data: {dates[0].date()} → {dates[-1].date()} ({n} days, {n/252:.1f} years)")
    
    # Correlation
    corr = np.corrcoef(btc_r, v4d_r)[0, 1]
    corr_v3b = np.corrcoef(btc_r, v3b_r)[0, 1]
    print(f"Correlation BTC_v7f vs Stock_v4d: {corr:.3f}")
    print(f"Correlation BTC_v7f vs Stock_v3b: {corr_v3b:.3f}")
    print()
    
    # Individual strategies
    m_btc = calc_metrics(btc_r)
    m_v4d = calc_metrics(v4d_r)
    m_v3b = calc_metrics(v3b_r)
    
    strategies = {}
    
    # Fixed weights
    for w in [0.3, 0.4, 0.5, 0.6]:
        name = f"Fixed_{int(w*100)}/{int((1-w)*100)}"
        r = portfolio_fixed(btc_r, v4d_r, w)
        strategies[name] = r
    
    # Risk parity
    for lb in [60, 126]:
        name = f"RP_{lb}d"
        strategies[name] = portfolio_risk_parity(btc_r, v4d_r, lb)
    
    # Momentum tilt
    for lb in [63, 126]:
        name = f"MomTilt_{lb}d"
        strategies[name] = portfolio_mom_tilt(btc_r, v4d_r, lb)
    
    # DD Responsive
    strategies["DD_Resp"] = portfolio_dd_responsive(btc_r, v4d_r, gld_r)
    
    # Corr adjusted
    strategies["Corr_Adj"] = portfolio_corr_adjusted(btc_r, v4d_r)
    
    # Print results
    print(f"{'Strategy':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'IS_Sh':>7} {'OOS_Sh':>7} {'WF_r':>6} {'Pass':>4} {'Score':>7}")
    print("-" * 105)
    
    # Baselines
    for name, r in [("BTC_v7f Solo", btc_r), ("Stock_v4d Solo", v4d_r), ("Stock_v3b Solo", v3b_r)]:
        m = calc_metrics(r)
        m_is, m_oos = walk_forward(r)
        wfr = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
        p = "✅" if wfr >= 0.70 else "❌"
        s = composite_score(m)
        print(f"{name:<18} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4} {s:>7.3f}")
    
    print("-" * 105)
    
    all_results = {}
    for name, r in sorted(strategies.items(), key=lambda x: composite_score(calc_metrics(x[1])), reverse=True):
        m = calc_metrics(r)
        m_is, m_oos = walk_forward(r)
        wfr = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
        p = "✅" if wfr >= 0.70 else "❌"
        s = composite_score(m)
        all_results[name] = {"m": m, "is": m_is, "oos": m_oos, "wfr": wfr, "score": s}
        print(f"{name:<18} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4} {s:>7.3f}")
    
    # WF multi-split test for top strategies
    print("\n=== Walk-Forward Multi-Split Test (top 3) ===")
    top3 = sorted(strategies.items(), key=lambda x: composite_score(calc_metrics(x[1])), reverse=True)[:3]
    
    splits = [0.5, 0.55, 0.6, 0.65]
    for name, r in top3:
        ratios = []
        for sp in splits:
            m_is, m_oos = walk_forward(r, sp)
            wfr = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
            ratios.append(wfr)
        avg_wf = np.mean(ratios)
        detail = " | ".join([f"{sp:.0%}:{r:.2f}" for sp, r in zip(splits, ratios)])
        print(f"  {name:<18}: {detail} | avg={avg_wf:.2f}")
    
    # Yearly returns for best
    best_name = max(all_results, key=lambda k: all_results[k]["score"])
    best_r = strategies[best_name]
    print(f"\n=== Yearly Returns: {best_name} vs Solo ===")
    
    years = sorted(set(d.year for d in dates))
    print(f"{'Year':<6} {'BTC_v7f':>9} {'Stock_v4d':>10} {'Portfolio':>10}")
    for yr in years:
        mask = np.array([d.year == yr for d in dates])
        if mask.sum() < 10:
            continue
        yr_btc = np.prod(1 + btc_r[mask]) - 1
        yr_stk = np.prod(1 + v4d_r[mask]) - 1
        yr_port = np.prod(1 + best_r[mask]) - 1
        print(f"{yr:<6} {yr_btc*100:>8.1f}% {yr_stk*100:>9.1f}% {yr_port*100:>9.1f}%")

if __name__ == "__main__":
    main()
