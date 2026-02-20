"""
Portfolio ETH3 — Balanced Three-Asset (30% BTC + 30% ETH + 40% Stock)
代码熊 🐻

核心思路：固定权重的三资产组合，用 DD Responsive 动态调整
- 基础配置：30% BTC, 30% ETH, 40% Stock v3b
- 回撤响应：当组合回撤 >-X% 时，增加 GLD 对冲
- 目标：通过分散化提升 Sharpe，同时保持可控回撤

测试方法：
1. Fixed 30/30/40
2. Risk Parity (inverse vol)
3. DD Responsive (回撤触发 GLD)
4. Momentum Tilt (倾向高动量资产)

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

def load_returns():
    """Load daily returns for BTC, ETH, Stock v3b, GLD."""
    # BTC returns (from v7f)
    btc_ret = pd.read_csv(BASE / "btc/codebear/v7f_daily_returns_2015_2025.csv",
                          parse_dates=["Date"]).set_index("Date")
    
    # Stock returns (v3b)
    stock_ret = pd.read_csv(BASE / "stocks/codebear/v3b_daily_returns.csv",
                            parse_dates=["Date"]).set_index("Date")
    
    # Load price data for ETH and GLD
    eth_price = pd.read_csv(BASE / "data_cache/ETH_USD.csv",
                           parse_dates=["Date"]).set_index("Date").sort_index()
    eth_ret = eth_price["Close"].pct_change()
    
    gld_price = pd.read_csv(BASE / "data_cache/GLD.csv",
                           parse_dates=["Date"]).set_index("Date").sort_index()
    gld_ret = gld_price["Close"].pct_change()
    
    # Align all
    common = (btc_ret.index
              .intersection(stock_ret.index)
              .intersection(eth_ret.index)
              .intersection(gld_ret.index))
    common = common.sort_values()
    
    btc_r = btc_ret.loc[common, "Return"].values
    stock_r = stock_ret.loc[common, "Return"].values
    eth_r = eth_ret.loc[common].values
    gld_r = gld_ret.loc[common].values
    
    return common, btc_r, eth_r, stock_r, gld_r

def calc_metrics(returns, rf=0.04):
    """Calculate performance metrics from daily returns."""
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
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe,
        "calmar": calmar, "final_value": equity[-1],
        "ann_vol": ann_vol, "years": years
    }

def composite_score(m):
    return m["sharpe"] * 0.4 + m["calmar"] * 0.4 + min(m["cagr"], 1.0) * 0.2

def walk_forward(returns, split=0.6):
    n = len(returns)
    s = int(n * split)
    m_is = calc_metrics(returns[:s])
    m_oos = calc_metrics(returns[s:])
    return m_is, m_oos

def portfolio_fixed(btc_r, eth_r, stock_r, w_btc=0.30, w_eth=0.30, w_stock=0.40):
    """Fixed weights."""
    return w_btc * btc_r + w_eth * eth_r + w_stock * stock_r

def portfolio_risk_parity(btc_r, eth_r, stock_r, lookback=126):
    """Inverse volatility weighting."""
    n = len(btc_r)
    port_r = np.zeros(n)
    
    for i in range(lookback, n):
        vol_b = np.std(btc_r[i-lookback:i]) * np.sqrt(252)
        vol_e = np.std(eth_r[i-lookback:i]) * np.sqrt(252)
        vol_s = np.std(stock_r[i-lookback:i]) * np.sqrt(252)
        
        inv_b = 1 / max(vol_b, 0.01)
        inv_e = 1 / max(vol_e, 0.01)
        inv_s = 1 / max(vol_s, 0.01)
        
        total_inv = inv_b + inv_e + inv_s
        w_b = inv_b / total_inv
        w_e = inv_e / total_inv
        w_s = inv_s / total_inv
        
        port_r[i] = w_b * btc_r[i] + w_e * eth_r[i] + w_s * stock_r[i]
    
    # Fill initial period
    port_r[:lookback] = portfolio_fixed(btc_r[:lookback], eth_r[:lookback],
                                       stock_r[:lookback])
    return port_r

def portfolio_dd_responsive(btc_r, eth_r, stock_r, gld_r):
    """Portfolio-level DD protection with GLD."""
    n = len(btc_r)
    port_r = np.zeros(n)
    equity = np.ones(n + 1)
    
    for i in range(n):
        peak = equity[:i+1].max()
        dd = (equity[i] - peak) / peak if peak > 0 else 0
        
        if dd < -0.15:
            # Heavy protection
            port_r[i] = (0.15 * btc_r[i] + 0.15 * eth_r[i] +
                        0.30 * stock_r[i] + 0.35 * gld_r[i])
        elif dd < -0.10:
            port_r[i] = (0.20 * btc_r[i] + 0.20 * eth_r[i] +
                        0.35 * stock_r[i] + 0.20 * gld_r[i])
        elif dd < -0.05:
            port_r[i] = (0.25 * btc_r[i] + 0.25 * eth_r[i] +
                        0.38 * stock_r[i] + 0.10 * gld_r[i])
        else:
            # Normal allocation
            port_r[i] = (0.30 * btc_r[i] + 0.30 * eth_r[i] +
                        0.40 * stock_r[i])
        
        equity[i+1] = equity[i] * (1 + port_r[i])
    
    return port_r

def portfolio_mom_tilt(btc_r, eth_r, stock_r, lookback=126):
    """Tilt toward higher momentum assets."""
    n = len(btc_r)
    port_r = np.zeros(n)
    
    for i in range(lookback, n):
        # Cumulative returns
        cum_b = np.prod(1 + btc_r[i-lookback:i]) - 1
        cum_e = np.prod(1 + eth_r[i-lookback:i]) - 1
        cum_s = np.prod(1 + stock_r[i-lookback:i]) - 1
        
        # Inverse vol for base
        vol_b = np.std(btc_r[i-lookback:i]) * np.sqrt(252)
        vol_e = np.std(eth_r[i-lookback:i]) * np.sqrt(252)
        vol_s = np.std(stock_r[i-lookback:i]) * np.sqrt(252)
        
        inv_b = 1 / max(vol_b, 0.01)
        inv_e = 1 / max(vol_e, 0.01)
        inv_s = 1 / max(vol_s, 0.01)
        
        total_inv = inv_b + inv_e + inv_s
        base_b = inv_b / total_inv
        base_e = inv_e / total_inv
        base_s = inv_s / total_inv
        
        # Momentum scores (normalized)
        total_mom = abs(cum_b) + abs(cum_e) + abs(cum_s) + 1e-6
        mom_b = max(0, cum_b) / total_mom
        mom_e = max(0, cum_e) / total_mom
        mom_s = max(0, cum_s) / total_mom
        
        # Blend: 50% base RP + 50% momentum
        w_b = 0.5 * base_b + 0.5 * mom_b
        w_e = 0.5 * base_e + 0.5 * mom_e
        w_s = 0.5 * base_s + 0.5 * mom_s
        
        # Normalize
        total = w_b + w_e + w_s
        w_b /= total
        w_e /= total
        w_s /= total
        
        port_r[i] = w_b * btc_r[i] + w_e * eth_r[i] + w_s * stock_r[i]
    
    port_r[:lookback] = portfolio_fixed(btc_r[:lookback], eth_r[:lookback],
                                       stock_r[:lookback])
    return port_r

def main():
    dates, btc_r, eth_r, stock_r, gld_r = load_returns()
    n = len(dates)
    print(f"Data: {dates[0].date()} → {dates[-1].date()} ({n} days, {n/252:.1f} years)")
    
    # Correlations
    corr_be = np.corrcoef(btc_r, eth_r)[0, 1]
    corr_bs = np.corrcoef(btc_r, stock_r)[0, 1]
    corr_es = np.corrcoef(eth_r, stock_r)[0, 1]
    print(f"Correlation BTC-ETH:   {corr_be:.3f}")
    print(f"Correlation BTC-Stock: {corr_bs:.3f}")
    print(f"Correlation ETH-Stock: {corr_es:.3f}")
    print()
    
    # Individual strategies
    m_btc = calc_metrics(btc_r)
    m_eth = calc_metrics(eth_r)
    m_stock = calc_metrics(stock_r)
    
    strategies = {}
    
    # Fixed weights (various combinations)
    for w_b in [0.25, 0.30, 0.35]:
        for w_e in [0.25, 0.30, 0.35]:
            w_s = 1.0 - w_b - w_e
            if w_s >= 0.30 and w_s <= 0.45:
                name = f"Fixed_{int(w_b*100)}/{int(w_e*100)}/{int(w_s*100)}"
                strategies[name] = portfolio_fixed(btc_r, eth_r, stock_r, w_b, w_e, w_s)
    
    # Risk Parity
    for lb in [60, 126, 252]:
        name = f"RP_{lb}d"
        strategies[name] = portfolio_risk_parity(btc_r, eth_r, stock_r, lb)
    
    # DD Responsive
    strategies["DD_Resp"] = portfolio_dd_responsive(btc_r, eth_r, stock_r, gld_r)
    
    # Momentum Tilt
    for lb in [63, 126]:
        name = f"MomTilt_{lb}d"
        strategies[name] = portfolio_mom_tilt(btc_r, eth_r, stock_r, lb)
    
    # Print results
    print(f"{'Strategy':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'IS_Sh':>7} {'OOS_Sh':>7} {'WF_r':>6} {'Pass':>4} {'Score':>7}")
    print("-" * 105)
    
    # Baselines
    for name, r in [("BTC Solo", btc_r), ("ETH Solo", eth_r), ("Stock v3b Solo", stock_r)]:
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
    
    # Multi-split WF test for top 3
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
    
    # Save best portfolio returns
    best_name = max(all_results, key=lambda k: all_results[k]["score"])
    best_r = strategies[best_name]
    output_path = BASE / "portfolio/codebear/portfolio_eth3_best_returns.csv"
    returns_df = pd.DataFrame({'Date': dates, 'Return': best_r})
    returns_df.to_csv(output_path, index=False)
    print(f"\nBest portfolio ({best_name}) returns saved to: {output_path}")

if __name__ == "__main__":
    main()
