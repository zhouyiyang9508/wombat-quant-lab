"""
Portfolio Ultimate — ETH2 (Crypto DualMom) + Stock v3b/v4d
代码熊 🐻

ETH2 是新的加密货币之王（Composite 1.493）
Stock v3b/v4d 是美股之王（Composite 1.173/1.356）

相关性极低：
- ETH2 vs Stock: 0.022
- 组合潜力巨大！

测试方法：
1. Fixed weights (various combinations)
2. Risk Parity (inverse vol)
3. DD Responsive (回撤触发 GLD)
4. Momentum Tilt

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[2]

def load_returns():
    """Load daily returns for ETH2, Stock v3b, Stock v4d (if exists), GLD."""
    # ETH2
    eth2_ret = pd.read_csv(BASE / "crypto/codebear/beast_eth2_daily_returns.csv",
                           parse_dates=["Date"]).set_index("Date")
    
    # Stock v3b
    stock_v3b_ret = pd.read_csv(BASE / "stocks/codebear/v3b_daily_returns.csv",
                                parse_dates=["Date"]).set_index("Date")
    
    # Stock v4d (if exists)
    try:
        stock_v4d_ret = pd.read_csv(BASE / "stocks/codebear/v4d_daily_returns.csv",
                                   parse_dates=["Date"]).set_index("Date")
        has_v4d = True
    except:
        has_v4d = False
    
    # GLD
    gld_price = pd.read_csv(BASE / "data_cache/GLD.csv",
                           parse_dates=["Date"]).set_index("Date").sort_index()
    gld_ret = gld_price["Close"].pct_change()
    
    # Align
    common = eth2_ret.index.intersection(stock_v3b_ret.index).intersection(gld_ret.index)
    if has_v4d:
        common = common.intersection(stock_v4d_ret.index)
    common = common.sort_values()
    
    eth2_r = eth2_ret.loc[common, "Return"].values
    stock_v3b_r = stock_v3b_ret.loc[common, "Return"].values
    stock_v4d_r = stock_v4d_ret.loc[common, "Return"].values if has_v4d else None
    gld_r = gld_ret.loc[common].values
    
    return common, eth2_r, stock_v3b_r, stock_v4d_r, gld_r

def calc_metrics(returns, rf=0.04):
    """Calculate performance metrics."""
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
    
    composite = sharpe * 0.4 + calmar * 0.4 + min(cagr, 1.0) * 0.2
    
    return {
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe,
        "calmar": calmar, "composite": composite,
        "ann_vol": ann_vol, "years": years
    }

def composite_score(m):
    return m["sharpe"] * 0.4 + m["calmar"] * 0.4 + min(m["cagr"], 1.0) * 0.2

def walk_forward(returns, split=0.6):
    n = len(returns)
    s = int(n * split)
    m_is = calc_metrics(returns[:s])
    m_oos = calc_metrics(returns[s:])
    wfr = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
    return m_is, m_oos, wfr

def portfolio_fixed(crypto_r, stock_r, w_crypto):
    """Fixed weight allocation."""
    return w_crypto * crypto_r + (1 - w_crypto) * stock_r

def portfolio_risk_parity(crypto_r, stock_r, lookback=126):
    """Inverse volatility weighting."""
    n = len(crypto_r)
    port_r = np.zeros(n)
    
    for i in range(lookback, n):
        vol_c = np.std(crypto_r[i-lookback:i]) * np.sqrt(252)
        vol_s = np.std(stock_r[i-lookback:i]) * np.sqrt(252)
        
        inv_c = 1 / max(vol_c, 0.01)
        inv_s = 1 / max(vol_s, 0.01)
        
        w_c = inv_c / (inv_c + inv_s)
        port_r[i] = w_c * crypto_r[i] + (1 - w_c) * stock_r[i]
    
    # Fill initial period with 50/50
    port_r[:lookback] = 0.5 * crypto_r[:lookback] + 0.5 * stock_r[:lookback]
    return port_r

def portfolio_dd_responsive(crypto_r, stock_r, gld_r):
    """Portfolio-level DD protection with GLD."""
    n = len(crypto_r)
    port_r = np.zeros(n)
    equity = np.ones(n + 1)
    
    for i in range(n):
        peak = equity[:i+1].max()
        dd = (equity[i] - peak) / peak if peak > 0 else 0
        
        if dd < -0.15:
            # Heavy protection
            port_r[i] = 0.25 * crypto_r[i] + 0.30 * stock_r[i] + 0.40 * gld_r[i]
        elif dd < -0.10:
            port_r[i] = 0.30 * crypto_r[i] + 0.40 * stock_r[i] + 0.25 * gld_r[i]
        elif dd < -0.05:
            port_r[i] = 0.35 * crypto_r[i] + 0.50 * stock_r[i] + 0.10 * gld_r[i]
        else:
            # Normal allocation
            port_r[i] = 0.45 * crypto_r[i] + 0.55 * stock_r[i]
        
        equity[i+1] = equity[i] * (1 + port_r[i])
    
    return port_r

def portfolio_mom_tilt(crypto_r, stock_r, lookback=126):
    """Tilt toward higher momentum asset."""
    n = len(crypto_r)
    port_r = np.zeros(n)
    
    for i in range(lookback, n):
        # Cumulative returns
        cum_c = np.prod(1 + crypto_r[i-lookback:i]) - 1
        cum_s = np.prod(1 + stock_r[i-lookback:i]) - 1
        
        # Base RP
        vol_c = np.std(crypto_r[i-lookback:i]) * np.sqrt(252)
        vol_s = np.std(stock_r[i-lookback:i]) * np.sqrt(252)
        
        inv_c = 1 / max(vol_c, 0.01)
        inv_s = 1 / max(vol_s, 0.01)
        
        base_c = inv_c / (inv_c + inv_s)
        
        # Momentum tilt
        mom_diff = cum_c - cum_s
        tilt = 1 / (1 + np.exp(-3 * mom_diff))  # Sigmoid
        
        # Blend: 50% base + 50% tilt
        w_c = 0.5 * base_c + 0.5 * tilt
        w_c = np.clip(w_c, 0.20, 0.70)
        
        port_r[i] = w_c * crypto_r[i] + (1 - w_c) * stock_r[i]
    
    port_r[:lookback] = 0.5 * crypto_r[:lookback] + 0.5 * stock_r[:lookback]
    return port_r

def main():
    dates, eth2_r, stock_v3b_r, stock_v4d_r, gld_r = load_returns()
    n = len(dates)
    
    print("=" * 95)
    print("ULTIMATE PORTFOLIO: ETH2 (Crypto DualMom) + Stock")
    print("代码熊 🐻 | 2026-02-20")
    print("=" * 95)
    print(f"\nData: {dates[0].date()} → {dates[-1].date()} ({n} days, {n/252:.1f} years)")
    
    # Correlations
    corr_eth2_v3b = np.corrcoef(eth2_r, stock_v3b_r)[0, 1]
    print(f"Correlation ETH2 vs Stock v3b: {corr_eth2_v3b:.3f}")
    
    if stock_v4d_r is not None:
        corr_eth2_v4d = np.corrcoef(eth2_r, stock_v4d_r)[0, 1]
        print(f"Correlation ETH2 vs Stock v4d: {corr_eth2_v4d:.3f}")
    print()
    
    # Individual baselines
    m_eth2 = calc_metrics(eth2_r)
    m_v3b = calc_metrics(stock_v3b_r)
    m_v4d = calc_metrics(stock_v4d_r) if stock_v4d_r is not None else None
    
    # Test with v3b
    print("=" * 95)
    print("ETH2 + Stock v3b Combinations")
    print("=" * 95)
    
    strategies_v3b = {}
    
    # Fixed weights
    for w in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        name = f"Fixed_{int(w*100)}/{int((1-w)*100)}"
        strategies_v3b[name] = portfolio_fixed(eth2_r, stock_v3b_r, w)
    
    # Risk parity
    for lb in [60, 126, 252]:
        name = f"RP_{lb}d"
        strategies_v3b[name] = portfolio_risk_parity(eth2_r, stock_v3b_r, lb)
    
    # DD Responsive
    strategies_v3b["DD_Resp"] = portfolio_dd_responsive(eth2_r, stock_v3b_r, gld_r)
    
    # Momentum tilt
    for lb in [63, 126]:
        name = f"MomTilt_{lb}d"
        strategies_v3b[name] = portfolio_mom_tilt(eth2_r, stock_v3b_r, lb)
    
    # Print results
    print(f"\n{'Strategy':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Comp':>7} {'IS_Sh':>7} {'OOS_Sh':>7} {'WF_r':>6} {'Pass':>4}")
    print("-" * 95)
    
    # Baselines
    _, _, wfr = walk_forward(eth2_r)
    m_is, m_oos, _ = walk_forward(eth2_r)
    p = "✅" if wfr >= 0.70 else "❌"
    print(f"{'ETH2 Solo':<18} {m_eth2['cagr']*100:>6.1f}% {m_eth2['maxdd']*100:>7.1f}% {m_eth2['sharpe']:>7.2f} {m_eth2['calmar']:>7.2f} {m_eth2['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    _, _, wfr = walk_forward(stock_v3b_r)
    m_is, m_oos, _ = walk_forward(stock_v3b_r)
    p = "✅" if wfr >= 0.70 else "❌"
    print(f"{'Stock v3b Solo':<18} {m_v3b['cagr']*100:>6.1f}% {m_v3b['maxdd']*100:>7.1f}% {m_v3b['sharpe']:>7.2f} {m_v3b['calmar']:>7.2f} {m_v3b['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    print("-" * 95)
    
    all_results = {}
    for name, r in sorted(strategies_v3b.items(), key=lambda x: composite_score(calc_metrics(x[1])), reverse=True):
        m = calc_metrics(r)
        m_is, m_oos, wfr = walk_forward(r)
        p = "✅" if wfr >= 0.70 else "❌"
        all_results[name] = {"m": m, "wfr": wfr}
        print(f"{name:<18} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    # Test with v4d if available
    if stock_v4d_r is not None:
        print("\n" + "=" * 95)
        print("ETH2 + Stock v4d Combinations")
        print("=" * 95)
        
        strategies_v4d = {}
        
        for w in [0.40, 0.45, 0.50, 0.55]:
            name = f"Fixed_{int(w*100)}/{int((1-w)*100)}"
            strategies_v4d[name] = portfolio_fixed(eth2_r, stock_v4d_r, w)
        
        for lb in [126, 252]:
            name = f"RP_{lb}d"
            strategies_v4d[name] = portfolio_risk_parity(eth2_r, stock_v4d_r, lb)
        
        strategies_v4d["DD_Resp"] = portfolio_dd_responsive(eth2_r, stock_v4d_r, gld_r)
        
        print(f"\n{'Strategy':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Comp':>7} {'WF_r':>6} {'Pass':>4}")
        print("-" * 80)
        
        _, _, wfr = walk_forward(stock_v4d_r)
        m_is, m_oos, _ = walk_forward(stock_v4d_r)
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"{'Stock v4d Solo':<18} {m_v4d['cagr']*100:>6.1f}% {m_v4d['maxdd']*100:>7.1f}% {m_v4d['sharpe']:>7.2f} {m_v4d['calmar']:>7.2f} {m_v4d['composite']:>7.3f} {wfr:>6.2f} {p:>4}")
        
        print("-" * 80)
        
        for name, r in sorted(strategies_v4d.items(), key=lambda x: composite_score(calc_metrics(x[1])), reverse=True):
            m = calc_metrics(r)
            _, _, wfr = walk_forward(r)
            p = "✅" if wfr >= 0.70 else "❌"
            print(f"{name:<18} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['composite']:>7.3f} {wfr:>6.2f} {p:>4}")
    
    # Top strategy summary
    print("\n" + "=" * 95)
    print("🏆 BEST PORTFOLIO COMBINATIONS")
    print("=" * 95)
    
    best = max(all_results.items(), key=lambda x: x[1]["m"]["composite"])
    best_name, best_data = best
    
    print(f"\nTop Strategy: {best_name}")
    print(f"  Composite Score: {best_data['m']['composite']:.3f}")
    print(f"  CAGR:            {best_data['m']['cagr']*100:.1f}%")
    print(f"  Max Drawdown:    {best_data['m']['maxdd']*100:.1f}%")
    print(f"  Sharpe Ratio:    {best_data['m']['sharpe']:.2f}")
    print(f"  Calmar Ratio:    {best_data['m']['calmar']:.2f}")
    print(f"  WF Ratio:        {best_data['wfr']:.2f}")
    
    # Save best returns
    best_r = strategies_v3b[best_name]
    output_path = BASE / "portfolio/codebear/portfolio_eth2_ultimate_returns.csv"
    returns_df = pd.DataFrame({'Date': dates, 'Return': best_r})
    returns_df.to_csv(output_path, index=False)
    print(f"\n✅ Best portfolio returns saved to: portfolio_eth2_ultimate_returns.csv")
    
    print("=" * 95)

if __name__ == "__main__":
    main()
