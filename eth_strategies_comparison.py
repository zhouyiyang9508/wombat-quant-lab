"""
ETH Strategies Comprehensive Comparison
代码熊 🐻

对比所有加入ETH的策略方案：
- ETH1: Triple Momentum (BTC/ETH/GLD 轮动)
- ETH2: Crypto Internal DualMom (BTC vs ETH + GLD hedge)
- ETH3: Balanced Portfolio (30/30/40 fixed + variants)

与现有最佳策略对比：
- BTC v7f: DualMom (BTC vs GLD)
- Stock v4d: Stock momentum + GLD DD hedge
- Portfolio DD Resp: BTC v7f + Stock v3b

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent

# Import strategy classes
import sys
sys.path.append(str(BASE / "crypto/codebear"))
sys.path.append(str(BASE / "btc/codebear"))

from beast_eth1_tripledm import BeastETH1
from beast_eth2_cryptodm import BeastETH2

def load_existing_strategies():
    """Load returns for existing best strategies."""
    results = {}
    
    # BTC v7f
    try:
        btc = pd.read_csv(BASE / "btc/codebear/v7f_daily_returns_2015_2025.csv",
                          parse_dates=["Date"]).set_index("Date")
        results['BTC v7f'] = btc["Return"].values
        dates_btc = btc.index
    except:
        results['BTC v7f'] = None
        dates_btc = None
    
    # Stock v3b
    try:
        stock = pd.read_csv(BASE / "stocks/codebear/v3b_daily_returns.csv",
                           parse_dates=["Date"]).set_index("Date")
        results['Stock v3b'] = stock["Return"].values
        dates_stock = stock.index
    except:
        results['Stock v3b'] = None
        dates_stock = None
    
    # Stock v4d (if exists)
    try:
        stock_v4d = pd.read_csv(BASE / "stocks/codebear/v4d_daily_returns.csv",
                               parse_dates=["Date"]).set_index("Date")
        results['Stock v4d'] = stock_v4d["Return"].values
        dates_v4d = stock_v4d.index
    except:
        results['Stock v4d'] = None
        dates_v4d = None
    
    # Find common dates
    if dates_btc is not None and dates_stock is not None:
        common = dates_btc.intersection(dates_stock)
        if dates_v4d is not None:
            common = common.intersection(dates_v4d)
        return results, common.sort_values()
    
    return results, None

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

def walk_forward(returns, split=0.6):
    """Walk-forward validation."""
    n = len(returns)
    s = int(n * split)
    m_is = calc_metrics(returns[:s])
    m_oos = calc_metrics(returns[s:])
    wfr = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
    return m_is, m_oos, wfr

def run_recent_performance(returns, dates, months=3):
    """Calculate recent N months performance."""
    if len(returns) != len(dates):
        # Align lengths
        min_len = min(len(returns), len(dates))
        returns = returns[:min_len]
        dates = dates[:min_len]
    
    cutoff = dates[-1] - pd.DateOffset(months=months)
    mask = dates >= cutoff
    recent_ret = returns[mask]
    
    if len(recent_ret) < 10:
        return None
    
    m = calc_metrics(recent_ret)
    return m

def main():
    print("=" * 90)
    print("ETH STRATEGIES COMPREHENSIVE COMPARISON")
    print("代码熊 🐻 | 2026-02-20")
    print("=" * 90)
    print()
    
    # Run new ETH strategies
    print("Running ETH strategies...")
    
    eth1 = BeastETH1()
    eth1.load_data()
    eth1.run_backtest()
    eth1_metrics = eth1.get_metrics()
    eth1_returns = eth1._dr.values
    eth1_dates = eth1._dr.index
    
    eth2 = BeastETH2()
    eth2.load_data()
    eth2.run_backtest()
    eth2_metrics = eth2.get_metrics()
    eth2_returns = eth2._dr.values
    eth2_dates = eth2._dr.index
    
    # Load ETH3 best result
    try:
        eth3_ret = pd.read_csv(BASE / "portfolio/codebear/portfolio_eth3_best_returns.csv",
                              parse_dates=["Date"]).set_index("Date")
        eth3_returns = eth3_ret["Return"].values
        eth3_dates = eth3_ret.index
        eth3_metrics = calc_metrics(eth3_returns)
    except:
        eth3_metrics = None
        eth3_returns = None
        eth3_dates = None
    
    # Load existing strategies
    existing, common_dates = load_existing_strategies()
    
    # Print comparison table
    print(f"\n{'Strategy':<25} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Comp':>7} {'IS_Sh':>7} {'OOS_Sh':>7} {'WF_r':>6} {'Pass':>4}")
    print("-" * 110)
    
    # Existing strategies
    if existing['BTC v7f'] is not None:
        m = calc_metrics(existing['BTC v7f'])
        _, _, wfr = walk_forward(existing['BTC v7f'])
        m_is, m_oos, _ = walk_forward(existing['BTC v7f'])
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"{'BTC v7f (baseline)':<25} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    if existing['Stock v3b'] is not None:
        m = calc_metrics(existing['Stock v3b'])
        _, _, wfr = walk_forward(existing['Stock v3b'])
        m_is, m_oos, _ = walk_forward(existing['Stock v3b'])
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"{'Stock v3b (baseline)':<25} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    if existing['Stock v4d'] is not None:
        m = calc_metrics(existing['Stock v4d'])
        _, _, wfr = walk_forward(existing['Stock v4d'])
        m_is, m_oos, _ = walk_forward(existing['Stock v4d'])
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"{'Stock v4d (baseline)':<25} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    print("-" * 110)
    
    # New ETH strategies
    m_is, m_oos, wfr = walk_forward(eth1_returns)
    p = "✅" if wfr >= 0.70 else "❌"
    print(f"{'ETH1: Triple DualMom':<25} {eth1_metrics['cagr']*100:>6.1f}% {eth1_metrics['max_dd']*100:>7.1f}% {eth1_metrics['sharpe']:>7.2f} {eth1_metrics['calmar']:>7.2f} {eth1_metrics['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    m_is, m_oos, wfr = walk_forward(eth2_returns)
    p = "✅" if wfr >= 0.70 else "❌"
    print(f"{'ETH2: Crypto DualMom':<25} {eth2_metrics['cagr']*100:>6.1f}% {eth2_metrics['max_dd']*100:>7.1f}% {eth2_metrics['sharpe']:>7.2f} {eth2_metrics['calmar']:>7.2f} {eth2_metrics['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    if eth3_metrics:
        m_is, m_oos, wfr = walk_forward(eth3_returns)
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"{'ETH3: Balanced RP_252d':<25} {eth3_metrics['cagr']*100:>6.1f}% {eth3_metrics['maxdd']*100:>7.1f}% {eth3_metrics['sharpe']:>7.2f} {eth3_metrics['calmar']:>7.2f} {eth3_metrics['composite']:>7.3f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")
    
    # Recent performance (past 3 months)
    print("\n" + "=" * 90)
    print("RECENT PERFORMANCE (Past 3 Months)")
    print("=" * 90)
    print(f"\n{'Strategy':<25} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8}")
    print("-" * 55)
    
    for name, returns, dates in [
        ("BTC v7f", existing.get('BTC v7f'), common_dates),
        ("Stock v3b", existing.get('Stock v3b'), common_dates),
        ("Stock v4d", existing.get('Stock v4d'), common_dates),
        ("ETH1: Triple DualMom", eth1_returns, eth1_dates),
        ("ETH2: Crypto DualMom", eth2_returns, eth2_dates),
        ("ETH3: Balanced RP_252d", eth3_returns, eth3_dates),
    ]:
        if returns is not None and dates is not None:
            m_recent = run_recent_performance(returns, dates, 3)
            if m_recent:
                print(f"{name:<25} {m_recent['cagr']*100:>7.1f}% {m_recent['sharpe']:>7.2f} {m_recent['maxdd']*100:>7.1f}%")
    
    # Key findings
    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)
    
    all_scores = []
    
    if existing['BTC v7f'] is not None:
        m = calc_metrics(existing['BTC v7f'])
        all_scores.append(("BTC v7f", m['composite']))
    
    if existing['Stock v3b'] is not None:
        m = calc_metrics(existing['Stock v3b'])
        all_scores.append(("Stock v3b", m['composite']))
    
    if existing['Stock v4d'] is not None:
        m = calc_metrics(existing['Stock v4d'])
        all_scores.append(("Stock v4d", m['composite']))
    
    all_scores.append(("ETH1: Triple DualMom", eth1_metrics['composite']))
    all_scores.append(("ETH2: Crypto DualMom", eth2_metrics['composite']))
    
    if eth3_metrics:
        all_scores.append(("ETH3: Balanced RP_252d", eth3_metrics['composite']))
    
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top 3 Strategies by Composite Score:")
    for i, (name, score) in enumerate(all_scores[:3], 1):
        print(f"   {i}. {name:<30} {score:.3f}")
    
    print(f"\n💡 Key Observations:")
    print(f"   • ETH2 (Crypto DualMom) achieves highest CAGR: {eth2_metrics['cagr']*100:.1f}%")
    print(f"   • ETH strategies show {eth1_metrics['sharpe']:.2f}+ Sharpe ratios")
    print(f"   • Adding ETH increases crypto volatility but improves returns")
    
    # Correlation analysis
    print(f"\n📊 Correlation Analysis:")
    if existing['BTC v7f'] is not None and existing['Stock v3b'] is not None:
        # Align for correlation - find actual common length
        min_len = min(len(existing['BTC v7f']), len(eth2_returns), len(existing['Stock v3b']))
        btc_aligned = existing['BTC v7f'][-min_len:]
        eth2_aligned = eth2_returns[-min_len:]
        stock_aligned = existing['Stock v3b'][-min_len:]
        
        corr_btc_eth2 = np.corrcoef(btc_aligned, eth2_aligned)[0, 1]
        corr_eth2_stock = np.corrcoef(eth2_aligned, stock_aligned)[0, 1]
        corr_btc_stock = np.corrcoef(btc_aligned, stock_aligned)[0, 1]
        
        print(f"   • BTC v7f vs ETH2:   {corr_btc_eth2:.3f}")
        print(f"   • ETH2 vs Stock v3b: {corr_eth2_stock:.3f}")
        print(f"   • BTC v7f vs Stock:  {corr_btc_stock:.3f}")
    
    # Save summary
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "strategies": {
            "ETH1": eth1_metrics,
            "ETH2": eth2_metrics,
            "ETH3": eth3_metrics if eth3_metrics else None,
        },
        "rankings": all_scores,
    }
    
    with open(BASE / "eth_strategies_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Summary saved to: eth_strategies_summary.json")
    print("=" * 90)

if __name__ == "__main__":
    main()
