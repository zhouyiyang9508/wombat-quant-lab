"""
Momentum Rotation v1: Equal-weight Top5 + Absolute Momentum Filter
Author: 代码熊 🐻

NO FUTURE FUNCTION: signals use t-1 data, execute at t.
"""
import numpy as np
import pandas as pd
from momentum_utils import (download_data, monthly_returns, momentum_score,
                             backtest_metrics, spy_benchmark, balanced_60_40,
                             walk_forward_split, print_results, TRADE_COST, TICKERS)

TOP_N = 3  # Optimized: top 3 concentrates on strongest momentum

def run_strategy(prices, top_n=TOP_N):
    """v1: Equal-weight top N with absolute momentum filter."""
    tradeable = [t for t in TICKERS if t != 'SHY' and t in prices.columns]
    monthly_p = prices[tradeable + ['SHY']].resample('ME').last().dropna(how='all')
    
    # Momentum scores (lagged: computed at month-end, applied next month)
    scores = momentum_score(monthly_p[tradeable])
    
    ret = monthly_returns(prices[tradeable + ['SHY']].dropna(how='all'))
    
    strat_returns = []
    prev_holdings = set()
    
    for i in range(13, len(ret)):  # need 12 months lookback
        date = ret.index[i]
        # Signal from previous month (t-1)
        sig_date = ret.index[i-1]
        if sig_date not in scores.index:
            continue
        
        row = scores.loc[sig_date].dropna()
        if len(row) == 0:
            strat_returns.append((date, 0.0))
            continue
        
        # Rank and select top N
        ranked = row.sort_values(ascending=False)
        selected = []
        for ticker in ranked.index[:top_n]:
            if ranked[ticker] > 0:  # absolute momentum filter
                selected.append(ticker)
            else:
                selected.append('SHY')  # replace with cash
        
        # Remove duplicates (multiple SHY)
        n_shy = selected.count('SHY')
        selected_unique = [t for t in selected if t != 'SHY']
        if n_shy > 0:
            selected_unique.append('SHY')
        
        # Equal weight
        weights = {}
        for t in selected:
            weights[t] = weights.get(t, 0) + 1.0 / top_n
        
        # Compute return
        month_ret = 0
        for t, w in weights.items():
            if t in ret.columns and not pd.isna(ret.loc[date, t]):
                month_ret += w * ret.loc[date, t]
        
        # Transaction costs
        current_holdings = set(weights.keys())
        turnover = len(current_holdings.symmetric_difference(prev_holdings)) / (2 * top_n)
        month_ret -= turnover * TRADE_COST
        prev_holdings = current_holdings
        
        strat_returns.append((date, month_ret))
    
    sr = pd.Series([r[1] for r in strat_returns], 
                   index=[r[0] for r in strat_returns])
    return sr

def main():
    print("Downloading data...")
    prices = download_data()
    print(f"Got {len(prices.columns)} assets, {len(prices)} days")
    
    # Run strategy
    strat_ret = run_strategy(prices)
    
    # Walk-forward
    is_ret, oos_ret = walk_forward_split(strat_ret)
    
    is_metrics = backtest_metrics(is_ret)
    oos_metrics = backtest_metrics(oos_ret)
    full_metrics = backtest_metrics(strat_ret)
    
    # Benchmarks
    spy_ret = spy_benchmark(prices)
    bal_ret = balanced_60_40(prices)
    common_idx = strat_ret.index.intersection(spy_ret.index)
    spy_metrics = backtest_metrics(spy_ret.loc[common_idx])
    bal_metrics = backtest_metrics(bal_ret.loc[bal_ret.index.intersection(common_idx)])
    
    print_results("Momentum v1 (Full)", full_metrics)
    print_results("Momentum v1 (IS)", is_metrics)
    print_results("Momentum v1 (OOS)", oos_metrics)
    print_results("SPY Buy&Hold", spy_metrics)
    print_results("60/40", bal_metrics)
    
    # Check OOS degradation
    if is_metrics.get('Sharpe', 0) > 0:
        degradation = 1 - oos_metrics.get('Sharpe', 0) / is_metrics['Sharpe']
        print(f"\nOOS Sharpe degradation: {degradation*100:.1f}%")
        print(f"Pass WF test: {'YES' if degradation < 0.3 else 'NO'}")
    
    return full_metrics, is_metrics, oos_metrics, spy_metrics, bal_metrics

if __name__ == '__main__':
    main()
