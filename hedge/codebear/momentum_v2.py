"""
Momentum Rotation v2: Risk-Parity Top5 + Absolute Momentum Filter
Author: 代码熊 🐻
"""
import numpy as np
import pandas as pd
from momentum_utils import (download_data, monthly_returns, momentum_score,
                             backtest_metrics, spy_benchmark, balanced_60_40,
                             walk_forward_split, print_results, TRADE_COST, TICKERS)

TOP_N = 3  # Optimized: top 3 with mid-term momentum emphasis

def run_strategy(prices, top_n=TOP_N):
    """v2: Inverse-vol weighted top N with absolute momentum filter."""
    tradeable = [t for t in TICKERS if t != 'SHY' and t in prices.columns]
    all_tickers = tradeable + ['SHY']
    monthly_p = prices[all_tickers].resample('ME').last().dropna(how='all')
    
    scores = momentum_score(monthly_p[tradeable], weights=[0.2, 0.4, 0.3, 0.1])
    ret = monthly_returns(prices[all_tickers].dropna(how='all'))
    
    # Rolling 6-month vol (annualized) for risk parity
    rolling_vol = ret[all_tickers].rolling(6).std() * np.sqrt(12)
    
    strat_returns = []
    prev_holdings = {}
    
    for i in range(13, len(ret)):
        date = ret.index[i]
        sig_date = ret.index[i-1]
        if sig_date not in scores.index:
            continue
        
        row = scores.loc[sig_date].dropna()
        if len(row) == 0:
            strat_returns.append((date, 0.0))
            continue
        
        ranked = row.sort_values(ascending=False)
        selected = []
        for ticker in ranked.index[:top_n]:
            if ranked[ticker] > 0:
                selected.append(ticker)
            else:
                selected.append('SHY')
        
        # Inverse vol weights
        weights = {}
        for t in selected:
            vol = rolling_vol.loc[sig_date, t] if sig_date in rolling_vol.index and t in rolling_vol.columns else 0.15
            if pd.isna(vol) or vol < 0.01:
                vol = 0.15
            inv_vol = 1.0 / vol
            weights[t] = weights.get(t, 0) + inv_vol
        
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {t: w/total_w for t, w in weights.items()}
        
        # Compute return
        month_ret = 0
        for t, w in weights.items():
            if t in ret.columns and not pd.isna(ret.loc[date, t]):
                month_ret += w * ret.loc[date, t]
        
        # Transaction costs
        turnover = sum(abs(weights.get(t, 0) - prev_holdings.get(t, 0)) for t in set(list(weights.keys()) + list(prev_holdings.keys()))) / 2
        month_ret -= turnover * TRADE_COST
        prev_holdings = weights.copy()
        
        strat_returns.append((date, month_ret))
    
    return pd.Series([r[1] for r in strat_returns], index=[r[0] for r in strat_returns])

def main():
    print("Downloading data...")
    prices = download_data()
    print(f"Got {len(prices.columns)} assets, {len(prices)} days")
    
    strat_ret = run_strategy(prices)
    is_ret, oos_ret = walk_forward_split(strat_ret)
    
    is_metrics = backtest_metrics(is_ret)
    oos_metrics = backtest_metrics(oos_ret)
    full_metrics = backtest_metrics(strat_ret)
    
    spy_ret = spy_benchmark(prices)
    bal_ret = balanced_60_40(prices)
    common_idx = strat_ret.index.intersection(spy_ret.index)
    spy_metrics = backtest_metrics(spy_ret.loc[common_idx])
    bal_metrics = backtest_metrics(bal_ret.loc[bal_ret.index.intersection(common_idx)])
    
    print_results("Momentum v2 (Full)", full_metrics)
    print_results("Momentum v2 (IS)", is_metrics)
    print_results("Momentum v2 (OOS)", oos_metrics)
    print_results("SPY Buy&Hold", spy_metrics)
    print_results("60/40", bal_metrics)
    
    if is_metrics.get('Sharpe', 0) > 0:
        degradation = 1 - oos_metrics.get('Sharpe', 0) / is_metrics['Sharpe']
        print(f"\nOOS Sharpe degradation: {degradation*100:.1f}%")
        print(f"Pass WF test: {'YES' if degradation < 0.3 else 'NO'}")
    
    return full_metrics, is_metrics, oos_metrics, spy_metrics, bal_metrics

if __name__ == '__main__':
    main()
