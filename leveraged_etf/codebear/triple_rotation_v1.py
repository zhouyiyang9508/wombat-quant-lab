"""
3x ETF 动量轮动策略 v1
在 TQQQ / SOXL / UPRO 之间根据相对强度+趋势过滤动态切换
防御时切入现金(SHY)或降低仓位

代码熊 🐻 2026-02-19
"""

import pandas as pd
import numpy as np
import os

def load(ticker, data_dir):
    path = os.path.join(data_dir, f'{ticker}.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    return df['Close'].dropna()

def compute_momentum(prices, lookbacks=[21, 63, 126]):
    """Blended momentum: 1M/3M/6M with weights 0.25/0.40/0.35"""
    weights = [0.25, 0.40, 0.35]
    mom = pd.Series(0.0, index=prices.index)
    for lb, w in zip(lookbacks, weights):
        m = prices.pct_change(lb)
        mom += w * m
    return mom

def run_rotation(data, params, cost_per_side=0.0005, slippage=0.001):
    """
    data: dict of ticker -> close prices (aligned DatetimeIndex)
    params: strategy parameters
    """
    tickers = list(data.keys())
    # Align all data
    df = pd.DataFrame(data)
    df = df.dropna()
    
    n = len(df)
    sma_period = params.get('sma_period', 200)
    rebalance_freq = params.get('rebalance_freq', 5)  # weekly
    top_k = params.get('top_k', 1)  # how many to hold
    defense_asset = params.get('defense_asset', None)  # if None, go cash
    
    # Precompute
    sma200 = df.rolling(sma_period).mean()
    momentum = pd.DataFrame({t: compute_momentum(df[t]) for t in tickers})
    vol20 = df.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Track
    holdings = {t: 0.0 for t in tickers}
    cash_weight = 1.0
    portfolio_ret = np.zeros(n)
    daily_ret = df.pct_change()
    
    positions_history = []
    
    for i in range(sma_period, n):
        # Daily return from yesterday's positions
        if i > sma_period:
            day_ret = 0.0
            for t in tickers:
                day_ret += holdings[t] * daily_ret[t].iloc[i]
            portfolio_ret[i] = day_ret
        
        # Rebalance check
        if (i - sma_period) % rebalance_freq != 0:
            continue
        
        # Which tickers are in uptrend (above SMA)?
        in_trend = {}
        for t in tickers:
            if not np.isnan(sma200[t].iloc[i]) and df[t].iloc[i] > sma200[t].iloc[i] * params.get('trend_threshold', 1.0):
                mom_val = momentum[t].iloc[i]
                if not np.isnan(mom_val) and mom_val > 0:  # absolute momentum filter
                    in_trend[t] = mom_val
        
        # Select top_k by momentum
        new_holdings = {t: 0.0 for t in tickers}
        
        if len(in_trend) > 0:
            sorted_tickers = sorted(in_trend.keys(), key=lambda t: in_trend[t], reverse=True)
            selected = sorted_tickers[:top_k]
            
            if top_k == 1:
                new_holdings[selected[0]] = 1.0
            else:
                # Inverse vol weighting among selected
                vols = {t: vol20[t].iloc[i] for t in selected if not np.isnan(vol20[t].iloc[i])}
                if vols:
                    inv_vol = {t: 1/v for t, v in vols.items() if v > 0}
                    total = sum(inv_vol.values())
                    for t in inv_vol:
                        new_holdings[t] = inv_vol[t] / total
                else:
                    for t in selected:
                        new_holdings[t] = 1.0 / len(selected)
        # else: all cash (new_holdings all 0)
        
        # Compute trading cost
        total_turnover = sum(abs(new_holdings[t] - holdings[t]) for t in tickers)
        trade_cost = total_turnover * (cost_per_side + slippage / 2)
        portfolio_ret[i] -= trade_cost
        
        holdings = new_holdings
        positions_history.append((df.index[i], dict(holdings)))
    
    return portfolio_ret

def calc_metrics(returns, rf_annual=0.04):
    r = returns[np.abs(returns) > 1e-15]
    if len(r) < 100:
        return {}
    cum = np.cumprod(1 + returns)
    n = len(returns)
    years = n / 252
    cagr = cum[-1] ** (1/years) - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = dd.min()
    daily_rf = (1 + rf_annual) ** (1/252) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * np.mean(excess) / np.std(excess) if np.std(excess) > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    return {'CAGR': cagr, 'MaxDD': maxdd, 'Sharpe': sharpe, 'Calmar': calmar, 'Final': cum[-1]}

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    
    # Load all 3x ETFs
    tickers_3x = ['TQQQ', 'SOXL', 'UPRO']
    data = {}
    for t in tickers_3x:
        data[t] = load(t, data_dir)
        print(f"{t}: {len(data[t])} rows")
    
    # Align
    df_all = pd.DataFrame(data).dropna()
    print(f"\nAligned: {len(df_all)} rows, {df_all.index[0].date()} to {df_all.index[-1].date()}")
    data_aligned = {t: df_all[t] for t in tickers_3x}
    
    # Walk-forward split
    n = len(df_all)
    split = int(n * 0.6)
    split_date = df_all.index[split]
    print(f"IS: {df_all.index[0].date()} to {df_all.index[split-1].date()}")
    print(f"OOS: {split_date.date()} to {df_all.index[-1].date()}")
    
    param_sets = [
        {'name': 'Top1_weekly', 'top_k': 1, 'rebalance_freq': 5, 'sma_period': 200, 'trend_threshold': 1.0},
        {'name': 'Top1_biweekly', 'top_k': 1, 'rebalance_freq': 10, 'sma_period': 200, 'trend_threshold': 1.0},
        {'name': 'Top1_monthly', 'top_k': 1, 'rebalance_freq': 21, 'sma_period': 200, 'trend_threshold': 1.0},
        {'name': 'Top2_RP_weekly', 'top_k': 2, 'rebalance_freq': 5, 'sma_period': 200, 'trend_threshold': 1.0},
        {'name': 'Top1_w_thr105', 'top_k': 1, 'rebalance_freq': 5, 'sma_period': 200, 'trend_threshold': 1.05},
        {'name': 'Top1_sma150', 'top_k': 1, 'rebalance_freq': 5, 'sma_period': 150, 'trend_threshold': 1.0},
    ]
    
    print(f"\n{'Name':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF?':>5}")
    print("-"*85)
    
    best_score = -1
    best_name = ""
    
    for p in param_sets:
        # Full
        ret_full = run_rotation(data_aligned, p)
        m_full = calc_metrics(ret_full[200:])
        
        # IS
        data_is = {t: df_all[t].iloc[:split] for t in tickers_3x}
        ret_is = run_rotation(data_is, p)
        m_is = calc_metrics(ret_is[200:])
        
        # OOS
        # Need enough history for SMA, so start from split-200
        start_oos = max(0, split - p['sma_period'])
        data_oos = {t: df_all[t].iloc[start_oos:] for t in tickers_3x}
        ret_oos = run_rotation(data_oos, p)
        # Only count from split point
        oos_offset = split - start_oos
        m_oos = calc_metrics(ret_oos[oos_offset:])
        
        if not m_full:
            continue
        
        wf = '✅' if m_oos.get('Sharpe',0) >= m_is.get('Sharpe',0) * 0.7 else '❌'
        print(f"{p['name']:<22} {m_full['CAGR']:>6.1%} {m_full['MaxDD']:>7.1%} {m_full['Sharpe']:>7.2f} {m_full['Calmar']:>7.2f} | {m_is.get('Sharpe',0):>6.2f} {m_oos.get('Sharpe',0):>7.2f} {wf:>5}")
        
        score = m_full['Sharpe']*0.4 + m_full['Calmar']*0.4 + m_full['CAGR']*0.2
        if score > best_score:
            best_score = score
            best_name = p['name']
            best_full = m_full
            best_oos = m_oos
    
    # Individual B&H for comparison
    print(f"\n--- Buy & Hold Baselines ---")
    for t in tickers_3x:
        ret = df_all[t].pct_change().values[200:]
        m = calc_metrics(ret)
        if m:
            print(f"{t+' B&H':<22} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f}")
    
    # Equal-weight B&H
    eq_ret = df_all.pct_change().mean(axis=1).values[200:]
    m_eq = calc_metrics(eq_ret)
    print(f"{'EqWt 3x B&H':<22} {m_eq['CAGR']:>6.1%} {m_eq['MaxDD']:>7.1%} {m_eq['Sharpe']:>7.2f} {m_eq['Calmar']:>7.2f}")
    
    print(f"\n⭐ Best Rotation: {best_name}")
    print(f"   Full: CAGR={best_full['CAGR']:.1%}, MaxDD={best_full['MaxDD']:.1%}, Sharpe={best_full['Sharpe']:.2f}, Calmar={best_full['Calmar']:.2f}")
    print(f"   OOS:  CAGR={best_oos['CAGR']:.1%}, MaxDD={best_oos['MaxDD']:.1%}, Sharpe={best_oos['Sharpe']:.2f}")
    print(f"   Composite Score: {best_score:.3f}")
