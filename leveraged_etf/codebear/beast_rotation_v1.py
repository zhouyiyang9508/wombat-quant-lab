"""
Beast Rotation v1 — 3x ETF动量轮动 + Beast熊市底仓
核心思想：牛市持有动量最强的3x ETF，熊市保持底仓捕捉V反弹
组合 TQQQ/SOXL/UPRO 的相对强度 + v8 beast的regime逻辑

代码熊 🐻 2026-02-19
"""

import pandas as pd
import numpy as np
import os

def load(ticker, data_dir):
    path = os.path.join(data_dir, f'{ticker}.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    return df['Close'].dropna()

def run_beast_rotation(df_all, params, cost_per_side=0.0005, slippage=0.001):
    tickers = list(df_all.columns)
    n = len(df_all)
    sma_p = params['sma_period']
    
    # Precompute per-ticker indicators
    sma200 = df_all.rolling(sma_p).mean()
    daily_ret = df_all.pct_change()
    
    # Blended momentum per ticker
    mom = pd.DataFrame(index=df_all.index)
    for t in tickers:
        m = 0.25 * df_all[t].pct_change(21) + 0.40 * df_all[t].pct_change(63) + 0.35 * df_all[t].pct_change(126)
        mom[t] = m
    
    # RSI per ticker
    rsi_all = pd.DataFrame(index=df_all.index)
    for t in tickers:
        delta = df_all[t].diff()
        gain = delta.clip(lower=0).rolling(10).mean()
        loss = (-delta.clip(upper=0)).rolling(10).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi_all[t] = 100 - 100 / (1 + rs)
    
    # Broad market regime: use SPY-proxy = average of underlyings / SMA
    avg_price = df_all.mean(axis=1)
    avg_sma = avg_price.rolling(sma_p).mean()
    
    bull_thr = params['bull_enter']
    bear_thr = params['bear_enter']
    bear_floor = params['bear_floor']
    rebal_freq = params['rebalance_freq']
    
    # Track
    holdings = {t: 0.0 for t in tickers}
    regime = 1  # 1=bull, 0=bear
    portfolio_ret = np.zeros(n)
    
    for i in range(sma_p, n):
        # Daily P&L from yesterday's holdings
        if i > sma_p:
            day_ret = sum(holdings[t] * daily_ret[t].iloc[i] for t in tickers)
            portfolio_ret[i] = day_ret
        
        # Rebalance?
        if (i - sma_p) % rebal_freq != 0:
            continue
        
        # Regime (hysteresis on broad)
        if not np.isnan(avg_sma.iloc[i]):
            ratio = avg_price.iloc[i] / avg_sma.iloc[i]
            if ratio > bull_thr:
                regime = 1
            elif ratio < bear_thr:
                regime = 0
        
        new_holdings = {t: 0.0 for t in tickers}
        
        if regime == 1:
            # Bull: pick strongest by momentum, full allocation
            valid = {}
            for t in tickers:
                m = mom[t].iloc[i]
                if not np.isnan(m):
                    valid[t] = m
            
            if valid:
                best = max(valid, key=valid.get)
                # Concentrate on best, but give some to #2
                sorted_t = sorted(valid.keys(), key=lambda x: valid[x], reverse=True)
                if len(sorted_t) >= 2 and valid[sorted_t[1]] > 0:
                    new_holdings[sorted_t[0]] = 0.70
                    new_holdings[sorted_t[1]] = 0.30
                else:
                    new_holdings[sorted_t[0]] = 1.0
        else:
            # Bear: floor position in strongest, RSI-based dip buying
            valid = {}
            for t in tickers:
                m = mom[t].iloc[i]
                if not np.isnan(m):
                    valid[t] = m
            
            if valid:
                sorted_t = sorted(valid.keys(), key=lambda x: valid[x], reverse=True)
                best = sorted_t[0]
                
                # Base: floor
                pos = bear_floor
                
                # RSI dip buying on best ticker
                rsi_val = rsi_all[best].iloc[i]
                if not np.isnan(rsi_val):
                    if rsi_val < 20:
                        pos = 0.80
                    elif rsi_val < 30:
                        pos = 0.60
                    elif rsi_val > 65:
                        pos = bear_floor
                
                new_holdings[best] = pos
        
        # Trading cost
        turnover = sum(abs(new_holdings[t] - holdings[t]) for t in tickers)
        trade_cost = turnover * (cost_per_side + slippage / 2)
        portfolio_ret[i] -= trade_cost
        
        holdings = new_holdings
    
    return portfolio_ret

def calc_metrics(returns, rf_annual=0.04):
    cum = np.cumprod(1 + returns)
    years = len(returns) / 252
    if years < 0.5: return {}
    cagr = cum[-1] ** (1/years) - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = dd.min()
    daily_rf = (1 + rf_annual) ** (1/252) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * np.mean(excess) / np.std(excess) if np.std(excess) > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    return {'CAGR': cagr, 'MaxDD': maxdd, 'Sharpe': sharpe, 'Calmar': calmar}

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    tickers = ['TQQQ', 'SOXL', 'UPRO']
    
    data = {t: load(t, data_dir) for t in tickers}
    df_all = pd.DataFrame(data).dropna()
    print(f"Aligned: {len(df_all)} rows, {df_all.index[0].date()} to {df_all.index[-1].date()}")
    
    n = len(df_all)
    split = int(n * 0.6)
    
    param_sets = [
        {'name': 'BeastRot_base', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90, 'bear_floor': 0.30, 'rebalance_freq': 5},
        {'name': 'BeastRot_wide', 'sma_period': 200, 'bull_enter': 1.08, 'bear_enter': 0.87, 'bear_floor': 0.25, 'rebalance_freq': 5},
        {'name': 'BeastRot_biweek', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90, 'bear_floor': 0.30, 'rebalance_freq': 10},
        {'name': 'BeastRot_month', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90, 'bear_floor': 0.30, 'rebalance_freq': 21},
        {'name': 'BeastRot_hfloor', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90, 'bear_floor': 0.40, 'rebalance_freq': 5},
        {'name': 'BeastRot_narrow', 'sma_period': 200, 'bull_enter': 1.03, 'bear_enter': 0.93, 'bear_floor': 0.35, 'rebalance_freq': 5},
    ]
    
    print(f"\n{'Name':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>4} {'Score':>6}")
    print("-"*90)
    
    for p in param_sets:
        sma_p = p['sma_period']
        
        # Full
        ret = run_beast_rotation(df_all, p)
        m = calc_metrics(ret[sma_p:])
        
        # IS / OOS
        m_is = calc_metrics(ret[sma_p:split])
        
        start_oos = max(0, split - sma_p)
        data_oos = {t: df_all[t].iloc[start_oos:] for t in tickers}
        df_oos = pd.DataFrame(data_oos)
        ret_oos = run_beast_rotation(df_oos, p)
        oos_offset = split - start_oos
        m_oos = calc_metrics(ret_oos[oos_offset:])
        
        if not m: continue
        
        wf = '✅' if m_oos.get('Sharpe',0) >= m_is.get('Sharpe',0) * 0.7 else '❌'
        score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
        print(f"{p['name']:<22} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {m_is.get('Sharpe',0):>6.2f} {m_oos.get('Sharpe',0):>7.2f} {wf:>4} {score:>6.3f}")
    
    # Baselines
    print(f"\n--- Baselines ---")
    for t in tickers:
        ret = df_all[t].pct_change().values[200:]
        m = calc_metrics(ret)
        if m:
            score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
            print(f"{t+' B&H':<22} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {'':>6} {'':>7} {'':>4} {score:>6.3f}")
