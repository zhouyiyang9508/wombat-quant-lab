"""
Dual Momentum 3x — Antonacci-style dual momentum with 3x ETFs
- Relative momentum: pick best among TQQQ/SOXL/UPRO
- Absolute momentum: if best has negative 6M return → switch to TLT or GLD (whichever stronger)
- If TLT and GLD both negative → 100% cash (SHY proxy: 0% return)
- Monthly rebalance
- Optional: add trend filter (SMA200) as confirmation

代码熊 🐻 2026-02-20
"""

import pandas as pd
import numpy as np
import os, json

def load(ticker, data_dir):
    path = os.path.join(data_dir, f'{ticker}.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    return df['Close'].dropna()

def calc_metrics(returns, rf_annual=0.04):
    ret = returns[returns != 0] if len(returns[returns != 0]) > 0 else returns
    cum = np.cumprod(1 + returns)
    years = len(returns) / 252
    if years < 0.5: return {}
    cagr = cum[-1] ** (1/years) - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = dd.min()
    daily_rf = (1 + rf_annual) ** (1/252) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * np.mean(excess) / (np.std(excess) + 1e-10)
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    return {'CAGR': cagr, 'MaxDD': maxdd, 'Sharpe': sharpe, 'Calmar': calmar}

def run_dual_momentum(df_all, off_tickers, def_tickers, params, cost=0.0005, slip=0.001):
    n = len(df_all)
    daily_ret = df_all.pct_change()
    
    mom_lookback = params['mom_lookback']  # in trading days
    rebal_freq = params['rebalance_freq']  # in trading days
    use_sma = params.get('use_sma_filter', False)
    sma_period = params.get('sma_period', 200)
    abs_mom_lookback = params.get('abs_mom_lookback', mom_lookback)
    
    # Precompute momentum
    mom = pd.DataFrame({t: df_all[t].pct_change(mom_lookback) for t in df_all.columns}, index=df_all.index)
    abs_mom = pd.DataFrame({t: df_all[t].pct_change(abs_mom_lookback) for t in df_all.columns}, index=df_all.index)
    
    if use_sma:
        sma = pd.DataFrame({t: df_all[t].rolling(sma_period).mean() for t in off_tickers}, index=df_all.index)
    
    all_tickers = list(df_all.columns)
    holdings = {t: 0.0 for t in all_tickers}
    portfolio_ret = np.zeros(n)
    warmup = max(mom_lookback, abs_mom_lookback, sma_period if use_sma else 0) + 1
    
    for i in range(warmup, n):
        # Daily PnL
        if i > warmup:
            portfolio_ret[i] = sum(holdings[t] * daily_ret[t].iloc[i] for t in all_tickers)
        
        if (i - warmup) % rebal_freq != 0:
            continue
        
        new_h = {t: 0.0 for t in all_tickers}
        
        # Step 1: Relative momentum among offense
        off_scores = {}
        for t in off_tickers:
            m = mom[t].iloc[i]
            if not np.isnan(m):
                off_scores[t] = m
        
        if not off_scores:
            holdings = new_h
            continue
        
        best_off = max(off_scores, key=off_scores.get)
        
        # Step 2: Absolute momentum filter
        abs_m = abs_mom[best_off].iloc[i]
        sma_ok = True
        if use_sma and best_off in off_tickers:
            sma_val = sma[best_off].iloc[i]
            if not np.isnan(sma_val):
                sma_ok = df_all[best_off].iloc[i] > sma_val
        
        if abs_m > 0 and sma_ok:
            # Offense: hold best
            new_h[best_off] = 1.0
        else:
            # Defense: pick best among TLT/GLD
            def_scores = {}
            for t in def_tickers:
                m = mom[t].iloc[i]
                if not np.isnan(m):
                    def_scores[t] = m
            
            if def_scores:
                best_def = max(def_scores, key=def_scores.get)
                if abs_mom[best_def].iloc[i] > 0:
                    new_h[best_def] = 1.0
                # else: cash
        
        # Cost
        turnover = sum(abs(new_h[t] - holdings[t]) for t in all_tickers)
        portfolio_ret[i] -= turnover * (cost + slip / 2)
        holdings = new_h
    
    return portfolio_ret, warmup

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    
    off_tickers = ['TQQQ', 'SOXL', 'UPRO']
    def_tickers = ['TLT', 'GLD']
    all_tickers = off_tickers + def_tickers
    
    data = {t: load(t, data_dir) for t in all_tickers}
    df_all = pd.DataFrame(data).dropna()
    print(f"Data: {len(df_all)} rows, {df_all.index[0].date()} to {df_all.index[-1].date()}")
    
    n = len(df_all)
    split = int(n * 0.6)
    
    param_sets = [
        {'name': 'DM_6m_monthly', 'mom_lookback': 126, 'abs_mom_lookback': 126, 'rebalance_freq': 21},
        {'name': 'DM_3m_monthly', 'mom_lookback': 63, 'abs_mom_lookback': 63, 'rebalance_freq': 21},
        {'name': 'DM_blend_monthly', 'mom_lookback': 126, 'abs_mom_lookback': 63, 'rebalance_freq': 21},
        {'name': 'DM_6m_weekly', 'mom_lookback': 126, 'abs_mom_lookback': 126, 'rebalance_freq': 5},
        {'name': 'DM_6m_sma_monthly', 'mom_lookback': 126, 'abs_mom_lookback': 126, 'rebalance_freq': 21, 'use_sma_filter': True, 'sma_period': 200},
        {'name': 'DM_3m_sma_weekly', 'mom_lookback': 63, 'abs_mom_lookback': 63, 'rebalance_freq': 5, 'use_sma_filter': True, 'sma_period': 200},
    ]
    
    print(f"\n{'Name':<24} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>4} {'Score':>6}")
    print("-"*92)
    
    results = []
    
    for p in param_sets:
        ret, warmup = run_dual_momentum(df_all, off_tickers, def_tickers, p)
        m = calc_metrics(ret[warmup:])
        
        # IS/OOS
        m_is = calc_metrics(ret[warmup:split])
        
        start_oos = max(0, split - max(p['mom_lookback'], p.get('abs_mom_lookback', p['mom_lookback']), p.get('sma_period', 0)) - 10)
        df_oos = df_all.iloc[start_oos:]
        ret_oos, warmup_oos = run_dual_momentum(df_oos, off_tickers, def_tickers, p)
        oos_offset = split - start_oos
        m_oos = calc_metrics(ret_oos[oos_offset:])
        
        if not m: continue
        
        wf = '✅' if m_oos.get('Sharpe',0) >= m_is.get('Sharpe',0) * 0.7 else '❌'
        score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
        
        print(f"{p['name']:<24} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {m_is.get('Sharpe',0):>6.2f} {m_oos.get('Sharpe',0):>7.2f} {wf:>4} {score:>6.3f}")
        
        results.append({'name': p['name'], 'full': {k: round(v,4) for k,v in m.items()},
                        'is': {k: round(v,4) for k,v in m_is.items()},
                        'oos': {k: round(v,4) for k,v in m_oos.items()},
                        'wf_pass': wf == '✅', 'composite': round(score,4)})
    
    # Baselines
    print(f"\n--- Baselines ---")
    for t in off_tickers:
        ret = df_all[t].pct_change().values[200:]
        m = calc_metrics(ret)
        if m:
            score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
            print(f"{t+' B&H':<24} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {'':>6} {'':>7} {'':>4} {score:>6.3f}")
    
    out_path = os.path.join(os.path.dirname(__file__), 'dual_momentum_3x_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
