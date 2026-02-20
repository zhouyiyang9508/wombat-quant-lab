"""
Beast Rotation v3 — 3x ETF轮动 + 熊市防御资产(TLT/GLD)
核心改进：熊市期间cash部分配置到TLT/GLD（按动量选优），而非纯现金
假设：牛市全仓3x ETF，熊市底仓3x + 剩余配TLT/GLD

代码熊 🐻 2026-02-20
"""

import pandas as pd
import numpy as np
import os, json

def load(ticker, data_dir):
    path = os.path.join(data_dir, f'{ticker}.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    return df['Close'].dropna()

def compute_rsi(series, period=10):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)

def blended_momentum(series, weights=(0.25, 0.40, 0.35)):
    """1M/3M/6M blended momentum"""
    return weights[0]*series.pct_change(21) + weights[1]*series.pct_change(63) + weights[2]*series.pct_change(126)

def run_strategy(df_offense, df_defense, params, cost=0.0005, slip=0.001):
    """
    df_offense: DataFrame of 3x ETF prices (TQQQ, SOXL, UPRO)
    df_defense: DataFrame of defensive asset prices (TLT, GLD)
    """
    off_tickers = list(df_offense.columns)
    def_tickers = list(df_defense.columns)
    all_tickers = off_tickers + def_tickers
    
    df_all = pd.concat([df_offense, df_defense], axis=1).dropna()
    n = len(df_all)
    sma_p = params['sma_period']
    
    # Precompute
    daily_ret = df_all.pct_change()
    mom = pd.DataFrame({t: blended_momentum(df_all[t]) for t in all_tickers}, index=df_all.index)
    rsi = pd.DataFrame({t: compute_rsi(df_all[t]) for t in off_tickers}, index=df_all.index)
    
    # Broad regime: average of offense tickers
    avg_price = df_offense.reindex(df_all.index).mean(axis=1)
    avg_sma = avg_price.rolling(sma_p).mean()
    
    bull_thr = params['bull_enter']
    bear_thr = params['bear_enter']
    bear_floor = params['bear_floor']
    rebal_freq = params['rebalance_freq']
    defense_mode = params.get('defense_mode', 'momentum')  # momentum | split | best_only
    
    holdings = {t: 0.0 for t in all_tickers}
    regime = 1
    portfolio_ret = np.zeros(n)
    
    for i in range(sma_p, n):
        if i > sma_p:
            day_ret = sum(holdings[t] * daily_ret[t].iloc[i] for t in all_tickers)
            portfolio_ret[i] = day_ret
        
        if (i - sma_p) % rebal_freq != 0:
            continue
        
        # Regime
        if not np.isnan(avg_sma.iloc[i]):
            ratio = avg_price.iloc[i] / avg_sma.iloc[i]
            if ratio > bull_thr:
                regime = 1
            elif ratio < bear_thr:
                regime = 0
        
        new_h = {t: 0.0 for t in all_tickers}
        
        # Get offense momentum ranking
        off_mom = {t: mom[t].iloc[i] for t in off_tickers if not np.isnan(mom[t].iloc[i])}
        
        if regime == 1:
            # Bull: full offense
            if off_mom:
                sorted_off = sorted(off_mom, key=off_mom.get, reverse=True)
                if len(sorted_off) >= 2 and off_mom[sorted_off[1]] > 0:
                    new_h[sorted_off[0]] = 0.70
                    new_h[sorted_off[1]] = 0.30
                else:
                    new_h[sorted_off[0]] = 1.0
        else:
            # Bear: floor in best offense + defense allocation
            if off_mom:
                sorted_off = sorted(off_mom, key=off_mom.get, reverse=True)
                best_off = sorted_off[0]
                
                pos = bear_floor
                rsi_val = rsi[best_off].iloc[i]
                if not np.isnan(rsi_val):
                    if rsi_val < 20:
                        pos = 0.80
                    elif rsi_val < 30:
                        pos = 0.60
                    elif rsi_val > 65:
                        pos = bear_floor
                
                new_h[best_off] = pos
                
                # Remaining goes to defense
                remaining = 1.0 - pos
                if remaining > 0.01:
                    def_mom = {t: mom[t].iloc[i] for t in def_tickers if not np.isnan(mom[t].iloc[i])}
                    
                    if defense_mode in ('none', 'none_internal'):
                        pass  # stay cash
                    elif defense_mode == 'momentum' and def_mom:
                        best_def = max(def_mom, key=def_mom.get)
                        if def_mom[best_def] > 0:
                            new_h[best_def] = remaining
                    elif defense_mode == 'split' and def_mom:
                        for t in def_tickers:
                            if t in def_mom and def_mom[t] > 0:
                                new_h[t] = remaining / len(def_tickers)
                    elif defense_mode == 'always_split':
                        for t in def_tickers:
                            new_h[t] = remaining / len(def_tickers)
        
        # Trading cost
        turnover = sum(abs(new_h[t] - holdings[t]) for t in all_tickers)
        portfolio_ret[i] -= turnover * (cost + slip / 2)
        holdings = new_h
    
    return portfolio_ret, df_all.index

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
    sharpe = np.sqrt(252) * np.mean(excess) / (np.std(excess) + 1e-10)
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    win_rate = np.mean(returns[returns != 0] > 0) if np.any(returns != 0) else 0
    vol = np.std(returns) * np.sqrt(252)
    return {'CAGR': cagr, 'MaxDD': maxdd, 'Sharpe': sharpe, 'Calmar': calmar, 'WinRate': win_rate, 'Vol': vol}

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    
    off_tickers = ['TQQQ', 'SOXL', 'UPRO']
    def_tickers = ['TLT', 'GLD']
    
    off_data = pd.DataFrame({t: load(t, data_dir) for t in off_tickers}).dropna()
    def_data = pd.DataFrame({t: load(t, data_dir) for t in def_tickers}).dropna()
    
    # Align
    common_idx = off_data.index.intersection(def_data.index)
    off_data = off_data.loc[common_idx]
    def_data = def_data.loc[common_idx]
    
    print(f"Data: {len(off_data)} rows, {off_data.index[0].date()} to {off_data.index[-1].date()}")
    
    n = len(off_data)
    sma_p = 200
    split = int(n * 0.6)
    
    param_sets = [
        # v1 baseline (no defense) for comparison
        {'name': 'v1_baseline', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'rebalance_freq': 5, 'defense_mode': 'none'},
        # v3a: momentum-pick best defensive
        {'name': 'v3a_mom_best', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'rebalance_freq': 5, 'defense_mode': 'momentum'},
        # v3b: always split TLT/GLD
        {'name': 'v3b_always_split', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'rebalance_freq': 5, 'defense_mode': 'always_split'},
        # v3c: split only if positive momentum
        {'name': 'v3c_mom_split', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'rebalance_freq': 5, 'defense_mode': 'split'},
        # v3d: wider bands + defense
        {'name': 'v3d_wide_mom', 'sma_period': 200, 'bull_enter': 1.08, 'bear_enter': 0.87,
         'bear_floor': 0.25, 'rebalance_freq': 5, 'defense_mode': 'momentum'},
        # v3e: higher floor + defense  
        {'name': 'v3e_hfloor_mom', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.40, 'rebalance_freq': 5, 'defense_mode': 'momentum'},
    ]
    
    print(f"\n{'Name':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>4} {'Score':>6}")
    print("-"*90)
    
    results = []
    
    for p in param_sets:
        # Handle v1 baseline (no defense)
        if p['defense_mode'] == 'none':
            # Just run offense only, remaining = cash
            def_empty = pd.DataFrame(index=def_data.index)
            ret, idx = run_strategy(off_data, def_data, {**p, 'defense_mode': 'none_internal'})
            # Actually let me just set defense_mode to something that won't allocate
            # Simpler: run with momentum but defense assets have 0 allocation since we skip
        
        ret, idx = run_strategy(off_data, def_data, p)
        m = calc_metrics(ret[sma_p:])
        
        # IS/OOS
        m_is = calc_metrics(ret[sma_p:split])
        
        # OOS: re-run from enough context
        start_oos = max(0, split - sma_p)
        off_oos = off_data.iloc[start_oos:]
        def_oos = def_data.iloc[start_oos:]
        ret_oos, _ = run_strategy(off_oos, def_oos, p)
        oos_offset = split - start_oos
        m_oos = calc_metrics(ret_oos[oos_offset:])
        
        if not m: continue
        
        wf = '✅' if m_oos.get('Sharpe',0) >= m_is.get('Sharpe',0) * 0.7 else '❌'
        score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
        
        print(f"{p['name']:<22} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {m_is.get('Sharpe',0):>6.2f} {m_oos.get('Sharpe',0):>7.2f} {wf:>4} {score:>6.3f}")
        
        results.append({
            'name': p['name'], 'params': {k:v for k,v in p.items() if k != 'name'},
            'full': {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            'is': {k: round(v, 4) for k, v in m_is.items()} if m_is else {},
            'oos': {k: round(v, 4) for k, v in m_oos.items()} if m_oos else {},
            'wf_pass': wf == '✅', 'composite': round(score, 4)
        })
    
    # Baselines
    print(f"\n--- Baselines ---")
    for t in off_tickers:
        ret = off_data[t].pct_change().values[sma_p:]
        m = calc_metrics(ret)
        if m:
            score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
            print(f"{t+' B&H':<22} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {'':>6} {'':>7} {'':>4} {score:>6.3f}")
    
    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'beast_rotation_v3_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
