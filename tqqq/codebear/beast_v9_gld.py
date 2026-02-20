"""
TQQQ Beast v9 — v8 core + GLD crisis hedge
Key idea: In bear market, allocate remaining (1-position) to GLD instead of cash
GLD tends to rally during equity crises (2020 COVID, 2022 inflation, 2025 tariffs)

This should reduce effective MaxDD by having GLD appreciation offset TQQQ losses.

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

def compute_rsi(series, period=10):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)

def run_beast_v9(tqqq_prices, gld_prices, params, cost=0.0005, slip=0.001):
    """Beast v8 logic on TQQQ + GLD hedge in bear market"""
    df = pd.DataFrame({'TQQQ': tqqq_prices, 'GLD': gld_prices}).dropna()
    n = len(df)
    
    sma_p = params['sma_period']
    bull_thr = params['bull_enter']
    bear_thr = params['bear_enter']
    bear_floor = params['bear_floor']
    gld_mode = params.get('gld_mode', 'bear_only')  # bear_only | always_partial | momentum
    gld_bull_alloc = params.get('gld_bull_alloc', 0.0)  # % of GLD in bull market
    
    tqqq = df['TQQQ']
    gld = df['GLD']
    tqqq_ret = tqqq.pct_change()
    gld_ret = gld.pct_change()
    
    sma200 = tqqq.rolling(sma_p).mean()
    rsi = compute_rsi(tqqq, 10)
    weekly_ret = tqqq.pct_change(5)
    
    regime = 1
    tqqq_w = 1.0
    gld_w = 0.0
    portfolio_ret = np.zeros(n)
    
    for i in range(sma_p, n):
        # Daily PnL
        portfolio_ret[i] = tqqq_w * tqqq_ret.iloc[i] + gld_w * gld_ret.iloc[i]
        
        # Regime (v8 hysteresis)
        sma_val = sma200.iloc[i]
        if np.isnan(sma_val): continue
        
        ratio = tqqq.iloc[i] / sma_val
        if ratio > bull_thr:
            regime = 1
        elif ratio < bear_thr:
            regime = 0
        
        # Position sizing (v8 logic)
        rsi_val = rsi.iloc[i]
        wk_ret = weekly_ret.iloc[i]
        
        if regime == 1:
            # Bull
            new_tqqq = 1.0
            if not np.isnan(rsi_val) and rsi_val > 80 and not np.isnan(wk_ret) and wk_ret > 0.15:
                new_tqqq = 0.80
            
            if gld_mode == 'always_partial':
                new_gld = gld_bull_alloc
                new_tqqq = min(new_tqqq, 1.0 - new_gld)
            else:
                new_gld = 0.0
        else:
            # Bear
            pos = bear_floor
            if not np.isnan(rsi_val):
                if rsi_val < 20 or (not np.isnan(wk_ret) and wk_ret < -0.12):
                    pos = 0.80
                elif rsi_val < 30:
                    pos = 0.60
                elif rsi_val > 65:
                    pos = bear_floor
            
            new_tqqq = pos
            remaining = 1.0 - pos
            
            if gld_mode in ('bear_only', 'always_partial', 'momentum'):
                # GLD gets the remaining
                new_gld = remaining
            else:
                new_gld = 0.0
        
        # Trading cost
        turnover = abs(new_tqqq - tqqq_w) + abs(new_gld - gld_w)
        portfolio_ret[i] -= turnover * (cost + slip / 2)
        
        tqqq_w = new_tqqq
        gld_w = new_gld
    
    return portfolio_ret, df.index

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    
    tqqq = load('TQQQ', data_dir)
    gld = load('GLD', data_dir)
    
    sma_p = 200
    
    param_sets = [
        # v8 baseline (no GLD)
        {'name': 'v8_baseline', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'gld_mode': 'none'},
        # v9a: GLD only in bear
        {'name': 'v9a_gld_bear', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'gld_mode': 'bear_only'},
        # v9b: wider bands + GLD bear
        {'name': 'v9b_wide_gld', 'sma_period': 200, 'bull_enter': 1.08, 'bear_enter': 0.87,
         'bear_floor': 0.25, 'gld_mode': 'bear_only'},
        # v9c: always 10% GLD
        {'name': 'v9c_always10', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'gld_mode': 'always_partial', 'gld_bull_alloc': 0.10},
        # v9d: higher floor + GLD
        {'name': 'v9d_hfloor_gld', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.35, 'gld_mode': 'bear_only'},
        # v9e: narrow bands + GLD (more time in bear → more GLD)
        {'name': 'v9e_narrow_gld', 'sma_period': 200, 'bull_enter': 1.03, 'bear_enter': 0.93,
         'bear_floor': 0.30, 'gld_mode': 'bear_only'},
    ]
    
    print(f"\n{'Name':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>4} {'Score':>6}")
    print("-"*90)
    
    results = []
    
    for p in param_sets:
        ret, idx = run_beast_v9(tqqq, gld, p)
        m = calc_metrics(ret[sma_p:])
        
        n = len(ret)
        split = int(n * 0.6)
        m_is = calc_metrics(ret[sma_p:split])
        m_oos = calc_metrics(ret[split:])
        
        if not m: continue
        
        wf = '✅' if m_oos.get('Sharpe',0) >= m_is.get('Sharpe',0) * 0.7 else '❌'
        score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
        
        print(f"{p['name']:<22} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {m_is.get('Sharpe',0):>6.2f} {m_oos.get('Sharpe',0):>7.2f} {wf:>4} {score:>6.3f}")
        
        results.append({'name': p['name'], 
                        'full': {k: round(v,4) for k,v in m.items()},
                        'is': {k: round(v,4) for k,v in m_is.items()},
                        'oos': {k: round(v,4) for k,v in m_oos.items()},
                        'wf_pass': wf == '✅', 'composite': round(score,4)})
    
    out_path = os.path.join(os.path.dirname(__file__), 'beast_v9_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
