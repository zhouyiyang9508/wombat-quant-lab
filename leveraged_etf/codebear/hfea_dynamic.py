"""
HFEA Dynamic — TQQQ + synthetic TMF (3x TLT) with dynamic allocation
Classic HFEA: 55% TQQQ / 45% TMF, quarterly rebalance
Dynamic: adjust ratio based on momentum/regime signals

Synthetic TMF: daily_ret = 3 × TLT_daily_ret (minus ~1% annual ER drag)

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
    vol = np.std(returns) * np.sqrt(252)
    return {'CAGR': cagr, 'MaxDD': maxdd, 'Sharpe': sharpe, 'Calmar': calmar, 'Vol': vol}

def build_synthetic_tmf(tlt_prices, er_annual=0.01):
    """Synthetic TMF = 3x daily TLT returns minus expense ratio drag"""
    tlt_ret = tlt_prices.pct_change()
    daily_er = er_annual / 252
    tmf_ret = 3 * tlt_ret - daily_er
    tmf_price = (1 + tmf_ret).cumprod() * 100
    tmf_price.iloc[0] = 100
    return tmf_price, tmf_ret

def run_hfea(tqqq_ret, tmf_ret, params, cost=0.0005, slip=0.001):
    """Run HFEA with given parameters"""
    n = len(tqqq_ret)
    tqqq_w = params['tqqq_weight']
    tmf_w = 1.0 - tqqq_w
    rebal_freq = params['rebalance_freq']
    rebal_band = params.get('rebal_band', 0.0)  # rebalance only if drift > band
    
    # Dynamic mode
    mode = params.get('mode', 'static')
    sma_period = params.get('sma_period', 200)
    
    # For SMA, need cumulative prices
    tqqq_cum = (1 + tqqq_ret).cumprod()
    tmf_cum = (1 + tmf_ret).cumprod()
    
    tqqq_sma = tqqq_cum.rolling(sma_period).mean() if mode != 'static' else None
    
    # Simulate
    portfolio_ret = np.zeros(n)
    curr_tqqq_w = tqqq_w
    curr_tmf_w = tmf_w
    warmup = sma_period if mode != 'static' else 1
    
    for i in range(warmup, n):
        # Daily return from current weights
        portfolio_ret[i] = curr_tqqq_w * tqqq_ret.iloc[i] + curr_tmf_w * tmf_ret.iloc[i]
        
        # Update weights for drift
        total = curr_tqqq_w * (1 + tqqq_ret.iloc[i]) + curr_tmf_w * (1 + tmf_ret.iloc[i])
        if total > 0:
            curr_tqqq_w = curr_tqqq_w * (1 + tqqq_ret.iloc[i]) / total
            curr_tmf_w = 1.0 - curr_tqqq_w
        
        # Rebalance check
        if (i - warmup) % rebal_freq != 0:
            continue
        
        # Determine target weights
        if mode == 'static':
            target_tqqq = tqqq_w
        elif mode == 'trend':
            # Above SMA → offense heavy, below → defense heavy
            if tqqq_sma is not None and not np.isnan(tqqq_sma.iloc[i]):
                if tqqq_cum.iloc[i] > tqqq_sma.iloc[i]:
                    target_tqqq = params.get('bull_tqqq', 0.70)
                else:
                    target_tqqq = params.get('bear_tqqq', 0.30)
            else:
                target_tqqq = tqqq_w
        elif mode == 'momentum':
            # Compare 3M momentum of TQQQ vs TMF
            if i >= 63:
                tqqq_mom = tqqq_cum.iloc[i] / tqqq_cum.iloc[i-63] - 1
                tmf_mom = tmf_cum.iloc[i] / tmf_cum.iloc[i-63] - 1
                if tqqq_mom > tmf_mom:
                    target_tqqq = params.get('bull_tqqq', 0.70)
                else:
                    target_tqqq = params.get('bear_tqqq', 0.30)
            else:
                target_tqqq = tqqq_w
        else:
            target_tqqq = tqqq_w
        
        target_tmf = 1.0 - target_tqqq
        
        # Only rebalance if drift exceeds band
        if abs(curr_tqqq_w - target_tqqq) > rebal_band:
            turnover = abs(curr_tqqq_w - target_tqqq) + abs(curr_tmf_w - target_tmf)
            portfolio_ret[i] -= turnover * (cost + slip / 2)
            curr_tqqq_w = target_tqqq
            curr_tmf_w = target_tmf
    
    return portfolio_ret, warmup

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    
    tqqq = load('TQQQ', data_dir)
    tlt = load('TLT', data_dir)
    
    # Align
    common = tqqq.index.intersection(tlt.index)
    tqqq = tqqq.loc[common]
    tlt = tlt.loc[common]
    
    tqqq_ret = tqqq.pct_change()
    tmf_price, tmf_ret = build_synthetic_tmf(tlt)
    
    # Align returns
    tqqq_ret = tqqq_ret.loc[common]
    tmf_ret = tmf_ret.loc[common]
    
    print(f"Data: {len(tqqq_ret)} rows, {common[0].date()} to {common[-1].date()}")
    
    n = len(tqqq_ret)
    split = int(n * 0.6)
    
    param_sets = [
        # Classic HFEA
        {'name': 'HFEA_55_45_Q', 'tqqq_weight': 0.55, 'rebalance_freq': 63, 'mode': 'static'},
        {'name': 'HFEA_55_45_M', 'tqqq_weight': 0.55, 'rebalance_freq': 21, 'mode': 'static'},
        {'name': 'HFEA_60_40_M', 'tqqq_weight': 0.60, 'rebalance_freq': 21, 'mode': 'static'},
        {'name': 'HFEA_70_30_M', 'tqqq_weight': 0.70, 'rebalance_freq': 21, 'mode': 'static'},
        # Trend-based: 70/30 in bull, 30/70 in bear
        {'name': 'HFEA_trend_70_30', 'tqqq_weight': 0.55, 'rebalance_freq': 21, 'mode': 'trend',
         'sma_period': 200, 'bull_tqqq': 0.70, 'bear_tqqq': 0.30},
        {'name': 'HFEA_trend_80_20', 'tqqq_weight': 0.55, 'rebalance_freq': 21, 'mode': 'trend',
         'sma_period': 200, 'bull_tqqq': 0.80, 'bear_tqqq': 0.20},
        # Momentum-based
        {'name': 'HFEA_mom_70_30', 'tqqq_weight': 0.55, 'rebalance_freq': 21, 'mode': 'momentum',
         'bull_tqqq': 0.70, 'bear_tqqq': 0.30},
        {'name': 'HFEA_mom_80_20', 'tqqq_weight': 0.55, 'rebalance_freq': 21, 'mode': 'momentum',
         'bull_tqqq': 0.80, 'bear_tqqq': 0.20},
        # Band rebalancing
        {'name': 'HFEA_55_band5', 'tqqq_weight': 0.55, 'rebalance_freq': 5, 'mode': 'static', 'rebal_band': 0.05},
    ]
    
    print(f"\n{'Name':<24} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Vol':>6} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>4} {'Score':>6}")
    print("-"*100)
    
    results = []
    
    for p in param_sets:
        ret, warmup = run_hfea(tqqq_ret, tmf_ret, p)
        m = calc_metrics(ret[warmup:])
        
        m_is = calc_metrics(ret[warmup:split])
        m_oos = calc_metrics(ret[split:])
        
        if not m: continue
        
        wf = '✅' if m_oos.get('Sharpe',0) >= m_is.get('Sharpe',0) * 0.7 else '❌'
        score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
        
        print(f"{p['name']:<24} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {m['Vol']:>5.1%} | {m_is.get('Sharpe',0):>6.2f} {m_oos.get('Sharpe',0):>7.2f} {wf:>4} {score:>6.3f}")
        
        results.append({'name': p['name'], 
                        'full': {k: round(v,4) for k,v in m.items()},
                        'is': {k: round(v,4) for k,v in m_is.items()},
                        'oos': {k: round(v,4) for k,v in m_oos.items()},
                        'wf_pass': wf == '✅', 'composite': round(score,4)})
    
    # TQQQ B&H baseline
    bh_ret = tqqq_ret.values[200:]
    m = calc_metrics(bh_ret)
    if m:
        score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
        print(f"\n{'TQQQ B&H':<24} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {m['Vol']:>5.1%} | {'':>6} {'':>7} {'':>4} {score:>6.3f}")
    
    out_path = os.path.join(os.path.dirname(__file__), 'hfea_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
