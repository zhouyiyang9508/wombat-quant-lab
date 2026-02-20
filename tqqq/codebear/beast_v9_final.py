"""
Beast v9 Final — Optimized combos with walk-forward validation
Testing best parameter combinations from robustness analysis

代码熊 🐻 2026-02-20
"""

import pandas as pd
import numpy as np
import os, json, sys
sys.path.insert(0, os.path.dirname(__file__))
from beast_v9_gld import load, calc_metrics, run_beast_v9, compute_rsi

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    tqqq = load('TQQQ', data_dir)
    gld = load('GLD', data_dir)
    
    # Align
    common = tqqq.index.intersection(gld.index)
    tqqq = tqqq.loc[common]
    gld = gld.loc[common]
    
    sma_p = 200
    n = len(tqqq)
    split = int(n * 0.6)
    
    combos = [
        # v8 reference (no GLD)
        {'name': 'v8_ref(no_gld)', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'gld_mode': 'none'},
        # v9a original
        {'name': 'v9a_original', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90,
         'bear_floor': 0.30, 'gld_mode': 'bear_only'},
        # v9f: narrow bear + GLD (best from sensitivity)
        {'name': 'v9f_narrow93', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.93,
         'bear_floor': 0.30, 'gld_mode': 'bear_only'},
        # v9g: narrow bear + low floor + GLD
        {'name': 'v9g_narrow93_lf', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.93,
         'bear_floor': 0.25, 'gld_mode': 'bear_only'},
        # v9h: narrow bear + always 10% GLD
        {'name': 'v9h_n93_a10', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.93,
         'bear_floor': 0.30, 'gld_mode': 'always_partial', 'gld_bull_alloc': 0.10},
        # v9i: moderate narrow + GLD
        {'name': 'v9i_narrow92', 'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.92,
         'bear_floor': 0.30, 'gld_mode': 'bear_only'},
        # v9j: wider bull + narrow bear
        {'name': 'v9j_b107_n93', 'sma_period': 200, 'bull_enter': 1.07, 'bear_enter': 0.93,
         'bear_floor': 0.30, 'gld_mode': 'bear_only'},
    ]
    
    print(f"Data: {n} rows, {tqqq.index[0].date()} to {tqqq.index[-1].date()}")
    print(f"Split: IS {tqqq.index[sma_p].date()}-{tqqq.index[split].date()}, OOS {tqqq.index[split].date()}-{tqqq.index[-1].date()}")
    
    print(f"\n{'Name':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>4} {'Score':>6}")
    print("-"*90)
    
    results = []
    
    for p in combos:
        ret, idx = run_beast_v9(tqqq, gld, p)
        m = calc_metrics(ret[sma_p:])
        m_is = calc_metrics(ret[sma_p:split])
        m_oos = calc_metrics(ret[split:])
        
        if not m: continue
        
        wf = '✅' if m_oos.get('Sharpe',0) >= m_is.get('Sharpe',0) * 0.7 else '❌'
        score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
        
        print(f"{p['name']:<22} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {m_is.get('Sharpe',0):>6.2f} {m_oos.get('Sharpe',0):>7.2f} {wf:>4} {score:>6.3f}")
        
        results.append({'name': p['name'], 'params': {k:v for k,v in p.items() if k != 'name'},
                        'full': {k: round(v,4) for k,v in m.items()},
                        'is': {k: round(v,4) for k,v in m_is.items()},
                        'oos': {k: round(v,4) for k,v in m_oos.items()},
                        'wf_pass': wf == '✅', 'composite': round(score,4)})
    
    out_path = os.path.join(os.path.dirname(__file__), 'beast_v9_final_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
