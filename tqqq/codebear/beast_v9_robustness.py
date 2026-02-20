"""
Beast v9 robustness check — parameter sensitivity analysis
Test v9a (GLD bear hedge) across parameter ranges

代码熊 🐻 2026-02-20
"""

import pandas as pd
import numpy as np
import os, json
import sys
sys.path.insert(0, os.path.dirname(__file__))
from beast_v9_gld import load, calc_metrics, run_beast_v9

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    tqqq = load('TQQQ', data_dir)
    gld = load('GLD', data_dir)
    
    sma_p = 200
    base = {'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90, 'bear_floor': 0.30, 'gld_mode': 'bear_only'}
    
    # Test each parameter independently
    tests = {
        'bull_enter': [1.02, 1.03, 1.05, 1.07, 1.08, 1.10],
        'bear_enter': [0.85, 0.87, 0.90, 0.92, 0.93, 0.95],
        'bear_floor': [0.20, 0.25, 0.30, 0.35, 0.40],
        'sma_period': [150, 175, 200, 225, 250],
    }
    
    for param_name, values in tests.items():
        print(f"\n=== Sensitivity: {param_name} ===")
        print(f"{'Value':>8} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Score':>7}")
        scores = []
        for v in values:
            p = {**base, param_name: v, 'name': f'{param_name}={v}'}
            ret, idx = run_beast_v9(tqqq, gld, p)
            m = calc_metrics(ret[sma_p:])
            if m:
                score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
                scores.append(score)
                marker = ' ◀' if v == base.get(param_name) else ''
                print(f"{v:>8} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {score:>7.3f}{marker}")
        
        if scores:
            print(f"  Range: {min(scores):.3f} - {max(scores):.3f}, Spread: {max(scores)-min(scores):.3f}")
    
    # Also compare v9a vs v9c (always 10% GLD) robustness
    print(f"\n=== v9c: GLD bull allocation sensitivity ===")
    print(f"{'GLD%':>8} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Score':>7}")
    for gld_pct in [0.0, 0.05, 0.10, 0.15, 0.20]:
        p = {**base, 'gld_mode': 'always_partial', 'gld_bull_alloc': gld_pct, 'name': f'gld_bull={gld_pct}'}
        ret, idx = run_beast_v9(tqqq, gld, p)
        m = calc_metrics(ret[sma_p:])
        if m:
            score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
            print(f"{gld_pct:>7.0%} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {score:>7.3f}")

if __name__ == '__main__':
    main()
