#!/usr/bin/env python3
"""
v9c 最终精细化扫描 — 找到最优 Vol + DD 组合
"""
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

# Import everything from v9c
import sys
sys.path.insert(0, str(Path(__file__).parent))
from momentum_v9c import (load_csv, load_stocks, precompute, evaluate, fmt,
                           CACHE, STOCK_CACHE)

tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
close_df = load_stocks(tickers + ['SPY'])
sectors  = json.load(open(CACHE / "sp500_sectors.json"))
gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
sig = precompute(close_df)
print(f"Loaded {len(close_df.columns)} tickers")

BASE_P = dict(
    mom_w=(0.20, 0.50, 0.20, 0.10),
    n_bull_secs=5, bull_sps=2, bear_sps=2,
    breadth_thresh=0.45,
    gld_thresh=0.70, gld_frac=0.20,
    cont_bonus=0.03, hi52_frac=0.60, use_shy=True,
    dd_params={-0.08: 0.30, -0.12: 0.50, -0.18: 0.60},
    vol_params=None,
)

print("=== Final Refinement: Vol + DD Combo ===")
combos = [
    ("vol30_45_baseline",  {0.30: 0.10, 0.45: 0.20}, {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}),
    ("vol25only_baseline", {0.25: 0.10},               {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}),
    ("vol25only+dd_agr",   {0.25: 0.10},               {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}),
    ("vol30_45+dd_agr",    {0.30: 0.10, 0.45: 0.20},  {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}),
    ("vol25only+dd5",      {0.25: 0.10},               {-0.05: 0.15, -0.08: 0.30, -0.12: 0.50, -0.18: 0.60}),
    ("vol30_45+dd5",       {0.30: 0.10, 0.45: 0.20},  {-0.05: 0.15, -0.08: 0.30, -0.12: 0.50, -0.18: 0.60}),
    ("vol25_40+dd_agr",    {0.25: 0.10, 0.40: 0.20},  {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}),
]

results = []
for label, vp, dp in combos:
    p = dict(BASE_P); p['vol_params'] = vp; p['dd_params'] = dp
    r = evaluate(close_df, sig, sectors, gld, shy, p)
    print(fmt(label, r))
    results.append({'label': label, **r})

valid = [r for r in results if r['wf'] >= 0.70]
best = max(valid, key=lambda x: x['comp'])
m = best['full']
print(f"\nBEST: {best['label']}")
print(f"  CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  DD {m['max_dd']:.1%}  Cal {m['calmar']:.2f}  WF {best['wf']:.2f}  Comp {best['comp']:.4f}")
print(f"  IS: {best['is_m']['sharpe']:.2f}, OOS: {best['oos_m']['sharpe']:.2f}")

# Save best params
print("\nBest vol_params and dd_params:")
for label, vp, dp in combos:
    if label == best['label']:
        print(f"  vol_params: {vp}")
        print(f"  dd_params:  {dp}")
