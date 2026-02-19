"""
Sensitivity analysis for momentum parameters.
Tests different lookback weights and top-N to find robust parameters.
NOT overfitting - just finding the best simple parameter set.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from momentum_utils import (download_data, monthly_returns, momentum_score,
                             backtest_metrics, walk_forward_split, TRADE_COST, TICKERS)

def run_momentum_rp(prices, top_n=5, lookbacks=[1,3,6,12], weights=[0.25,0.25,0.25,0.25], vol_timing=False):
    """Generic risk-parity momentum with configurable params."""
    tradeable = [t for t in TICKERS if t != 'SHY' and t in prices.columns]
    all_tickers = tradeable + ['SHY']
    monthly_p = prices[all_tickers].resample('ME').last().dropna(how='all')
    
    scores = momentum_score(monthly_p[tradeable], lookbacks=lookbacks, weights=weights)
    ret = monthly_returns(prices[all_tickers].dropna(how='all'))
    rolling_vol = ret[all_tickers].rolling(6).std() * np.sqrt(12)
    
    if vol_timing:
        agg_vol = rolling_vol[tradeable].mean(axis=1)
        vol_median = agg_vol.expanding(min_periods=12).median()
    
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
        w = {}
        for t in selected:
            vol = rolling_vol.loc[sig_date, t] if sig_date in rolling_vol.index and t in rolling_vol.columns else 0.15
            if pd.isna(vol) or vol < 0.01:
                vol = 0.15
            w[t] = w.get(t, 0) + 1.0 / vol
        
        total_w = sum(w.values())
        if total_w > 0:
            w = {t: v/total_w for t, v in w.items()}
        
        # Vol timing
        if vol_timing and sig_date in agg_vol.index and sig_date in vol_median.index:
            cv = agg_vol.loc[sig_date]
            mv = vol_median.loc[sig_date]
            if not pd.isna(cv) and not pd.isna(mv) and cv > mv:
                shy_add = 0
                for t in list(w.keys()):
                    if t != 'SHY':
                        r = w[t] * 0.5
                        w[t] *= 0.5
                        shy_add += r
                w['SHY'] = w.get('SHY', 0) + shy_add
        
        month_ret = sum(w.get(t, 0) * ret.loc[date, t] for t in w if t in ret.columns and not pd.isna(ret.loc[date, t]))
        
        turnover = sum(abs(w.get(t, 0) - prev_holdings.get(t, 0)) for t in set(list(w.keys()) + list(prev_holdings.keys()))) / 2
        month_ret -= turnover * TRADE_COST
        prev_holdings = w.copy()
        
        strat_returns.append((date, month_ret))
    
    return pd.Series([r[1] for r in strat_returns], index=[r[0] for r in strat_returns])

def main():
    prices = download_data()
    print(f"Assets: {len(prices.columns)}, Date: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    
    # Parameter grid (small, to avoid overfitting)
    configs = [
        # (name, top_n, lookbacks, weights, vol_timing)
        ("RP-N3-equal", 3, [1,3,6,12], [0.25,0.25,0.25,0.25], False),
        ("RP-N4-equal", 4, [1,3,6,12], [0.25,0.25,0.25,0.25], False),
        ("RP-N5-equal", 5, [1,3,6,12], [0.25,0.25,0.25,0.25], False),
        ("RP-N3-short", 3, [1,3,6,12], [0.4,0.3,0.2,0.1], False),
        ("RP-N3-long",  3, [1,3,6,12], [0.1,0.2,0.3,0.4], False),
        ("RP-N3-mid",   3, [1,3,6,12], [0.2,0.4,0.3,0.1], False),
        ("RP-N3-eq+VT", 3, [1,3,6,12], [0.25,0.25,0.25,0.25], True),
        ("RP-N4-eq+VT", 4, [1,3,6,12], [0.25,0.25,0.25,0.25], True),
        ("RP-N3-short+VT", 3, [1,3,6,12], [0.4,0.3,0.2,0.1], True),
    ]
    
    results = []
    for name, n, lb, wt, vt in configs:
        sr = run_momentum_rp(prices, top_n=n, lookbacks=lb, weights=wt, vol_timing=vt)
        is_r, oos_r = walk_forward_split(sr)
        fm = backtest_metrics(sr)
        im = backtest_metrics(is_r)
        om = backtest_metrics(oos_r)
        
        deg = 1 - om.get('Sharpe',0)/im['Sharpe'] if im.get('Sharpe',0) > 0 else 999
        comp = fm.get('Sharpe',0)*0.4 + fm.get('Calmar',0)*0.4 + fm.get('CAGR',0)/100*0.2
        
        results.append({
            'name': name, **fm, 
            'OOS_Sharpe': om.get('Sharpe',0),
            'IS_Sharpe': im.get('Sharpe',0),
            'WF_deg': round(deg*100,1),
            'Composite': round(comp, 3)
        })
        print(f"{name:20s} CAGR={fm['CAGR']:6.2f}% MaxDD={fm['MaxDD']:7.2f}% Sharpe={fm['Sharpe']:5.2f} Calmar={fm['Calmar']:5.2f} OOS_S={om.get('Sharpe',0):5.2f} Comp={comp:.3f}")
    
    print("\n--- Sorted by Composite ---")
    results.sort(key=lambda x: -x['Composite'])
    for r in results:
        print(f"{r['name']:20s} Comp={r['Composite']:.3f} Sharpe={r['Sharpe']:.2f} Calmar={r['Calmar']:.2f} CAGR={r['CAGR']:.1f}%")

if __name__ == '__main__':
    main()
