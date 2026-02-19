"""
Beast Rotation v2 — 波动率感知降仓优化
在 v1 基础上叠加波动率分档降仓，降低极端回撤

v2a（保守）: vol阈值 [25,35,45], 降仓系数 [0.85,0.7,0.5]
v2b（激进）: vol阈值 [20,30,40], 降仓系数 [0.9,0.8,0.7]
v2c（动态最低仓）: v2b + RSI动态最低仓位

代码熊 🐻 2026-02-19
"""

import pandas as pd
import numpy as np
import os

def load(ticker, data_dir):
    path = os.path.join(data_dir, f'{ticker}.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    return df['Close'].dropna()

def run_beast_rotation_v2(df_all, params, vol_params, cost_per_side=0.0005, slippage=0.001):
    """
    vol_params: dict with keys:
      - vol_thresholds: list of 3 vol levels (ascending)
      - vol_scales: list of 3 scale factors for each band
      - min_pos_bull: minimum position in bull regime after vol scaling
      - min_pos_bear: minimum position in bear regime after vol scaling
      - dynamic_min: bool, if True use RSI to adjust min position
    """
    tickers = list(df_all.columns)
    n = len(df_all)
    sma_p = params['sma_period']
    
    # Precompute
    sma200 = df_all.rolling(sma_p).mean()
    daily_ret = df_all.pct_change()
    
    # Blended momentum
    mom = pd.DataFrame(index=df_all.index)
    for t in tickers:
        m = 0.25 * df_all[t].pct_change(21) + 0.40 * df_all[t].pct_change(63) + 0.35 * df_all[t].pct_change(126)
        mom[t] = m
    
    # RSI
    rsi_all = pd.DataFrame(index=df_all.index)
    for t in tickers:
        delta = df_all[t].diff()
        gain = delta.clip(lower=0).rolling(10).mean()
        loss = (-delta.clip(upper=0)).rolling(10).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi_all[t] = 100 - 100 / (1 + rs)
    
    # Broad market
    avg_price = df_all.mean(axis=1)
    avg_sma = avg_price.rolling(sma_p).mean()
    
    # Realized vol on portfolio proxy (equal-weight avg daily return)
    avg_daily_ret = daily_ret.mean(axis=1)
    realized_vol = avg_daily_ret.rolling(20).std() * np.sqrt(252)
    
    bull_thr = params['bull_enter']
    bear_thr = params['bear_enter']
    bear_floor = params['bear_floor']
    rebal_freq = params['rebalance_freq']
    
    vt = vol_params['vol_thresholds']
    vs = vol_params['vol_scales']
    min_bull = vol_params['min_pos_bull']
    min_bear = vol_params['min_pos_bear']
    dynamic_min = vol_params.get('dynamic_min', False)
    
    holdings = {t: 0.0 for t in tickers}
    regime = 1
    portfolio_ret = np.zeros(n)
    
    for i in range(sma_p, n):
        if i > sma_p:
            day_ret = sum(holdings[t] * daily_ret[t].iloc[i] for t in tickers)
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
        
        new_holdings = {t: 0.0 for t in tickers}
        
        if regime == 1:
            valid = {}
            for t in tickers:
                m = mom[t].iloc[i]
                if not np.isnan(m):
                    valid[t] = m
            if valid:
                sorted_t = sorted(valid.keys(), key=lambda x: valid[x], reverse=True)
                if len(sorted_t) >= 2 and valid[sorted_t[1]] > 0:
                    new_holdings[sorted_t[0]] = 0.70
                    new_holdings[sorted_t[1]] = 0.30
                else:
                    new_holdings[sorted_t[0]] = 1.0
        else:
            valid = {}
            for t in tickers:
                m = mom[t].iloc[i]
                if not np.isnan(m):
                    valid[t] = m
            if valid:
                sorted_t = sorted(valid.keys(), key=lambda x: valid[x], reverse=True)
                best = sorted_t[0]
                pos = bear_floor
                rsi_val = rsi_all[best].iloc[i]
                if not np.isnan(rsi_val):
                    if rsi_val < 20:
                        pos = 0.80
                    elif rsi_val < 30:
                        pos = 0.60
                    elif rsi_val > 65:
                        pos = bear_floor
                new_holdings[best] = pos
        
        # --- Vol-aware scaling ---
        vol_now = realized_vol.iloc[i]
        if not np.isnan(vol_now):
            if vol_now >= vt[2]:
                scale = vs[2]
            elif vol_now >= vt[1]:
                scale = vs[1]
            elif vol_now >= vt[0]:
                scale = vs[0]
            else:
                scale = 1.0
            
            # Apply scale
            for t in tickers:
                new_holdings[t] *= scale
            
            # Enforce minimum position
            total_pos = sum(new_holdings[t] for t in tickers)
            if total_pos > 0:
                # Determine min floor
                if dynamic_min:
                    # Use best ticker's RSI to raise min in oversold
                    best_t = max(new_holdings, key=new_holdings.get)
                    rsi_v = rsi_all[best_t].iloc[i]
                    if not np.isnan(rsi_v) and rsi_v < 30:
                        cur_min = 0.30 if regime == 1 else 0.25
                    else:
                        cur_min = min_bull if regime == 1 else min_bear
                else:
                    cur_min = min_bull if regime == 1 else min_bear
                
                if total_pos < cur_min:
                    # Scale up proportionally to meet minimum
                    factor = cur_min / total_pos
                    for t in tickers:
                        new_holdings[t] *= factor
        
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
    
    base_params = {'sma_period': 200, 'bull_enter': 1.05, 'bear_enter': 0.90, 'bear_floor': 0.30, 'rebalance_freq': 5}
    
    vol_variants = {
        'v2a_conservative': {
            'vol_thresholds': [0.75, 0.95, 1.15],
            'vol_scales': [0.85, 0.70, 0.50],
            'min_pos_bull': 0.20,
            'min_pos_bear': 0.15,
            'dynamic_min': False,
        },
        'v2b_balanced': {
            'vol_thresholds': [0.50, 0.80, 1.00],
            'vol_scales': [0.90, 0.75, 0.55],
            'min_pos_bull': 0.25,
            'min_pos_bear': 0.15,
            'dynamic_min': False,
        },
        'v2c_dynamic': {
            'vol_thresholds': [0.50, 0.80, 1.00],
            'vol_scales': [0.90, 0.75, 0.55],
            'min_pos_bull': 0.25,
            'min_pos_bear': 0.15,
            'dynamic_min': True,
        },
    }
    
    sma_p = base_params['sma_period']
    
    print(f"\n{'Name':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>4} {'Composite':>9}")
    print("-"*95)
    
    results = {}
    
    for vname, vp in vol_variants.items():
        # Full period
        ret = run_beast_rotation_v2(df_all, base_params, vp)
        m = calc_metrics(ret[sma_p:])
        
        # IS
        m_is = calc_metrics(ret[sma_p:split])
        
        # OOS
        start_oos = max(0, split - sma_p)
        data_oos = {t: df_all[t].iloc[start_oos:] for t in tickers}
        df_oos = pd.DataFrame(data_oos)
        ret_oos = run_beast_rotation_v2(df_oos, base_params, vp)
        oos_offset = split - start_oos
        m_oos = calc_metrics(ret_oos[oos_offset:])
        
        if not m: continue
        
        wf = '✅' if m_oos.get('Sharpe',0) >= m_is.get('Sharpe',0) * 0.7 else '❌'
        score = m['Sharpe']*0.4 + m['Calmar']*0.4 + m['CAGR']*0.2
        
        results[vname] = {**m, 'IS_Sharpe': m_is.get('Sharpe',0), 'OOS_Sharpe': m_oos.get('Sharpe',0), 'WF': wf, 'Composite': score}
        
        print(f"{vname:<22} {m['CAGR']:>6.1%} {m['MaxDD']:>7.1%} {m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} | {m_is.get('Sharpe',0):>6.2f} {m_oos.get('Sharpe',0):>7.2f} {wf:>4} {score:>9.3f}")
    
    # v1 baseline for comparison
    print(f"\n--- v1 baseline ---")
    from beast_rotation_v1 import run_beast_rotation
    ret_v1 = run_beast_rotation(df_all, base_params)
    m_v1 = calc_metrics(ret_v1[sma_p:])
    score_v1 = m_v1['Sharpe']*0.4 + m_v1['Calmar']*0.4 + m_v1['CAGR']*0.2
    print(f"{'v1_base':<22} {m_v1['CAGR']:>6.1%} {m_v1['MaxDD']:>7.1%} {m_v1['Sharpe']:>7.2f} {m_v1['Calmar']:>7.2f} | {'':>6} {'':>7} {'':>4} {score_v1:>9.3f}")
    
    # Best selection
    print(f"\n--- Best v2 ---")
    best_name = max(results, key=lambda k: results[k]['Composite'] if results[k]['MaxDD'] > -0.45 else results[k]['Composite'] - 1)
    b = results[best_name]
    print(f"Winner: {best_name}")
    print(f"  CAGR={b['CAGR']:.1%}, MaxDD={b['MaxDD']:.1%}, Sharpe={b['Sharpe']:.2f}, Calmar={b['Calmar']:.2f}, Composite={b['Composite']:.3f}")
    print(f"  WF: IS={b['IS_Sharpe']:.2f}, OOS={b['OOS_Sharpe']:.2f}, Pass={b['WF']}")
    meets_target = b['MaxDD'] > -0.45 and b['CAGR'] > 0.35
    print(f"  Target met (MaxDD<-45%, CAGR>35%): {'✅' if meets_target else '❌'}")
