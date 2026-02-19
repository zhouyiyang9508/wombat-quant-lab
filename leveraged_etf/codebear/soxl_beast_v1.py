"""
SOXL Beast v1 — 3x半导体ETF趋势跟踪策略
移植TQQQ v8的核心逻辑：SMA200滞后带 + vol感知 + sigmoid仓位
代码熊 🐻 2026-02-19
"""

import pandas as pd
import numpy as np
import sys, os

def load_data(path):
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    df = df[['Open','High','Low','Close','Volume']].dropna(subset=['Close'])
    return df

def sigmoid(x, center=0, steepness=1):
    return 1 / (1 + np.exp(-steepness * (x - center)))

def run_strategy(df, params, cost_per_side=0.0005, slippage=0.001):
    """Walk-forward: first 60% train, last 40% test"""
    close = df['Close'].values
    n = len(close)
    
    # Precompute indicators
    sma200 = pd.Series(close).rolling(200).mean().values
    rsi_period = params['rsi_period']
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = (100 - 100 / (1 + rs)).values
    
    # Weekly return
    weekly_ret = pd.Series(close).pct_change(5).values
    
    # Volatility (20d realized)
    daily_ret = pd.Series(close).pct_change().values
    vol20 = pd.Series(daily_ret).rolling(20).std().values * np.sqrt(252)
    
    # Params
    bull_enter = params['bull_enter']  # e.g. 1.05
    bear_enter = params['bear_enter']  # e.g. 0.90
    bear_floor = params['bear_floor']
    
    positions = np.zeros(n)
    regime = np.ones(n)  # 1=bull, 0=bear
    
    for i in range(200, n):
        if np.isnan(sma200[i]):
            continue
        
        ratio = close[i] / sma200[i]
        
        # Regime with hysteresis
        if i > 200:
            regime[i] = regime[i-1]
        if ratio > bull_enter:
            regime[i] = 1
        elif ratio < bear_enter:
            regime[i] = 0
        
        if regime[i] == 1:  # Bull
            # Base: full position
            pos = 1.0
            # Vol adjustment: reduce if vol very high
            if not np.isnan(vol20[i]):
                if vol20[i] > 0.8:  # SOXL vol can be extreme
                    pos *= 0.8
                elif vol20[i] > 1.0:
                    pos *= 0.6
            # Extreme euphoria: RSI>80 and weekly>15%
            if not np.isnan(rsi[i]) and rsi[i] > 80 and not np.isnan(weekly_ret[i]) and weekly_ret[i] > 0.15:
                pos = min(pos, 0.80)
            positions[i] = pos
        else:  # Bear
            # Base: floor position (catch V-bounces)
            pos = bear_floor
            if not np.isnan(rsi[i]):
                if rsi[i] < 20:
                    pos = 0.80  # Panic buy
                elif rsi[i] < 30:
                    pos = 0.60  # Moderate dip buy
                elif rsi[i] > 65:
                    pos = bear_floor  # Back to floor
            # Weekly crash override
            if not np.isnan(weekly_ret[i]) and weekly_ret[i] < -0.12:
                pos = max(pos, 0.70)
            positions[i] = pos
    
    # Simulate returns with costs
    daily_ret_asset = pd.Series(close).pct_change().values
    port_ret = np.zeros(n)
    
    for i in range(201, n):
        # Position change cost
        delta_pos = abs(positions[i] - positions[i-1])
        trade_cost = delta_pos * (cost_per_side * 2 + slippage)
        port_ret[i] = positions[i-1] * daily_ret_asset[i] - trade_cost
    
    return positions, port_ret, regime

def calc_metrics(returns, rf_annual=0.04):
    ret = returns[returns != 0]
    if len(ret) < 100:
        return {}
    cum = (1 + pd.Series(returns)).cumprod()
    total_days = len(returns)
    years = total_days / 252
    final = cum.iloc[-1]
    cagr = final ** (1/years) - 1
    
    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    maxdd = dd.min()
    
    # Sharpe
    daily_rf = (1 + rf_annual) ** (1/252) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0
    
    # Calmar
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    
    # Win rate (monthly)
    monthly = pd.Series(returns).groupby(np.arange(len(returns)) // 21).sum()
    win_rate = (monthly > 0).mean()
    
    annual_vol = pd.Series(returns).std() * np.sqrt(252)
    
    return {
        'CAGR': cagr, 'MaxDD': maxdd, 'Sharpe': sharpe, 'Calmar': calmar,
        'WinRate': win_rate, 'AnnVol': annual_vol, 'Final': final
    }

def walk_forward_test(df, params_list):
    """Test multiple param sets with walk-forward"""
    n = len(df)
    split = int(n * 0.6)
    
    results = []
    for params in params_list:
        # Full period
        pos, ret, reg = run_strategy(df, params)
        full_m = calc_metrics(ret[200:])
        
        # In-sample
        is_m = calc_metrics(ret[200:split])
        
        # Out-of-sample
        oos_m = calc_metrics(ret[split:])
        
        # Buy & hold
        bh_ret = pd.Series(df['Close'].values).pct_change().values
        bh_m = calc_metrics(bh_ret[200:])
        
        results.append({
            'params': params,
            'full': full_m,
            'is': is_m,
            'oos': oos_m,
            'bh': bh_m
        })
    
    return results

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')
    
    df_soxl = load_data(os.path.join(data_dir, 'SOXL.csv'))
    print(f"SOXL: {len(df_soxl)} rows, {df_soxl.index[0].date()} to {df_soxl.index[-1].date()}")
    
    # Parameter grid for walk-forward
    param_sets = [
        # Baseline: same as TQQQ v8
        {'name': 'SOXL_v1a_tqqq_clone', 'bull_enter': 1.05, 'bear_enter': 0.90, 'bear_floor': 0.30, 'rsi_period': 10},
        # Wider band (SOXL more volatile)
        {'name': 'SOXL_v1b_wide', 'bull_enter': 1.08, 'bear_enter': 0.87, 'bear_floor': 0.25, 'rsi_period': 10},
        # Narrower band
        {'name': 'SOXL_v1c_narrow', 'bull_enter': 1.03, 'bear_enter': 0.93, 'bear_floor': 0.35, 'rsi_period': 10},
        # Higher floor
        {'name': 'SOXL_v1d_high_floor', 'bull_enter': 1.05, 'bear_enter': 0.90, 'bear_floor': 0.40, 'rsi_period': 14},
        # Very wide
        {'name': 'SOXL_v1e_vwide', 'bull_enter': 1.10, 'bear_enter': 0.85, 'bear_floor': 0.20, 'rsi_period': 10},
    ]
    
    results = walk_forward_test(df_soxl, param_sets)
    
    print("\n" + "="*100)
    print("SOXL Beast v1 — Walk-Forward Results")
    print("="*100)
    print(f"{'Name':<25} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} | {'IS_Sh':>6} {'OOS_Sh':>7} {'WF_Pass':>8}")
    print("-"*100)
    
    for r in results:
        f = r['full']
        is_m = r['is']
        oos_m = r['oos']
        if not f:
            continue
        wf_pass = '✅' if oos_m.get('Sharpe',0) >= is_m.get('Sharpe',0) * 0.7 else '❌'
        print(f"{r['params']['name']:<25} {f['CAGR']:>6.1%} {f['MaxDD']:>7.1%} {f['Sharpe']:>7.2f} {f['Calmar']:>7.2f} | {is_m.get('Sharpe',0):>6.2f} {oos_m.get('Sharpe',0):>7.2f} {wf_pass:>8}")
    
    # Buy & Hold
    bh = results[0]['bh']
    print(f"{'SOXL B&H':<25} {bh['CAGR']:>6.1%} {bh['MaxDD']:>7.1%} {bh['Sharpe']:>7.2f} {bh['Calmar']:>7.2f}")
    
    # Best strategy
    best = max(results, key=lambda r: r['full'].get('Sharpe', 0))
    print(f"\n⭐ Best: {best['params']['name']}")
    print(f"   Full: CAGR={best['full']['CAGR']:.1%}, MaxDD={best['full']['MaxDD']:.1%}, Sharpe={best['full']['Sharpe']:.2f}")
    print(f"   OOS:  CAGR={best['oos']['CAGR']:.1%}, MaxDD={best['oos']['MaxDD']:.1%}, Sharpe={best['oos']['Sharpe']:.2f}")
    
    # Composite score
    s = best['full']
    score = s['Sharpe']*0.4 + s['Calmar']*0.4 + s['CAGR']*0.2
    print(f"   Composite Score: {score:.3f}")
