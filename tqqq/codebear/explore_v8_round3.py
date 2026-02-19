#!/usr/bin/env python3
"""
Round 3: Fine-tune Combo2 (Sigmoid + Vol-aware) — the breakthrough candidate.
Also try combining with wider bands and other fine-tuning.
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tqqq_daily.csv')

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
    return df[['Close', 'High', 'Low']].dropna()

def compute_sma(prices, window):
    return prices.rolling(window).mean()

def compute_rsi(prices, period=10):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_weekly_ret(prices, period=5):
    return prices.pct_change(period)

def compute_realized_vol(prices, window=20):
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)

def sigmoid(x, center, steepness):
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

def calc_metrics(portfolio_values):
    pv = pd.Series(portfolio_values) if not isinstance(portfolio_values, pd.Series) else portfolio_values
    final_val = pv.iloc[-1]
    start_val = pv.iloc[0]
    years = len(pv) / 252.0
    cagr = (final_val / start_val) ** (1 / years) - 1
    peak = pv.cummax()
    dd = (pv - peak) / peak
    max_dd = dd.min()
    dr = pv.pct_change().dropna()
    rf = 0.045 / 252
    excess = dr - rf
    sharpe = (excess.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
    downside = dr[dr < 0].std()
    sortino = (excess.mean() / downside) * np.sqrt(252) if downside > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'final_val': final_val, 'cagr': cagr, 'max_dd': max_dd,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar}


def run_v5(data, initial_capital=10000):
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]; wret = weekly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False
        if not in_bear:
            target_pct = 0.80 if (not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15) else 1.00
        else:
            if not pd.isna(rsi) and rsi < 20: target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12: target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30: target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65: target_pct = 0.30
            else:
                cv = cash + shares * price
                target_pct = max((shares * price) / cv if cv > 0 else 0.30, 0.30)
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


def run_combo2(data, initial_capital=10000,
               bear_floor=0.25, bear_ceiling=0.95,
               rsi_center=30, rsi_steepness=-0.20,
               vol_high=0.65, vol_reduce=0.85,
               vol_window=20,
               bear_band=0.90, bull_band=1.05,
               crash_threshold=-0.12, crash_target=0.80):
    """Sigmoid bear + Vol-aware bull trim."""
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    vol = compute_realized_vol(prices, vol_window)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]; v = vol.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * bear_band: in_bear = True
            elif in_bear and price > sma * bull_band: in_bear = False
        if not in_bear:
            target_pct = 1.00
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            if not pd.isna(v) and v > vol_high:
                target_pct = min(target_pct, vol_reduce)
        else:
            if not pd.isna(rsi):
                sig_val = sigmoid(rsi, rsi_center, rsi_steepness)
                target_pct = bear_floor + (bear_ceiling - bear_floor) * sig_val
                if not pd.isna(wret) and wret < crash_threshold:
                    target_pct = max(target_pct, crash_target)
            else:
                target_pct = bear_floor
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


def main():
    data = load_data()
    v5_pv = run_v5(data)
    v5_m = calc_metrics(v5_pv)
    
    print(f"📊 Data: {data.index[0].date()} → {data.index[-1].date()}, {len(data)} rows")
    print(f"V5 Baseline: CAGR {v5_m['cagr']*100:.1f}%, MaxDD {v5_m['max_dd']*100:.1f}%, Sharpe {v5_m['sharpe']:.2f}, Calmar {v5_m['calmar']:.2f}, Sortino {v5_m['sortino']:.2f}")
    
    # ── Sweep 1: Vol threshold and reduce level ──
    print(f"\n{'='*100}")
    print(f"🔬 Sweep 1: Vol threshold (0.50-0.80) × Vol reduce level (0.75-0.95)")
    print(f"{'vol_high':<10} {'vol_red':<10} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Sortino':>8} {'Final$':>12} {'wins':>5}")
    
    for vh in [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]:
        for vr in [0.75, 0.80, 0.85, 0.90, 0.95]:
            pv = run_combo2(data, vol_high=vh, vol_reduce=vr)
            m = calc_metrics(pv)
            wins = 0
            if m['cagr'] - v5_m['cagr'] > 0.02: wins += 1
            if m['max_dd'] - v5_m['max_dd'] > 0.03: wins += 1
            if m['sharpe'] - v5_m['sharpe'] > 0.05: wins += 1
            if m['calmar'] - v5_m['calmar'] > 0.05: wins += 1
            marker = ' ★' if wins >= 2 else ''
            if m['calmar'] > v5_m['calmar'] - 0.02:
                print(f"{vh:<10} {vr:<10} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['sortino']:>7.2f} ${m['final_val']:>10,.0f} {wins:>4}{marker}")
    
    # ── Sweep 2: Vol window ──
    print(f"\n{'='*100}")
    print(f"🔬 Sweep 2: Vol window (10-60) with best vol_high/reduce from above")
    print(f"{'vol_win':<10} {'vol_high':<10} {'vol_red':<10} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    
    for vw in [10, 15, 20, 30, 40, 60]:
        for vh in [0.60, 0.65, 0.70]:
            for vr in [0.80, 0.85, 0.90]:
                pv = run_combo2(data, vol_high=vh, vol_reduce=vr, vol_window=vw)
                m = calc_metrics(pv)
                if m['calmar'] > v5_m['calmar'] + 0.03:
                    print(f"{vw:<10} {vh:<10} {vr:<10} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")
    
    # ── Sweep 3: Sigmoid params with best vol settings ──
    print(f"\n{'='*100}")
    print(f"🔬 Sweep 3: Sigmoid params with vol_high=0.65, vol_reduce=0.85")
    print(f"{'floor':<7} {'ceil':<7} {'ctr':<6} {'steep':<7} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    
    for floor in [0.15, 0.20, 0.25, 0.30]:
        for ceil in [0.85, 0.90, 0.95, 1.00]:
            for ctr in [25, 30, 35]:
                for steep in [-0.15, -0.20, -0.25, -0.30]:
                    pv = run_combo2(data, bear_floor=floor, bear_ceiling=ceil,
                                    rsi_center=ctr, rsi_steepness=steep,
                                    vol_high=0.65, vol_reduce=0.85)
                    m = calc_metrics(pv)
                    if m['calmar'] > 0.72:  # Only show good results
                        print(f"{floor:<7} {ceil:<7} {ctr:<6} {steep:<7} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")
    
    # ── Sweep 4: Hysteresis bands with vol overlay ──
    print(f"\n{'='*100}")
    print(f"🔬 Sweep 4: Bands + Vol overlay (best sigmoid params)")
    print(f"{'bear_b':<8} {'bull_b':<8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    
    for bb in [0.88, 0.89, 0.90, 0.91, 0.92, 0.93]:
        for blb in [1.03, 1.04, 1.05, 1.06, 1.07]:
            pv = run_combo2(data, bear_band=bb, bull_band=blb,
                           vol_high=0.65, vol_reduce=0.85)
            m = calc_metrics(pv)
            if m['calmar'] > 0.70:
                print(f"{bb:<8} {blb:<8} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")
    
    # ── Final best candidate ──
    print(f"\n{'='*100}")
    print(f"🏆 FINAL COMPARISON — Top Candidates vs V5")
    print(f"{'='*100}")
    print(f"{'Strategy':<40} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Sortino':>8} {'Final$':>12}")
    print('-' * 100)
    
    candidates = {
        'V5 Baseline':                  run_v5(data),
        'Combo2(vh=0.65,vr=0.85)':      run_combo2(data, vol_high=0.65, vol_reduce=0.85),
        'Combo2(vh=0.60,vr=0.85)':      run_combo2(data, vol_high=0.60, vol_reduce=0.85),
        'Combo2(vh=0.65,vr=0.80)':      run_combo2(data, vol_high=0.65, vol_reduce=0.80),
        'Combo2(vh=0.65,vr=0.90)':      run_combo2(data, vol_high=0.65, vol_reduce=0.90),
        'Combo2(vh=0.62,vr=0.85)':      run_combo2(data, vol_high=0.62, vol_reduce=0.85),
        'Combo2(vh=0.68,vr=0.85)':      run_combo2(data, vol_high=0.68, vol_reduce=0.85),
        'Combo2+bands92/105':           run_combo2(data, vol_high=0.65, vol_reduce=0.85, bear_band=0.92, bull_band=1.05),
        'Combo2+bands93/105':           run_combo2(data, vol_high=0.65, vol_reduce=0.85, bear_band=0.93, bull_band=1.05),
        'Combo2+sig(20/95/30/-0.25)':   run_combo2(data, vol_high=0.65, vol_reduce=0.85,
                                                     bear_floor=0.20, bear_ceiling=0.95, rsi_center=30, rsi_steepness=-0.25),
        'Combo2+sig(15/100/30/-0.30)':  run_combo2(data, vol_high=0.65, vol_reduce=0.85,
                                                     bear_floor=0.15, bear_ceiling=1.00, rsi_center=30, rsi_steepness=-0.30),
    }
    
    for name, pv in candidates.items():
        m = calc_metrics(pv)
        
        # Count wins vs v5
        wins = 0; details = []
        cd = m['cagr'] - v5_m['cagr']
        dd = m['max_dd'] - v5_m['max_dd']
        sd = m['sharpe'] - v5_m['sharpe']
        cld = m['calmar'] - v5_m['calmar']
        
        if cd > 0.02: wins += 1; details.append(f'CAGR+{cd*100:.1f}%')
        if dd > 0.03: wins += 1; details.append(f'DD+{dd*100:.1f}%')
        if sd > 0.05: wins += 1; details.append(f'Sh+{sd:.2f}')
        if cld > 0.05: wins += 1; details.append(f'Cal+{cld:.2f}')
        
        marker = f' ★ WINS({wins})' if wins >= 2 else f' [{wins}]'
        if name == 'V5 Baseline':
            marker = ' ★BASE'
        
        print(f"{name:<40} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['sortino']:>7.2f} ${m['final_val']:>10,.0f}{marker} {', '.join(details)}")


if __name__ == '__main__':
    main()
