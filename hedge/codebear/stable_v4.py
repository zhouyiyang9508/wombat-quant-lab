"""
stable_v4.py — Protective Momentum Allocation (PMA)
代码熊 🐻 | 2026-02-19

灵感: Keller's PAA + Antonacci Dual Momentum + Drawdown Control

资产: QQQ (equity), TLT (bonds), GLD (gold)
安全资产: TLT (flight to quality)

核心:
1. Breadth momentum: 计算几个资产的13W SMA趋势方向（上/下）
2. 如果多数趋势向下 → 防御模式（重仓TLT+GLD）
3. 如果趋势向上 → 进攻模式（按动量排名加权）
4. 月度再平衡 + 仓位平滑（避免whipsaw）
5. Drawdown protection: 回撤>8%时自动降仓

关键: 通过SMA breadth而非单一指标判断regime，减少假信号
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"
COMMISSION = 0.0005; SLIPPAGE = 0.001; RF = 0.04; INIT = 10000; CASH_D = RF/252

# --- PARAMETER GRID (for sensitivity analysis) ---
DEFAULT_PARAMS = {
    'sma_period': 65,        # ~13 weeks SMA for breadth
    'mom_lookback': 126,     # 6M momentum
    'mom_lookback2': 252,    # 12M momentum
    'mom_w1': 0.6,           # weight on 6M
    'offense_top1': 0.55,    # top asset weight in offense
    'offense_top2': 0.30,    # 2nd asset weight
    'offense_top3': 0.15,    # 3rd asset weight
    'defense_tlt': 0.45,     # TLT weight in defense
    'defense_gld': 0.35,     # GLD weight in defense  
    'defense_cash': 0.20,    # Cash in defense
    'dd_start': -0.08,       # Start reducing risk
    'dd_max': -0.18,         # Maximum reduction
    'smooth': 0.5,           # Weight smoothing (0=no smooth, 1=full smooth)
}


def load():
    a = {}
    for t in ['QQQ','TLT','GLD']:
        df = pd.read_csv(DATA_DIR/f"{t}.csv", parse_dates=['Date']).set_index('Date').sort_index()
        a[t] = df['Close']
    p = pd.DataFrame(a).dropna()
    return p, p.pct_change().dropna()


def backtest(prices, returns, params=None, start_idx=260):
    P = params or DEFAULT_PARAMS
    tickers = ['QQQ','TLT','GLD']
    dates = prices.index[start_idx:]
    
    hold = {t:0 for t in tickers}; hold['CASH'] = INIT
    prev_target = None
    last_m = None; pvl = []; costs = 0; wh = []; peak = INIT
    
    for i, dt in enumerate(dates):
        idx = start_idx + i
        if i > 0:
            for t in tickers:
                r = returns[t].iloc[idx-1] if idx-1 < len(returns) else 0
                hold[t] *= (1+r)
            hold['CASH'] *= (1+CASH_D)
        
        total = sum(hold.values())
        if total > peak: peak = total
        
        m = (dt.year, dt.month)
        if m != last_m and idx >= P['mom_lookback2']:
            last_m = m
            
            # 1. Breadth: how many assets above their SMA?
            n_above = 0
            for t in tickers:
                sma = prices[t].iloc[idx-P['sma_period']:idx].mean()
                if prices[t].iloc[idx] > sma:
                    n_above += 1
            
            # 2. Momentum scores (blended 6M + 12M)
            scores = {}
            for t in tickers:
                m6 = prices[t].iloc[idx] / prices[t].iloc[idx-P['mom_lookback']] - 1
                m12 = prices[t].iloc[idx] / prices[t].iloc[idx-P['mom_lookback2']] - 1
                scores[t] = P['mom_w1'] * m6 + (1-P['mom_w1']) * m12
            
            ranked = sorted(tickers, key=lambda t: scores[t], reverse=True)
            
            # 3. Determine regime and weights
            target = {t: 0 for t in tickers}; target['CASH'] = 0
            
            if n_above >= 2:
                # OFFENSE: most assets trending up
                wts = [P['offense_top1'], P['offense_top2'], P['offense_top3']]
                for j, t in enumerate(ranked):
                    # Only allocate if momentum is positive
                    if scores[t] > 0:
                        target[t] = wts[j]
                    else:
                        target['CASH'] += wts[j]
            elif n_above == 1:
                # MIXED: only 1 asset trending up
                # Top asset gets moderate allocation
                if scores[ranked[0]] > 0:
                    target[ranked[0]] = 0.40
                target['TLT'] = max(target.get('TLT',0), P['defense_tlt'] * 0.6)
                target['GLD'] = max(target.get('GLD',0), P['defense_gld'] * 0.6)
                target['CASH'] = 1.0 - sum(target[t] for t in tickers)
            else:
                # DEFENSE: all trending down
                target['TLT'] = P['defense_tlt']
                target['GLD'] = P['defense_gld']
                target['CASH'] = P['defense_cash']
            
            # Normalize
            s = sum(target.values())
            if s > 0:
                for k in target: target[k] /= s
            
            # 4. Drawdown protection
            dd = (total - peak) / peak
            if dd < P['dd_start']:
                scale = max(0.3, 1.0 + (dd - P['dd_start']) / (P['dd_start'] - P['dd_max']))
                for t in tickers:
                    freed = target[t] * (1 - scale)
                    target[t] *= scale
                    target['CASH'] += freed
            
            # 5. Smooth weights (reduce turnover)
            if prev_target is not None and P['smooth'] > 0:
                sm = P['smooth']
                for k in target:
                    target[k] = sm * prev_target.get(k,0) + (1-sm) * target[k]
            prev_target = target.copy()
            
            # Execute
            for t in tickers:
                cw = hold[t]/total if total>0 else 0
                c = abs(target[t]-cw)*total*(COMMISSION+SLIPPAGE)
                costs += c; total -= c
            for t in tickers: hold[t] = total*target[t]
            hold['CASH'] = total*target.get('CASH',0)
            wh.append({'Date':dt,**target})
        
        pvl.append(sum(hold.values()))
    
    return pd.Series(pvl, index=dates), wh, costs


def met(pv, label=""):
    r = pv.pct_change().dropna(); y = len(r)/252
    cagr = (pv.iloc[-1]/pv.iloc[0])**(1/y)-1
    vol = r.std()*np.sqrt(252)
    sh = (cagr-RF)/vol if vol>0 else 0
    dd = (pv/pv.cummax()-1).min()
    cal = cagr/abs(dd) if dd!=0 else 0
    mr = pv.resample('ME').last().pct_change().dropna()
    return {'l':label,'f':pv.iloc[-1],'c':cagr,'v':vol,'s':sh,'d':dd,'k':cal,'w':(mr>0).mean()}

def pr(m):
    return (f"  {m['l']:30s} | ${m['f']:>10,.0f} | CAGR {m['c']:6.1%} | Vol {m['v']:5.1%} | "
            f"MaxDD {m['d']:6.1%} | Sharpe {m['s']:5.2f} | Calmar {m['k']:5.2f} | WR {m['w']:5.1%}")

def bh(prices, si=260):
    r = prices[['QQQ','TLT']].pct_change().iloc[si:]
    return INIT*(1+r['QQQ']*0.6+r['TLT']*0.4).cumprod()


def sensitivity_test(prices, returns):
    """Test parameter sensitivity to check robustness."""
    print("\n=== SENSITIVITY ANALYSIS ===")
    base = DEFAULT_PARAMS.copy()
    
    tests = {
        'sma_period': [45, 55, 65, 75, 85],
        'mom_lookback': [63, 90, 126, 168, 200],
        'smooth': [0, 0.3, 0.5, 0.7],
        'offense_top1': [0.45, 0.55, 0.65, 0.75],
    }
    
    for param, values in tests.items():
        print(f"\n  {param}:")
        for v in values:
            p = base.copy(); p[param] = v
            pv, _, _ = backtest(prices, returns, p)
            m = met(pv)
            marker = " ←base" if v == base[param] else ""
            print(f"    {v:>6} → Sharpe {m['s']:5.2f} | CAGR {m['c']:5.1%} | MaxDD {m['d']:6.1%}{marker}")


def main():
    prices, returns = load()
    n = len(prices); sp = int(n*0.6)
    print(f"Data: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Split: train→{prices.index[sp-1].date()} | test {prices.index[sp].date()}→\n")
    
    pv, wh, costs = backtest(prices, returns)
    bhp = bh(prices)
    
    print("=== FULL ===")
    mf = met(pv,"Stable v4 PMA (Full)"); mb = met(bhp,"60/40 B&H")
    print(pr(mf)); print(pr(mb))
    
    te = prices.index[sp]
    pvtr = pv.loc[:te]
    pvte = pv.loc[te:]; pvte = pvte/pvte.iloc[0]*INIT
    bhte = bhp.loc[te:]; bhte = bhte/bhte.iloc[0]*INIT
    
    mtr = met(pvtr,"v4 Train"); mte = met(pvte,"v4 Test (OOS)")
    print("\n=== TRAIN ==="); print(pr(mtr))
    print("\n=== TEST (OOS) ==="); print(pr(mte)); print(pr(met(bhte,"60/40 Test")))
    
    deg = (mtr['s']-mte['s'])/abs(mtr['s'])*100 if mtr['s']!=0 else 0
    print(f"\nOverfit: {mtr['s']:.2f}→{mte['s']:.2f} Δ{deg:.0f}% {'✅' if abs(deg)<30 else '⚠️'}")
    score = mf['s']*0.4+mf['k']*0.4+mf['c']*0.2
    print(f"Composite: {score:.3f} | Costs: ${costs:,.0f}")
    
    wdf = pd.DataFrame(wh)
    print("\nWeights:")
    for c in ['QQQ','TLT','GLD','CASH']:
        if c in wdf: print(f"  {c}: {wdf[c].mean():.1%}")
    
    # Sensitivity
    sensitivity_test(prices, returns)


if __name__ == "__main__":
    main()
