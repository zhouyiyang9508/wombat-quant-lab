"""
stable_v3.py — Concentrated Dual Momentum with Regime Overlay
代码熊 🐻 | 2026-02-19

核心: Antonacci-style dual momentum, but 3 assets
- 每月选最强1-2资产，集中持仓
- 绝对动量滤波: 只在全部资产负动量时转现金
- Regime: 用 QQQ SMA200 判断风险偏好
  - Risk-on: 倾向QQQ
  - Risk-off: 倾向TLT/GLD

目标: CAGR 15%+, MaxDD <20%, Sharpe >1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"

COMMISSION = 0.0005
SLIPPAGE = 0.001
RISK_FREE = 0.04
INITIAL = 10000
CASH_DAILY = 0.04 / 252

# Parameters
MOM_FAST = 63    # 3M momentum
MOM_SLOW = 252   # 12M momentum
MOM_BLEND = 0.5  # weight on fast vs slow


def load_data():
    assets = {}
    for t in ['QQQ', 'TLT', 'GLD']:
        df = pd.read_csv(DATA_DIR / f"{t}.csv", parse_dates=['Date'])
        df = df.set_index('Date').sort_index()
        assets[t] = df['Close']
    prices = pd.DataFrame(assets).dropna()
    returns = prices.pct_change().dropna()
    return prices, returns


def momentum(series, lookback):
    return series.iloc[-1] / series.iloc[-lookback] - 1 if len(series) >= lookback else 0


def backtest(prices, returns, start_idx=252):
    tickers = ['QQQ', 'TLT', 'GLD']
    dates = prices.index[start_idx:]
    
    holdings = {t: 0.0 for t in tickers}
    holdings['CASH'] = INITIAL
    last_month = None
    pv_list = []
    costs = 0
    weight_hist = []
    
    for i, date in enumerate(dates):
        idx = start_idx + i
        
        if i > 0:
            for t in tickers:
                r = returns[t].iloc[idx-1] if idx-1 < len(returns) else 0
                holdings[t] *= (1 + r)
            holdings['CASH'] *= (1 + CASH_DAILY)
        
        total = sum(holdings.values())
        month = (date.year, date.month)
        
        if month != last_month and idx >= MOM_SLOW:
            last_month = month
            
            # Blended momentum scores
            scores = {}
            for t in tickers:
                p = prices[t].iloc[:idx+1]
                m_fast = momentum(p, MOM_FAST)
                m_slow = momentum(p, MOM_SLOW)
                scores[t] = MOM_BLEND * m_fast + (1 - MOM_BLEND) * m_slow
            
            # Regime: QQQ above SMA200?
            qqq_sma = prices['QQQ'].iloc[idx-200:idx].mean()
            risk_on = prices['QQQ'].iloc[idx] > qqq_sma
            
            # Sort by score
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Absolute momentum: is the best asset positive?
            best_t, best_s = ranked[0]
            second_t, second_s = ranked[1]
            
            target = {t: 0.0 for t in tickers}
            target['CASH'] = 0.0
            
            if best_s <= 0 and second_s <= 0:
                # All weak → defensive: small GLD + cash
                target['GLD'] = 0.3
                target['TLT'] = 0.2
                target['CASH'] = 0.5
            elif risk_on:
                # Risk-on: concentrated in top 1-2
                if best_s > 0:
                    target[best_t] = 0.70
                if second_s > 0:
                    target[second_t] = 0.30
                else:
                    target[best_t] = 0.85
                    target['CASH'] = 0.15
            else:
                # Risk-off: favor safety assets
                if best_t in ['TLT', 'GLD']:
                    target[best_t] = 0.60
                    if second_s > 0:
                        target[second_t] = 0.25
                    target['CASH'] = 1.0 - sum(target.values())
                else:
                    # QQQ is strongest but risk-off → reduce
                    target['QQQ'] = 0.35
                    # Pick best safe asset
                    safe_scores = {t: scores[t] for t in ['TLT', 'GLD']}
                    best_safe = max(safe_scores, key=safe_scores.get)
                    target[best_safe] = 0.40
                    target['CASH'] = 0.25
            
            # Ensure weights sum to ~1
            s = sum(target.values())
            if s > 0 and abs(s - 1.0) > 0.01:
                for k in target:
                    target[k] /= s
            
            # Apply costs
            for t in tickers:
                cw = holdings[t] / total if total > 0 else 0
                turn = abs(target[t] - cw)
                c = turn * total * (COMMISSION + SLIPPAGE)
                costs += c
                total -= c
            
            for t in tickers:
                holdings[t] = total * target[t]
            holdings['CASH'] = total * target.get('CASH', 0)
            weight_hist.append({'Date': date, **target})
        
        pv_list.append(sum(holdings.values()))
    
    return pd.Series(pv_list, index=dates), weight_hist, costs


def met(pv, label=""):
    r = pv.pct_change().dropna()
    y = len(r)/252
    cagr = (pv.iloc[-1]/pv.iloc[0])**(1/y)-1
    vol = r.std()*np.sqrt(252)
    sh = (cagr-RISK_FREE)/vol if vol>0 else 0
    dd = (pv/pv.cummax()-1).min()
    cal = cagr/abs(dd) if dd!=0 else 0
    mr = pv.resample('ME').last().pct_change().dropna()
    return {'l':label,'f':pv.iloc[-1],'c':cagr,'v':vol,'s':sh,'d':dd,'k':cal,'w':(mr>0).mean()}

def pr(m):
    return (f"  {m['l']:28s} | ${m['f']:>10,.0f} | CAGR {m['c']:6.1%} | Vol {m['v']:5.1%} | "
            f"MaxDD {m['d']:6.1%} | Sharpe {m['s']:5.2f} | Calmar {m['k']:5.2f} | WR {m['w']:5.1%}")

def bh6040(prices, si=252):
    r = prices[['QQQ','TLT']].pct_change().iloc[si:]
    return INITIAL*(1+r['QQQ']*0.6+r['TLT']*0.4).cumprod()

def main():
    prices, returns = load_data()
    n = len(prices); sp = int(n*0.6)
    print(f"Data: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Split: train→{prices.index[sp-1].date()} | test {prices.index[sp].date()}→\n")
    
    pv, wh, costs = backtest(prices, returns, 252)
    bh = bh6040(prices)
    
    mf = met(pv,"Stable v3 (Full)"); mb = met(bh,"60/40 B&H")
    print("=== FULL ==="); print(pr(mf)); print(pr(mb))
    
    te = prices.index[sp]
    pvtr = pv.loc[:te]; pvte = pv.loc[te:]; pvte = pvte/pvte.iloc[0]*INITIAL
    bhte = bh.loc[te:]; bhte = bhte/bhte.iloc[0]*INITIAL
    
    mtr = met(pvtr,"v3 Train"); mte = met(pvte,"v3 Test")
    print("\n=== TRAIN ==="); print(pr(mtr))
    print("\n=== TEST (OOS) ==="); print(pr(mte)); print(pr(met(bhte,"60/40 Test")))
    
    deg = (mtr['s']-mte['s'])/abs(mtr['s'])*100 if mtr['s']!=0 else 0
    print(f"\nOverfit: {mtr['s']:.2f}→{mte['s']:.2f} Δ{deg:.0f}% {'✅' if abs(deg)<30 else '⚠️'}")
    score = mf['s']*0.4+mf['k']*0.4+mf['c']*0.2
    print(f"Score: {score:.3f} | Costs: ${costs:,.0f}")
    
    wdf = pd.DataFrame(wh)
    print("\nWeights:")
    for c in ['QQQ','TLT','GLD','CASH']:
        if c in wdf: print(f"  {c}: {wdf[c].mean():.1%}")

if __name__ == "__main__":
    main()
