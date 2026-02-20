#!/usr/bin/env python3
"""
动量轮动 v3c — 5-Sector Diversified + SMA50
代码熊 🐻

最低回撤版本: MaxDD -14.7% (vs v2d -21.9%, v3b -17.7%)

核心创新: 5行业分散 + SMA50趋势过滤
Bull: 5行业 × 3股 = 15只, 100% invested
Bear: 4行业 × 2股 = 8只, 85% invested

Results:
  Full (2015-2025): CAGR 22.4%, Sharpe 1.27, MaxDD -14.7%, Calmar 1.52
  IS: 1.22  OOS: 1.22  WF: 1.00 ✅ (完美WF!)
  Turnover: 62.7%  Composite: 1.162
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

def load_csv(fp):
    df = pd.read_csv(fp)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']); df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index); df = df.sort_index()
    for c in ['Close','Volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def load_all_data(tickers):
    d = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500: continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200: d[t] = df['Close'].dropna()
        except: pass
    return pd.DataFrame(d)

def precompute(close_df):
    r1 = close_df / close_df.shift(22) - 1
    r3 = close_df / close_df.shift(63) - 1
    r6 = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1
    lr = np.log(close_df / close_df.shift(1))
    v30 = lr.rolling(30).std() * np.sqrt(252)
    spy = close_df['SPY'] if 'SPY' in close_df.columns else None
    return {'r1':r1,'r3':r3,'r6':r6,'r12':r12,'v30':v30,
            'spy_sma200': spy.rolling(200).mean() if spy is not None else None,
            'spy': spy, 'sma50': close_df.rolling(50).mean(), 'close': close_df}

def regime(sig, date):
    s = sig['spy_sma200']
    if s is None: return 'bull'
    v = s.loc[:date].dropna(); sp = sig['spy'].loc[:date].dropna()
    if len(v)==0 or len(sp)==0: return 'bull'
    return 'bull' if sp.iloc[-1] > v.iloc[-1] else 'bear'

def select(sig, sectors, date, prev):
    c = sig['close']; idx = c.index[c.index <= date]
    if len(idx)==0: return {}
    idx = idx[-1]; r = regime(sig, date)
    mom = sig['r1'].loc[idx]*0.20 + sig['r3'].loc[idx]*0.40 + sig['r6'].loc[idx]*0.30 + sig['r12'].loc[idx]*0.10
    df = pd.DataFrame({'m':mom,'a6':sig['r6'].loc[idx],'v':sig['v30'].loc[idx],'p':c.loc[idx],'s50':sig['sma50'].loc[idx]})
    df = df.dropna(subset=['m','s50'])
    df = df[(df['p']>=5)&(df.index!='SPY')&(df['a6']>0)&(df['v']<0.65)&(df['p']>df['s50'])]
    df['sec'] = df.index.map(lambda t: sectors.get(t,'?'))
    for t in df.index:
        if t in prev: df.loc[t,'m'] += 0.03
    sm = df.groupby('sec')['m'].mean().sort_values(ascending=False)
    if r == 'bull':
        ts, sps, cash = sm.head(5).index.tolist(), 3, 0.0
    else:
        ts, sps, cash = sm.head(4).index.tolist(), 2, 0.15
    sel = []
    for s in ts:
        sd = df[df['sec']==s].sort_values('m',ascending=False)
        sel.extend(sd.index[:sps].tolist())
    if not sel: return {}
    iv = {t: 1.0/max(df.loc[t,'v'],0.10) for t in sel}
    tt = sum(iv.values()); inv = 1.0 - cash
    return {t: (v/tt)*inv for t,v in iv.items()}

def backtest(close_df, sig, sectors, start='2015-01-01', end='2025-12-31'):
    cr = close_df.loc[start:end].dropna(how='all')
    me = cr.resample('ME').last().index
    pv,pd_=[],[]
    to_l=[]; pw,ph={},set(); cv=1.0; hh={}
    for i in range(len(me)-1):
        d,nd = me[i],me[i+1]
        nw = select(sig, sectors, d, ph)
        at = set(list(nw.keys())+list(pw.keys()))
        to = sum(abs(nw.get(t,0)-pw.get(t,0)) for t in at)/2
        to_l.append(to); pw=nw.copy(); ph=set(nw.keys())
        hh[d.strftime('%Y-%m')]=list(nw.keys())
        pr = sum((close_df[t].loc[d:nd].dropna().iloc[-1]/close_df[t].loc[d:nd].dropna().iloc[0]-1)*w
                 for t,w in nw.items() if len(close_df[t].loc[d:nd].dropna())>=2)
        pr -= to*0.0015*2; cv *= (1+pr); pv.append(cv); pd_.append(nd)
    return pd.Series(pv, index=pd.DatetimeIndex(pd_)), hh, np.mean(to_l)

def metrics(eq):
    y = (eq.index[-1]-eq.index[0]).days/365.25
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/y)-1
    m = eq.pct_change().dropna()
    sh = m.mean()/m.std()*np.sqrt(12) if m.std()>0 else 0
    dd = (eq-eq.cummax())/eq.cummax(); mdd = dd.min()
    cal = cagr/abs(mdd) if mdd!=0 else 0
    return cagr, mdd, sh, cal

def main():
    print("🐻 v3c — 5-Sector Diversified + SMA50")
    tk = (CACHE/"sp500_tickers.txt").read_text().strip().split('\n')
    cd = load_all_data(tk+['SPY']); sec = json.load(open(CACHE/"sp500_sectors.json"))
    sig = precompute(cd)
    eq,h,to = backtest(cd,sig,sec); c,d,s,cal = metrics(eq)
    eq_is,_,_ = backtest(cd,sig,sec,'2015-01-01','2020-12-31')
    eq_oos,_,_ = backtest(cd,sig,sec,'2021-01-01','2025-12-31')
    _,_,si,_ = metrics(eq_is); _,_,so,_ = metrics(eq_oos)
    wf = so/si if si else 0
    print(f"CAGR {c:.1%} | Sharpe {s:.2f} | MaxDD {d:.1%} | Calmar {cal:.2f}")
    print(f"WF: IS {si:.2f} → OOS {so:.2f} = {wf:.2f} {'✅' if wf>=0.70 else '❌'}")
    print(f"T/O: {to:.1%} | Composite: {s*0.4+cal*0.4+c*0.2:.3f}")

if __name__ == '__main__': main()
