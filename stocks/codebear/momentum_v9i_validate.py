#!/usr/bin/env python3
"""
v9i 验证 — vol targeting精调 + 不同lookback窗口 + IS/OOS分解
目的: 确认 vt=0.11 的 Composite 1.973 是否鲁棒

担忧:
1. 3月lookback估计vol太嘈杂
2. 是否过拟合?
3. IS vs OOS分解

测试:
- lookback: 3m, 6m, 12m
- target_vol: 0.09, 0.10, 0.11, 0.12, 0.13
- 详细报告 IS/OOS 分解
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

N_BULL_SECS    = 5; N_BULL_SECS_HI = 4; BREADTH_CONC = 0.65
BULL_SPS = 2; BEAR_SPS = 2; BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
CONT_BONUS = 0.03; HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18
MOM_W = (0.20, 0.50, 0.20, 0.10)

def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    df = df.set_index(c).sort_index()
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df

def load_stocks(tickers):
    d = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500: continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except: pass
    return pd.DataFrame(d)

def precompute(close_df):
    r1 = close_df/close_df.shift(22)-1; r3 = close_df/close_df.shift(63)-1
    r6 = close_df/close_df.shift(126)-1; r12 = close_df/close_df.shift(252)-1
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df/close_df.shift(1))
    vol5 = log_r.rolling(5).std()*np.sqrt(252)
    vol30 = log_r.rolling(30).std()*np.sqrt(252)
    spy = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200 = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1,r3=r3,r6=r6,r12=r12,r52w_hi=r52w_hi,
                vol5=vol5,vol30=vol30,spy=spy,s200=s200,sma50=sma50,close=close_df)

def compute_breadth(sig, date):
    close=sig['close'].loc[:date].dropna(how='all'); sma50=sig['sma50'].loc[:date].dropna(how='all')
    if len(close)<50: return 1.0
    lc=close.iloc[-1]; ls=sma50.iloc[-1]; mask=(lc>ls).dropna()
    return float(mask.sum()/len(mask)) if len(mask)>0 else 1.0

def get_regime(sig, date):
    if sig['s200'] is None: return 'bull'
    sn=sig['spy'].loc[:date].dropna(); s2=sig['s200'].loc[:date].dropna()
    if len(sn)==0 or len(s2)==0: return 'bull'
    return 'bear' if (sn.iloc[-1]<s2.iloc[-1] and compute_breadth(sig,date)<BREADTH_NARROW) else 'bull'

def asset_compete(sig, date, prices, thresh, frac, lb=127):
    r6=sig['r6']; idx=r6.index[r6.index<=date]
    if len(idx)<1: return 0.0
    d=idx[-1]; sr6=r6.loc[d].dropna(); sr6=sr6[sr6>0]
    if len(sr6)<10: return 0.0
    hist=prices.loc[:d].dropna()
    if len(hist)<lb+3: return 0.0
    ar6=hist.iloc[-1]/hist.iloc[-lb]-1
    return frac if ar6>=sr6.mean()*thresh else 0.0

def select(sig, sectors, date, prev_hold, gld_p, gdx_p):
    close=sig['close']; idx=close.index[close.index<=date]
    if len(idx)==0: return {}
    d=idx[-1]; w1,w3,w6,w12=MOM_W
    mom=(sig['r1'].loc[d]*w1+sig['r3'].loc[d]*w3+sig['r6'].loc[d]*w6+sig['r12'].loc[d]*w12)
    df=pd.DataFrame({'mom':mom,'r6':sig['r6'].loc[d],'vol':sig['vol30'].loc[d],
                     'price':close.loc[d],'sma50':sig['sma50'].loc[d],'hi52':sig['r52w_hi'].loc[d]}).dropna(subset=['mom','sma50'])
    df=df[(df['price']>=5)&(df.index!='SPY')]
    df=df[(df['r6']>0)&(df['vol']<0.65)]; df=df[df['price']>df['sma50']]
    df=df[df['price']>=df['hi52']*HI52_FRAC]
    df['sector']=df.index.map(lambda t:sectors.get(t,'Unknown')); df=df[df['sector']!='Unknown']
    for t in df.index:
        if t in prev_hold: df.loc[t,'mom']+=CONT_BONUS
    if len(df)==0: return {}
    sec_mom=df.groupby('sector')['mom'].mean().sort_values(ascending=False)
    gld_a=asset_compete(sig,date,gld_p,GLD_AVG_THRESH,GLD_COMPETE_FRAC)
    gdx_a=asset_compete(sig,date,gdx_p,GDX_AVG_THRESH,GDX_COMPETE_FRAC)
    tc=gld_a+gdx_a; nc=(1 if gld_a>0 else 0)+(1 if gdx_a>0 else 0)
    reg=get_regime(sig,date); breadth=compute_breadth(sig,date)
    if reg=='bull':
        nb=N_BULL_SECS_HI if breadth>BREADTH_CONC else N_BULL_SECS
        n_secs=max(nb-nc,1); sps,cash=BULL_SPS,0.0
    else:
        n_secs=max(3-nc,1); sps,cash=BEAR_SPS,0.20
    selected=[]
    for sec in sec_mom.head(n_secs).index:
        sdf=df[df['sector']==sec].sort_values('mom',ascending=False)
        selected.extend(sdf.index[:sps].tolist())
    sf=max(1.0-cash-tc,0.0)
    if not selected:
        w={}
        if gld_a>0: w['GLD']=gld_a
        if gdx_a>0: w['GDX']=gdx_a
        return w
    iv={t:1.0/max(df.loc[t,'vol'],0.10) for t in selected}
    ivt=sum(iv.values()); ivw={t:v/ivt for t,v in iv.items()}
    mn=min(df.loc[t,'mom'] for t in selected); sh=max(-mn+0.01,0)
    mw={t:df.loc[t,'mom']+sh for t in selected}
    mwt=sum(mw.values()); mww={t:v/mwt for t,v in mw.items()}
    wts={t:(0.70*ivw[t]+0.30*mww[t])*sf for t in selected}
    if gld_a>0: wts['GLD']=gld_a
    if gdx_a>0: wts['GDX']=gdx_a
    return wts

def apply_ov(weights, spy_vol, dd, port_vol_ann, target_vol):
    if spy_vol>=GDXJ_VOL_HI_THRESH: gv=GDXJ_VOL_HI_FRAC
    elif spy_vol>=GDXJ_VOL_LO_THRESH: gv=GDXJ_VOL_LO_FRAC
    else: gv=0.0
    gd=max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd<th),default=0.0)
    tot=gv+gd
    if tot>0 and weights:
        sf=max(1.0-tot,0.01); t=sum(weights.values())
        if t>0: weights={k:w/t*sf for k,w in weights.items()}
        if gd>0: weights['GLD']=weights.get('GLD',0)+gd
        if gv>0: weights['GDXJ']=weights.get('GDXJ',0)+gv
    shy_b=0.0
    if target_vol is not None and port_vol_ann>0.01:
        scale=min(target_vol/max(port_vol_ann,0.01),1.0)
        if scale<0.98:
            eq_keys=[t for t in weights if t not in ('GLD','GDX','GDXJ')]
            eq_frac=sum(weights[t] for t in eq_keys)
            if eq_frac>0:
                for t in eq_keys: weights[t]*=scale
                shy_b=eq_frac*(1-scale)
    return weights, shy_b

def run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, target_vol, lookback,
           start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng=close_df.loc[start:end].dropna(how='all')
    ends=rng.resample('ME').last().index
    vals,dates,tos,scale_hist=[],[],[],[]
    prev_w,prev_h={},set()
    val=1.0; peak=1.0; prets=[]
    SPY_VOL=sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None
    for i in range(len(ends)-1):
        dt,ndt=ends[i],ends[i+1]
        dd=(val-peak)/peak if peak>0 else 0
        sv=float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna())>0 else 0.15
        if len(prets)>=lookback:
            pv=np.std(prets[-lookback:],ddof=1)*np.sqrt(12)
        else:
            pv=0.20
        w=select(sig,sectors,dt,prev_h,gld_p,gdx_p)
        w,shy_b=apply_ov(w,sv,dd,pv,target_vol)
        if target_vol is not None and pv>0.01:
            scale_hist.append(min(target_vol/max(pv,0.01),1.0))
        all_t=set(w)|set(prev_w)
        to=sum(abs(w.get(t,0)-prev_w.get(t,0)) for t in all_t)/2
        tos.append(to); prev_w=w.copy()
        prev_h={k for k in w if k not in ('GLD','GDX','GDXJ')}
        inv=sum(w.values()); cf=max(1.0-inv,0.0)
        ret=0.0
        for t,wt in w.items():
            if t=='GLD': s=gld_p.loc[dt:ndt].dropna()
            elif t=='GDX': s=gdx_p.loc[dt:ndt].dropna()
            elif t=='GDXJ': s=gdxj_p.loc[dt:ndt].dropna()
            elif t in close_df.columns: s=close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s)>=2: ret+=(s.iloc[-1]/s.iloc[0]-1)*wt
        total_shy=shy_b+(cf if USE_SHY else 0.0)
        if total_shy>0 and shy_p is not None:
            s=shy_p.loc[dt:ndt].dropna()
            if len(s)>=2: ret+=(s.iloc[-1]/s.iloc[0]-1)*total_shy
        ret-=to*cost*2; val*=(1+ret)
        if val>peak: peak=val
        vals.append(val); dates.append(ndt); prets.append(ret)
    eq=pd.Series(vals,index=pd.DatetimeIndex(dates))
    avg_scale=np.mean(scale_hist) if scale_hist else 1.0
    pct_scaled=np.mean([s<0.99 for s in scale_hist]) if scale_hist else 0.0
    return eq, avg_scale, pct_scaled

def cm(eq):
    if len(eq)<3: return dict(cagr=0,max_dd=0,sharpe=0,calmar=0)
    yrs=(eq.index[-1]-eq.index[0]).days/365.25
    if yrs<0.5: return dict(cagr=0,max_dd=0,sharpe=0,calmar=0)
    cagr=(eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1
    mo=eq.pct_change().dropna()
    sh=mo.mean()/mo.std()*np.sqrt(12) if mo.std()>0 else 0
    dd=((eq-eq.cummax())/eq.cummax()).min()
    cal=cagr/abs(dd) if dd!=0 else 0
    return dict(cagr=float(cagr),max_dd=float(dd),sharpe=float(sh),calmar=float(cal))

def main():
    print("=" * 75)
    print("🐻 v9i 验证 — vol targeting 精调 + 不同lookback窗口")
    print("=" * 75)

    tickers=(CACHE/"sp500_tickers.txt").read_text().strip().split('\n')
    close_df=load_stocks(tickers+['SPY'])
    sectors=json.load(open(CACHE/"sp500_sectors.json"))
    gld_p=load_csv(CACHE/"GLD.csv")['Close'].dropna()
    gdx_p=load_csv(CACHE/"GDX.csv")['Close'].dropna()
    gdxj_p=load_csv(CACHE/"GDXJ.csv")['Close'].dropna()
    shy_p=load_csv(CACHE/"SHY.csv")['Close'].dropna()
    sig=precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers\n")

    results={}
    for lb in [3, 6]:
        for tv in [0.09, 0.10, 0.11, 0.12, 0.13, 0.14]:
            lbl=f"lb{lb}_vt{int(tv*100)}"
            print(f"--- {lbl} ---", flush=True)
            eq_f, avg_sc, pct_sc = run_bt(close_df,sig,sectors,gld_p,gdx_p,gdxj_p,shy_p,tv,lb)
            eq_is,_,_ = run_bt(close_df,sig,sectors,gld_p,gdx_p,gdxj_p,shy_p,tv,lb,'2015-01-01','2020-12-31')
            eq_oo,_,_ = run_bt(close_df,sig,sectors,gld_p,gdx_p,gdxj_p,shy_p,tv,lb,'2021-01-01','2025-12-31')
            m=cm(eq_f); mi=cm(eq_is); mo=cm(eq_oo)
            wf=mo['sharpe']/mi['sharpe'] if mi['sharpe']>0 else 0
            comp=m['sharpe']*0.4+m['calmar']*0.4+m['cagr']*0.2
            results[lbl]={'full':m,'is':mi,'oos':mo,'wf':wf,'composite':comp,
                         'avg_scale':avg_sc,'pct_scaled':pct_sc}
            tag='🚨' if comp>1.95 else ('🚀🚀' if comp>1.85 else ('🚀' if comp>1.759 else ''))
            print(f"  Comp={comp:.4f} {tag} | Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%} "
                  f"Cal={m['calmar']:.2f} WF={wf:.2f} | IS={mi['sharpe']:.2f} OOS={mo['sharpe']:.2f}")
            print(f"  [vol_info] avg_scale={avg_sc:.3f} pct_scaled={pct_sc:.0%}")

    # Add baseline
    eq_b,_,_=run_bt(close_df,sig,sectors,gld_p,gdx_p,gdxj_p,shy_p,None,3)
    eq_bis,_,_=run_bt(close_df,sig,sectors,gld_p,gdx_p,gdxj_p,shy_p,None,3,'2015-01-01','2020-12-31')
    eq_boo,_,_=run_bt(close_df,sig,sectors,gld_p,gdx_p,gdxj_p,shy_p,None,3,'2021-01-01','2025-12-31')
    m=cm(eq_b); mi=cm(eq_bis); mo=cm(eq_boo)
    wf=mo['sharpe']/mi['sharpe'] if mi['sharpe']>0 else 0
    comp=m['sharpe']*0.4+m['calmar']*0.4+m['cagr']*0.2
    results['baseline']={'full':m,'is':mi,'oos':mo,'wf':wf,'composite':comp,'avg_scale':1.0,'pct_scaled':0.0}
    print(f"\nBaseline: Comp={comp:.4f} | Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%} WF={wf:.2f}")

    print("\n" + "=" * 75)
    print("📊 FINAL RANKINGS")
    print("=" * 75)
    ranked=sorted(results.items(),key=lambda x:x[1]['composite'],reverse=True)
    for i,(n,r) in enumerate(ranked[:10]):
        m=r['full']; mi=r['is']; mo=r['oos']
        tag='🚨' if r['composite']>1.95 else ('🚀🚀' if r['composite']>1.85 else ('🚀' if r['composite']>1.759 else ''))
        print(f"  #{i+1} {n:16s}: Comp={r['composite']:.4f} {tag} | "
              f"Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%} "
              f"WF={r['wf']:.2f} | IS={mi['sharpe']:.2f} OOS={mo['sharpe']:.2f}")

    best=max(results.items(),key=lambda x:x[1]['composite'])
    b=best[1]
    print(f"\n🏆 Best: {best[0]} → Composite {b['composite']:.4f}")
    if b['composite']>1.95:
        print("🚨🚨🚨 近乎【重大突破】Composite接近2.0!")
    elif b['composite']>1.80:
        print("🚀🚀 突破1.80!")

    jf=Path(__file__).parent/"momentum_v9i_validate_results.json"
    jf.write_text(json.dumps({k:{'full':v['full'],'is':v['is'],'oos':v['oos'],
        'wf':v['wf'],'composite':v['composite'],'avg_scale':v.get('avg_scale',1.0),
        'pct_scaled':v.get('pct_scaled',0.0)} for k,v in results.items()},indent=2))
    print(f"💾 → {jf}")

if __name__=='__main__':
    main()
