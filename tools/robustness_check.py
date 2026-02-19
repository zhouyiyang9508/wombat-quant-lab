"""
Robustness Check — Is the vol targeting strategy overfit?
Tests:
1. Walk-forward: train on first half, test on second half
2. Parameter sensitivity: change thresholds ±20%
3. Simple vs Complex: does adding more parameters actually help?
4. Sub-period consistency: does it work in every 3-year window?
"""
import pandas as pd
import numpy as np

tqqq = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
gld = pd.read_csv('data_cache/GLD.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
tlt = pd.read_csv('data_cache/TLT.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()

common = tqqq.index.intersection(gld.index).intersection(tlt.index)
tqqq = tqqq.loc[common]; gld = gld.loc[common]; tlt = tlt.loc[common]

vol20 = tqqq.pct_change().rolling(20).std() * np.sqrt(252)
sma200 = tqqq.rolling(200).mean()
mayer = tqqq / sma200

def calc_rsi(s, p):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100/(1+g/l)
rsi14 = calc_rsi(tqqq, 14)
ath = tqqq.cummax()
dd_from_ath = (tqqq - ath) / ath
gld_mom = gld.pct_change(63)
tlt_mom = tlt.pct_change(63)

def run_on_period(strategy_fn, start, end, monthly_amount=1000):
    """Run strategy on a sub-period."""
    mask = (common >= start) & (common <= end)
    dates = common[mask]
    if len(dates) < 252: return None
    
    monthly = tqqq.loc[dates].groupby(tqqq.loc[dates].index.to_period('M')).apply(lambda x: x.index[0])
    rebal = set(monthly)
    
    tqqq_sh = 0; gld_sh = 0; tlt_sh = 0; cash = 0; total_inv = 0
    vals = []
    
    for date in dates:
        tp = tqqq.loc[date]; gp = gld.loc[date]; tlp = tlt.loc[date]
        if date in rebal:
            cash += monthly_amount; total_inv += monthly_amount
            pv = tqqq_sh*tp + gld_sh*gp + tlt_sh*tlp + cash
            ind = {
                'vol': vol20.get(date, np.nan), 'mayer': mayer.get(date, np.nan),
                'rsi14': rsi14.get(date, np.nan), 'dd': dd_from_ath.get(date, np.nan),
                'gld_mom': gld_mom.get(date, np.nan), 'tlt_mom': tlt_mom.get(date, np.nan),
            }
            alloc = strategy_fn(date, tp, ind)
            cash += tqqq_sh*tp + gld_sh*gp + tlt_sh*tlp
            tqqq_sh=0; gld_sh=0; tlt_sh=0
            t_t = pv*alloc.get('tqqq',0); t_g = pv*alloc.get('gld',0); t_l = pv*alloc.get('tlt',0)
            tqqq_sh=t_t/tp; gld_sh=t_g/gp; tlt_sh=t_l/tlp
            cash -= (t_t+t_g+t_l)
        pv = tqqq_sh*tp + gld_sh*gp + tlt_sh*tlp + cash
        vals.append((date, pv, total_inv))
    
    h = pd.DataFrame(vals, columns=['Date','Value','Invested']).set_index('Date')
    f = h['Value'].iloc[-1]; inv = h['Invested'].iloc[-1]
    mult = f/inv; years = (h.index[-1]-h.index[0]).days/365.25
    pk = h['Value'].cummax(); mdd = ((h['Value']-pk)/pk).min()
    mr = h['Value'].resample('ME').last().dropna().pct_change().dropna()
    sh = mr.mean()/mr.std()*np.sqrt(12) if len(mr)>1 and mr.std()>0 else 0
    return {'final': f, 'invested': inv, 'mult': mult, 'mdd': mdd, 'sharpe': sh, 'years': years}

# ── Strategies (from simple to complex) ──

def strat_bh(d, p, i):
    return {'tqqq': 1.0}

# S1: Simplest possible — just vol threshold, 2 params only
def make_simple_vol(high_thresh, low_alloc):
    def fn(d, p, i):
        v = i['vol']
        if pd.isna(v): return {'tqqq': 1.0}
        if v > high_thresh: return {'tqqq': low_alloc}
        return {'tqqq': 1.0}
    return fn

# S2: 3-tier vol (what we've been using), 4 params
def make_3tier_vol(t1, t2, t3, a1, a2, a3):
    def fn(d, p, i):
        v = i['vol']
        if pd.isna(v): return {'tqqq': 1.0}
        if v > t1: return {'tqqq': a1}
        elif v > t2: return {'tqqq': a2}
        elif v > t3: return {'tqqq': a3}
        return {'tqqq': 1.0}
    return fn

# S3: v4 Adaptive (many params — the complex one)
def strat_adaptive(d, p, i):
    v = i['vol']; mm = i['mayer']; r = i['rsi14']; dd = i['dd']
    gm = i['gld_mom']; tm = i['tlt_mom']
    if pd.isna(v): return {'tqqq': 1.0}
    if v > 1.2: t = 0.25
    elif v > 0.9: t = 0.45
    elif v > 0.7: t = 0.75
    else: t = 1.0
    if not pd.isna(mm) and mm < 0.7: t = max(t, 0.80)
    if not pd.isna(r) and r < 25: t = max(t, 0.85)
    if not pd.isna(dd) and dd < -0.60: t = max(t, 0.90)
    rest = 1.0 - t
    if rest <= 0: return {'tqqq': 1.0}
    if not pd.isna(gm) and not pd.isna(tm):
        if gm > tm: return {'tqqq': t, 'gld': rest*0.7, 'tlt': rest*0.3}
        else: return {'tqqq': t, 'tlt': rest*0.7, 'gld': rest*0.3}
    return {'tqqq': t, 'gld': rest*0.5, 'tlt': rest*0.5}

# ═══════════════════════════════════════
# TEST 1: Walk-forward (first half vs second half)
# ═══════════════════════════════════════
mid = '2018-02-01'
print("=" * 100)
print("TEST 1: Walk-Forward — First Half (2010-2018) vs Second Half (2018-2026)")
print("=" * 100)

strategies = [
    ("B&H", strat_bh),
    ("Simple Vol (>0.9→30%)", make_simple_vol(0.9, 0.30)),
    ("Simple Vol (>1.0→30%)", make_simple_vol(1.0, 0.30)),
    ("3-Tier Vol", make_3tier_vol(1.2, 0.9, 0.7, 0.30, 0.50, 0.80)),
    ("Adaptive Multi-Asset", strat_adaptive),
]

print(f"\n  {'Strategy':35s} | {'1st Half mult':>12s} {'DD':>7s} {'Sh':>5s} | {'2nd Half mult':>12s} {'DD':>7s} {'Sh':>5s} | {'Full mult':>10s}")
print("  " + "-" * 110)

for name, fn in strategies:
    r1 = run_on_period(fn, '2010-02-11', '2018-01-31')
    r2 = run_on_period(fn, '2018-02-01', '2026-02-18')
    rf = run_on_period(fn, '2010-02-11', '2026-02-18')
    if r1 and r2 and rf:
        print(f"  {name:35s} | {r1['mult']:8.1f}x {r1['mdd']*100:6.1f}% {r1['sharpe']:5.2f} | {r2['mult']:8.1f}x {r2['mdd']*100:6.1f}% {r2['sharpe']:5.2f} | {rf['mult']:6.1f}x")

# ═══════════════════════════════════════
# TEST 2: Parameter Sensitivity
# ═══════════════════════════════════════
print("\n" + "=" * 100)
print("TEST 2: Parameter Sensitivity — vary vol thresholds ±20%")
print("=" * 100)

base_thresholds = (1.2, 0.9, 0.7)
variations = [
    ("Base (1.2/0.9/0.7)", 1.2, 0.9, 0.7),
    ("-20% (0.96/0.72/0.56)", 0.96, 0.72, 0.56),
    ("+20% (1.44/1.08/0.84)", 1.44, 1.08, 0.84),
    ("-10% (1.08/0.81/0.63)", 1.08, 0.81, 0.63),
    ("+10% (1.32/0.99/0.77)", 1.32, 0.99, 0.77),
    ("Wider (1.5/1.0/0.6)", 1.5, 1.0, 0.6),
    ("Tighter (1.0/0.8/0.65)", 1.0, 0.8, 0.65),
]

print(f"\n  {'Thresholds':35s} | {'Final':>12s} | {'Mult':>6s} | {'MaxDD':>7s} | {'Sharpe':>6s}")
print("  " + "-" * 80)

for label, t1, t2, t3 in variations:
    fn = make_3tier_vol(t1, t2, t3, 0.30, 0.50, 0.80)
    r = run_on_period(fn, '2010-02-11', '2026-02-18')
    if r:
        print(f"  {label:35s} | ${r['final']:>10,.0f} | {r['mult']:5.1f}x | {r['mdd']*100:6.1f}% | {r['sharpe']:5.2f}")

# ═══════════════════════════════════════
# TEST 3: Rolling 3-year windows
# ═══════════════════════════════════════
print("\n" + "=" * 100)
print("TEST 3: Rolling 3-Year Windows — Consistency Check")
print("=" * 100)

windows = [
    ('2010-2013', '2010-02-11', '2013-02-10'),
    ('2013-2016', '2013-02-11', '2016-02-10'),
    ('2016-2019', '2016-02-11', '2019-02-10'),
    ('2019-2022', '2019-02-11', '2022-02-10'),
    ('2022-2025', '2022-02-11', '2025-02-10'),
]

print(f"\n  {'Window':12s} | {'B&H mult':>9s} {'DD':>7s} | {'Simple Vol':>10s} {'DD':>7s} | {'3-Tier':>7s} {'DD':>7s} | {'Adaptive':>9s} {'DD':>7s} | Win?")
print("  " + "-" * 110)

for label, start, end in windows:
    bh = run_on_period(strat_bh, start, end)
    sv = run_on_period(make_simple_vol(0.9, 0.30), start, end)
    tv = run_on_period(make_3tier_vol(1.2, 0.9, 0.7, 0.30, 0.50, 0.80), start, end)
    ad = run_on_period(strat_adaptive, start, end)
    
    if bh and sv and tv and ad:
        best = max([(bh['mult'],'B&H'), (sv['mult'],'SimpleV'), (tv['mult'],'3Tier'), (ad['mult'],'Adapt')], key=lambda x:x[0])
        print(f"  {label:12s} | {bh['mult']:6.1f}x {bh['mdd']*100:6.1f}% | {sv['mult']:7.1f}x {sv['mdd']*100:6.1f}% | {tv['mult']:4.1f}x {tv['mdd']*100:6.1f}% | {ad['mult']:6.1f}x {ad['mdd']*100:6.1f}% | {best[1]}")

# ═══════════════════════════════════════
# TEST 4: Simplicity comparison
# ═══════════════════════════════════════
print("\n" + "=" * 100)
print("TEST 4: Simplicity vs Complexity (fewer params = less overfit risk)")
print("=" * 100)

complexity = [
    ("B&H (0 params)", strat_bh, 0),
    ("Simple: vol>0.9→30% (2 params)", make_simple_vol(0.9, 0.30), 2),
    ("Simple: vol>1.0→30% (2 params)", make_simple_vol(1.0, 0.30), 2),
    ("3-Tier vol (6 params)", make_3tier_vol(1.2, 0.9, 0.7, 0.30, 0.50, 0.80), 6),
    ("Adaptive Multi (12+ params)", strat_adaptive, 12),
]

print(f"\n  {'Strategy':40s} | {'Params':>6s} | {'Mult':>6s} | {'MaxDD':>7s} | {'Sharpe':>6s} | Overfit Risk")
print("  " + "-" * 100)

for name, fn, params in complexity:
    r = run_on_period(fn, '2010-02-11', '2026-02-18')
    if r:
        risk = "🟢 Low" if params <= 2 else "🟡 Med" if params <= 6 else "🔴 High"
        print(f"  {name:40s} | {params:6d} | {r['mult']:5.1f}x | {r['mdd']*100:6.1f}% | {r['sharpe']:5.2f} | {risk}")
