"""
DCA Research v5 — Downside Volatility Only
Key insight: only reduce position on DOWNSIDE vol, ignore upside vol.
This prevents selling during V-shape rallies where total vol is high but
it's all upside moves.
"""
import pandas as pd
import numpy as np

# Load data
tqqq = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
gld = pd.read_csv('data_cache/GLD.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
tlt = pd.read_csv('data_cache/TLT.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()

common = tqqq.index.intersection(gld.index).intersection(tlt.index)
tqqq = tqqq.loc[common]; gld = gld.loc[common]; tlt = tlt.loc[common]

daily_ret = tqqq.pct_change()

# ── Volatility variants ──
# Standard vol (both directions)
vol20_total = daily_ret.rolling(20).std() * np.sqrt(252)

# Downside vol only (only negative returns count)
downside_ret = daily_ret.clip(upper=0)
vol20_down = downside_ret.rolling(20).std() * np.sqrt(252)

# Semi-deviation (downside only, vs mean)
def semi_deviation(returns, window):
    """Annualized downside semi-deviation."""
    result = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(returns)):
        chunk = returns.iloc[i-window:i]
        neg = chunk[chunk < 0]
        if len(neg) > 0:
            result.iloc[i] = neg.std() * np.sqrt(252)
        else:
            result.iloc[i] = 0
    return result

# Pre-compute semi-dev (faster version)
vol20_semi = daily_ret.rolling(20).apply(
    lambda x: np.sqrt(np.mean(np.minimum(x, 0)**2)) * np.sqrt(252), raw=True
)

# Ratio of downside vol to upside vol
upside_ret = daily_ret.clip(lower=0)
vol20_up = upside_ret.rolling(20).std() * np.sqrt(252)
vol_ratio = vol20_down / vol20_up.replace(0, np.nan)  # >1 means more downside

# Other indicators
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

monthly_dates = tqqq.groupby(tqqq.index.to_period('M')).apply(lambda x: x.index[0])

print(f"Data: {common[0].date()} → {common[-1].date()}")

# Quick stats
print(f"\nVol comparison (mean):")
print(f"  Total vol20: {vol20_total.mean()*100:.0f}%")
print(f"  Downside vol20: {vol20_down.mean()*100:.0f}%")
print(f"  Semi-dev vol20: {vol20_semi.mean()*100:.0f}%")

# Check 2020 V-shape
v_shape = pd.DataFrame({
    'total': vol20_total, 'down': vol20_down, 'semi': vol20_semi, 'ratio': vol_ratio
}).loc['2020-03-20':'2020-05-15']
print(f"\n2020 V-shape recovery vol comparison (monthly avg):")
m = v_shape.resample('ME').mean()
print(m.to_string())

def run_strategy(strategy_fn, name, monthly_amount=1000):
    tqqq_sh = 0.0; gld_sh = 0.0; tlt_sh = 0.0; cash = 0.0; total_inv = 0.0
    daily_values = []
    rebal = set(monthly_dates)
    
    for date in common:
        tp = tqqq.loc[date]; gp = gld.loc[date]; tlp = tlt.loc[date]
        if date in rebal:
            cash += monthly_amount; total_inv += monthly_amount
            pv = tqqq_sh*tp + gld_sh*gp + tlt_sh*tlp + cash
            ind = {
                'vol_total': vol20_total.get(date, np.nan),
                'vol_down': vol20_down.get(date, np.nan),
                'vol_semi': vol20_semi.get(date, np.nan),
                'vol_ratio': vol_ratio.get(date, np.nan),
                'mayer': mayer.get(date, np.nan),
                'rsi14': rsi14.get(date, np.nan),
                'dd': dd_from_ath.get(date, np.nan),
                'gld_mom': gld_mom.get(date, np.nan),
                'tlt_mom': tlt_mom.get(date, np.nan),
            }
            alloc = strategy_fn(date, tp, ind)
            
            cash += tqqq_sh*tp + gld_sh*gp + tlt_sh*tlp
            tqqq_sh = 0; gld_sh = 0; tlt_sh = 0
            t_tqqq = pv * alloc.get('tqqq', 0)
            t_gld = pv * alloc.get('gld', 0)
            t_tlt = pv * alloc.get('tlt', 0)
            tqqq_sh = t_tqqq/tp; cash -= t_tqqq
            gld_sh = t_gld/gp; cash -= t_gld
            tlt_sh = t_tlt/tlp; cash -= t_tlt
        
        pv = tqqq_sh*tp + gld_sh*gp + tlt_sh*tlp + cash
        daily_values.append((date, pv, total_inv))
    
    hist = pd.DataFrame(daily_values, columns=['Date','Value','Invested']).set_index('Date')
    f = hist['Value'].iloc[-1]; inv = hist['Invested'].iloc[-1]; m = f/inv
    pk = hist['Value'].cummax(); mdd = ((hist['Value']-pk)/pk).min()
    mr = hist['Value'].resample('ME').last().dropna().pct_change().dropna()
    sh = mr.mean()/mr.std()*np.sqrt(12) if len(mr)>1 else 0
    print(f"  {name:55s} | ${f:>11,.0f} ({m:5.1f}x) | DD:{mdd*100:5.1f}% | Sh:{sh:.2f}")
    return hist

# ══════════════════════════════════════════════
# STRATEGIES
# ══════════════════════════════════════════════

def strat_bh(d, p, i):
    return {'tqqq': 1.0}

# A. Original v6 (total vol + cash)
def strat_v6_total_vol(d, p, i):
    v = i['vol_total']; mm = i['mayer']; r = i['rsi14']; dd = i['dd']
    if pd.isna(v): return {'tqqq': 1.0}
    if v > 1.2: b = 0.30
    elif v > 0.9: b = 0.50
    elif v > 0.7: b = 0.80
    else: b = 1.0
    if not pd.isna(mm) and mm < 0.7: b = max(b, 0.80)
    if not pd.isna(r) and r < 25: b = max(b, 0.85)
    if not pd.isna(dd) and dd < -0.60: b = max(b, 0.90)
    return {'tqqq': b}

# B. Downside vol only (main innovation!)
def strat_downvol(d, p, i):
    v = i['vol_down']; mm = i['mayer']; r = i['rsi14']; dd = i['dd']
    if pd.isna(v): return {'tqqq': 1.0}
    # Downside vol thresholds (lower than total vol since it's half the distribution)
    if v > 0.70: b = 0.30
    elif v > 0.55: b = 0.50
    elif v > 0.40: b = 0.80
    else: b = 1.0
    if not pd.isna(mm) and mm < 0.7: b = max(b, 0.80)
    if not pd.isna(r) and r < 25: b = max(b, 0.85)
    if not pd.isna(dd) and dd < -0.60: b = max(b, 0.90)
    return {'tqqq': b}

# C. Semi-deviation targeting
def strat_semi_dev(d, p, i):
    v = i['vol_semi']; mm = i['mayer']; r = i['rsi14']; dd = i['dd']
    if pd.isna(v): return {'tqqq': 1.0}
    target_semi = 0.30
    b = min(1.0, target_semi / v) if v > 0 else 1.0
    if not pd.isna(mm) and mm < 0.7: b = max(b, 0.80)
    if not pd.isna(r) and r < 25: b = max(b, 0.85)
    if not pd.isna(dd) and dd < -0.60: b = max(b, 0.90)
    return {'tqqq': max(0.15, b)}

# D. Vol ratio: only reduce when downside vol dominates upside vol
def strat_vol_ratio(d, p, i):
    vr = i['vol_ratio']; vt = i['vol_total']; mm = i['mayer']; r = i['rsi14']; dd = i['dd']
    if pd.isna(vr) or pd.isna(vt): return {'tqqq': 1.0}
    
    # If downside vol > upside vol (ratio > 1) AND total vol is high → reduce
    # If upside vol dominates (ratio < 1) → stay in even if total vol is high
    if vr > 1.3 and vt > 0.9: b = 0.30     # strong downside dominance + high vol
    elif vr > 1.1 and vt > 0.9: b = 0.50   # mild downside dominance + high vol
    elif vr > 1.3 and vt > 0.7: b = 0.60   # downside dominance + moderate vol
    elif vt > 1.2 and vr < 0.8: b = 0.80   # high vol but UPSIDE dominant → stay!
    elif vt > 0.9 and vr < 0.9: b = 0.90   # moderate vol, upside dominant → stay!
    elif vt > 0.7: b = 0.80
    else: b = 1.0
    
    if not pd.isna(mm) and mm < 0.7: b = max(b, 0.80)
    if not pd.isna(r) and r < 25: b = max(b, 0.85)
    if not pd.isna(dd) and dd < -0.60: b = max(b, 0.90)
    return {'tqqq': b}

# E. Downside vol + Gold hedge (best of both worlds)
def strat_downvol_gold(d, p, i):
    v = i['vol_down']; mm = i['mayer']; r = i['rsi14']; dd = i['dd']
    gm = i['gld_mom']; tm = i['tlt_mom']
    if pd.isna(v): return {'tqqq': 1.0}
    
    if v > 0.70: tqqq_pct = 0.30
    elif v > 0.55: tqqq_pct = 0.50
    elif v > 0.40: tqqq_pct = 0.80
    else: tqqq_pct = 1.0
    
    if not pd.isna(mm) and mm < 0.7: tqqq_pct = max(tqqq_pct, 0.80)
    if not pd.isna(r) and r < 25: tqqq_pct = max(tqqq_pct, 0.85)
    if not pd.isna(dd) and dd < -0.60: tqqq_pct = max(tqqq_pct, 0.90)
    
    rest = 1.0 - tqqq_pct
    if rest <= 0: return {'tqqq': 1.0}
    if not pd.isna(gm) and not pd.isna(tm):
        if gm > tm: return {'tqqq': tqqq_pct, 'gld': rest*0.7, 'tlt': rest*0.3}
        else: return {'tqqq': tqqq_pct, 'tlt': rest*0.7, 'gld': rest*0.3}
    return {'tqqq': tqqq_pct, 'gld': rest*0.5, 'tlt': rest*0.5}

# F. Vol ratio + Gold hedge (the ultimate?)
def strat_volratio_gold(d, p, i):
    vr = i['vol_ratio']; vt = i['vol_total']; mm = i['mayer']; r = i['rsi14']; dd = i['dd']
    gm = i['gld_mom']; tm = i['tlt_mom']
    if pd.isna(vr) or pd.isna(vt): return {'tqqq': 1.0}
    
    if vr > 1.3 and vt > 0.9: tqqq_pct = 0.25
    elif vr > 1.1 and vt > 0.9: tqqq_pct = 0.45
    elif vr > 1.3 and vt > 0.7: tqqq_pct = 0.55
    elif vt > 1.2 and vr < 0.8: tqqq_pct = 0.85  # high vol but upside → hold!
    elif vt > 0.9 and vr < 0.9: tqqq_pct = 0.95  # upside dominant → almost full
    elif vt > 0.7: tqqq_pct = 0.80
    else: tqqq_pct = 1.0
    
    if not pd.isna(mm) and mm < 0.7: tqqq_pct = max(tqqq_pct, 0.80)
    if not pd.isna(r) and r < 25: tqqq_pct = max(tqqq_pct, 0.85)
    if not pd.isna(dd) and dd < -0.60: tqqq_pct = max(tqqq_pct, 0.90)
    
    rest = 1.0 - tqqq_pct
    if rest <= 0: return {'tqqq': 1.0}
    if not pd.isna(gm) and not pd.isna(tm):
        if gm > tm: return {'tqqq': tqqq_pct, 'gld': rest*0.7, 'tlt': rest*0.3}
        else: return {'tqqq': tqqq_pct, 'tlt': rest*0.7, 'gld': rest*0.3}
    return {'tqqq': tqqq_pct, 'gld': rest*0.5, 'tlt': rest*0.5}

# G. Adaptive Multi-Asset with downside vol (upgrade from v4)
def strat_adaptive_downvol(d, p, i):
    vd = i['vol_down']; vt = i['vol_total']; vr = i['vol_ratio']
    mm = i['mayer']; r = i['rsi14']; dd = i['dd']
    gm = i['gld_mom']; tm = i['tlt_mom']
    
    if pd.isna(vd) or pd.isna(vt): return {'tqqq': 1.0}
    
    # Use downside vol as primary signal
    if vd > 0.70: base = 0.25
    elif vd > 0.55: base = 0.45
    elif vd > 0.40: base = 0.75
    else: base = 1.0
    
    # If vol ratio < 1 (more upside than downside), boost allocation
    if not pd.isna(vr) and vr < 0.8 and vt > 0.7:
        base = min(1.0, base + 0.25)  # upside vol dominant, hold more
    
    # Crash overrides
    if not pd.isna(mm) and mm < 0.7: base = max(base, 0.80)
    if not pd.isna(r) and r < 25: base = max(base, 0.85)
    if not pd.isna(dd) and dd < -0.60: base = max(base, 0.90)
    
    # Bubble protection
    if not pd.isna(mm) and mm > 2.5: base = min(base, 0.40)
    
    rest = 1.0 - base
    if rest <= 0: return {'tqqq': 1.0}
    if not pd.isna(gm) and not pd.isna(tm):
        if gm > tm: return {'tqqq': base, 'gld': rest*0.7, 'tlt': rest*0.3}
        else: return {'tqqq': base, 'tlt': rest*0.7, 'gld': rest*0.3}
    return {'tqqq': base, 'gld': rest*0.5, 'tlt': rest*0.5}

# H. Previous best (Adaptive Multi with total vol) for comparison
def strat_v4_adaptive(d, p, i):
    v = i['vol_total']; mm = i['mayer']; r = i['rsi14']; dd = i['dd']
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

print(f"\n  {'Strategy':55s} | {'Final':>16s} | {'DD':>7s} | {'Sh':>5s}")
print("  " + "=" * 100)

results = {}
for name, fn in [
    ("📈 100% TQQQ B&H DCA", strat_bh),
    ("A. v6 Total Vol (cash)", strat_v6_total_vol),
    ("B. Downside Vol Only (cash)", strat_downvol),
    ("C. Semi-Dev Targeting (cash)", strat_semi_dev),
    ("D. Vol Ratio (up vs down) (cash)", strat_vol_ratio),
    ("E. Downside Vol + Gold/TLT Hedge", strat_downvol_gold),
    ("F. Vol Ratio + Gold/TLT Hedge", strat_volratio_gold),
    ("G. 🏆 Adaptive Downside Vol + Gold/TLT", strat_adaptive_downvol),
    ("H. v4 Adaptive (total vol) for comparison", strat_v4_adaptive),
]:
    results[name] = run_strategy(fn, name)

bh_f = results["📈 100% TQQQ B&H DCA"]['Value'].iloc[-1]
print(f"\n  Ranked (vs B&H ${bh_f:,.0f}):")
print("  " + "-" * 100)
ranked = sorted(results.items(), key=lambda x: x[1]['Value'].iloc[-1], reverse=True)
for i, (n, h) in enumerate(ranked):
    f = h['Value'].iloc[-1]; m = f/h['Invested'].iloc[-1]
    pk = h['Value'].cummax(); dd = ((h['Value']-pk)/pk).min()
    beat = "✅" if f > bh_f else "❌"
    print(f"  #{i+1} {beat} {n:53s} ${f:>11,.0f} ({m:5.1f}x) DD:{dd*100:5.1f}% vs B&H:{(f/bh_f-1)*100:+.1f}%")
