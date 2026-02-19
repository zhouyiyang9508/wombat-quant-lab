"""
Deep robustness test for the Simple Vol strategy (2 params: threshold + allocation)
Exhaustive parameter sweep + multiple validation methods
"""
import pandas as pd
import numpy as np
from itertools import product

tqqq = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
vol20 = tqqq.pct_change().rolling(20).std() * np.sqrt(252)

monthly_dates = tqqq.groupby(tqqq.index.to_period('M')).apply(lambda x: x.index[0])

def run_simple_vol(start, end, vol_thresh, bear_alloc, monthly_amount=1000):
    mask = (tqqq.index >= start) & (tqqq.index <= end)
    dates = tqqq.index[mask]
    if len(dates) < 200: return None
    
    rebal = set(monthly_dates[monthly_dates.isin(dates)])
    shares = 0.0; cash = 0.0; total_inv = 0.0; vals = []
    
    for date in dates:
        p = tqqq.loc[date]
        if date in rebal:
            cash += monthly_amount; total_inv += monthly_amount
            pv = shares*p + cash
            v = vol20.get(date, np.nan)
            target = bear_alloc if (not pd.isna(v) and v > vol_thresh) else 1.0
            target_eq = pv * target
            diff = target_eq - shares*p
            if diff > 0:
                buy = min(diff, cash); shares += buy/p; cash -= buy
            elif diff < 0:
                sell = abs(diff); shares -= sell/p; cash += sell
        pv = shares*p + cash
        vals.append((date, pv, total_inv))
    
    h = pd.DataFrame(vals, columns=['Date','Value','Invested']).set_index('Date')
    f = h['Value'].iloc[-1]; inv = h['Invested'].iloc[-1]
    mult = f/inv; years = (h.index[-1]-h.index[0]).days/365.25
    pk = h['Value'].cummax(); mdd = ((h['Value']-pk)/pk).min()
    mr = h['Value'].resample('ME').last().dropna().pct_change().dropna()
    sh = mr.mean()/mr.std()*np.sqrt(12) if len(mr)>1 and mr.std()>0 else 0
    return {'mult': mult, 'mdd': mdd, 'sharpe': sh, 'final': f, 'invested': inv, 'years': years}

def run_bh(start, end, monthly_amount=1000):
    return run_simple_vol(start, end, 999, 1.0, monthly_amount)

# ════════════════════════════════════════════════════════
# TEST 1: Full parameter sweep (threshold x allocation)
# ════════════════════════════════════════════════════════
print("=" * 110)
print("TEST 1: Parameter Sweep — every combo of vol threshold × bear allocation")
print("=" * 110)

thresholds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
allocations = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

bh = run_bh('2010-02-11', '2026-02-18')
bh_mult = bh['mult']

print(f"\nB&H baseline: {bh_mult:.1f}x | MaxDD: {bh['mdd']*100:.1f}%\n")

# Heatmap: rows = threshold, cols = allocation
print(f"{'':>8s}", end="")
for a in allocations:
    print(f" alloc={a:.0%}".rjust(10), end="")
print()

beat_count = 0
total_count = 0
all_results = []

for t in thresholds:
    print(f"vol>{t:.1f}", end=" ")
    for a in allocations:
        r = run_simple_vol('2010-02-11', '2026-02-18', t, a)
        total_count += 1
        if r and r['mult'] > bh_mult:
            beat_count += 1
            print(f" {r['mult']:7.1f}x✅", end="")
        elif r:
            print(f" {r['mult']:7.1f}x  ", end="")
        all_results.append({'thresh': t, 'alloc': a, **r} if r else None)
    print()

print(f"\n{beat_count}/{total_count} combos beat B&H ({beat_count/total_count*100:.0f}%)")

# ════════════════════════════════════════════════════════
# TEST 2: Walk-forward on the best parameter region
# ════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("TEST 2: Walk-Forward validation — 3 periods")
print("=" * 110)

periods = [
    ("2010-2015", "2010-02-11", "2015-12-31"),
    ("2016-2020", "2016-01-01", "2020-12-31"),
    ("2021-2026", "2021-01-01", "2026-02-18"),
]

test_configs = [
    ("vol>0.8, alloc=30%", 0.8, 0.30),
    ("vol>0.9, alloc=20%", 0.9, 0.20),
    ("vol>0.9, alloc=30%", 0.9, 0.30),
    ("vol>0.9, alloc=40%", 0.9, 0.40),
    ("vol>1.0, alloc=30%", 1.0, 0.30),
    ("vol>1.1, alloc=30%", 1.1, 0.30),
]

print(f"\n  {'Config':25s}", end="")
for label, _, _ in periods:
    print(f" | {label:>22s}", end="")
print(f" |  {'Wins':>4s}")
print("  " + "-" * 100)

for name, thresh, alloc in test_configs:
    wins = 0
    print(f"  {name:25s}", end="")
    for label, start, end in periods:
        r = run_simple_vol(start, end, thresh, alloc)
        b = run_bh(start, end)
        if r and b:
            beat = r['mult'] > b['mult']
            if beat: wins += 1
            mark = "✅" if beat else "❌"
            print(f" | {r['mult']:5.1f}x vs {b['mult']:4.1f}x {mark}", end="")
    print(f" |  {wins}/3")

# ════════════════════════════════════════════════════════
# TEST 3: Every single year
# ════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("TEST 3: Year-by-Year — vol>0.9→30% vs B&H")
print("=" * 110)

print(f"\n  {'Year':6s} | {'B&H':>8s} | {'Simple':>8s} | {'Δ':>8s} | Win?")
print("  " + "-" * 50)

yearly_wins = 0; yearly_total = 0
for year in range(2011, 2026):
    start = f"{year}-01-01"; end = f"{year}-12-31"
    r = run_simple_vol(start, end, 0.9, 0.30)
    b = run_bh(start, end)
    if r and b:
        yearly_total += 1
        delta = r['mult'] - b['mult']
        win = delta > 0
        if win: yearly_wins += 1
        print(f"  {year:6d} | {b['mult']:6.2f}x | {r['mult']:6.2f}x | {delta:+6.2f}x | {'✅' if win else '❌'}")

print(f"\n  Won {yearly_wins}/{yearly_total} years ({yearly_wins/yearly_total*100:.0f}%)")

# ════════════════════════════════════════════════════════
# TEST 4: Different vol windows (10, 20, 30, 60 days)
# ════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("TEST 4: Vol Window Sensitivity — is 20-day the best lookback?")
print("=" * 110)

for window in [10, 15, 20, 30, 40, 60]:
    vol_w = tqqq.pct_change().rolling(window).std() * np.sqrt(252)
    
    # Override vol20 temporarily
    old_vol = vol20.copy()
    
    # Run manually
    mask = (tqqq.index >= '2010-02-11') & (tqqq.index <= '2026-02-18')
    dates = tqqq.index[mask]
    rebal = set(monthly_dates[monthly_dates.isin(dates)])
    shares = 0.0; cash = 0.0; total_inv = 0.0; vals = []
    
    for date in dates:
        p = tqqq.loc[date]
        if date in rebal:
            cash += 1000; total_inv += 1000
            pv = shares*p + cash
            v = vol_w.get(date, np.nan)
            target = 0.30 if (not pd.isna(v) and v > 0.9) else 1.0
            target_eq = pv * target
            diff = target_eq - shares*p
            if diff > 0:
                buy = min(diff, cash); shares += buy/p; cash -= buy
            elif diff < 0:
                sell = abs(diff); shares -= sell/p; cash += sell
        vals.append((date, shares*p + cash, total_inv))
    
    h = pd.DataFrame(vals, columns=['Date','Value','Invested']).set_index('Date')
    f = h['Value'].iloc[-1]; inv = h['Invested'].iloc[-1]; mult = f/inv
    pk = h['Value'].cummax(); mdd = ((h['Value']-pk)/pk).min()
    mr = h['Value'].resample('ME').last().dropna().pct_change().dropna()
    sh = mr.mean()/mr.std()*np.sqrt(12) if len(mr)>1 else 0
    
    print(f"  Vol window {window:2d}d | {mult:5.1f}x | MaxDD: {mdd*100:5.1f}% | Sharpe: {sh:.2f}")

# ════════════════════════════════════════════════════════
# TEST 5: Bootstrap — random 8-year sub-samples
# ════════════════════════════════════════════════════════
print("\n" + "=" * 110)
print("TEST 5: Bootstrap — 50 random 8-year sub-samples")
print("=" * 110)

np.random.seed(42)
n_bootstrap = 50
wins = 0; total = 0
deltas = []

all_dates_list = tqqq.index.tolist()
max_start_idx = len(all_dates_list) - 252*8  # need 8 years

for i in range(n_bootstrap):
    start_idx = np.random.randint(0, max_start_idx)
    start = all_dates_list[start_idx]
    end = all_dates_list[min(start_idx + 252*8, len(all_dates_list)-1)]
    
    r = run_simple_vol(str(start.date()), str(end.date()), 0.9, 0.30)
    b = run_bh(str(start.date()), str(end.date()))
    
    if r and b:
        total += 1
        if r['mult'] > b['mult']:
            wins += 1
        deltas.append(r['mult'] - b['mult'])

deltas = np.array(deltas)
print(f"  Wins: {wins}/{total} ({wins/total*100:.0f}%)")
print(f"  Avg outperformance: {deltas.mean():.2f}x")
print(f"  Median outperformance: {np.median(deltas):.2f}x")
print(f"  Worst: {deltas.min():.2f}x | Best: {deltas.max():.2f}x")
print(f"  % positive: {(deltas > 0).sum()/len(deltas)*100:.0f}%")
