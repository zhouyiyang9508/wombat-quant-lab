"""
BTC DCA v6 — Refine Halving Cycle Strategy
Explore variations on cycle timing + sell triggers.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data_cache/BTC_USD.csv', parse_dates=['Date']).set_index('Date').sort_index()
prices = df['Close'].dropna()

sma200 = prices.rolling(200).mean()
mayer = prices / sma200
sma111 = prices.rolling(111).mean()
sma350x2 = prices.rolling(350).mean() * 2

def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

rsi14 = calc_rsi(prices, 14)
ath = prices.cummax()
dd_from_ath = (prices - ath) / ath

halving_dates = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

def days_since_halving(date):
    for h in reversed(halving_dates):
        if date >= h:
            return (date - h).days
    return 9999

monthly_dates = prices.groupby(prices.index.to_period('M')).apply(lambda x: x.index[0])
all_dates = prices.index

def run_strategy(strategy_fn, name, monthly_amount=1000):
    shares = 0.0; cash = 0.0; total_invested = 0.0
    daily_values = []
    rebalance_dates = set(monthly_dates)
    
    for date in all_dates:
        p = prices.loc[date]
        if date in rebalance_dates:
            cash += monthly_amount
            total_invested += monthly_amount
            pv = shares * p + cash
            ind = {
                'sma200': sma200.get(date, np.nan),
                'mayer': mayer.get(date, np.nan),
                'sma111': sma111.get(date, np.nan),
                'sma350x2': sma350x2.get(date, np.nan),
                'rsi14': rsi14.get(date, np.nan),
                'dd_from_ath': dd_from_ath.get(date, np.nan),
                'dsh': days_since_halving(date),
            }
            target_pct = strategy_fn(date, p, ind)
            target_eq = pv * target_pct
            curr_eq = shares * p
            diff = target_eq - curr_eq
            if diff > 0 and cash >= diff:
                shares += diff / p; cash -= diff
            elif diff > 0:
                shares += cash / p; cash = 0
            elif diff < 0:
                sell = abs(diff); shares -= sell / p; cash += sell
        daily_values.append((date, shares * p + cash, total_invested))
    
    hist = pd.DataFrame(daily_values, columns=['Date', 'Value', 'Invested']).set_index('Date')
    final = hist['Value'].iloc[-1]
    invested = hist['Invested'].iloc[-1]
    mult = final / invested
    peak = hist['Value'].cummax()
    max_dd = ((hist['Value'] - peak) / peak).min()
    monthly_vals = hist['Value'].resample('ME').last().dropna()
    monthly_rets = monthly_vals.pct_change().dropna()
    sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12)
    
    print(f"  {name:50s} | ${final:>10,.0f} ({mult:5.1f}x) | DD:{max_dd*100:5.1f}% | Sh:{sharpe:.2f}")
    return hist

def strat_bh(date, price, ind):
    return 1.0

# ── Halving Cycle Variations ──

# V1: Original (from research)
def strat_v1_orig(date, price, ind):
    dsh = ind['dsh']; mm = ind['mayer']; r14 = ind['rsi14']
    if dsh < 180: return 1.0
    elif dsh < 365: return 1.0
    elif dsh < 540:
        if not pd.isna(mm) and mm > 2.5: return 0.30
        elif not pd.isna(mm) and mm > 2.0: return 0.50
        elif not pd.isna(mm) and mm > 1.5: return 0.70
        else: return 0.90
    elif dsh < 900:
        if not pd.isna(r14) and r14 < 25: return 1.0
        elif not pd.isna(mm) and mm < 0.7: return 1.0
        else: return 0.50
    else: return 1.0

# V2: More aggressive top selling
def strat_v2_aggressive_sell(date, price, ind):
    dsh = ind['dsh']; mm = ind['mayer']; r14 = ind['rsi14']; dd = ind['dd_from_ath']
    if dsh < 180: return 1.0
    elif dsh < 365: return 1.0
    elif dsh < 540:
        if not pd.isna(mm) and mm > 2.5: return 0.15
        elif not pd.isna(mm) and mm > 2.0: return 0.30
        elif not pd.isna(mm) and mm > 1.5: return 0.50
        else: return 0.85
    elif dsh < 900:
        if not pd.isna(r14) and r14 < 25: return 1.0
        elif not pd.isna(mm) and mm < 0.7: return 1.0
        elif not pd.isna(dd) and dd < -0.60: return 1.0
        else: return 0.30
    else: return 1.0

# V3: Earlier sell, earlier buy
def strat_v3_early(date, price, ind):
    dsh = ind['dsh']; mm = ind['mayer']; r14 = ind['rsi14']; dd = ind['dd_from_ath']
    if dsh < 180: return 1.0
    elif dsh < 300:  # start selling earlier
        if not pd.isna(mm) and mm > 2.0: return 0.40
        return 1.0
    elif dsh < 540:
        if not pd.isna(mm) and mm > 2.0: return 0.20
        elif not pd.isna(mm) and mm > 1.5: return 0.40
        else: return 0.70
    elif dsh < 750:  # buy back earlier
        if not pd.isna(dd) and dd < -0.50: return 1.0
        elif not pd.isna(r14) and r14 < 30: return 1.0
        elif not pd.isna(mm) and mm < 0.8: return 1.0
        else: return 0.40
    else: return 1.0

# V4: Halving + Pi Cycle
def strat_v4_halving_pi(date, price, ind):
    dsh = ind['dsh']; mm = ind['mayer']; r14 = ind['rsi14']
    s111 = ind['sma111']; s350x2 = ind['sma350x2']
    dd = ind['dd_from_ath']
    
    # Pi Cycle signal
    pi_top = False
    if not pd.isna(s111) and not pd.isna(s350x2):
        pi_top = s111 / s350x2 > 0.92
    
    if dsh < 180: return 1.0
    elif dsh < 365:
        if pi_top and not pd.isna(mm) and mm > 2.0: return 0.30  # early top
        return 1.0
    elif dsh < 540:
        if pi_top: return 0.20
        if not pd.isna(mm) and mm > 2.5: return 0.25
        elif not pd.isna(mm) and mm > 2.0: return 0.40
        elif not pd.isna(mm) and mm > 1.5: return 0.60
        else: return 0.90
    elif dsh < 900:
        if not pd.isna(dd) and dd < -0.60: return 1.0
        if not pd.isna(r14) and r14 < 25: return 1.0
        elif not pd.isna(mm) and mm < 0.7: return 1.0
        else: return 0.40
    else: return 1.0

# V5: Mayer-weighted within cycle framework
def strat_v5_mayer_cycle(date, price, ind):
    dsh = ind['dsh']; mm = ind['mayer']; r14 = ind['rsi14']; dd = ind['dd_from_ath']
    if pd.isna(mm): return 1.0
    
    # Cycle phase determines how aggressive Mayer signals are
    if dsh < 365:
        # Early cycle: accumulate regardless
        return 1.0
    elif dsh < 600:
        # Mid-to-late cycle: Mayer drives allocation aggressively
        if mm > 3.0: return 0.10
        elif mm > 2.5: return 0.20
        elif mm > 2.0: return 0.35
        elif mm > 1.5: return 0.60
        else: return 1.0
    else:
        # Bear/accumulation phase
        if not pd.isna(dd) and dd < -0.65: return 1.0
        if mm < 0.6: return 1.0
        elif mm < 0.8: return 1.0
        elif mm > 1.5: return 0.50  # bear rally, reduce
        else: return 0.70

# V6: Gentle approach — mostly hold, only sell extreme tops
def strat_v6_gentle(date, price, ind):
    dsh = ind['dsh']; mm = ind['mayer']; r14 = ind['rsi14']
    if pd.isna(mm): return 1.0
    
    # Only act on extreme signals
    # Sell: late cycle + extreme Mayer
    if dsh > 300 and dsh < 600 and mm > 2.5:
        return 0.30
    elif dsh > 300 and dsh < 600 and mm > 2.0:
        return 0.60
    # Buy: deep bear
    elif dsh > 600 and mm < 0.7:
        return 1.0  # all in
    else:
        return 1.0

# V7: Phase-based with gradual transitions
def strat_v7_gradual(date, price, ind):
    dsh = ind['dsh']; mm = ind['mayer']; r14 = ind['rsi14']; dd = ind['dd_from_ath']
    if pd.isna(mm): return 1.0
    
    # Phase allocation (base)
    if dsh < 180: base = 1.0
    elif dsh < 365: base = 1.0
    elif dsh < 450: base = 0.80  # start cautious
    elif dsh < 540: base = 0.60  # peak zone
    elif dsh < 720: base = 0.40  # early bear
    elif dsh < 900: base = 0.70  # accumulation starts
    else: base = 1.0  # deep accumulation
    
    # Mayer overlay
    if mm > 2.5: base = min(base, 0.20)
    elif mm > 2.0: base = min(base, 0.40)
    elif mm < 0.6: base = max(base, 1.0)
    elif mm < 0.8: base = max(base, 0.90)
    
    # RSI overlay
    if not pd.isna(r14):
        if r14 < 25: base = max(base, 1.0)
        elif r14 > 85: base = min(base, 0.30)
    
    # Crash override
    if not pd.isna(dd) and dd < -0.65:
        base = max(base, 1.0)
    
    return base

print(f"BTC Halving Cycle Variants — {prices.index[0].date()} → {prices.index[-1].date()}\n")
print(f"  {'Strategy':50s} | {'Final':>16s} | {'DD':>7s} | {'Sh':>5s}")
print("  " + "=" * 95)

results = {}
for name, fn in [
    ("📈 BTC DCA Buy & Hold", strat_bh),
    ("V1: Original Halving Cycle", strat_v1_orig),
    ("V2: Aggressive Top Selling", strat_v2_aggressive_sell),
    ("V3: Early Sell + Early Buy", strat_v3_early),
    ("V4: Halving + Pi Cycle", strat_v4_halving_pi),
    ("V5: Mayer-Weighted Cycle", strat_v5_mayer_cycle),
    ("V6: Gentle (extreme tops only)", strat_v6_gentle),
    ("V7: Gradual Phase Transition", strat_v7_gradual),
]:
    results[name] = run_strategy(fn, name)

bh_f = results["📈 BTC DCA Buy & Hold"]['Value'].iloc[-1]
print(f"\n  Ranked (vs B&H ${bh_f:,.0f}):")
ranked = sorted(results.items(), key=lambda x: x[1]['Value'].iloc[-1], reverse=True)
for i, (n, h) in enumerate(ranked):
    f = h['Value'].iloc[-1]; m = f / h['Invested'].iloc[-1]
    pk = h['Value'].cummax(); dd = ((h['Value'] - pk) / pk).min()
    beat = "✅" if f > bh_f else "❌"
    print(f"  #{i+1} {beat} {n:48s} ${f:>10,.0f} ({m:5.1f}x) DD:{dd*100:5.1f}% vs B&H:{(f/bh_f-1)*100:+.1f}%")
