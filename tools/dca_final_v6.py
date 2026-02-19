"""
DCA Final Strategy — Beast v6 (Volatility Targeting)
Core insight: reduce exposure when vol is high (sell into panic), 
increase when vol is low (compound in calm).
Override: stay in during deep crashes (Mayer < 0.7, RSI < 25, DD < -60%).
"""
import pandas as pd
import numpy as np

def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def run_dca_full(prices, strategy_fn, monthly_amount=1000):
    """Full portfolio management DCA with daily tracking."""
    sma200 = prices.rolling(200).mean()
    mayer = prices / sma200
    rsi14 = calc_rsi(prices, 14)
    vol20 = prices.pct_change().rolling(20).std() * np.sqrt(252)
    ath = prices.cummax()
    dd_from_ath = (prices - ath) / ath
    
    monthly_dates = set(prices.groupby(prices.index.to_period('M')).apply(lambda x: x.index[0]))
    
    shares = 0.0; cash = 0.0; total_invested = 0.0
    daily_values = []
    
    for date in prices.index:
        p = prices.loc[date]
        if date in monthly_dates:
            cash += monthly_amount
            total_invested += monthly_amount
            pv = shares * p + cash
            
            ind = {
                'sma200': sma200.get(date, np.nan),
                'mayer': mayer.get(date, np.nan),
                'rsi14': rsi14.get(date, np.nan),
                'vol20': vol20.get(date, np.nan),
                'dd_from_ath': dd_from_ath.get(date, np.nan),
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
                sell = abs(diff)
                shares -= sell / p; cash += sell
        
        daily_values.append((date, shares * p + cash, total_invested))
    
    hist = pd.DataFrame(daily_values, columns=['Date', 'Value', 'Invested']).set_index('Date')
    return hist

def metrics(hist, name, bh_final=None):
    final = hist['Value'].iloc[-1]
    invested = hist['Invested'].iloc[-1]
    mult = final / invested
    years = (hist.index[-1] - hist.index[0]).days / 365.25
    peak = hist['Value'].cummax()
    dd = (hist['Value'] - peak) / peak
    max_dd = dd.min()
    monthly_vals = hist['Value'].resample('ME').last().dropna()
    monthly_rets = monthly_vals.pct_change().dropna()
    sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12) if len(monthly_rets) > 1 else 0
    calmar_r = (mult ** (1/years) - 1)
    calmar = calmar_r / abs(max_dd) if max_dd != 0 else 0
    
    vs = f"vs B&H: {(final/bh_final-1)*100:+.1f}%" if bh_final else ""
    print(f"  {name:42s} | ${final:>11,.0f} ({mult:5.1f}x) | MaxDD: {max_dd*100:5.1f}% | Sharpe: {sharpe:.2f} | Calmar: {calmar:.2f} | {vs}")
    return final

# ══════════════════════════════════════════════
# STRATEGIES
# ══════════════════════════════════════════════

def strat_bh(date, price, ind):
    return 1.0

# V6 Beast — Vol Targeting with crash overrides
def strat_v6_beast(date, price, ind):
    v = ind['vol20']
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    
    if pd.isna(v): return 1.0
    
    # Base: inverse vol with threshold breakpoints
    if v > 1.2:
        base = 0.30
    elif v > 0.9:
        base = 0.50
    elif v > 0.7:
        base = 0.80
    else:
        base = 1.0
    
    # Override 1: Deep value — stay in even during high vol
    if not pd.isna(mm) and mm < 0.7:
        base = max(base, 0.80)
    
    # Override 2: RSI extreme oversold
    if not pd.isna(r14) and r14 < 25:
        base = max(base, 0.85)
    
    # Override 3: Extreme crash
    if not pd.isna(dd) and dd < -0.60:
        base = max(base, 0.90)
    
    return base

# V6b — Smoother vol targeting (no hard thresholds)
def strat_v6b_smooth(date, price, ind):
    v = ind['vol20']
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    
    if pd.isna(v): return 1.0
    
    # Smooth vol targeting: target = 0.55 annualized
    base = min(1.0, 0.55 / v)
    
    # Crash overrides
    if not pd.isna(mm) and mm < 0.7:
        base = max(base, 0.80)
    if not pd.isna(dd) and dd < -0.60:
        base = max(base, 0.90)
    if not pd.isna(r14) and r14 < 25:
        base = max(base, 0.85)
    
    # Bubble protection
    if not pd.isna(mm) and mm > 2.5:
        base = min(base, 0.40)
    
    return max(0.15, base)

# V6c — Vol targeting + DD override (simpler, fewer params)
def strat_v6c_voldd(date, price, ind):
    v = ind['vol20']
    dd = ind['dd_from_ath']
    
    if pd.isna(v): return 1.0
    
    # Vol targeting
    base = min(1.0, 0.55 / v)
    
    # Deep drawdown override: be greedy
    if not pd.isna(dd):
        if dd < -0.65: base = max(base, 1.0)
        elif dd < -0.50: base = max(base, 0.80)
        elif dd < -0.35: base = max(base, 0.60)
    
    return max(0.10, base)

# V6d — Threshold vol + enhanced crash buying
def strat_v6d_enhanced(date, price, ind):
    v = ind['vol20']
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    
    if pd.isna(v): return 1.0
    
    # Threshold-based (like original but tweaked)
    if v > 1.1: base = 0.25
    elif v > 0.85: base = 0.50
    elif v > 0.65: base = 0.80
    else: base = 1.0
    
    # Aggressive crash overrides
    if not pd.isna(dd) and dd < -0.65:
        base = max(base, 1.0)
    elif not pd.isna(mm) and mm < 0.6:
        base = max(base, 0.90)
    elif not pd.isna(r14) and r14 < 20:
        base = max(base, 0.90)
    elif not pd.isna(mm) and mm < 0.7:
        base = max(base, 0.80)
    elif not pd.isna(r14) and r14 < 30:
        base = max(base, 0.70)
    
    return base

# ══════════════════════════════════════════════
# RUN TQQQ
# ══════════════════════════════════════════════
print("=" * 120)
print("TQQQ DCA COMPARISON ($1,000/month)")
print("=" * 120)

tqqq = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date').sort_index()['Close'].dropna()

tqqq_results = {}
for name, fn in [
    ("📈 DCA Buy & Hold", strat_bh),
    ("🐻 v6 Beast (threshold)", strat_v6_beast),
    ("🐻 v6b Smooth (target=0.55)", strat_v6b_smooth),
    ("🐻 v6c Vol+DD", strat_v6c_voldd),
    ("🐻 v6d Enhanced", strat_v6d_enhanced),
]:
    tqqq_results[name] = run_dca_full(tqqq, fn)

bh_t = tqqq_results["📈 DCA Buy & Hold"]['Value'].iloc[-1]
print(f"\n  {'Strategy':42s} | {'Final (mult)':>20s} | {'MaxDD':>8s} | {'Sharpe':>7s} | {'Calmar':>7s} |")
print("  " + "-" * 115)
for name, hist in tqqq_results.items():
    metrics(hist, name, bh_t)

# ══════════════════════════════════════════════
# RUN BTC
# ══════════════════════════════════════════════
print("\n" + "=" * 120)
print("BTC DCA COMPARISON ($1,000/month)")
print("=" * 120)

btc = pd.read_csv('data_cache/BTC_USD.csv', parse_dates=['Date']).set_index('Date').sort_index()['Close'].dropna()

btc_results = {}
for name, fn in [
    ("📈 DCA Buy & Hold", strat_bh),
    ("🐻 v6 Beast (threshold)", strat_v6_beast),
    ("🐻 v6b Smooth (target=0.55)", strat_v6b_smooth),
    ("🐻 v6c Vol+DD", strat_v6c_voldd),
    ("🐻 v6d Enhanced", strat_v6d_enhanced),
]:
    btc_results[name] = run_dca_full(btc, fn)

bh_b = btc_results["📈 DCA Buy & Hold"]['Value'].iloc[-1]
print(f"\n  {'Strategy':42s} | {'Final (mult)':>20s} | {'MaxDD':>8s} | {'Sharpe':>7s} | {'Calmar':>7s} |")
print("  " + "-" * 115)
for name, hist in btc_results.items():
    metrics(hist, name, bh_b)
