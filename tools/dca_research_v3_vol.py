"""
DCA Research v3 — Deep dive into Volatility-based strategies
The Inverse Vol approach beat B&H in v2. Now let's optimize and validate.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date').sort_index()
prices = df['Close'].dropna()

# Indicators
sma200 = prices.rolling(200).mean()
mayer = prices / sma200

def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

rsi14 = calc_rsi(prices, 14)
ath = prices.cummax()
dd_from_ath = (prices - ath) / ath

# Multiple vol windows
vol10 = prices.pct_change().rolling(10).std() * np.sqrt(252)
vol20 = prices.pct_change().rolling(20).std() * np.sqrt(252)
vol30 = prices.pct_change().rolling(30).std() * np.sqrt(252)
vol60 = prices.pct_change().rolling(60).std() * np.sqrt(252)

monthly_dates = prices.groupby(prices.index.to_period('M')).apply(lambda x: x.index[0])
all_dates = prices.index

print(f"Data: {prices.index[0].date()} → {prices.index[-1].date()}\n")

# Vol percentile (rolling 1-year)
vol20_pct = vol20.rolling(252).rank(pct=True)

def run_strategy(strategy_fn, name, monthly_amount=1000):
    shares = 0.0
    cash = 0.0
    total_invested = 0.0
    daily_values = []
    rebalance_dates = set(monthly_dates)
    
    for date in all_dates:
        p = prices.loc[date]
        
        if date in rebalance_dates:
            cash += monthly_amount
            total_invested += monthly_amount
            portfolio_val = shares * p + cash
            
            ind = {
                'sma200': sma200.get(date, np.nan),
                'mayer': mayer.get(date, np.nan),
                'rsi14': rsi14.get(date, np.nan),
                'dd_from_ath': dd_from_ath.get(date, np.nan),
                'vol10': vol10.get(date, np.nan),
                'vol20': vol20.get(date, np.nan),
                'vol30': vol30.get(date, np.nan),
                'vol60': vol60.get(date, np.nan),
                'vol20_pct': vol20_pct.get(date, np.nan),
                'price': p
            }
            
            target_pct = strategy_fn(date, p, ind)
            target_equity = portfolio_val * target_pct
            current_equity = shares * p
            diff = target_equity - current_equity
            
            if diff > 0 and cash >= diff:
                shares += diff / p; cash -= diff
            elif diff > 0:
                shares += cash / p; cash = 0
            elif diff < 0:
                sell = abs(diff)
                shares -= sell / p; cash += sell
        
        portfolio_val = shares * p + cash
        daily_values.append((date, portfolio_val, total_invested))
    
    hist = pd.DataFrame(daily_values, columns=['Date', 'Value', 'Invested']).set_index('Date')
    final = hist['Value'].iloc[-1]
    invested = hist['Invested'].iloc[-1]
    mult = final / invested
    peak = hist['Value'].cummax()
    dd_series = (hist['Value'] - peak) / peak
    max_dd = dd_series.min()
    monthly_vals = hist['Value'].resample('ME').last().dropna()
    monthly_rets = monthly_vals.pct_change().dropna()
    sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12) if len(monthly_rets) > 1 else 0
    calmar = (mult ** (1 / ((hist.index[-1] - hist.index[0]).days / 365.25)) - 1) / abs(max_dd) if max_dd != 0 else 0
    
    print(f"{name:45s} | ${final:>11,.0f} | {mult:5.1f}x | {max_dd*100:6.1f}% | {sharpe:5.2f} | {calmar:5.2f}")
    return hist

# ══════════════════════════════════════════════
# Baseline
# ══════════════════════════════════════════════
def strat_bh(date, price, ind):
    return 1.0

# ══════════════════════════════════════════════
# Vol-based strategies — systematic exploration
# ══════════════════════════════════════════════

# V1: Original inverse vol (from v2)
def strat_inv_vol_orig(date, price, ind):
    v = ind['vol20']
    mm = ind['mayer']
    if pd.isna(v): return 1.0
    if v > 1.2:
        if not pd.isna(mm) and mm < 0.7: return 0.80
        return 0.30
    elif v > 0.9: return 0.50
    elif v > 0.7: return 0.80
    else: return 1.0

# V2: Vol percentile-based (adaptive to regime)
def strat_vol_percentile(date, price, ind):
    vp = ind['vol20_pct']
    if pd.isna(vp): return 1.0
    # Linear scaling: 0th pct → 100%, 100th pct → 20%
    return max(0.20, 1.0 - 0.8 * vp)

# V3: Vol threshold with smoother transitions
def strat_vol_smooth(date, price, ind):
    v = ind['vol20']
    if pd.isna(v): return 1.0
    # Sigmoid-like mapping: low vol→100%, high vol→20%
    # Target vol of 0.60 (annualized) → TQQQ normal is ~50-60%
    target_vol = 0.60
    alloc = min(1.0, target_vol / v)
    return max(0.15, alloc)

# V4: Vol targeting + mean reversion (hybrid)
def strat_vol_target_mr(date, price, ind):
    v = ind['vol20']
    mm = ind['mayer']
    if pd.isna(v): return 1.0
    
    # Vol targeting base
    target_vol = 0.55
    base_alloc = min(1.0, target_vol / v)
    
    # Mean reversion adjustment
    if not pd.isna(mm):
        if mm < 0.6: base_alloc = min(1.0, base_alloc + 0.30)
        elif mm < 0.8: base_alloc = min(1.0, base_alloc + 0.15)
        elif mm > 2.0: base_alloc = max(0.10, base_alloc - 0.20)
        elif mm > 1.5: base_alloc = max(0.15, base_alloc - 0.10)
    
    return max(0.10, min(1.0, base_alloc))

# V5: Vol targeting + RSI (don't sell when oversold)
def strat_vol_rsi(date, price, ind):
    v = ind['vol20']
    r14 = ind['rsi14']
    if pd.isna(v): return 1.0
    
    target_vol = 0.55
    base_alloc = min(1.0, target_vol / v)
    
    # RSI override: when oversold in high vol, INCREASE allocation
    if not pd.isna(r14):
        if r14 < 25: base_alloc = min(1.0, base_alloc + 0.40)
        elif r14 < 35: base_alloc = min(1.0, base_alloc + 0.20)
        elif r14 > 80: base_alloc = max(0.20, base_alloc - 0.15)
    
    return max(0.10, min(1.0, base_alloc))

# V6: Different target vols
def make_vol_target(tv):
    def strat(date, price, ind):
        v = ind['vol20']
        if pd.isna(v): return 1.0
        return max(0.15, min(1.0, tv / v))
    return strat

# V7: Vol + DD combo
def strat_vol_dd(date, price, ind):
    v = ind['vol20']
    dd = ind['dd_from_ath']
    if pd.isna(v): return 1.0
    
    target_vol = 0.55
    base = min(1.0, target_vol / v)
    
    # Deep drawdown override: be greedy when others are fearful
    if not pd.isna(dd):
        if dd < -0.65: base = max(base, 0.90)  # huge crash = override vol, buy
        elif dd < -0.50: base = max(base, 0.70)
        elif dd < -0.35: base = max(base, 0.50)
    
    return max(0.10, min(1.0, base))

# V8: Vol with 60-day window (slower, less whipsaw)
def strat_vol60(date, price, ind):
    v = ind['vol60']
    dd = ind['dd_from_ath']
    if pd.isna(v): return 1.0
    
    target_vol = 0.55
    base = min(1.0, target_vol / v)
    
    if not pd.isna(dd):
        if dd < -0.60: base = max(base, 0.85)
        elif dd < -0.45: base = max(base, 0.60)
    
    return max(0.10, min(1.0, base))

# V9: Multi-timeframe vol (10 + 60 consensus)
def strat_vol_multi(date, price, ind):
    v10 = ind['vol10']
    v60 = ind['vol60']
    dd = ind['dd_from_ath']
    
    if pd.isna(v10) or pd.isna(v60): return 1.0
    
    # Use max of short and long vol for conservative estimate
    v = max(v10, v60)
    target_vol = 0.55
    base = min(1.0, target_vol / v)
    
    # If short vol < long vol, vol is declining → increase
    if v10 < v60 * 0.7:
        base = min(1.0, base + 0.20)  # vol declining, recovery
    
    if not pd.isna(dd):
        if dd < -0.60: base = max(base, 0.90)
    
    return max(0.10, min(1.0, base))

# V10: Best combo — vol target + RSI + DD (kitchen sink but refined)
def strat_best_combo(date, price, ind):
    v = ind['vol20']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    mm = ind['mayer']
    
    if pd.isna(v): return 1.0
    
    # Base: vol targeting
    target_vol = 0.55
    base = min(1.0, target_vol / v)
    
    # RSI adjustment
    if not pd.isna(r14):
        if r14 < 20: base = min(1.0, base + 0.50)
        elif r14 < 30: base = min(1.0, base + 0.30)
        elif r14 < 40: base = min(1.0, base + 0.10)
        elif r14 > 80: base = max(0.15, base - 0.15)
    
    # DD override (greedy when fearful)
    if not pd.isna(dd):
        if dd < -0.65: base = max(base, 1.0)
        elif dd < -0.50: base = max(base, 0.80)
    
    # Mayer extreme protection
    if not pd.isna(mm):
        if mm > 2.5: base = min(base, 0.40)
        elif mm > 2.0: base = min(base, 0.60)
    
    return max(0.10, min(1.0, base))

print(f"{'Strategy':45s} | {'Final':>12s} | {'Mult':>5s} | {'MaxDD':>7s} | {'Shpe':>5s} | {'Calmar':>5s}")
print("=" * 100)

strategies = [
    ("📈 DCA Buy & Hold", strat_bh),
    ("📊 InvVol Original", strat_inv_vol_orig),
    ("📊 Vol Percentile", strat_vol_percentile),
    ("📊 Vol Smooth (target=0.60)", strat_vol_smooth),
    ("📊 Vol Target + MeanRev", strat_vol_target_mr),
    ("📊 Vol + RSI Override", strat_vol_rsi),
    ("📊 Vol Target 0.45", make_vol_target(0.45)),
    ("📊 Vol Target 0.50", make_vol_target(0.50)),
    ("📊 Vol Target 0.55", make_vol_target(0.55)),
    ("📊 Vol Target 0.60", make_vol_target(0.60)),
    ("📊 Vol Target 0.65", make_vol_target(0.65)),
    ("📊 Vol + DD Override", strat_vol_dd),
    ("📊 Vol60 (slower)", strat_vol60),
    ("📊 Vol Multi-TF", strat_vol_multi),
    ("🏆 Best Combo (Vol+RSI+DD)", strat_best_combo),
]

results = {}
for name, fn in strategies:
    results[name] = run_strategy(fn, name)

print("\n" + "=" * 100)
print("\nRanked by Final Value:")
ranked = sorted(results.items(), key=lambda x: x[1]['Value'].iloc[-1], reverse=True)
bh_final = results["📈 DCA Buy & Hold"]['Value'].iloc[-1]
for i, (name, h) in enumerate(ranked):
    final = h['Value'].iloc[-1]
    invested = h['Invested'].iloc[-1]
    mult = final / invested
    peak = h['Value'].cummax()
    dd = ((h['Value'] - peak) / peak).min()
    beat = "✅" if final > bh_final else "❌"
    pct_diff = (final / bh_final - 1) * 100
    print(f"  #{i+1} {beat} {name:42s} ${final:>11,.0f}  ({mult:5.1f}x)  MaxDD:{dd*100:5.1f}%  vs B&H: {pct_diff:+.1f}%")
