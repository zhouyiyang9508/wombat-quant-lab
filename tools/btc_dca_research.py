"""
BTC DCA Research — Find strategy to beat B&H
BTC-specific signals: halving cycle, Mayer Multiple, long-term trend
Vol targeting doesn't work for BTC (always high vol even in bull).
Need different approach.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data_cache/BTC_USD.csv', parse_dates=['Date']).set_index('Date').sort_index()
prices = df['Close'].dropna()

# Indicators
sma200 = prices.rolling(200).mean()
sma50 = prices.rolling(50).mean()
sma100 = prices.rolling(100).mean()
mayer = prices / sma200

def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

rsi14 = calc_rsi(prices, 14)
vol20 = prices.pct_change().rolling(20).std() * np.sqrt(252)
ath = prices.cummax()
dd_from_ath = (prices - ath) / ath

# Mayer Multiple percentile (350-day rolling)
mayer_pct = mayer.rolling(350).rank(pct=True)

# 200-day SMA slope (% change over 30 days)
sma200_slope = sma200.pct_change(30)

# Pi Cycle indicator: SMA111 vs SMA350x2
sma111 = prices.rolling(111).mean()
sma350x2 = prices.rolling(350).mean() * 2

# Halving dates (approximate)
halving_dates = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),  # approximate
]

def days_since_halving(date):
    """Days since last halving."""
    for h in reversed(halving_dates):
        if date >= h:
            return (date - h).days
    return 9999

monthly_dates = prices.groupby(prices.index.to_period('M')).apply(lambda x: x.index[0])
all_dates = prices.index

print(f"BTC Data: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Monthly dates: {len(monthly_dates)}\n")

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
                'sma50': sma50.get(date, np.nan),
                'sma100': sma100.get(date, np.nan),
                'sma111': sma111.get(date, np.nan),
                'sma350x2': sma350x2.get(date, np.nan),
                'mayer': mayer.get(date, np.nan),
                'mayer_pct': mayer_pct.get(date, np.nan),
                'rsi14': rsi14.get(date, np.nan),
                'vol20': vol20.get(date, np.nan),
                'dd_from_ath': dd_from_ath.get(date, np.nan),
                'sma200_slope': sma200_slope.get(date, np.nan),
                'days_since_halving': days_since_halving(date),
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
    dd_series = (hist['Value'] - peak) / peak
    max_dd = dd_series.min()
    monthly_vals = hist['Value'].resample('ME').last().dropna()
    monthly_rets = monthly_vals.pct_change().dropna()
    sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12) if len(monthly_rets) > 1 else 0
    
    print(f"  {name:45s} | ${final:>10,.0f} ({mult:5.1f}x) | MaxDD: {max_dd*100:5.1f}% | Sharpe: {sharpe:.2f}")
    return hist

# ══════════════════════════════════════════════
# STRATEGIES
# ══════════════════════════════════════════════

def strat_bh(date, price, ind):
    return 1.0

# S1: Halving Cycle — accumulate in bear (18-36 months post-halving), go light late cycle
def strat_halving_cycle(date, price, ind):
    dsh = ind['days_since_halving']
    mm = ind['mayer']
    r14 = ind['rsi14']
    
    # Halving cycle phases:
    # 0-6 months: accumulation (still cheap)
    # 6-12 months: early bull
    # 12-18 months: peak zone
    # 18-30 months: bear market
    # 30-48 months: deep accumulation
    
    if dsh < 180:  # 0-6 months after halving
        base = 1.0
    elif dsh < 365:  # 6-12 months
        base = 1.0
    elif dsh < 540:  # 12-18 months: peak zone
        # Reduce exposure
        if not pd.isna(mm) and mm > 2.5: base = 0.30
        elif not pd.isna(mm) and mm > 2.0: base = 0.50
        elif not pd.isna(mm) and mm > 1.5: base = 0.70
        else: base = 0.90
    elif dsh < 900:  # 18-30 months: bear
        if not pd.isna(r14) and r14 < 25: base = 1.0  # oversold buy
        elif not pd.isna(mm) and mm < 0.7: base = 1.0  # deep value
        else: base = 0.50  # reduce
    else:  # 30+ months: deep accumulation
        base = 1.0  # buy everything
    
    return base

# S2: Mayer Multiple mean reversion
def strat_mayer_mr(date, price, ind):
    mm = ind['mayer']
    if pd.isna(mm): return 1.0
    
    # Buy heavily below mean, sell above
    if mm < 0.5: return 1.0
    elif mm < 0.7: return 1.0
    elif mm < 0.9: return 1.0
    elif mm < 1.1: return 0.90
    elif mm < 1.4: return 0.70
    elif mm < 1.8: return 0.50
    elif mm < 2.2: return 0.35
    elif mm < 2.8: return 0.20
    else: return 0.10

# S3: Pi Cycle + Mayer (sell when Pi Cycle tops)
def strat_pi_cycle(date, price, ind):
    s111 = ind['sma111']
    s350x2 = ind['sma350x2']
    mm = ind['mayer']
    r14 = ind['rsi14']
    
    if pd.isna(s111) or pd.isna(s350x2): return 1.0
    
    # Pi Cycle top: SMA111 crosses above SMA350x2
    ratio = s111 / s350x2
    
    if ratio > 0.95:  # approaching or at top
        if not pd.isna(mm) and mm > 2.0:
            return 0.20  # strong sell signal
        return 0.50
    elif ratio > 0.85:
        if not pd.isna(mm) and mm > 1.8:
            return 0.60
        return 0.80
    else:
        # Below Pi Cycle top zone
        if not pd.isna(mm) and mm < 0.7:
            return 1.0  # deep value
        return 1.0

# S4: SMA200 slope — bull when slope positive, cautious when negative
def strat_sma_slope(date, price, ind):
    slope = ind['sma200_slope']
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    
    if pd.isna(slope): return 1.0
    
    if slope > 0.05:  # strong uptrend
        if not pd.isna(mm) and mm > 2.5: return 0.50  # bubble
        return 1.0
    elif slope > 0:  # mild uptrend
        return 0.90
    elif slope > -0.05:  # mild downtrend
        if not pd.isna(r14) and r14 < 30: return 1.0  # oversold
        return 0.60
    else:  # strong downtrend
        if not pd.isna(dd) and dd < -0.60: return 0.80  # crash buy
        if not pd.isna(r14) and r14 < 25: return 0.80
        return 0.30

# S5: Combo — Halving + Mayer + Pi Cycle
def strat_btc_combo(date, price, ind):
    dsh = ind['days_since_halving']
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    s111 = ind['sma111']
    s350x2 = ind['sma350x2']
    
    if pd.isna(mm): return 1.0
    
    # Score system
    score = 0  # positive = buy, negative = sell
    
    # Mayer Multiple
    if mm < 0.6: score += 3
    elif mm < 0.8: score += 2
    elif mm < 1.0: score += 1
    elif mm > 2.5: score -= 4
    elif mm > 2.0: score -= 3
    elif mm > 1.5: score -= 1
    
    # Halving cycle
    if dsh > 900: score += 2  # late bear, accumulate
    elif dsh > 600: score += 1
    elif 365 < dsh < 540:  # peak zone
        score -= 2
    
    # RSI
    if not pd.isna(r14):
        if r14 < 25: score += 2
        elif r14 < 35: score += 1
        elif r14 > 80: score -= 2
        elif r14 > 70: score -= 1
    
    # Pi Cycle
    if not pd.isna(s111) and not pd.isna(s350x2):
        ratio = s111 / s350x2
        if ratio > 0.95: score -= 3
        elif ratio > 0.85: score -= 1
    
    # DD
    if not pd.isna(dd):
        if dd < -0.70: score += 3
        elif dd < -0.50: score += 2
        elif dd < -0.30: score += 1
    
    # Map score to allocation
    if score >= 5: return 1.0
    elif score >= 3: return 1.0
    elif score >= 1: return 0.90
    elif score >= 0: return 0.80
    elif score >= -2: return 0.60
    elif score >= -4: return 0.35
    else: return 0.15

# S6: Minimal sell — only sell at extreme tops
def strat_minimal_sell(date, price, ind):
    mm = ind['mayer']
    r14 = ind['rsi14']
    dsh = ind['days_since_halving']
    
    if pd.isna(mm): return 1.0
    
    # Only sell when multiple signals agree on a top
    sell_signals = 0
    if mm > 2.5: sell_signals += 2
    elif mm > 2.0: sell_signals += 1
    if not pd.isna(r14) and r14 > 85: sell_signals += 1
    if 365 < dsh < 600 and mm > 1.5: sell_signals += 1
    
    if sell_signals >= 3: return 0.20
    elif sell_signals >= 2: return 0.50
    elif sell_signals >= 1: return 0.80
    else: return 1.0

# S7: Pure Mayer with wider bands (less trading)
def strat_mayer_wide(date, price, ind):
    mm = ind['mayer']
    if pd.isna(mm): return 1.0
    
    if mm > 2.8: return 0.15
    elif mm > 2.0: return 0.40
    elif mm < 0.5: return 1.0
    else: return 1.0

# S8: Drawdown buyer — normal 100% but sell on extreme overvaluation
def strat_dd_buyer(date, price, ind):
    mm = ind['mayer']
    dd = ind['dd_from_ath']
    r14 = ind['rsi14']
    
    if pd.isna(mm): return 1.0
    
    # Sell only at extreme bubble
    if mm > 3.0: return 0.20
    elif mm > 2.5: return 0.40
    elif mm > 2.0 and not pd.isna(r14) and r14 > 80: return 0.50
    else: return 1.0

print(f"  {'Strategy':45s} | {'Final':>16s} | {'MaxDD':>8s} | {'Sharpe':>6s}")
print("  " + "=" * 90)

results = {}
for name, fn in [
    ("📈 BTC DCA Buy & Hold", strat_bh),
    ("🔄 Halving Cycle", strat_halving_cycle),
    ("💎 Mayer Mean Reversion", strat_mayer_mr),
    ("🥧 Pi Cycle + Mayer", strat_pi_cycle),
    ("📐 SMA200 Slope", strat_sma_slope),
    ("🎯 BTC Combo (all signals)", strat_btc_combo),
    ("🔧 Minimal Sell (tops only)", strat_minimal_sell),
    ("💎 Mayer Wide Bands", strat_mayer_wide),
    ("📉 DD Buyer", strat_dd_buyer),
]:
    results[name] = run_strategy(fn, name)

bh_final = results["📈 BTC DCA Buy & Hold"]['Value'].iloc[-1]
print(f"\n  Ranked by Final Value (vs B&H ${bh_final:,.0f}):")
print("  " + "-" * 90)
ranked = sorted(results.items(), key=lambda x: x[1]['Value'].iloc[-1], reverse=True)
for i, (name, h) in enumerate(ranked):
    final = h['Value'].iloc[-1]
    mult = final / h['Invested'].iloc[-1]
    peak = h['Value'].cummax()
    dd = ((h['Value'] - peak) / peak).min()
    beat = "✅" if final > bh_final else "❌"
    pct = (final / bh_final - 1) * 100
    print(f"  #{i+1} {beat} {name:42s} ${final:>10,.0f} ({mult:5.1f}x) MaxDD:{dd*100:5.1f}% vs B&H:{pct:+.1f}%")
