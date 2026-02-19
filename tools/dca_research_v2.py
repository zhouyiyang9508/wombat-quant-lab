"""
DCA Strategy Research v2 — Full Portfolio Management
Each month: receive $1000, then decide target allocation for ENTIRE portfolio.
Can SELL existing shares (not just decide what to do with new $1000).
This is the real-world scenario: you manage a portfolio and add $1000/month.
"""
import pandas as pd
import numpy as np

# ── Load Data ──
df = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date').sort_index()
prices = df['Close'].dropna()

# ── Indicators ──
sma200 = prices.rolling(200).mean()
sma50 = prices.rolling(50).mean()

def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

rsi14 = calc_rsi(prices, 14)
rsi10 = calc_rsi(prices, 10)
mayer = prices / sma200
weekly_ret = prices.pct_change(5)
vol20 = prices.pct_change().rolling(20).std() * np.sqrt(252)
ath = prices.cummax()
dd_from_ath = (prices - ath) / ath

# Monthly dates
monthly_dates = prices.groupby(prices.index.to_period('M')).apply(lambda x: x.index[0])

# Also get daily data between monthly dates for tracking portfolio value
all_dates = prices.index

print(f"Data: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Monthly dates: {len(monthly_dates)}\n")

def run_strategy(strategy_fn, name, monthly_amount=1000):
    """
    Full portfolio management DCA.
    strategy_fn(date, price, indicators) -> target_pct (0.0 to 1.0)
    target_pct = fraction of TOTAL portfolio value in TQQQ
    Each month: add $1000 to portfolio, then rebalance to target.
    """
    shares = 0.0
    cash = 0.0
    total_invested = 0.0
    
    # Track daily portfolio value for proper drawdown
    daily_values = []
    monthly_log = []
    
    rebalance_dates = set(monthly_dates)
    
    # We need to track between monthly dates too
    last_target = 1.0  # default full in
    
    for date in all_dates:
        p = prices.loc[date]
        portfolio_val = shares * p + cash
        
        if date in rebalance_dates:
            # Add monthly contribution
            cash += monthly_amount
            total_invested += monthly_amount
            portfolio_val = shares * p + cash
            
            # Get indicators
            s200 = sma200.loc[date] if date in sma200.index else np.nan
            s50 = sma50.loc[date] if date in sma50.index else np.nan
            r14 = rsi14.loc[date] if date in rsi14.index else np.nan
            r10 = rsi10.loc[date] if date in rsi10.index else np.nan
            mm = mayer.loc[date] if date in mayer.index else np.nan
            wr = weekly_ret.loc[date] if date in weekly_ret.index else np.nan
            v = vol20.loc[date] if date in vol20.index else np.nan
            dd = dd_from_ath.loc[date] if date in dd_from_ath.index else np.nan
            
            indicators = {
                'sma200': s200, 'sma50': s50,
                'rsi14': r14, 'rsi10': r10,
                'mayer': mm, 'weekly_ret': wr,
                'vol20': v, 'dd_from_ath': dd, 'price': p
            }
            
            target_pct = strategy_fn(date, p, indicators)
            last_target = target_pct
            
            # Rebalance
            target_equity = portfolio_val * target_pct
            current_equity = shares * p
            diff = target_equity - current_equity
            
            if diff > 0 and cash >= diff:
                shares += diff / p
                cash -= diff
            elif diff > 0:
                shares += cash / p
                cash = 0
            elif diff < 0:
                sell = abs(diff)
                shares -= sell / p
                cash += sell
        
        portfolio_val = shares * p + cash
        daily_values.append((date, portfolio_val, total_invested))
    
    hist = pd.DataFrame(daily_values, columns=['Date', 'Value', 'Invested']).set_index('Date')
    
    final = hist['Value'].iloc[-1]
    invested = hist['Invested'].iloc[-1]
    mult = final / invested
    
    # Max drawdown
    peak = hist['Value'].cummax()
    dd_series = (hist['Value'] - peak) / peak
    max_dd = dd_series.min()
    
    # Monthly returns for Sharpe
    monthly_vals = hist['Value'].resample('ME').last().dropna()
    monthly_rets = monthly_vals.pct_change().dropna()
    sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12) if len(monthly_rets) > 1 else 0
    
    print(f"{name:40s} | ${final:>11,.0f} | {mult:5.1f}x | {max_dd*100:6.1f}% | {sharpe:5.2f}")
    return hist

# ══════════════════════════════════════════════
# STRATEGIES — return target % in TQQQ
# ══════════════════════════════════════════════

# 1. Buy & Hold DCA
def strat_bh(date, price, ind):
    return 1.0

# 2. v5 style — hysteresis band
_regime2 = {'r': 'bull'}
def strat_v5_full(date, price, ind):
    sma = ind['sma200']
    r10 = ind['rsi10']
    if pd.isna(sma): return 1.0
    
    if _regime2['r'] == 'bull' and price < sma * 0.90:
        _regime2['r'] = 'bear'
    elif _regime2['r'] == 'bear' and price > sma * 1.05:
        _regime2['r'] = 'bull'
    
    if _regime2['r'] == 'bull':
        return 1.0
    else:
        if not pd.isna(r10) and r10 < 20: return 0.80
        elif not pd.isna(r10) and r10 < 30: return 0.60
        else: return 0.30

# 3. Aggressive Regime — sell more in bear, buy aggressively on fear
_regime3 = {'r': 'bull'}
def strat_aggressive_regime(date, price, ind):
    sma = ind['sma200']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    if pd.isna(sma): return 1.0
    
    if _regime3['r'] == 'bull' and price < sma * 0.92:
        _regime3['r'] = 'bear'
    elif _regime3['r'] == 'bear' and price > sma * 1.03:
        _regime3['r'] = 'bull'
    
    if _regime3['r'] == 'bull':
        return 1.0
    else:
        # Aggressive bear regime
        if not pd.isna(dd) and dd < -0.60:
            return 1.0  # massive crash = go all in
        elif not pd.isna(r14) and r14 < 20:
            return 0.90
        elif not pd.isna(r14) and r14 < 30:
            return 0.70
        elif not pd.isna(dd) and dd < -0.40:
            return 0.50
        else:
            return 0.10  # minimal in bear

# 4. Mayer Multiple allocation
def strat_mayer_alloc(date, price, ind):
    mm = ind['mayer']
    r14 = ind['rsi14']
    if pd.isna(mm): return 1.0
    
    if mm < 0.5: return 1.0
    elif mm < 0.7: return 1.0
    elif mm < 0.85: return 1.0
    elif mm < 1.0: return 0.90
    elif mm < 1.2: return 0.80
    elif mm < 1.5: return 0.60
    elif mm < 2.0: return 0.40
    else: return 0.20

# 5. Fear/Greed DCA — multi-signal scoring
def strat_fear_greed(date, price, ind):
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    v = ind['vol20']
    
    if pd.isna(mm) or pd.isna(r14): return 1.0
    
    score = 0  # positive = fear (buy more), negative = greed (buy less)
    
    if mm < 0.6: score += 4
    elif mm < 0.8: score += 2
    elif mm < 1.0: score += 1
    elif mm > 2.0: score -= 3
    elif mm > 1.5: score -= 2
    elif mm > 1.2: score -= 1
    
    if r14 < 20: score += 3
    elif r14 < 30: score += 2
    elif r14 < 40: score += 1
    elif r14 > 80: score -= 2
    elif r14 > 70: score -= 1
    
    if not pd.isna(dd):
        if dd < -0.60: score += 3
        elif dd < -0.40: score += 2
        elif dd < -0.20: score += 1
        elif dd > -0.02: score -= 1
    
    # Map score to allocation
    if score >= 7: return 1.0
    elif score >= 4: return 1.0
    elif score >= 2: return 0.95
    elif score >= 0: return 0.85
    elif score >= -2: return 0.65
    elif score >= -4: return 0.45
    else: return 0.25

# 6. Crash Hunter — stay 100% but sell to 0% on bubble, rebuy on crash
def strat_crash_hunter(date, price, ind):
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    
    if pd.isna(mm): return 1.0
    
    # Only reduce when extreme bubble
    if mm > 2.5 and not pd.isna(r14) and r14 > 80:
        return 0.30  # extreme bubble
    elif mm > 2.0 and not pd.isna(r14) and r14 > 75:
        return 0.50  # bubble
    elif mm > 1.8:
        return 0.70
    # Crash buying
    elif not pd.isna(dd) and dd < -0.60:
        return 1.0
    elif mm < 0.6:
        return 1.0
    else:
        return 1.0  # default: stay in

# 7. Minimal timing — only exit on extreme overvaluation
def strat_minimal_timing(date, price, ind):
    mm = ind['mayer']
    r14 = ind['rsi14']
    if pd.isna(mm): return 1.0
    
    # Only sell when Mayer > 2.0 + RSI > 75 (extreme)
    if mm > 2.0 and not pd.isna(r14) and r14 > 75:
        return 0.50
    elif mm > 2.5:
        return 0.40
    else:
        return 1.0

# 8. SMA crossover + position management
_regime8 = {'r': 'bull', 'prev_dd': 0}
def strat_sma_cross(date, price, ind):
    s50 = ind['sma50']
    s200 = ind['sma200']
    r10 = ind['rsi10']
    dd = ind['dd_from_ath']
    if pd.isna(s50) or pd.isna(s200): return 1.0
    
    if s50 > s200 and price > s200:
        return 1.0  # golden cross zone
    elif s50 > s200:
        return 0.80  # crossing but price weak
    elif price > s200:
        return 0.70
    else:
        # Both bearish
        if not pd.isna(r10) and r10 < 25:
            return 0.80  # oversold bounce
        elif not pd.isna(dd) and dd < -0.50:
            return 0.70  # deep crash
        else:
            return 0.15  # minimal

# 9. NEW: Inverse Volatility — reduce when vol is high (panic), increase when calm
def strat_inv_vol(date, price, ind):
    v = ind['vol20']
    mm = ind['mayer']
    if pd.isna(v): return 1.0
    
    # TQQQ annualized vol is typically 50-100%
    if v > 1.2:  # extreme vol
        if not pd.isna(mm) and mm < 0.7:
            return 0.80  # high vol but cheap = buy
        return 0.30
    elif v > 0.9:
        return 0.50
    elif v > 0.7:
        return 0.80
    else:
        return 1.0  # low vol = calm market, stay in

# 10. NEW: Drawdown-adaptive — aggressively buy drawdowns with full portfolio rebalance
_regime10 = {'r': 'bull', 'bear_start_price': None}
def strat_dd_adaptive(date, price, ind):
    sma = ind['sma200']
    dd = ind['dd_from_ath']
    r14 = ind['rsi14']
    mm = ind['mayer']
    if pd.isna(sma) or pd.isna(dd): return 1.0
    
    if _regime10['r'] == 'bull' and price < sma * 0.88:
        _regime10['r'] = 'bear'
        _regime10['bear_start_price'] = price
    elif _regime10['r'] == 'bear' and price > sma * 1.02:
        _regime10['r'] = 'bull'
    
    if _regime10['r'] == 'bull':
        # In bull: almost always 100%
        if not pd.isna(mm) and mm > 2.5:
            return 0.70  # extreme bubble only
        return 1.0
    else:
        # Bear: AGGRESSIVE buying schedule based on drawdown depth
        if dd < -0.70: return 1.0   # 70%+ down = back up the truck
        elif dd < -0.55: return 0.90
        elif dd < -0.40: return 0.75
        elif dd < -0.25: return 0.50
        else: return 0.20  # early bear, mostly cash

print(f"{'Strategy':40s} | {'Final':>12s} | {'Mult':>5s} | {'MaxDD':>7s} | {'Sharpe':>5s}")
print("=" * 85)

# Reset states
_regime2['r'] = 'bull'
_regime3['r'] = 'bear'  # start conservative
_regime8['r'] = 'bull'
_regime10['r'] = 'bull'
_regime10['bear_start_price'] = None

strategies = [
    ("📈 DCA Buy & Hold", strat_bh),
    ("🐻 v5 Full Mgmt", strat_v5_full),
    ("⚔️ Aggressive Regime", strat_aggressive_regime),
    ("💎 Mayer Allocation", strat_mayer_alloc),
    ("😱 Fear/Greed Score", strat_fear_greed),
    ("🎯 Crash Hunter", strat_crash_hunter),
    ("🔧 Minimal Timing", strat_minimal_timing),
    ("✂️ SMA Crossover", strat_sma_cross),
    ("📊 Inverse Volatility", strat_inv_vol),
    ("📉 Drawdown Adaptive", strat_dd_adaptive),
]

results = {}
for name, fn in strategies:
    results[name] = run_strategy(fn, name)

print("\n" + "=" * 85)
print("\nRanked by Final Value:")
ranked = sorted(results.items(), key=lambda x: x[1]['Value'].iloc[-1], reverse=True)
for i, (name, h) in enumerate(ranked):
    final = h['Value'].iloc[-1]
    invested = h['Invested'].iloc[-1]
    mult = final / invested
    peak = h['Value'].cummax()
    dd = ((h['Value'] - peak) / peak).min()
    print(f"  #{i+1} {name:40s} ${final:>11,.0f}  ({mult:5.1f}x)  MaxDD: {dd*100:.1f}%")
