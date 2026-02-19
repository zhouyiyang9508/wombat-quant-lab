"""
DCA Strategy Research — Beat Buy & Hold with $1000/month
No look-ahead bias. All signals use only past data.
"""
import pandas as pd
import numpy as np
from datetime import datetime

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

# Mayer Multiple
mayer = prices / sma200

# Weekly return
weekly_ret = prices.pct_change(5)

# Monthly return
monthly_ret = prices.pct_change(21)

# Volatility (20-day)
vol20 = prices.pct_change().rolling(20).std() * np.sqrt(252)

# Distance from ATH
ath = prices.cummax()
dd_from_ath = (prices - ath) / ath

# ── Get first trading day of each month ──
monthly_dates = prices.groupby(prices.index.to_period('M')).apply(lambda x: x.index[0])

print(f"Data: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Monthly investment dates: {len(monthly_dates)}")
print(f"Total months of investing: {len(monthly_dates)}")
print()

# ══════════════════════════════════════════════
# STRATEGY FRAMEWORK
# ══════════════════════════════════════════════

def run_dca_strategy(strategy_fn, name, monthly_amount=1000):
    """
    Run a DCA strategy.
    strategy_fn(date, price, indicators) -> (buy_amount, cash_to_deploy)
    - buy_amount: how much of this month's $1000 goes to TQQQ (rest stays cash)
    - cash_to_deploy: extra accumulated cash to deploy NOW (for "save and deploy" strategies)
    
    Returns: final portfolio value, total invested
    """
    shares = 0.0
    cash_reserve = 0.0  # accumulated uninvested cash
    total_invested = 0.0
    portfolio_history = []
    
    for date in monthly_dates:
        if date not in prices.index:
            continue
        
        p = prices.loc[date]
        s200 = sma200.loc[date] if date in sma200.index else np.nan
        s50 = sma50.loc[date] if date in sma50.index else np.nan
        r14 = rsi14.loc[date] if date in rsi14.index else np.nan
        r10 = rsi10.loc[date] if date in rsi10.index else np.nan
        mm = mayer.loc[date] if date in mayer.index else np.nan
        wr = weekly_ret.loc[date] if date in weekly_ret.index else np.nan
        mr = monthly_ret.loc[date] if date in monthly_ret.index else np.nan
        v = vol20.loc[date] if date in vol20.index else np.nan
        dd = dd_from_ath.loc[date] if date in dd_from_ath.index else np.nan
        
        indicators = {
            'sma200': s200, 'sma50': s50,
            'rsi14': r14, 'rsi10': r10,
            'mayer': mm, 'weekly_ret': wr, 'monthly_ret': mr,
            'vol20': v, 'dd_from_ath': dd,
            'price': p
        }
        
        total_invested += monthly_amount
        
        buy_pct, deploy_reserve = strategy_fn(date, p, indicators, cash_reserve)
        
        # This month's buy
        buy_amount = monthly_amount * buy_pct
        save_amount = monthly_amount * (1 - buy_pct)
        
        # Deploy from reserve
        deploy_amount = min(deploy_reserve, cash_reserve)
        
        total_buy = buy_amount + deploy_amount
        shares += total_buy / p
        cash_reserve += save_amount - deploy_amount
        
        portfolio_val = shares * p + cash_reserve
        portfolio_history.append((date, portfolio_val, shares * p, cash_reserve, total_invested))
    
    hist = pd.DataFrame(portfolio_history, columns=['Date', 'Total', 'Equity', 'Cash', 'Invested'])
    hist.set_index('Date', inplace=True)
    
    final = hist['Total'].iloc[-1]
    invested = hist['Invested'].iloc[-1]
    years = (hist.index[-1] - hist.index[0]).days / 365.25
    
    # TWR approximation using monthly values
    total_return = final / invested - 1
    
    # Max drawdown on total portfolio
    peak = hist['Total'].cummax()
    dd_series = (hist['Total'] - peak) / peak
    max_dd = dd_series.min()
    
    # Profit multiple
    profit_mult = final / invested
    
    print(f"{name:35s} | Final: ${final:>12,.0f} | Invested: ${invested:>9,.0f} | "
          f"Profit: {profit_mult:.2f}x | MaxDD: {max_dd*100:.1f}% | Cash: ${cash_reserve:>8,.0f}")
    
    return hist

# ══════════════════════════════════════════════
# STRATEGIES
# ══════════════════════════════════════════════

# 1. Pure DCA Buy & Hold — baseline
def strat_bh(date, price, ind, reserve):
    return 1.0, 0  # always buy 100%

# 2. v5 DCA — buy according to v5 position sizing
_v5_regime = {'regime': 'bull'}
def strat_v5(date, price, ind, reserve):
    sma = ind['sma200']
    r10 = ind['rsi10']
    wr = ind['weekly_ret']
    if pd.isna(sma):
        return 1.0, 0
    
    if _v5_regime['regime'] == 'bull' and price < sma * 0.90:
        _v5_regime['regime'] = 'bear'
    elif _v5_regime['regime'] == 'bear' and price > sma * 1.05:
        _v5_regime['regime'] = 'bull'
    
    if _v5_regime['regime'] == 'bull':
        return 1.0, 0
    else:
        # Bear: save cash, deploy on RSI signals
        if not pd.isna(r10) and r10 < 20:
            return 1.0, reserve  # deploy everything!
        elif not pd.isna(r10) and r10 < 30:
            return 1.0, reserve * 0.5
        else:
            return 0.3, 0  # save 70%

# 3. Smart DCA — RSI-weighted buying
def strat_smart_dca(date, price, ind, reserve):
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    if pd.isna(r14) or pd.isna(dd):
        return 1.0, 0
    
    # More aggressive buying when oversold / far from ATH
    if r14 < 25 or dd < -0.50:
        return 1.0, reserve  # full deploy
    elif r14 < 35 or dd < -0.30:
        return 1.0, reserve * 0.5
    elif r14 > 75 and dd > -0.05:
        return 0.3, 0  # save 70% near ATH + overbought
    elif r14 > 65:
        return 0.5, 0  # save 50%
    else:
        return 1.0, 0  # normal

# 4. Value DCA — buy more when price is below trend, less when above
def strat_value_dca(date, price, ind, reserve):
    mm = ind['mayer']  # Mayer Multiple (price/SMA200)
    if pd.isna(mm):
        return 1.0, 0
    
    if mm < 0.6:
        return 1.0, reserve  # way below trend, deploy all
    elif mm < 0.8:
        return 1.0, reserve * 0.5  # below trend, deploy half reserve
    elif mm < 1.0:
        return 1.0, 0  # slightly below, normal buy
    elif mm < 1.2:
        return 0.7, 0  # slightly above, save some
    elif mm < 1.5:
        return 0.4, 0  # well above trend
    else:
        return 0.2, 0  # far above trend, mostly save

# 5. Crash Accumulator — normal DCA but 2x when crashed
def strat_crash_acc(date, price, ind, reserve):
    dd = ind['dd_from_ath']
    r14 = ind['rsi14']
    if pd.isna(dd):
        return 1.0, 0
    
    if dd < -0.60:
        return 1.0, reserve  # massive crash, all in
    elif dd < -0.40:
        return 1.0, reserve * 0.7
    elif dd < -0.25:
        return 1.0, reserve * 0.3
    elif dd > -0.05 and not pd.isna(r14) and r14 > 70:
        return 0.5, 0  # near ATH + overbought, save
    else:
        return 1.0, 0

# 6. Dual Momentum DCA — only buy when price > SMA50 AND SMA200, save otherwise
_dm_cash = {'reserve': 0}
def strat_dual_mom(date, price, ind, reserve):
    s50 = ind['sma50']
    s200 = ind['sma200']
    r10 = ind['rsi10']
    if pd.isna(s50) or pd.isna(s200):
        return 1.0, 0
    
    if price > s50 and price > s200:
        # Strong uptrend - buy and deploy reserve
        return 1.0, reserve * 0.3
    elif price > s200:
        # Mild uptrend
        return 1.0, 0
    else:
        # Downtrend - save unless oversold
        if not pd.isna(r10) and r10 < 25:
            return 1.0, reserve * 0.8
        elif not pd.isna(r10) and r10 < 35:
            return 0.5, reserve * 0.3
        else:
            return 0.0, 0  # full save

# 7. Aggressive Mean Reversion DCA
def strat_mean_rev(date, price, ind, reserve):
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    if pd.isna(mm):
        return 1.0, 0
    
    # Scale buying inversely with Mayer Multiple
    if mm < 0.5:
        return 1.0, reserve  # extreme discount
    elif mm < 0.7:
        return 1.0, reserve * 0.8
    elif mm < 0.85:
        return 1.0, reserve * 0.3
    elif mm < 1.0:
        return 1.0, 0
    elif mm < 1.3:
        return 0.6, 0
    elif mm < 1.6:
        return 0.3, 0
    elif mm < 2.0:
        return 0.15, 0
    else:
        return 0.0, 0  # extreme premium, save everything

# 8. Hybrid: Mayer + RSI + DD (kitchen sink)
def strat_hybrid(date, price, ind, reserve):
    mm = ind['mayer']
    r14 = ind['rsi14']
    dd = ind['dd_from_ath']
    v = ind['vol20']
    
    if pd.isna(mm) or pd.isna(r14) or pd.isna(dd):
        return 1.0, 0
    
    # Score system: higher = more bearish/cheap = buy more
    score = 0
    
    # Mayer Multiple
    if mm < 0.6: score += 3
    elif mm < 0.8: score += 2
    elif mm < 1.0: score += 1
    elif mm > 1.5: score -= 2
    elif mm > 1.2: score -= 1
    
    # RSI
    if r14 < 25: score += 3
    elif r14 < 35: score += 2
    elif r14 < 45: score += 1
    elif r14 > 75: score -= 2
    elif r14 > 65: score -= 1
    
    # Drawdown from ATH
    if dd < -0.50: score += 3
    elif dd < -0.30: score += 2
    elif dd < -0.15: score += 1
    elif dd > -0.03: score -= 1
    
    # Deploy decision
    if score >= 6:
        return 1.0, reserve
    elif score >= 4:
        return 1.0, reserve * 0.6
    elif score >= 2:
        return 1.0, reserve * 0.2
    elif score >= 0:
        return 1.0, 0
    elif score >= -2:
        return 0.5, 0
    else:
        return 0.2, 0

# ══════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════

print(f"{'Strategy':35s} | {'Final':>14s} | {'Invested':>11s} | "
      f"{'Profit':>7s} | {'MaxDD':>7s} | {'Cash':>10s}")
print("=" * 110)

# Reset stateful strategies
_v5_regime['regime'] = 'bull'

results = {}
for name, fn in [
    ("📈 DCA Buy & Hold", strat_bh),
    ("🐻 v5 DCA", strat_v5),
    ("🧠 Smart DCA (RSI+DD)", strat_smart_dca),
    ("💎 Value DCA (Mayer)", strat_value_dca),
    ("💥 Crash Accumulator", strat_crash_acc),
    ("🔀 Dual Momentum DCA", strat_dual_mom),
    ("📉 Mean Reversion DCA", strat_mean_rev),
    ("🎯 Hybrid Score DCA", strat_hybrid),
]:
    h = run_dca_strategy(fn, name)
    results[name] = h

print("\n" + "=" * 110)
print("\nRanked by Final Value:")
ranked = sorted(results.items(), key=lambda x: x[1]['Total'].iloc[-1], reverse=True)
for i, (name, h) in enumerate(ranked):
    final = h['Total'].iloc[-1]
    invested = h['Invested'].iloc[-1]
    mult = final / invested
    print(f"  #{i+1} {name:35s} → ${final:>12,.0f}  ({mult:.2f}x)")
