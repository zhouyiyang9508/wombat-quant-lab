"""
Options DCA vs TQQQ DCA
Buy ATM QQQ calls each month instead of buying TQQQ.
Simulate with Black-Scholes pricing (no actual options data).
"""
import pandas as pd
import numpy as np
from scipy.stats import norm

qqq = pd.read_csv('data_cache/QQQ.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
tqqq = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()

common = qqq.index.intersection(tqqq.index)
qqq = qqq.loc[common]; tqqq = tqqq.loc[common]

# Historical vol for BS pricing
qqq_vol20 = qqq.pct_change().rolling(20).std() * np.sqrt(252)
qqq_vol60 = qqq.pct_change().rolling(60).std() * np.sqrt(252)

# Approximate risk-free rate by period
def get_rfr(date):
    """Approximate Fed Funds rate."""
    y = date.year + date.month/12
    if y < 2016: return 0.002
    elif y < 2019: return 0.015 + (y-2016)*0.005
    elif y < 2020.2: return 0.02
    elif y < 2022: return 0.001
    elif y < 2023: return 0.03
    elif y < 2024.5: return 0.05
    else: return 0.045

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    """Call delta."""
    if T <= 0 or sigma <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

monthly_dates = qqq.groupby(qqq.index.to_period('M')).apply(lambda x: x.index[0])
monthly_list = sorted(monthly_dates.tolist())

def run_options_dca(expiry_months, monthly_amount=1000, otm_pct=0.0):
    """
    Each month: buy ATM (or OTM) QQQ calls with $1000.
    At expiration: collect payoff. Reinvest everything + new $1000.
    
    expiry_months: 1, 3, or 6
    otm_pct: 0 = ATM, 0.05 = 5% OTM, etc.
    """
    cash = 0.0  # accumulated profits
    positions = []  # list of (strike, num_contracts, expiry_date_idx)
    total_invested = 0.0
    daily_values = []
    
    for i, date in enumerate(monthly_list):
        if date not in qqq.index: continue
        S = qqq.loc[date]
        
        # Check expiring positions
        new_positions = []
        for strike, n_contracts, exp_idx in positions:
            if i >= exp_idx:
                # Expired — collect payoff
                payoff = max(S - strike, 0) * n_contracts * 100
                cash += payoff
            else:
                new_positions.append((strike, n_contracts, exp_idx))
        positions = new_positions
        
        # Add new monthly investment
        total_invested += monthly_amount
        budget = cash + monthly_amount
        cash = 0
        
        # Buy new calls
        K = S * (1 + otm_pct)  # strike
        T = expiry_months / 12.0
        r = get_rfr(date)
        sigma = qqq_vol60.get(date, 0.20)
        if pd.isna(sigma): sigma = 0.20
        
        premium_per_share = bs_call_price(S, K, T, r, sigma)
        premium_per_contract = premium_per_share * 100  # 100 shares per contract
        
        if premium_per_contract > 0:
            n_contracts = budget / premium_per_contract
            positions.append((K, n_contracts, i + expiry_months))
        else:
            cash += budget
        
        # Value positions mark-to-market
        pos_value = 0
        for strike, nc, exp_idx in positions:
            remaining_T = max(0, (exp_idx - i) / 12.0)
            pos_value += bs_call_price(S, strike, remaining_T, r, sigma) * nc * 100
        
        total_value = pos_value + cash
        daily_values.append((date, total_value, total_invested))
    
    # Final expiration of remaining positions
    final_price = qqq.iloc[-1]
    for strike, nc, exp_idx in positions:
        cash += max(final_price - strike, 0) * nc * 100
    
    if daily_values:
        daily_values[-1] = (daily_values[-1][0], cash + sum(
            max(final_price - s, 0) * nc * 100 
            for s, nc, ei in positions if ei > len(monthly_list)-1
        ), daily_values[-1][2])
    
    h = pd.DataFrame(daily_values, columns=['Date','Value','Invested']).set_index('Date')
    return h

def run_tqqq_dca(monthly_amount=1000):
    shares = 0; total_inv = 0; vals = []
    rebal = set(monthly_dates)
    for date in common:
        p = tqqq.loc[date]
        if date in rebal:
            total_inv += monthly_amount
            shares += monthly_amount / p
        vals.append((date, shares*p, total_inv))
    return pd.DataFrame(vals, columns=['Date','Value','Invested']).set_index('Date')

def run_qqq_dca(monthly_amount=1000):
    shares = 0; total_inv = 0; vals = []
    rebal = set(monthly_dates)
    for date in common:
        p = qqq.loc[date]
        if date in rebal:
            total_inv += monthly_amount
            shares += monthly_amount / p
        vals.append((date, shares*p, total_inv))
    return pd.DataFrame(vals, columns=['Date','Value','Invested']).set_index('Date')

def report(h, name):
    f = h['Value'].iloc[-1]; inv = h['Invested'].iloc[-1]; m = f/inv
    years = (h.index[-1]-h.index[0]).days/365.25
    cagr = m**(1/years)-1 if years > 0 else 0
    pk = h['Value'].cummax(); mdd = ((h['Value']-pk)/pk).min()
    print(f"  {name:40s} | ${f:>11,.0f} ({m:5.1f}x) | CAGR≈{cagr*100:5.1f}% | MaxDD:{mdd*100:5.1f}%")
    return f

print(f"Data: {common[0].date()} → {common[-1].date()}")
print(f"Monthly DCA: $1,000/month, Total invested: $193,000\n")

print(f"  {'Strategy':40s} | {'Final':>16s} | {'CAGR':>8s} | {'MaxDD':>8s}")
print("  " + "=" * 90)

# Baselines
qqq_h = run_qqq_dca()
report(qqq_h, "📈 QQQ DCA (no leverage)")

tqqq_h = run_tqqq_dca()
tqqq_f = report(tqqq_h, "📈 TQQQ DCA (3x leverage)")

print("  " + "-" * 90)

# ATM Calls
for months in [1, 3, 6]:
    h = run_options_dca(months, otm_pct=0.0)
    report(h, f"📞 ATM Call {months}M expiry")

print("  " + "-" * 90)

# OTM Calls (cheaper, more leverage)
for months in [1, 3, 6]:
    h = run_options_dca(months, otm_pct=0.05)
    report(h, f"📞 5% OTM Call {months}M expiry")

print("  " + "-" * 90)

# ITM Calls (more delta, less theta decay)
for months in [3, 6]:
    h = run_options_dca(months, otm_pct=-0.05)
    report(h, f"📞 5% ITM Call {months}M expiry")

# Deep ITM (like synthetic stock but with leverage)
for months in [6]:
    h = run_options_dca(months, otm_pct=-0.10)
    report(h, f"📞 10% ITM Call {months}M (synthetic)")

print("\n\n--- Analysis ---")
print("Options leverage = price_of_stock / price_of_option")
# Show effective leverage at various points
print("\nEffective leverage of ATM calls (approx):")
for months in [1, 3, 6]:
    T = months/12
    # Use average vol and a sample price
    S = 100; K = 100; r = 0.03; sigma = 0.20
    premium = bs_call_price(S, K, T, r, sigma)
    delta = bs_delta(S, K, T, r, sigma)
    leverage = delta * S / premium
    theta_pct = bs_call_price(S, K, T, r, sigma) / S * 100
    print(f"  {months}M ATM: premium={premium:.1f}% of stock, delta={delta:.2f}, leverage={leverage:.1f}x, cost={theta_pct:.1f}% of stock")
