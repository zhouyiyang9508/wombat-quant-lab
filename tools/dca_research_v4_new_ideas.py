"""
DCA Research v4 — New Strategies from Web Research
1. TQQQ + Gold rebalancing (Hedgefundie-style)
2. QQQ indicators for TQQQ (cleaner signals)
3. Dual Momentum (absolute + relative)
4. TQQQ/Gold/TLT multi-asset rotation
5. Volatility targeting using QQQ vol (not TQQQ vol)
"""
import pandas as pd
import numpy as np

# ── Load all data ──
tqqq = pd.read_csv('data_cache/TQQQ.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
gld = pd.read_csv('data_cache/GLD.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
qqq = pd.read_csv('data_cache/QQQ.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()
tlt = pd.read_csv('data_cache/TLT.csv', parse_dates=['Date']).set_index('Date')['Close'].dropna()

# Align dates
common_dates = tqqq.index.intersection(gld.index).intersection(qqq.index).intersection(tlt.index)
tqqq = tqqq.loc[common_dates]
gld = gld.loc[common_dates]
qqq = qqq.loc[common_dates]
tlt = tlt.loc[common_dates]

print(f"Data: {common_dates[0].date()} → {common_dates[-1].date()}, {len(common_dates)} days")

# ── Pre-compute indicators ──
def calc_rsi(s, p):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100/(1+g/l)

tqqq_sma200 = tqqq.rolling(200).mean()
tqqq_vol20 = tqqq.pct_change().rolling(20).std() * np.sqrt(252)
tqqq_rsi14 = calc_rsi(tqqq, 14)
tqqq_mayer = tqqq / tqqq_sma200
tqqq_ath = tqqq.cummax()
tqqq_dd = (tqqq - tqqq_ath) / tqqq_ath

# QQQ indicators (cleaner for regime detection!)
qqq_sma200 = qqq.rolling(200).mean()
qqq_sma50 = qqq.rolling(50).mean()
qqq_vol20 = qqq.pct_change().rolling(20).std() * np.sqrt(252)
qqq_rsi14 = calc_rsi(qqq, 14)

# Gold indicators
gld_sma200 = gld.rolling(200).mean()
gld_mom = gld.pct_change(63)  # 3-month momentum

# TLT indicators
tlt_sma200 = tlt.rolling(200).mean()
tlt_mom = tlt.pct_change(63)

# Daily returns
tqqq_ret = tqqq.pct_change()
gld_ret = gld.pct_change()
tlt_ret = tlt.pct_change()

monthly_dates = tqqq.groupby(tqqq.index.to_period('M')).apply(lambda x: x.index[0])

def run_multi_asset_dca(strategy_fn, name, monthly_amount=1000):
    """
    Multi-asset DCA. strategy_fn returns dict: {'tqqq': pct, 'gld': pct, 'tlt': pct, 'cash': pct}
    Rebalances monthly. Tracks daily values.
    """
    # Holdings
    tqqq_shares = 0.0
    gld_shares = 0.0
    tlt_shares = 0.0
    cash = 0.0
    total_invested = 0.0
    
    rebalance_dates = set(monthly_dates)
    daily_values = []
    
    for date in common_dates:
        tp = tqqq.loc[date]
        gp = gld.loc[date]
        tlp = tlt.loc[date]
        
        if date in rebalance_dates:
            cash += monthly_amount
            total_invested += monthly_amount
            pv = tqqq_shares * tp + gld_shares * gp + tlt_shares * tlp + cash
            
            ind = {
                'tqqq_sma200': tqqq_sma200.get(date, np.nan),
                'tqqq_vol20': tqqq_vol20.get(date, np.nan),
                'tqqq_rsi14': tqqq_rsi14.get(date, np.nan),
                'tqqq_mayer': tqqq_mayer.get(date, np.nan),
                'tqqq_dd': tqqq_dd.get(date, np.nan),
                'qqq_sma200': qqq_sma200.get(date, np.nan),
                'qqq_sma50': qqq_sma50.get(date, np.nan),
                'qqq_vol20': qqq_vol20.get(date, np.nan),
                'qqq_rsi14': qqq_rsi14.get(date, np.nan),
                'gld_sma200': gld_sma200.get(date, np.nan),
                'gld_mom': gld_mom.get(date, np.nan),
                'tlt_sma200': tlt_sma200.get(date, np.nan),
                'tlt_mom': tlt_mom.get(date, np.nan),
                'qqq_price': qqq.loc[date],
                'gld_price': gp,
                'tlt_price': tlp,
            }
            
            alloc = strategy_fn(date, tp, ind)
            
            # Rebalance to target
            target_tqqq = pv * alloc.get('tqqq', 0)
            target_gld = pv * alloc.get('gld', 0)
            target_tlt = pv * alloc.get('tlt', 0)
            
            # Sell everything first
            cash += tqqq_shares * tp + gld_shares * gp + tlt_shares * tlp
            tqqq_shares = 0; gld_shares = 0; tlt_shares = 0
            
            # Buy targets
            tqqq_shares = target_tqqq / tp; cash -= target_tqqq
            gld_shares = target_gld / gp; cash -= target_gld
            tlt_shares = target_tlt / tlp; cash -= target_tlt
        
        pv = tqqq_shares * tp + gld_shares * gp + tlt_shares * tlp + cash
        daily_values.append((date, pv, total_invested))
    
    hist = pd.DataFrame(daily_values, columns=['Date', 'Value', 'Invested']).set_index('Date')
    final = hist['Value'].iloc[-1]
    invested = hist['Invested'].iloc[-1]
    mult = final / invested
    peak = hist['Value'].cummax()
    max_dd = ((hist['Value'] - peak) / peak).min()
    monthly_vals = hist['Value'].resample('ME').last().dropna()
    monthly_rets = monthly_vals.pct_change().dropna()
    sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12) if len(monthly_rets) > 1 else 0
    
    print(f"  {name:50s} | ${final:>11,.0f} ({mult:5.1f}x) | DD:{max_dd*100:5.1f}% | Sh:{sharpe:.2f}")
    return hist

# ══════════════════════════════════════════════
# STRATEGIES
# ══════════════════════════════════════════════

# 0. Baseline: 100% TQQQ B&H DCA
def strat_bh(date, price, ind):
    return {'tqqq': 1.0}

# 1. TQQQ + Gold 50/50 rebalance (Hedgefundie-inspired)
def strat_tqqq_gold_5050(date, price, ind):
    return {'tqqq': 0.50, 'gld': 0.50}

# 2. TQQQ + Gold 60/40
def strat_tqqq_gold_6040(date, price, ind):
    return {'tqqq': 0.60, 'gld': 0.40}

# 3. TQQQ + Gold 70/30
def strat_tqqq_gold_7030(date, price, ind):
    return {'tqqq': 0.70, 'gld': 0.30}

# 4. TQQQ + TLT (classic Hedgefundie 55/45)
def strat_tqqq_tlt_5545(date, price, ind):
    return {'tqqq': 0.55, 'tlt': 0.45}

# 5. TQQQ + Gold + TLT (three-way)
def strat_three_way(date, price, ind):
    return {'tqqq': 0.50, 'gld': 0.25, 'tlt': 0.25}

# 6. Vol targeting with QQQ vol (not TQQQ vol — cleaner signal!)
def strat_qqq_vol_target(date, price, ind):
    v = ind['qqq_vol20']
    if pd.isna(v): return {'tqqq': 1.0}
    
    # QQQ normal vol: ~15-20%. Crisis: 30-50%
    target_vol = 0.18  # 18% annualized for QQQ
    alloc = min(1.0, target_vol / v)
    alloc = max(0.15, alloc)
    
    # Override: deep value
    if not pd.isna(ind['tqqq_dd']) and ind['tqqq_dd'] < -0.60:
        alloc = max(alloc, 0.90)
    if not pd.isna(ind['qqq_rsi14']) and ind['qqq_rsi14'] < 25:
        alloc = max(alloc, 0.85)
    
    return {'tqqq': alloc, 'gld': (1 - alloc) * 0.5}  # put rest in gold

# 7. Vol targeting + gold hedge (rest goes to gold instead of cash)
def strat_vol_gold_hedge(date, price, ind):
    v = ind['tqqq_vol20']
    if pd.isna(v): return {'tqqq': 1.0}
    
    if v > 1.2: tqqq_pct = 0.30
    elif v > 0.9: tqqq_pct = 0.50
    elif v > 0.7: tqqq_pct = 0.80
    else: tqqq_pct = 1.0
    
    # Override: deep value
    mm = ind['tqqq_mayer']
    if not pd.isna(mm) and mm < 0.7:
        tqqq_pct = max(tqqq_pct, 0.80)
    if not pd.isna(ind['tqqq_rsi14']) and ind['tqqq_rsi14'] < 25:
        tqqq_pct = max(tqqq_pct, 0.85)
    if not pd.isna(ind['tqqq_dd']) and ind['tqqq_dd'] < -0.60:
        tqqq_pct = max(tqqq_pct, 0.90)
    
    gold_pct = 1.0 - tqqq_pct  # rest goes to gold
    return {'tqqq': tqqq_pct, 'gld': gold_pct}

# 8. Dual Momentum: buy TQQQ if QQQ > SMA200 AND QQQ momentum > GLD momentum, else buy GLD
def strat_dual_momentum(date, price, ind):
    qqq_above_sma = not pd.isna(ind['qqq_sma200']) and ind['qqq_price'] > ind['qqq_sma200']
    
    gm = ind['gld_mom']
    # QQQ 3-month momentum
    qqq_mom = qqq.loc[:date].iloc[-1] / qqq.loc[:date].iloc[-min(63, len(qqq.loc[:date]))] - 1 if len(qqq.loc[:date]) > 63 else 0
    
    if qqq_above_sma and (pd.isna(gm) or qqq_mom > gm):
        return {'tqqq': 1.0}
    elif qqq_above_sma:
        return {'tqqq': 0.60, 'gld': 0.40}
    else:
        if not pd.isna(gm) and gm > 0:
            return {'gld': 0.70, 'tqqq': 0.30}
        return {'tlt': 0.50, 'gld': 0.30, 'tqqq': 0.20}

# 9. QQQ Golden/Death Cross regime + vol
def strat_qqq_cross_vol(date, price, ind):
    s50 = ind['qqq_sma50']
    s200 = ind['qqq_sma200']
    v = ind['qqq_vol20']
    
    if pd.isna(s50) or pd.isna(s200): return {'tqqq': 1.0}
    
    golden_cross = s50 > s200
    
    if golden_cross:
        # Bull: full TQQQ
        return {'tqqq': 1.0}
    else:
        # Death cross: reduce TQQQ, add gold
        if not pd.isna(ind['qqq_rsi14']) and ind['qqq_rsi14'] < 25:
            return {'tqqq': 0.80, 'gld': 0.20}  # oversold, buy
        return {'tqqq': 0.25, 'gld': 0.50, 'tlt': 0.25}

# 10. Adaptive multi-asset — best combo of vol + gold + regime
def strat_adaptive_multi(date, price, ind):
    v = ind['tqqq_vol20']
    mm = ind['tqqq_mayer']
    r14 = ind['tqqq_rsi14']
    dd = ind['tqqq_dd']
    qqq_above = not pd.isna(ind['qqq_sma200']) and ind['qqq_price'] > ind['qqq_sma200']
    
    if pd.isna(v): return {'tqqq': 1.0}
    
    # Vol targeting base
    if v > 1.2: tqqq_pct = 0.25
    elif v > 0.9: tqqq_pct = 0.45
    elif v > 0.7: tqqq_pct = 0.75
    else: tqqq_pct = 1.0
    
    # Crash overrides
    if not pd.isna(mm) and mm < 0.7: tqqq_pct = max(tqqq_pct, 0.80)
    if not pd.isna(r14) and r14 < 25: tqqq_pct = max(tqqq_pct, 0.85)
    if not pd.isna(dd) and dd < -0.60: tqqq_pct = max(tqqq_pct, 0.90)
    
    # Distribute remainder
    rest = 1.0 - tqqq_pct
    if rest <= 0:
        return {'tqqq': 1.0}
    
    # Gold vs TLT: use momentum to decide
    gm = ind['gld_mom']
    tm = ind['tlt_mom']
    if not pd.isna(gm) and not pd.isna(tm):
        if gm > tm:
            return {'tqqq': tqqq_pct, 'gld': rest * 0.7, 'tlt': rest * 0.3}
        else:
            return {'tqqq': tqqq_pct, 'tlt': rest * 0.7, 'gld': rest * 0.3}
    return {'tqqq': tqqq_pct, 'gld': rest * 0.5, 'tlt': rest * 0.5}

# 11. Original v6 (TQQQ vol + cash) for comparison
def strat_v6_orig(date, price, ind):
    v = ind['tqqq_vol20']
    mm = ind['tqqq_mayer']
    r14 = ind['tqqq_rsi14']
    dd = ind['tqqq_dd']
    if pd.isna(v): return {'tqqq': 1.0}
    
    if v > 1.2: base = 0.30
    elif v > 0.9: base = 0.50
    elif v > 0.7: base = 0.80
    else: base = 1.0
    
    if not pd.isna(mm) and mm < 0.7: base = max(base, 0.80)
    if not pd.isna(r14) and r14 < 25: base = max(base, 0.85)
    if not pd.isna(dd) and dd < -0.60: base = max(base, 0.90)
    
    return {'tqqq': base}  # rest is cash

# ══════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════
print(f"\n  {'Strategy':50s} | {'Final':>16s} | {'DD':>7s} | {'Sh':>5s}")
print("  " + "=" * 95)

results = {}
for name, fn in [
    ("📈 100% TQQQ B&H DCA", strat_bh),
    ("🐻 v6 Vol Target (cash)", strat_v6_orig),
    ("🥇 TQQQ+Gold 50/50", strat_tqqq_gold_5050),
    ("🥇 TQQQ+Gold 60/40", strat_tqqq_gold_6040),
    ("🥇 TQQQ+Gold 70/30", strat_tqqq_gold_7030),
    ("📊 TQQQ+TLT 55/45 (Hedgefundie)", strat_tqqq_tlt_5545),
    ("🌐 TQQQ+Gold+TLT 50/25/25", strat_three_way),
    ("📊 QQQ Vol Target → TQQQ+Gold", strat_qqq_vol_target),
    ("🐻 Vol Target + Gold Hedge", strat_vol_gold_hedge),
    ("🔀 Dual Momentum TQQQ/GLD", strat_dual_momentum),
    ("✂️ QQQ Cross + Vol", strat_qqq_cross_vol),
    ("🏆 Adaptive Multi-Asset", strat_adaptive_multi),
]:
    results[name] = run_multi_asset_dca(fn, name)

bh_final = results["📈 100% TQQQ B&H DCA"]['Value'].iloc[-1]
print(f"\n  Ranked (vs B&H ${bh_final:,.0f}):")
print("  " + "-" * 95)
ranked = sorted(results.items(), key=lambda x: x[1]['Value'].iloc[-1], reverse=True)
for i, (name, h) in enumerate(ranked):
    f = h['Value'].iloc[-1]
    m = f / h['Invested'].iloc[-1]
    pk = h['Value'].cummax()
    dd = ((h['Value'] - pk) / pk).min()
    beat = "✅" if f > bh_final else "❌"
    print(f"  #{i+1} {beat} {name:48s} ${f:>11,.0f} ({m:5.1f}x) DD:{dd*100:5.1f}% vs B&H:{(f/bh_final-1)*100:+.1f}%")
