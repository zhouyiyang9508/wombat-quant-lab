#!/usr/bin/env python3
"""
Round 2: Deep exploration on the most promising directions.
Focus on Sigmoid (C) with extreme params + combinations.
"""

import pandas as pd
import numpy as np
import os, sys

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tqqq_daily.csv')

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
    return df[['Close', 'High', 'Low']].dropna()

def compute_sma(prices, window):
    return prices.rolling(window).mean()

def compute_rsi(prices, period=10):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_weekly_ret(prices, period=5):
    return prices.pct_change(period)

def compute_realized_vol(prices, window=20):
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)

def sigmoid(x, center, steepness):
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

def calc_metrics(portfolio_values):
    pv = pd.Series(portfolio_values) if not isinstance(portfolio_values, pd.Series) else portfolio_values
    final_val = pv.iloc[-1]
    start_val = pv.iloc[0]
    years = len(pv) / 252.0
    cagr = (final_val / start_val) ** (1 / years) - 1
    peak = pv.cummax()
    dd = (pv - peak) / peak
    max_dd = dd.min()
    dr = pv.pct_change().dropna()
    rf = 0.045 / 252
    excess = dr - rf
    sharpe = (excess.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
    downside = dr[dr < 0].std()
    sortino = (excess.mean() / downside) * np.sqrt(252) if downside > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'final_val': final_val, 'cagr': cagr, 'max_dd': max_dd,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar}


def run_v5(data, initial_capital=10000):
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]; wret = weekly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False
        if not in_bear:
            target_pct = 0.80 if (not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15) else 1.00
        else:
            if not pd.isna(rsi) and rsi < 20: target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12: target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30: target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65: target_pct = 0.30
            else:
                cv = cash + shares * price
                target_pct = max((shares * price) / cv if cv > 0 else 0.30, 0.30)
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY C-EXTREME: Sigmoid with extreme parameters
# ══════════════════════════════════════════════════════════════════

def run_sigmoid_extreme(data, initial_capital=10000,
                        bear_floor=0.20, bear_ceiling=1.0,
                        rsi_center=25, rsi_steepness=-0.20,
                        bull_euphoria_trim=0.80,
                        crash_boost_threshold=-0.12, crash_target=0.85):
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]; wret = weekly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False
        if not in_bear:
            target_pct = bull_euphoria_trim if (not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15) else 1.00
        else:
            if not pd.isna(rsi):
                sig_val = sigmoid(rsi, rsi_center, rsi_steepness)
                target_pct = bear_floor + (bear_ceiling - bear_floor) * sig_val
                if not pd.isna(wret) and wret < crash_boost_threshold:
                    target_pct = max(target_pct, crash_target)
                if not pd.isna(wret) and wret < -0.20:
                    target_pct = max(target_pct, 0.95)
            else:
                target_pct = bear_floor
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY COMBO-1: Sigmoid + Momentum pre-defense
# ══════════════════════════════════════════════════════════════════

def run_combo1(data, initial_capital=10000,
               bear_floor=0.25, bear_ceiling=0.95,
               rsi_center=30, rsi_steepness=-0.20):
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    monthly_ret = prices.pct_change(21)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]; mret = monthly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False
        if not in_bear:
            target_pct = 1.00
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            # Momentum pre-defense: if monthly return is very negative, reduce
            if not pd.isna(mret) and mret < -0.20:
                target_pct = min(target_pct, 0.60)
            elif not pd.isna(mret) and mret < -0.15:
                target_pct = min(target_pct, 0.75)
        else:
            if not pd.isna(rsi):
                sig_val = sigmoid(rsi, rsi_center, rsi_steepness)
                target_pct = bear_floor + (bear_ceiling - bear_floor) * sig_val
                if not pd.isna(wret) and wret < -0.12:
                    target_pct = max(target_pct, 0.80)
            else:
                target_pct = bear_floor
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY COMBO-2: Sigmoid + Vol-aware sizing in bull
# ══════════════════════════════════════════════════════════════════

def run_combo2(data, initial_capital=10000,
               bear_floor=0.25, bear_ceiling=0.95,
               rsi_center=30, rsi_steepness=-0.20,
               vol_high=0.65, vol_reduce=0.85):
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    vol = compute_realized_vol(prices, 20)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]; v = vol.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False
        if not in_bear:
            target_pct = 1.00
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            # Vol-aware: if vol is very high in bull, slight reduce
            if not pd.isna(v) and v > vol_high:
                target_pct = min(target_pct, vol_reduce)
        else:
            if not pd.isna(rsi):
                sig_val = sigmoid(rsi, rsi_center, rsi_steepness)
                target_pct = bear_floor + (bear_ceiling - bear_floor) * sig_val
                if not pd.isna(wret) and wret < -0.12:
                    target_pct = max(target_pct, 0.80)
            else:
                target_pct = bear_floor
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY TURBO: RSI-only bear without hysteresis overhead
# ══════════════════════════════════════════════════════════════════

def run_turbo(data, initial_capital=10000):
    """
    What if we just stay 100% in bull and only use RSI+weekly signals to
    buy MORE aggressively in bear (including 100% floor)?
    Essentially v5 but with bear floor at 50% and max at 100%.
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]; wret = weekly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False
        if not in_bear:
            target_pct = 1.00
        else:
            # Higher floor + aggressive knife catching
            if not pd.isna(rsi) and rsi < 15:
                target_pct = 1.00  # Full send on extreme panic
            elif not pd.isna(rsi) and rsi < 25:
                target_pct = 0.85
            elif not pd.isna(wret) and wret < -0.15:
                target_pct = 0.90
            elif not pd.isna(wret) and wret < -0.10:
                target_pct = 0.75
            elif not pd.isna(rsi) and rsi > 65:
                target_pct = 0.50
            else:
                cv = cash + shares * price
                target_pct = max((shares * price) / cv if cv > 0 else 0.50, 0.50)
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY NO-TRIM: v5 but remove euphoria trim (always 100% in bull)
# ══════════════════════════════════════════════════════════════════

def run_no_trim(data, initial_capital=10000):
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]; wret = weekly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90: in_bear = True
            elif in_bear and price > sma * 1.05: in_bear = False
        if not in_bear:
            target_pct = 1.00  # Always 100% in bull, no trim
        else:
            if not pd.isna(rsi) and rsi < 20: target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12: target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30: target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65: target_pct = 0.30
            else:
                cv = cash + shares * price
                target_pct = max((shares * price) / cv if cv > 0 else 0.30, 0.30)
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY WIDER-BANDS: try even wider bands (85%/110%)
# ══════════════════════════════════════════════════════════════════

def run_wider_bands(data, initial_capital=10000, bear_pct=0.85, bull_pct=1.10):
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma = sma200.iloc[i]; rsi = rsi10.iloc[i]; wret = weekly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * bear_pct: in_bear = True
            elif in_bear and price > sma * bull_pct: in_bear = False
        if not in_bear:
            target_pct = 0.80 if (not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15) else 1.00
        else:
            if not pd.isna(rsi) and rsi < 20: target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12: target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30: target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65: target_pct = 0.30
            else:
                cv = cash + shares * price
                target_pct = max((shares * price) / cv if cv > 0 else 0.30, 0.30)
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY SMA-CROSS: Use SMA50/SMA200 cross instead of price-based bands
# ══════════════════════════════════════════════════════════════════

def run_sma_cross(data, initial_capital=10000):
    prices = data['Close']
    sma50 = compute_sma(prices, 50)
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    cash = initial_capital; shares = 0.0; in_bear = False; pv = []
    for i in range(len(prices)):
        price = prices.iloc[i]; sma50v = sma50.iloc[i]; sma200v = sma200.iloc[i]
        rsi = rsi10.iloc[i]; wret = weekly_ret.iloc[i]
        if i >= 200 and not pd.isna(sma50v) and not pd.isna(sma200v):
            if not in_bear and sma50v < sma200v * 0.97:  # death cross with buffer
                in_bear = True
            elif in_bear and sma50v > sma200v * 1.03:  # golden cross with buffer
                in_bear = False
        if not in_bear:
            target_pct = 0.80 if (not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15) else 1.00
        else:
            if not pd.isna(rsi) and rsi < 20: target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12: target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30: target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65: target_pct = 0.30
            else:
                cv = cash + shares * price
                target_pct = max((shares * price) / cv if cv > 0 else 0.30, 0.30)
        cv = cash + shares * price; te = cv * target_pct; diff = te - shares * price
        if diff > 0 and cash > 0: buy = min(diff, cash); shares += buy / price; cash -= buy
        elif diff < 0: sell = abs(diff); shares -= sell / price; cash += sell
        pv.append(cash + shares * price)
    return pd.Series(pv, index=prices.index)


def main():
    data = load_data()
    print(f"📊 Data: {data.index[0].date()} → {data.index[-1].date()}, {len(data)} rows")
    
    v5_pv = run_v5(data)
    v5_m = calc_metrics(v5_pv)
    
    print(f"\n{'='*95}")
    print(f"V5 Baseline: CAGR {v5_m['cagr']*100:.1f}%, MaxDD {v5_m['max_dd']*100:.1f}%, Sharpe {v5_m['sharpe']:.2f}, Calmar {v5_m['calmar']:.2f}")
    print(f"{'='*95}")
    
    # ── Round 2A: Sigmoid extreme parameter sweep ──
    print(f"\n🔬 Round 2A: Sigmoid Extreme Sweep (floor<0.25, ceiling>0.95, center<30)")
    print(f"{'floor':<7} {'ceil':<7} {'ctr':<6} {'steep':<7} {'crash':<7} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Final$':>12}")
    
    best_cagr = 0
    best_params = None
    
    for floor in [0.15, 0.20, 0.25]:
        for ceiling in [0.90, 0.95, 1.00]:
            for center in [20, 25, 30]:
                for steep in [-0.15, -0.20, -0.25, -0.30]:
                    for crash_t in [0.80, 0.85, 0.90]:
                        pv = run_sigmoid_extreme(data, bear_floor=floor, bear_ceiling=ceiling,
                                                 rsi_center=center, rsi_steepness=steep,
                                                 crash_target=crash_t)
                        m = calc_metrics(pv)
                        if m['cagr'] > v5_m['cagr'] + 0.005:  # Print only if >0.5% better
                            print(f"{floor:<7} {ceiling:<7} {center:<6} {steep:<7} {crash_t:<7} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} ${m['final_val']:>10,.0f}")
                        if m['cagr'] > best_cagr:
                            best_cagr = m['cagr']
                            best_params = (floor, ceiling, center, steep, crash_t)
    
    print(f"\n  🏆 Best sigmoid: floor={best_params[0]}, ceiling={best_params[1]}, center={best_params[2]}, steep={best_params[3]}, crash={best_params[4]} → CAGR {best_cagr*100:.1f}%")
    
    # ── Round 2B: Other strategies ──
    print(f"\n{'='*95}")
    print(f"🔬 Round 2B: Alternative Strategies")
    print(f"{'Strategy':<28} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Sortino':>8} {'Final$':>12}")
    print('-' * 95)
    
    strategies = {
        'V5 Baseline':              v5_pv,
        'No-Trim (always 100%)':    run_no_trim(data),
        'Turbo (50% floor)':        run_turbo(data),
        'SMA50/200 Cross':          run_sma_cross(data),
    }
    
    for bp, bpv in [(0.85, 1.08), (0.85, 1.10), (0.85, 1.12), (0.88, 1.08), (0.88, 1.10)]:
        strategies[f'Bands {int(bp*100)}/{int(bpv*100)}'] = run_wider_bands(data, bear_pct=bp, bull_pct=bpv)
    
    for name, pv in strategies.items():
        m = calc_metrics(pv)
        marker = ' ★' if name == 'V5 Baseline' else ''
        print(f"{name:<28} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['sortino']:>7.2f} ${m['final_val']:>10,.0f}{marker}")
    
    # ── Round 2C: Combo strategies ──
    print(f"\n{'='*95}")
    print(f"🔬 Round 2C: Combination Strategies (Sigmoid + overlays)")
    print(f"{'Strategy':<40} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Final$':>12}")
    print('-' * 95)
    
    combo_results = {}
    
    # Sigmoid base (best from round 1): floor=0.25, ceiling=0.95, center=30, steep=-0.20
    combo_results['Sigmoid(25/95/30/-0.2)'] = calc_metrics(
        run_sigmoid_extreme(data, bear_floor=0.25, bear_ceiling=0.95, rsi_center=30, rsi_steepness=-0.20, crash_target=0.85))
    
    # Combo 1: Sigmoid + Momentum defense
    combo_results['Combo1: Sig+Momentum'] = calc_metrics(run_combo1(data))
    
    # Combo 2: Sigmoid + Vol-aware
    combo_results['Combo2: Sig+Vol(0.65)'] = calc_metrics(run_combo2(data, vol_high=0.65))
    combo_results['Combo2: Sig+Vol(0.70)'] = calc_metrics(run_combo2(data, vol_high=0.70))
    combo_results['Combo2: Sig+Vol(0.75)'] = calc_metrics(run_combo2(data, vol_high=0.75))
    
    for name, m in combo_results.items():
        print(f"{name:<40} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} ${m['final_val']:>10,.0f}")

    # ── Hysteresis band sweep ──
    print(f"\n{'='*95}")
    print(f"🔬 Round 2D: Hysteresis Band Fine-Tune")
    print(f"{'Bear%':<8} {'Bull%':<8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    for bear in [0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]:
        for bull in [1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]:
            pv = run_wider_bands(data, bear_pct=bear, bull_pct=bull)
            m = calc_metrics(pv)
            if m['cagr'] > v5_m['cagr'] - 0.005:  # Within 0.5% of v5
                print(f"{bear:<8} {bull:<8} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")


if __name__ == '__main__':
    main()
