#!/usr/bin/env python3
"""Verify v8 vs v5 head-to-head with CSV data."""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tqqq_daily.csv')

def load_csv():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
    return df[['Close']].dropna()

# Import both strategies
from beast_v5 import WombatBeastV5
from beast_v8_final import WombatBeastV8

data = load_csv()
print(f"Data: {data.index[0].date()} → {data.index[-1].date()}, {len(data)} rows\n")

# v5
v5 = WombatBeastV5(initial_capital=10000)
v5.data = data
v5.run_backtest()
m5 = v5.show_metrics()

# v8
v8 = WombatBeastV8(initial_capital=10000)
v8.data = data
v8.run_backtest()
m8 = v8.show_metrics()

# Head to head
print(f"\n{'='*50}")
print(f"{'Metric':<15} {'v5.0':>10} {'v8.0':>10} {'Δ':>10} {'Win?':>6}")
print(f"-"*50)
for key, label, thresh, higher_better in [
    ('cagr', 'CAGR', 0.02, True),
    ('max_dd', 'MaxDD', 0.03, True),  # less negative = better, diff > 0 = better
    ('sharpe', 'Sharpe', 0.05, True),
    ('calmar', 'Calmar', 0.05, True),
    ('sortino', 'Sortino', 0.05, True),
]:
    v5v = m5[key]
    v8v = m8[key]
    diff = v8v - v5v
    sig = '✅' if diff > thresh else ('❌' if diff < -thresh else '—')
    if key in ('cagr', 'max_dd'):
        print(f"{label:<15} {v5v*100:>9.1f}% {v8v*100:>9.1f}% {diff*100:>+9.1f}% {sig:>6}")
    else:
        print(f"{label:<15} {v5v:>10.2f} {v8v:>10.2f} {diff:>+10.2f} {sig:>6}")

wins = 0
if m8['cagr'] - m5['cagr'] > 0.02: wins += 1
if m8['max_dd'] - m5['max_dd'] > 0.03: wins += 1
if m8['sharpe'] - m5['sharpe'] > 0.05: wins += 1
if m8['calmar'] - m5['calmar'] > 0.05: wins += 1

print(f"\n🏆 Significant wins: {wins}/4 {'→ v8 IS THE NEW CHAMPION!' if wins >= 2 else '→ v5 remains champion'}")

# Current signal
sig = v8.get_current_signal()
if sig:
    print(f"\n📊 Current Signal ({sig['date'].date()}):")
    print(f"  Price: ${sig['price']:.2f}")
    print(f"  SMA200: ${sig['sma200']:.2f}")
    print(f"  RSI(10): {sig['rsi10']:.1f}")
    print(f"  Weekly Return: {sig['weekly_ret']*100:.1f}%")
    print(f"  Realized Vol: {sig['realized_vol']*100:.1f}%")
    print(f"  Regime: {sig['regime']}")
    print(f"  Bear Trigger: ${sig['bear_trigger']:.2f} ({sig['pct_to_bear']:+.1f}% away)")
    if sig['pct_to_bull'] is not None:
        print(f"  Bull Trigger: ${sig['bull_trigger']:.2f} ({sig['pct_to_bull']:.1f}% needed)")
