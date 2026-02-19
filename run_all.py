"""
Run all 4 strategies using cached CSV data (bypassing yfinance rate limits).
"""
import pandas as pd
import numpy as np
import sys, os

CACHE = '/root/.openclaw/workspace/wombat-quant-lab/data_cache'

def load_btc():
    df = pd.read_csv(f'{CACHE}/BTC_USD.csv', index_col=0, parse_dates=True)
    return df

def load_tqqq():
    df = pd.read_csv(f'{CACHE}/TQQQ.csv', index_col=0, parse_dates=True)
    return df

# Monkey-patch yfinance before importing strategies
import yfinance as yf
_btc_cache = None
_tqqq_cache = None

def patched_download(ticker, *args, **kwargs):
    global _btc_cache, _tqqq_cache
    if 'BTC' in str(ticker):
        if _btc_cache is None:
            _btc_cache = load_btc()
        return _btc_cache.copy()
    else:
        if _tqqq_cache is None:
            _tqqq_cache = load_tqqq()
        return _tqqq_cache.copy()

yf.download = patched_download

# Now import strategies
print("="*60)
print("🏃 BTC Beast v2.0 (Baseline)")
print("="*60)
from btc_beast_3q80_mode import BTCBeastStrategy
bot = BTCBeastStrategy(use_yfinance=True)
bot.load_data()
bot.calculate_indicators()
bot.generate_report()

print("\n" + "="*60)
print("🏃 BTC Beast v3.0 (Multi-Factor)")
print("="*60)
from btc_beast_v3_ml import fetch_data as btc_fetch, add_indicators, backtest as btc_bt, compute_metrics as btc_m
df = btc_fetch()
df = add_indicators(df)
p, inv = btc_bt(df)
btc_m(p, df.index, inv, "BTC Beast v3.0 (Multi-Factor)")

print("\n" + "="*60)
print("🏃 TQQQ Ultimate Wombat v2.0 (Baseline)")
print("="*60)
from tqqq_ultimate_wombat_mode import UltimateWombat
bot2 = UltimateWombat()
bot2.run_backtest(switch_threshold=1_000_000)
bot2.show_metrics()

print("\n" + "="*60)
print("🏃 TQQQ Ultimate Wombat v3.0 (ML)")
print("="*60)
from tqqq_v3_ml import fetch_data as tqqq_fetch, add_features, walk_forward_predict, backtest as tqqq_bt, compute_metrics as tqqq_m
df2 = tqqq_fetch()
df2 = add_features(df2)
print(f"Running walk-forward ML...", flush=True)
preds, probs = walk_forward_predict(df2)
print(f"ML predictions: {probs.notna().sum()} days", flush=True)
p2, inv2 = tqqq_bt(df2, preds, probs)
tqqq_m(p2, df2.index, inv2, "TQQQ Ultimate Wombat v3.0 (ML)")

print("\n✅ All strategies completed!")
