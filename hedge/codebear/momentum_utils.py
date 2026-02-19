"""
Shared utilities for momentum rotation strategies.
Author: 代码熊 🐻
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data_cache')

TICKERS = ['SPY','QQQ','IWM','EFA','EEM','TLT','IEF','HYG','TIP',
           'GLD','SLV','USO','DBC','VNQ','SHY']

RISK_FREE_RATE = 0.04
COMMISSION = 0.0005  # 0.05% per side
SLIPPAGE = 0.001     # 0.1%
TRADE_COST = 2 * COMMISSION + SLIPPAGE  # round-trip approx per rebalance

def download_data(tickers=TICKERS, start='2012-01-01', end='2026-02-18'):
    """Load cached price data (stooq or yfinance format)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    prices = {}
    for t in tickers:
        cache = os.path.join(DATA_DIR, f'{t}.csv')
        if os.path.exists(cache) and os.path.getsize(cache) > 1000:
            df = pd.read_csv(cache, parse_dates=['Date'] if 'Date' in pd.read_csv(cache, nrows=0).columns else [0])
            # Normalize: ensure Date is index
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            elif df.columns[0] != 'Close':
                df.index = pd.to_datetime(df.iloc[:, 0])
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            col = 'Close' if 'Close' in df.columns else 'Adj Close'
            if col in df.columns and len(df) > 100:
                prices[t] = df[col].astype(float)
    return pd.DataFrame(prices).dropna(how='all')

def monthly_returns(prices):
    """Resample to month-end prices and compute returns."""
    monthly = prices.resample('ME').last()
    return monthly.pct_change()

def momentum_score(monthly_prices, lookbacks=[1,3,6,12], weights=[0.25,0.25,0.25,0.25]):
    """Multi-period momentum score. Uses LAGGED data (no future leak)."""
    scores = pd.DataFrame(index=monthly_prices.index, columns=monthly_prices.columns, dtype=float)
    scores[:] = 0
    for lb, w in zip(lookbacks, weights):
        ret = monthly_prices / monthly_prices.shift(lb) - 1
        scores += ret * w
    return scores

def backtest_metrics(returns, rf=RISK_FREE_RATE):
    """Compute strategy metrics from a return series."""
    if len(returns) < 12:
        return {}
    cum = (1 + returns).cumprod()
    total_years = len(returns) / 12
    cagr = cum.iloc[-1] ** (1/total_years) - 1
    
    running_max = cum.cummax()
    dd = cum / running_max - 1
    maxdd = dd.min()
    
    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    
    win_rate = (returns > 0).mean()
    
    return {
        'CAGR': round(cagr * 100, 2),
        'MaxDD': round(maxdd * 100, 2),
        'Sharpe': round(sharpe, 2),
        'Calmar': round(calmar, 2),
        'WinRate': round(win_rate * 100, 1),
        'AnnVol': round(ann_vol * 100, 2),
    }

def spy_benchmark(prices):
    """SPY buy & hold monthly returns."""
    return monthly_returns(prices[['SPY']].dropna())['SPY']

def balanced_60_40(prices):
    """60/40 SPY/IEF benchmark."""
    mr = monthly_returns(prices[['SPY','IEF']].dropna())
    return mr['SPY'] * 0.6 + mr['IEF'] * 0.4

def walk_forward_split(returns, is_frac=0.6):
    """Split into in-sample and out-of-sample."""
    n = len(returns)
    split = int(n * is_frac)
    return returns.iloc[:split], returns.iloc[split:]

def print_results(name, metrics):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v}")
