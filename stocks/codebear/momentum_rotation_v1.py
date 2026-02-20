#!/usr/bin/env python3
"""
动量轮动选股策略 v1 — 代码熊 🐻
S&P 500 月度动量 Top 10 等权持仓

幸存者偏差声明：使用当前 S&P 500 成分股列表，结果会偏乐观。
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

def load_csv(filepath):
    """Load CSV, handle both stooq and yfinance formats."""
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    # Ensure numeric
    for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_all_data(tickers):
    """Load all cached data into DataFrames."""
    close_dict = {}
    volume_dict = {}
    loaded = 0
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                close_dict[t] = df['Close'].dropna()
                if 'Volume' in df.columns:
                    volume_dict[t] = df['Volume'].dropna()
                loaded += 1
        except:
            pass
    
    close_df = pd.DataFrame(close_dict)
    volume_df = pd.DataFrame(volume_dict)
    print(f"Loaded {loaded} stocks with sufficient data")
    return close_df, volume_df

def run_backtest(close_df, volume_df, start='2015-01-01', end='2025-12-31',
                 top_n=10, cost_per_trade=0.0015):
    """Monthly momentum rotation backtest."""
    
    # Get month-end dates within range
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    
    portfolio_values = []
    portfolio_dates = []
    holdings_history = {}
    turnover_list = []
    prev_holdings = set()
    
    current_value = 1.0
    
    for i in range(len(month_ends) - 1):
        date = month_ends[i]
        next_date = month_ends[i + 1]
        
        # Calculate momentum scores
        scores = {}
        for ticker in close_df.columns:
            try:
                prices = close_df[ticker].loc[:date].dropna()
                if len(prices) < 130:
                    continue
                
                current_price = prices.iloc[-1]
                if np.isnan(current_price) or current_price < 5:
                    continue
                
                # Volume filter (relaxed)
                if ticker in volume_df.columns:
                    vol = volume_df[ticker].loc[:date].dropna()
                    if len(vol) >= 20:
                        vol_20 = vol.iloc[-20:].mean()
                        if vol.iloc[-1] < vol_20 * 0.3:
                            continue
                
                # Momentum
                p = prices.values
                n = len(p)
                ret_1m = p[-1] / p[-22] - 1 if n > 22 else np.nan
                ret_3m = p[-1] / p[-63] - 1 if n > 63 else np.nan
                ret_6m = p[-1] / p[-126] - 1 if n > 126 else np.nan
                
                if np.isnan(ret_1m) or np.isnan(ret_3m) or np.isnan(ret_6m):
                    continue
                
                momentum = 0.25 * ret_1m + 0.40 * ret_3m + 0.35 * ret_6m
                scores[ticker] = momentum
            except:
                continue
        
        if len(scores) < top_n:
            continue
        
        # Select top N
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, s in sorted_scores[:top_n]]
        selected_set = set(selected)
        
        # Turnover
        if prev_holdings:
            changed = len(selected_set - prev_holdings) + len(prev_holdings - selected_set)
            turnover = changed / (2 * top_n)
        else:
            turnover = 1.0
        turnover_list.append(turnover)
        
        holdings_history[date.strftime('%Y-%m')] = selected
        
        # Calculate period return
        period_returns = []
        for t in selected:
            try:
                t_prices = close_df[t].loc[date:next_date].dropna()
                if len(t_prices) >= 2:
                    ret = t_prices.iloc[-1] / t_prices.iloc[0] - 1
                    period_returns.append(ret)
                else:
                    period_returns.append(0)
            except:
                period_returns.append(0)
        
        port_ret = np.mean(period_returns)
        cost = turnover * cost_per_trade * 2
        port_ret -= cost
        
        current_value *= (1 + port_ret)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)
        
        prev_holdings = selected_set
    
    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    return equity, holdings_history, turnover_list

def compute_metrics(equity, name="Strategy"):
    if len(equity) < 2:
        return {'name': name, 'cagr': 0, 'total_return': 0, 'max_dd': 0, 
                'sharpe': 0, 'calmar': 0, 'win_rate': 0}
    
    total_days = (equity.index[-1] - equity.index[0]).days
    total_years = total_days / 365.25
    
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/total_years) - 1
    
    monthly = equity.pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0
    
    return {
        'name': name, 'cagr': cagr, 'total_return': total_return,
        'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar, 'win_rate': win_rate,
    }

def main():
    print("=" * 60)
    print("🐻 代码熊 — 动量轮动选股策略 v1")
    print("=" * 60)
    
    # Load tickers
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    print(f"Tickers: {len(tickers)}")
    
    # Load data
    close_df, volume_df = load_all_data(tickers + ['SPY'])
    
    # Full backtest
    print("\n🔄 Running full backtest (2015-2025)...")
    eq_full, hold_full, to_full = run_backtest(close_df, volume_df, '2015-01-01', '2025-12-31')
    
    # Walk-forward
    print("🔬 Walk-forward: IS (2015-2020)...")
    eq_is, hold_is, to_is = run_backtest(close_df, volume_df, '2015-01-01', '2020-12-31')
    print("🔬 Walk-forward: OOS (2021-2025)...")
    eq_oos, hold_oos, to_oos = run_backtest(close_df, volume_df, '2021-01-01', '2025-12-31')
    
    # SPY benchmark
    spy_prices = close_df['SPY'].loc['2015-01-01':'2025-12-31'].dropna()
    spy_monthly = spy_prices.resample('ME').last()
    spy_eq = spy_monthly / spy_monthly.iloc[0]
    
    spy_is = spy_prices.loc['2015-01-01':'2020-12-31'].resample('ME').last()
    spy_is = spy_is / spy_is.iloc[0]
    spy_oos = spy_prices.loc['2021-01-01':'2025-12-31'].resample('ME').last()
    spy_oos = spy_oos / spy_oos.iloc[0]
    
    # Metrics
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    results_list = [
        compute_metrics(eq_full, "Momentum Top10 (Full 2015-2025)"),
        compute_metrics(spy_eq, "SPY Buy&Hold (Full 2015-2025)"),
        compute_metrics(eq_is, "Momentum (IS 2015-2020)"),
        compute_metrics(spy_is, "SPY (IS 2015-2020)"),
        compute_metrics(eq_oos, "Momentum (OOS 2021-2025)"),
        compute_metrics(spy_oos, "SPY (OOS 2021-2025)"),
    ]
    
    for m in results_list:
        print(f"\n--- {m['name']} ---")
        print(f"  CAGR:      {m['cagr']:.1%}")
        print(f"  Total Ret: {m['total_return']:.1%}")
        print(f"  MaxDD:     {m['max_dd']:.1%}")
        print(f"  Sharpe:    {m['sharpe']:.2f}")
        print(f"  Calmar:    {m['calmar']:.2f}")
        print(f"  Win Rate:  {m['win_rate']:.1%}")
    
    avg_to = np.mean(to_full) if to_full else 0
    print(f"\n📊 Avg Monthly Turnover: {avg_to:.1%}")
    
    # Walk-forward check
    m_is = results_list[2]
    m_oos = results_list[4]
    ratio = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
    print(f"\n🔬 OOS/IS Sharpe ratio: {ratio:.2f} (target >= 0.70)")
    print(f"   {'✅ PASS' if ratio >= 0.7 else '❌ FAIL'}")
    
    # Most selected stocks
    print("\n📋 Most Selected Stocks (Full Period):")
    all_h = []
    for stocks in hold_full.values():
        all_h.extend(stocks)
    freq = Counter(all_h).most_common(20)
    total_months = len(hold_full)
    for ticker, count in freq:
        print(f"  {ticker:6s}: {count:3d}/{total_months} months ({count/total_months:.0%})")
    
    # Sample holdings
    for ym in ['2020-03', '2023-06', '2024-12', '2024-06']:
        if ym in hold_full:
            print(f"\n📅 {ym} Top 10: {', '.join(hold_full[ym])}")
    
    # Hot stocks in 2023-2024
    print("\n🔍 Hot stocks (NVDA/TSLA/META/AVGO/AMD) in 2023-2024:")
    hot_set = {'NVDA', 'TSLA', 'META', 'AMZN', 'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AVGO', 'AMD'}
    for ym in sorted(hold_full.keys()):
        if ym.startswith('2023') or ym.startswith('2024'):
            hot = [s for s in hold_full[ym] if s in hot_set]
            if hot:
                print(f"  {ym}: {', '.join(hot)}")
    
    # Save results
    results = {
        'full': results_list[0], 'spy': results_list[1],
        'is': results_list[2], 'oos': results_list[4],
        'spy_is': results_list[3], 'spy_oos': results_list[5],
        'avg_turnover': avg_to, 'wf_ratio': ratio,
        'top_stocks': freq[:15],
        'holdings': hold_full,
    }
    
    out = BASE / "stocks" / "codebear" / "momentum_v1_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved to {out}")
    
    return results

if __name__ == '__main__':
    results = main()
