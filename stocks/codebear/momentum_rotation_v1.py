#!/usr/bin/env python3
"""
动量轮动选股策略 v1 — 代码熊 🐻
S&P 500 月度动量 Top 10 等权持仓

幸存者偏差声明：使用当前 S&P 500 成分股列表，结果会偏乐观。
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"
STOCK_CACHE.mkdir(parents=True, exist_ok=True)

# ─── Step 1: Get S&P 500 tickers ───
def get_sp500_tickers():
    cache_file = CACHE / "sp500_tickers.txt"
    if cache_file.exists():
        tickers = cache_file.read_text().strip().split('\n')
        if len(tickers) > 400:
            print(f"Loaded {len(tickers)} tickers from cache")
            return tickers
    
    # Scrape from Wikipedia
    import urllib.request
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=30).read().decode()
        # Parse table
        tickers = []
        rows = html.split('<tbody>')[1].split('</tbody>')[0].split('<tr>')
        for row in rows[1:]:
            cols = row.split('<td>')
            if len(cols) > 1:
                ticker = cols[1].split('</td>')[0].strip()
                # Remove HTML tags
                import re
                ticker = re.sub(r'<[^>]+>', '', ticker).strip()
                if ticker:
                    tickers.append(ticker.replace('.', '-'))  # BRK.B -> BRK-B for yfinance
        cache_file.write_text('\n'.join(tickers))
        print(f"Scraped {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"Error scraping: {e}")
        # Fallback: use pandas read_html
        try:
            tables = pd.read_html(url)
            tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            cache_file.write_text('\n'.join(tickers))
            print(f"Got {len(tickers)} tickers via pandas")
            return tickers
        except:
            raise RuntimeError("Cannot get S&P 500 tickers")

# ─── Step 2: Download data ───
def download_data(tickers, start='2014-06-01', end='2025-12-31'):
    """Download daily OHLCV for all tickers. Cache to CSV."""
    missing = []
    cached = 0
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if f.exists() and f.stat().st_size > 1000:
            cached += 1
            continue
        missing.append(t)
    
    print(f"Cached: {cached}, To download: {len(missing)}")
    
    if not missing:
        return
    
    # Download one by one with delays to avoid rate limiting
    for idx, t in enumerate(missing):
        f = STOCK_CACHE / f"{t}.csv"
        if idx % 20 == 0:
            print(f"Downloading {idx+1}/{len(missing)}: {t}...")
        try:
            data = yf.download(t, start=start, end=end, auto_adjust=True, 
                             progress=False, timeout=15)
            if not data.empty and len(data) > 100:
                data.to_csv(f)
        except Exception as e:
            if idx % 50 == 0:
                print(f"  Error {t}: {e}")
        time.sleep(0.5)  # 500ms delay between requests
        # Extra pause every 50 to avoid rate limit
        if (idx + 1) % 50 == 0:
            print(f"  Pausing 30s to avoid rate limit...")
            time.sleep(30)

def load_all_data(tickers):
    """Load all cached data into a dict of DataFrames."""
    close_dict = {}
    volume_dict = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if 'Close' in df.columns and 'Volume' in df.columns:
                s = df['Close'].dropna()
                v = df['Volume'].dropna()
                if len(s) > 200:
                    close_dict[t] = s
                    volume_dict[t] = v
        except:
            pass
    
    close_df = pd.DataFrame(close_dict)
    volume_df = pd.DataFrame(volume_dict)
    print(f"Loaded {len(close_dict)} stocks with sufficient data")
    return close_df, volume_df

# ─── Step 3: Backtest Engine ───
def run_backtest(close_df, volume_df, start='2015-01-01', end='2025-12-31',
                 top_n=10, cost_per_side=0.0015):
    """
    Monthly momentum rotation backtest.
    
    momentum = 0.25 * ret_1m + 0.40 * ret_3m + 0.35 * ret_6m
    """
    close = close_df.loc[start:end].copy()
    volume = volume_df.reindex(close.index)
    
    # Monthly end dates
    monthly = close.resample('ME').last()
    monthly_dates = monthly.index
    
    # We need 6 months of lookback, so start selecting from month 6
    # But we need data from before start for lookback
    full_close = close_df.copy()
    
    portfolio_values = [1.0]
    portfolio_dates = [pd.Timestamp(start)]
    holdings_history = {}
    turnover_list = []
    prev_weights = {}
    
    # Get all month-end dates in the backtest period
    rebal_dates = monthly_dates[monthly_dates >= start]
    
    for i, date in enumerate(rebal_dates):
        # Calculate momentum scores
        scores = {}
        for ticker in close.columns:
            try:
                # Get price history up to this date
                prices = full_close[ticker].loc[:date].dropna()
                if len(prices) < 130:  # need ~6 months
                    continue
                
                current_price = prices.iloc[-1]
                
                # Price filter
                if current_price < 5:
                    continue
                
                # Volume filter: current volume vs 20-day average
                vol_series = volume[ticker].loc[:date].dropna()
                if len(vol_series) < 20:
                    continue
                vol_20 = vol_series.iloc[-20:].mean()
                if vol_series.iloc[-1] < vol_20 * 0.5:  # relaxed filter
                    continue
                
                # Momentum calculation
                ret_1m = prices.iloc[-1] / prices.iloc[-21] - 1 if len(prices) > 21 else np.nan
                ret_3m = prices.iloc[-1] / prices.iloc[-63] - 1 if len(prices) > 63 else np.nan
                ret_6m = prices.iloc[-1] / prices.iloc[-126] - 1 if len(prices) > 126 else np.nan
                
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
        
        # Equal weight
        new_weights = {t: 1.0/top_n for t in selected}
        
        # Calculate turnover
        all_tickers = set(list(prev_weights.keys()) + list(new_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers) / 2
        turnover_list.append(turnover)
        
        # Record holdings
        holdings_history[date.strftime('%Y-%m')] = selected
        
        # Calculate return for this month (from this rebal to next)
        if i + 1 < len(rebal_dates):
            next_date = rebal_dates[i + 1]
        else:
            next_date = close.index[-1]
        
        # Portfolio return for this period
        period_returns = []
        for t in selected:
            try:
                t_prices = close[t].loc[date:next_date].dropna()
                if len(t_prices) >= 2:
                    ret = t_prices.iloc[-1] / t_prices.iloc[0] - 1
                    period_returns.append(ret)
                else:
                    period_returns.append(0)
            except:
                period_returns.append(0)
        
        if period_returns:
            port_ret = np.mean(period_returns)  # equal weight
            # Subtract transaction costs
            cost = turnover * cost_per_side * 2  # buy + sell
            port_ret -= cost
            
            portfolio_values.append(portfolio_values[-1] * (1 + port_ret))
            portfolio_dates.append(next_date)
        
        prev_weights = new_weights
    
    # Build equity curve
    equity = pd.Series(portfolio_values, index=portfolio_dates)
    
    return equity, holdings_history, turnover_list

def compute_metrics(equity, name="Strategy"):
    """Compute standard performance metrics."""
    total_days = (equity.index[-1] - equity.index[0]).days
    total_years = total_days / 365.25
    
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/total_years) - 1
    
    # Monthly returns for Sharpe
    monthly = equity.resample('ME').last().pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    
    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate (monthly)
    win_rate = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0
    
    return {
        'name': name,
        'cagr': cagr,
        'total_return': total_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'win_rate': win_rate,
        'total_years': total_years,
    }

def get_spy_benchmark(start='2015-01-01', end='2025-12-31'):
    """Get SPY buy & hold equity curve."""
    spy_file = CACHE / "stocks" / "SPY.csv"
    if not spy_file.exists():
        data = yf.download('SPY', start='2014-06-01', end=end, auto_adjust=True, progress=False)
        data.to_csv(spy_file)
    
    df = pd.read_csv(spy_file, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices = df['Close'].loc[start:end].dropna()
    equity = prices / prices.iloc[0]
    return equity

# ─── Main ───
def main():
    print("=" * 60)
    print("🐻 代码熊 — 动量轮动选股策略 v1")
    print("=" * 60)
    
    # Step 1: Get tickers
    print("\n📋 Step 1: Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    
    # Step 2: Download data
    print("\n📥 Step 2: Downloading stock data...")
    download_data(tickers)
    # Also ensure SPY is downloaded
    download_data(['SPY'])
    
    # Step 3: Load data
    print("\n📊 Step 3: Loading data...")
    close_df, volume_df = load_all_data(tickers + ['SPY'])
    
    # Step 4: Run full backtest
    print("\n🔄 Step 4: Running full backtest (2015-2025)...")
    equity_full, holdings_full, turnover_full = run_backtest(close_df, volume_df, 
                                                              start='2015-01-01', end='2025-12-31')
    
    # Step 5: Walk-forward
    print("\n🔬 Step 5: Walk-forward validation...")
    equity_is, holdings_is, turnover_is = run_backtest(close_df, volume_df,
                                                        start='2015-01-01', end='2020-12-31')
    equity_oos, holdings_oos, turnover_oos = run_backtest(close_df, volume_df,
                                                           start='2021-01-01', end='2025-12-31')
    
    # Step 6: Benchmarks
    print("\n📈 Step 6: Computing benchmarks...")
    spy_full = get_spy_benchmark('2015-01-01', '2025-12-31')
    spy_is = get_spy_benchmark('2015-01-01', '2020-12-31')
    spy_oos = get_spy_benchmark('2021-01-01', '2025-12-31')
    
    # Step 7: Metrics
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    m_full = compute_metrics(equity_full, "Momentum Top10 (Full)")
    m_spy = compute_metrics(spy_full, "SPY Buy&Hold (Full)")
    m_is = compute_metrics(equity_is, "Momentum (IS 2015-2020)")
    m_oos = compute_metrics(equity_oos, "Momentum (OOS 2021-2025)")
    m_spy_is = compute_metrics(spy_is, "SPY (IS 2015-2020)")
    m_spy_oos = compute_metrics(spy_oos, "SPY (OOS 2021-2025)")
    
    for m in [m_full, m_spy, m_is, m_oos, m_spy_is, m_spy_oos]:
        print(f"\n--- {m['name']} ---")
        print(f"  CAGR:      {m['cagr']:.1%}")
        print(f"  Total Ret: {m['total_return']:.1%}")
        print(f"  MaxDD:     {m['max_dd']:.1%}")
        print(f"  Sharpe:    {m['sharpe']:.2f}")
        print(f"  Calmar:    {m['calmar']:.2f}")
        print(f"  Win Rate:  {m['win_rate']:.1%}")
    
    # Turnover
    avg_turnover = np.mean(turnover_full) if turnover_full else 0
    print(f"\n📊 Average Monthly Turnover: {avg_turnover:.1%}")
    
    # Walk-forward check
    ratio = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
    print(f"\n🔬 Walk-forward: OOS Sharpe / IS Sharpe = {ratio:.2f} (target >= 0.70)")
    print(f"   {'✅ PASS' if ratio >= 0.7 else '❌ FAIL'}")
    
    # Holdings analysis
    print("\n📋 Most Selected Stocks:")
    all_holdings = []
    for month, stocks in holdings_full.items():
        all_holdings.extend(stocks)
    from collections import Counter
    freq = Counter(all_holdings).most_common(20)
    for ticker, count in freq:
        total_months = len(holdings_full)
        print(f"  {ticker:6s}: {count:3d}/{total_months} months ({count/total_months:.0%})")
    
    # Sample holdings
    for year_month in ['2020-03', '2023-06', '2024-12']:
        if year_month in holdings_full:
            print(f"\n📅 {year_month} Top 10: {', '.join(holdings_full[year_month])}")
    
    # Check 2023-2024 for NVDA, TSLA etc
    print("\n🔍 NVDA/TSLA appearances in 2023-2024:")
    for ym, stocks in sorted(holdings_full.items()):
        if ym.startswith('2023') or ym.startswith('2024'):
            hot = [s for s in stocks if s in ['NVDA', 'TSLA', 'META', 'AMZN', 'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AVGO', 'AMD']]
            if hot:
                print(f"  {ym}: {', '.join(hot)}")
    
    # Save results to JSON for report
    results = {
        'full': m_full, 'spy': m_spy,
        'is': m_is, 'oos': m_oos,
        'spy_is': m_spy_is, 'spy_oos': m_spy_oos,
        'avg_turnover': avg_turnover,
        'wf_ratio': ratio,
        'top_stocks': freq[:15],
        'holdings_sample': {k: v for k, v in holdings_full.items() if k in ['2020-03','2023-06','2024-12']},
    }
    
    results_file = BASE / "stocks" / "codebear" / "momentum_v1_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_file}")
    
    return results

if __name__ == '__main__':
    results = main()
