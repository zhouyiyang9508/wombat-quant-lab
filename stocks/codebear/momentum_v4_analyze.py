#!/usr/bin/env python3
"""
Deep analysis of v4d DD Responsive - understand where MaxDD improved.
Also test v4d parameter sensitivity.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Import from main file
sys.path.insert(0, str(Path(__file__).parent))
from momentum_v4_gld import (
    load_all_data, load_gld, load_spy, precompute_signals,
    run_backtest_v4, run_v3b_backtest, compute_metrics,
    select_stocks_v3b, select_v4d, DDResponsiveTracker,
    CACHE, STOCK_CACHE
)


def find_maxdd_period(equity, name=""):
    """Find the exact period of maximum drawdown"""
    dd = (equity - equity.cummax()) / equity.cummax()
    maxdd_date = dd.idxmin()
    # Find the peak before the trough
    peak_date = equity.loc[:maxdd_date].idxmax()
    # Find recovery
    post = equity.loc[maxdd_date:]
    peak_val = equity.loc[peak_date]
    recovery = post[post >= peak_val]
    recovery_date = recovery.index[0] if len(recovery) > 0 else None

    print(f"  {name}: MaxDD {dd.min():.1%} from {peak_date.date()} to {maxdd_date.date()}"
          f" (recovery: {recovery_date.date() if recovery_date else 'not yet'})")
    return peak_date, maxdd_date, recovery_date


def analyze_gld_allocation(close_df, signals, sectors, gld_prices, spy_prices):
    """Track when GLD is actually allocated in v4d"""
    close_range = close_df.loc['2015-01-01':'2025-12-31'].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    dd_tracker = DDResponsiveTracker()
    prev_holdings = set()
    prev_weights = {}
    current_value = 1.0
    gld_range = gld_prices.loc['2015-01-01':'2025-12-31'].dropna()

    gld_months = []
    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]
        new_weights = select_v4d(signals, sectors, date, prev_holdings, dd_tracker)

        gld_w = new_weights.get('GLD', 0.0)
        if gld_w > 0:
            gld_months.append((date.strftime('%Y-%m'), gld_w, dd_tracker.get_drawdown()))

        # Calculate return for DD tracker
        port_ret = 0.0
        for t, w in new_weights.items():
            if t == 'GLD':
                gld_slice = gld_range.loc[date:next_date].dropna()
                if len(gld_slice) >= 2:
                    port_ret += (gld_slice.iloc[-1] / gld_slice.iloc[0] - 1) * w
            else:
                if t in close_df.columns:
                    stock_slice = close_df[t].loc[date:next_date].dropna()
                    if len(stock_slice) >= 2:
                        port_ret += (stock_slice.iloc[-1] / stock_slice.iloc[0] - 1) * w

        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        port_ret -= turnover * 0.0015 * 2
        current_value *= (1 + port_ret)
        dd_tracker.update(port_ret)
        prev_weights = new_weights.copy()
        prev_holdings = set(k for k in new_weights.keys() if k != 'GLD')

    print(f"\n📅 GLD Allocation Months ({len(gld_months)} total):")
    for ym, w, dd in gld_months:
        print(f"  {ym}: GLD {w:.0%} (DD: {dd:.1%})")


def run_v4d_sensitivity():
    """Test v4d with different DD thresholds"""
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)
    gld_prices = load_gld()
    spy_prices = load_spy()

    # Find MaxDD periods for v3b and v4d
    print("=" * 60)
    print("🔍 MaxDD Period Analysis")
    print("=" * 60)

    eq_v3b, _ = run_v3b_backtest(close_df, signals, sectors)
    find_maxdd_period(eq_v3b, "v3b")

    eq_v4d, _ = run_backtest_v4(close_df, signals, sectors, gld_prices, spy_prices, variant='v4d')
    find_maxdd_period(eq_v4d, "v4d")

    eq_v4e, _ = run_backtest_v4(close_df, signals, sectors, gld_prices, spy_prices, variant='v4e')
    find_maxdd_period(eq_v4e, "v4e")

    # Analyze GLD allocation pattern
    print("\n" + "=" * 60)
    print("📊 v4d GLD Allocation Tracking")
    print("=" * 60)
    analyze_gld_allocation(close_df, signals, sectors, gld_prices, spy_prices)

    # Year-by-year performance comparison
    print("\n" + "=" * 60)
    print("📅 Year-by-Year Return Comparison")
    print("=" * 60)
    print(f"{'Year':<8} {'v3b':>8} {'v4d':>8} {'v4e':>8} {'Diff(v4d)':>10}")
    print("-" * 42)

    for year in range(2015, 2026):
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        for name, eq in [('v3b', eq_v3b), ('v4d', eq_v4d), ('v4e', eq_v4e)]:
            eq_yr = eq.loc[start:end]
            if len(eq_yr) >= 2:
                locals()[f'ret_{name}'] = eq_yr.iloc[-1] / eq_yr.iloc[0] - 1
            else:
                locals()[f'ret_{name}'] = 0.0

        r3b = locals().get('ret_v3b', 0)
        r4d = locals().get('ret_v4d', 0)
        r4e = locals().get('ret_v4e', 0)
        diff = r4d - r3b
        print(f"{year:<8} {r3b:>7.1%} {r4d:>7.1%} {r4e:>7.1%} {diff:>+9.1%}")


if __name__ == '__main__':
    run_v4d_sensitivity()
