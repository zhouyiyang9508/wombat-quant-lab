#!/usr/bin/env python3
"""
动量轮动选股策略 v2 — 全面优化探索
代码熊 🐻

v2a: Regime Filter + Absolute Momentum (bear market protection)
v2b: Bimonthly Rebalance + Skip-1M Momentum + Patience Filter (low turnover)
v2c: Sector-Diversified + Vol-Weighted + Regime + Abs Momentum (diversification)
v2d: Adaptive Regime + Dynamic Size + Sector + Vol (best combo)
v2e: Conservative Combo (bimonthly + regime + sector + vol)
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ─── Data Loading ───────────────────────────────────────────────

def load_csv(filepath):
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_all_data(tickers):
    close_dict, volume_dict = {}, {}
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
    return pd.DataFrame(close_dict), pd.DataFrame(volume_dict), loaded

def load_sectors():
    sf = CACHE / "sp500_sectors.json"
    if sf.exists():
        with open(sf) as f:
            return json.load(f)
    return {}

# ─── Momentum Scoring ──────────────────────────────────────────

def compute_momentum_scores(close_df, volume_df, date, 
                            weights=(0.25, 0.40, 0.35, 0.0),
                            skip_recent_month=False,
                            min_price=5.0, min_history=130):
    """
    Compute momentum scores for all stocks as of `date`.
    weights: (1M, 3M, 6M, 12M) momentum weights.
    skip_recent_month: if True, use prices from 22 days ago as "current" price.
    """
    scores = {}
    for ticker in close_df.columns:
        if ticker == 'SPY':
            continue
        try:
            prices = close_df[ticker].loc[:date].dropna()
            if len(prices) < max(min_history, 252):
                continue
            
            p = prices.values
            n = len(p)
            
            # Current price reference
            if skip_recent_month and n > 22:
                ref_idx = -22  # Skip most recent month
            else:
                ref_idx = -1
            
            current_price = p[ref_idx] if ref_idx == -1 else p[-1]
            if np.isnan(current_price) or current_price < min_price:
                continue
            
            # Volume filter
            if ticker in volume_df.columns:
                vol = volume_df[ticker].loc[:date].dropna()
                if len(vol) >= 20:
                    vol_20 = vol.iloc[-20:].mean()
                    if vol.iloc[-1] < vol_20 * 0.3:
                        continue
            
            # Momentum returns
            w1m, w3m, w6m, w12m = weights
            
            if skip_recent_month:
                # Skip recent month: all lookbacks measured from 22 days ago
                base = n - 22
                ret_1m = p[base] / p[base - 22] - 1 if base > 22 else np.nan
                ret_3m = p[base] / p[base - 63] - 1 if base > 63 else np.nan
                ret_6m = p[base] / p[base - 126] - 1 if base > 126 else np.nan
                ret_12m = p[base] / p[base - 252] - 1 if base > 252 else np.nan
            else:
                ret_1m = p[-1] / p[-22] - 1 if n > 22 else np.nan
                ret_3m = p[-1] / p[-63] - 1 if n > 63 else np.nan
                ret_6m = p[-1] / p[-126] - 1 if n > 126 else np.nan
                ret_12m = p[-1] / p[-252] - 1 if n > 252 else np.nan
            
            # Compute blended momentum
            parts = []
            total_w = 0
            for r, w in [(ret_1m, w1m), (ret_3m, w3m), (ret_6m, w6m), (ret_12m, w12m)]:
                if not np.isnan(r) and w > 0:
                    parts.append(r * w)
                    total_w += w
            
            if total_w < 0.5:
                continue
            
            momentum = sum(parts) / total_w
            
            # Also compute 6M absolute return for filtering
            abs_6m = p[-1] / p[-126] - 1 if n > 126 else 0
            
            # 30-day annualized volatility
            if n > 30:
                daily_rets = np.diff(np.log(p[-31:]))
                vol_30d = np.std(daily_rets) * np.sqrt(252)
            else:
                vol_30d = 0.5  # default high
            
            scores[ticker] = {
                'momentum': momentum,
                'abs_6m': abs_6m,
                'vol_30d': vol_30d,
                'price': p[-1],
            }
        except:
            continue
    
    return scores

def get_spy_regime(close_df, date, sma_period=200):
    """Check if SPY is above/below SMA200."""
    if 'SPY' not in close_df.columns:
        return 'bull'
    spy = close_df['SPY'].loc[:date].dropna()
    if len(spy) < sma_period:
        return 'bull'
    sma = spy.iloc[-sma_period:].mean()
    current = spy.iloc[-1]
    return 'bull' if current > sma else 'bear'


# ─── Strategy Variants ─────────────────────────────────────────

def strategy_v1_baseline(close_df, volume_df, sectors, date, prev_holdings, **kwargs):
    """v1 baseline: Pure momentum Top 10, equal weight, monthly."""
    scores = compute_momentum_scores(close_df, volume_df, date,
                                     weights=(0.25, 0.40, 0.35, 0.0))
    sorted_s = sorted(scores.items(), key=lambda x: x[1]['momentum'], reverse=True)
    selected = [t for t, _ in sorted_s[:10]]
    weights = {t: 1.0/len(selected) for t in selected} if selected else {}
    return weights


def strategy_v2a_regime_absmom(close_df, volume_df, sectors, date, prev_holdings, **kwargs):
    """
    v2a: Market Regime + Absolute Momentum Filter
    - Bull (SPY > SMA200): Top 10 momentum stocks with abs_6m > 0
    - Bear (SPY < SMA200): 100% cash
    - If fewer than 10 pass filter, hold fewer + cash
    """
    regime = get_spy_regime(close_df, date)
    
    if regime == 'bear':
        return {}  # 100% cash
    
    scores = compute_momentum_scores(close_df, volume_df, date,
                                     weights=(0.25, 0.40, 0.35, 0.0))
    
    # Absolute momentum filter
    filtered = {t: s for t, s in scores.items() if s['abs_6m'] > 0}
    
    sorted_s = sorted(filtered.items(), key=lambda x: x[1]['momentum'], reverse=True)
    selected = [t for t, _ in sorted_s[:10]]
    weights = {t: 1.0/max(len(selected), 1) for t in selected}
    return weights


def strategy_v2b_lowturnover(close_df, volume_df, sectors, date, prev_holdings, **kwargs):
    """
    v2b: Low Turnover Focus
    - Skip-1-month momentum (academic 12-1 style)
    - Absolute momentum filter
    - Regime filter (SPY SMA200)
    - Holdover bonus: stocks already in portfolio get +5% momentum bonus
    - Top 10, equal weight
    """
    regime = get_spy_regime(close_df, date)
    if regime == 'bear':
        return {}
    
    # Skip recent month momentum weights: emphasize 3M and 6M
    scores = compute_momentum_scores(close_df, volume_df, date,
                                     weights=(0.0, 0.40, 0.35, 0.25),
                                     skip_recent_month=True)
    
    # Absolute momentum filter
    filtered = {t: s for t, s in scores.items() if s['abs_6m'] > 0}
    
    # Holdover bonus: reduce turnover by giving incumbents a boost
    for t in filtered:
        if t in prev_holdings:
            filtered[t] = dict(filtered[t])
            filtered[t]['momentum'] = filtered[t]['momentum'] + 0.05
    
    sorted_s = sorted(filtered.items(), key=lambda x: x[1]['momentum'], reverse=True)
    selected = [t for t, _ in sorted_s[:10]]
    weights = {t: 1.0/max(len(selected), 1) for t in selected}
    return weights


def strategy_v2c_sector_vol(close_df, volume_df, sectors, date, prev_holdings, **kwargs):
    """
    v2c: Sector Diversified + Vol-Weighted + Regime + Abs Momentum
    - Max 3 per sector
    - Inverse-vol weighting
    - Regime filter
    - Absolute momentum filter
    """
    regime = get_spy_regime(close_df, date)
    if regime == 'bear':
        return {}
    
    scores = compute_momentum_scores(close_df, volume_df, date,
                                     weights=(0.25, 0.40, 0.35, 0.0))
    
    # Absolute momentum filter
    filtered = {t: s for t, s in scores.items() if s['abs_6m'] > 0}
    
    # Sort by momentum
    sorted_s = sorted(filtered.items(), key=lambda x: x[1]['momentum'], reverse=True)
    
    # Sector-diversified selection
    selected = []
    sector_count = Counter()
    max_per_sector = 3
    
    for ticker, data in sorted_s:
        sector = sectors.get(ticker, 'Unknown')
        if sector_count[sector] < max_per_sector:
            selected.append((ticker, data))
            sector_count[sector] += 1
        if len(selected) >= 10:
            break
    
    if not selected:
        return {}
    
    # Inverse-vol weighting
    inv_vols = {}
    for ticker, data in selected:
        vol = max(data['vol_30d'], 0.10)  # floor at 10%
        inv_vols[ticker] = 1.0 / vol
    
    total_inv = sum(inv_vols.values())
    weights = {t: v / total_inv for t, v in inv_vols.items()}
    return weights


def strategy_v2d_adaptive(close_df, volume_df, sectors, date, prev_holdings, **kwargs):
    """
    v2d: Adaptive All-in-One
    - Bull: Top 12, sector cap 3, inverse-vol weighted
    - Bear: Top 5 sector cap 2, 50% invested + 50% cash
    - Absolute momentum filter
    - Vol filter: exclude stocks with 30d vol > 65%
    - Holdover bonus
    """
    regime = get_spy_regime(close_df, date)
    
    scores = compute_momentum_scores(close_df, volume_df, date,
                                     weights=(0.20, 0.40, 0.30, 0.10))
    
    # Absolute momentum filter
    filtered = {t: s for t, s in scores.items() 
                if s['abs_6m'] > 0 and s['vol_30d'] < 0.65}
    
    # Holdover bonus
    for t in filtered:
        if t in prev_holdings:
            filtered[t] = dict(filtered[t])
            filtered[t]['momentum'] = filtered[t]['momentum'] + 0.03
    
    sorted_s = sorted(filtered.items(), key=lambda x: x[1]['momentum'], reverse=True)
    
    if regime == 'bull':
        top_n = 12
        max_sector = 3
        cash_frac = 0.0
    else:
        top_n = 5
        max_sector = 2
        cash_frac = 0.50
    
    # Sector-diversified selection
    selected = []
    sector_count = Counter()
    for ticker, data in sorted_s:
        sector = sectors.get(ticker, 'Unknown')
        if sector_count[sector] < max_sector:
            selected.append((ticker, data))
            sector_count[sector] += 1
        if len(selected) >= top_n:
            break
    
    if not selected:
        return {}
    
    # Inverse-vol weighting
    inv_vols = {}
    for ticker, data in selected:
        vol = max(data['vol_30d'], 0.10)
        inv_vols[ticker] = 1.0 / vol
    
    total_inv = sum(inv_vols.values())
    invested_frac = 1.0 - cash_frac
    weights = {t: (v / total_inv) * invested_frac for t, v in inv_vols.items()}
    return weights


def strategy_v2e_conservative(close_df, volume_df, sectors, date, prev_holdings, **kwargs):
    """
    v2e: Conservative Best Combo
    - Regime filter (bear → 100% cash)
    - Absolute momentum filter
    - Skip-1-month momentum
    - Sector cap 3
    - Inverse-vol weighting
    - Holdover bonus (larger, 8%)
    - Top 10
    """
    regime = get_spy_regime(close_df, date)
    if regime == 'bear':
        return {}
    
    scores = compute_momentum_scores(close_df, volume_df, date,
                                     weights=(0.0, 0.35, 0.35, 0.30),
                                     skip_recent_month=True)
    
    # Absolute momentum + vol filter
    filtered = {t: s for t, s in scores.items() 
                if s['abs_6m'] > 0 and s['vol_30d'] < 0.60}
    
    # Holdover bonus
    for t in filtered:
        if t in prev_holdings:
            filtered[t] = dict(filtered[t])
            filtered[t]['momentum'] = filtered[t]['momentum'] + 0.08
    
    sorted_s = sorted(filtered.items(), key=lambda x: x[1]['momentum'], reverse=True)
    
    # Sector-diversified
    selected = []
    sector_count = Counter()
    for ticker, data in sorted_s:
        sector = sectors.get(ticker, 'Unknown')
        if sector_count[sector] < 3:
            selected.append((ticker, data))
            sector_count[sector] += 1
        if len(selected) >= 10:
            break
    
    if not selected:
        return {}
    
    inv_vols = {}
    for ticker, data in selected:
        vol = max(data['vol_30d'], 0.10)
        inv_vols[ticker] = 1.0 / vol
    
    total_inv = sum(inv_vols.values())
    weights = {t: v / total_inv for t, v in inv_vols.items()}
    return weights


# ─── Backtest Engine ────────────────────────────────────────────

def run_backtest(close_df, volume_df, sectors, strategy_fn, 
                 start='2015-01-01', end='2025-12-31',
                 rebalance_months=1, cost_per_trade=0.0015):
    """
    Generic backtest engine.
    rebalance_months: 1=monthly, 2=bimonthly, 3=quarterly
    """
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    
    # Filter rebalance dates
    if rebalance_months > 1:
        rebal_dates = month_ends[::rebalance_months]
    else:
        rebal_dates = month_ends
    
    portfolio_values = []
    portfolio_dates = []
    holdings_history = {}
    turnover_list = []
    prev_weights = {}
    prev_holdings = set()
    
    current_value = 1.0
    last_rebal_idx = -rebalance_months  # Force first rebalance
    
    for i in range(len(month_ends) - 1):
        date = month_ends[i]
        next_date = month_ends[i + 1]
        
        # Check if this is a rebalance date
        is_rebal = (date in rebal_dates) or (i == 0)
        
        if is_rebal:
            new_weights = strategy_fn(close_df, volume_df, sectors, date, prev_holdings)
            
            # Compute turnover
            all_tickers = set(list(new_weights.keys()) + list(prev_weights.keys()))
            turnover = 0
            for t in all_tickers:
                old_w = prev_weights.get(t, 0)
                new_w = new_weights.get(t, 0)
                turnover += abs(new_w - old_w)
            turnover /= 2  # One-way turnover
            turnover_list.append(turnover)
            
            current_weights = new_weights
            prev_weights = new_weights.copy()
            prev_holdings = set(new_weights.keys())
            
            selected = list(new_weights.keys())
            holdings_history[date.strftime('%Y-%m')] = selected
        else:
            current_weights = prev_weights
        
        # Calculate period return
        invested_frac = sum(current_weights.values())
        cash_frac = 1.0 - invested_frac
        
        period_returns = []
        weight_sum = 0
        for t, w in current_weights.items():
            try:
                t_prices = close_df[t].loc[date:next_date].dropna()
                if len(t_prices) >= 2:
                    ret = t_prices.iloc[-1] / t_prices.iloc[0] - 1
                    period_returns.append(ret * w)
                    weight_sum += w
            except:
                pass
        
        port_ret = sum(period_returns) + cash_frac * 0.0  # Cash earns 0%
        
        # Transaction cost
        if is_rebal:
            cost = turnover * cost_per_trade * 2
            port_ret -= cost
        
        current_value *= (1 + port_ret)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)
    
    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    avg_turnover = np.mean(turnover_list) if turnover_list else 0
    
    # For bimonthly/quarterly, annualize turnover differently
    periods_per_year = 12 / rebalance_months
    annual_turnover = avg_turnover * periods_per_year
    monthly_equiv_turnover = avg_turnover  # Per rebalance period
    
    return equity, holdings_history, avg_turnover, turnover_list


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
    max_dd_date = drawdown.idxmin()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0
    
    return {
        'name': name, 'cagr': cagr, 'total_return': total_return,
        'max_dd': max_dd, 'max_dd_date': str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
        'sharpe': sharpe, 'calmar': calmar, 'win_rate': win_rate,
    }


# ─── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🐻 代码熊 — 动量轮动选股策略 v2 全面优化")
    print("=" * 70)
    
    # Load data
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df, volume_df, loaded = load_all_data(tickers + ['SPY'])
    sectors = load_sectors()
    print(f"Loaded {loaded} stocks, {len(sectors)} sectors mapped")
    
    # Strategy definitions
    strategies = {
        'v1_baseline': {
            'fn': strategy_v1_baseline,
            'rebal_months': 1,
            'desc': 'v1 原版: 纯动量Top10等权月度'
        },
        'v2a_regime': {
            'fn': strategy_v2a_regime_absmom,
            'rebal_months': 1,
            'desc': 'v2a: Regime过滤 + 绝对动量'
        },
        'v2b_lowturn': {
            'fn': strategy_v2b_lowturnover,
            'rebal_months': 2,
            'desc': 'v2b: 双月+Skip1M+持仓惯性'
        },
        'v2c_sector': {
            'fn': strategy_v2c_sector_vol,
            'rebal_months': 1,
            'desc': 'v2c: 行业分散+Vol加权+Regime'
        },
        'v2d_adaptive': {
            'fn': strategy_v2d_adaptive,
            'rebal_months': 1,
            'desc': 'v2d: 自适应持仓+Regime+行业+Vol'
        },
        'v2e_conservative': {
            'fn': strategy_v2e_conservative,
            'rebal_months': 2,
            'desc': 'v2e: 保守最优组合(双月+Skip1M+行业+Vol)'
        },
    }
    
    all_results = {}
    
    for key, cfg in strategies.items():
        print(f"\n{'─' * 60}")
        print(f"📊 Running {key}: {cfg['desc']}")
        print(f"{'─' * 60}")
        
        # Full period
        eq_full, hold_full, to_full, to_list_full = run_backtest(
            close_df, volume_df, sectors, cfg['fn'],
            '2015-01-01', '2025-12-31', cfg['rebal_months'])
        
        # IS period
        eq_is, hold_is, to_is, _ = run_backtest(
            close_df, volume_df, sectors, cfg['fn'],
            '2015-01-01', '2020-12-31', cfg['rebal_months'])
        
        # OOS period
        eq_oos, hold_oos, to_oos, _ = run_backtest(
            close_df, volume_df, sectors, cfg['fn'],
            '2021-01-01', '2025-12-31', cfg['rebal_months'])
        
        m_full = compute_metrics(eq_full, key)
        m_is = compute_metrics(eq_is, f"{key}_IS")
        m_oos = compute_metrics(eq_oos, f"{key}_OOS")
        
        wf_ratio = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
        wf_pass = wf_ratio >= 0.70
        
        composite = m_full['sharpe'] * 0.4 + m_full['calmar'] * 0.4 + m_full['cagr'] * 0.2
        
        result = {
            'desc': cfg['desc'],
            'full': m_full,
            'is': m_is,
            'oos': m_oos,
            'wf_ratio': wf_ratio,
            'wf_pass': wf_pass,
            'avg_turnover': to_full,
            'composite': composite,
            'holdings': hold_full,
            'rebal_months': cfg['rebal_months'],
        }
        all_results[key] = result
        
        print(f"  Full:  CAGR {m_full['cagr']:.1%}  MaxDD {m_full['max_dd']:.1%}  "
              f"Sharpe {m_full['sharpe']:.2f}  Calmar {m_full['calmar']:.2f}")
        print(f"  IS:    Sharpe {m_is['sharpe']:.2f}")
        print(f"  OOS:   Sharpe {m_oos['sharpe']:.2f}")
        print(f"  WF:    {wf_ratio:.2f} {'✅' if wf_pass else '❌'}")
        print(f"  TO:    {to_full:.1%}/period  Composite: {composite:.3f}")
        print(f"  MaxDD date: {m_full.get('max_dd_date', 'N/A')}")
    
    # SPY benchmark
    print(f"\n{'─' * 60}")
    print(f"📊 SPY Buy & Hold")
    spy_prices = close_df['SPY'].loc['2015-01-01':'2025-12-31'].dropna()
    spy_monthly = spy_prices.resample('ME').last()
    spy_eq = spy_monthly / spy_monthly.iloc[0]
    m_spy = compute_metrics(spy_eq, "SPY B&H")
    print(f"  CAGR {m_spy['cagr']:.1%}  MaxDD {m_spy['max_dd']:.1%}  Sharpe {m_spy['sharpe']:.2f}")
    
    # ─── Comparison Table ─────────────────────────────────────
    print("\n" + "=" * 110)
    print("📊 COMPARISON TABLE")
    print("=" * 110)
    print(f"{'Version':<16} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Turnover':>9} "
          f"{'IS Sh':>7} {'OOS Sh':>8} {'WF':>6} {'Composite':>10}")
    print("-" * 110)
    
    for key, r in all_results.items():
        wf_mark = '✅' if r['wf_pass'] else '❌'
        print(f"{key:<16} "
              f"{r['full']['cagr']:>6.1%} "
              f"{r['full']['max_dd']:>7.1%} "
              f"{r['full']['sharpe']:>7.2f} "
              f"{r['avg_turnover']:>8.1%} "
              f"{r['is']['sharpe']:>7.2f} "
              f"{r['oos']['sharpe']:>8.2f} "
              f"{wf_mark:>4} "
              f"{r['composite']:>10.3f}")
    
    print(f"{'SPY B&H':<16} "
          f"{m_spy['cagr']:>6.1%} "
          f"{m_spy['max_dd']:>7.1%} "
          f"{m_spy['sharpe']:>7.2f} "
          f"{'—':>9} "
          f"{'—':>7} "
          f"{'—':>8} "
          f"{'—':>4} "
          f"{'—':>10}")
    
    # ─── Find best ──────────────────────────────────────────
    # Priority: WF pass first, then composite
    wf_passed = {k: v for k, v in all_results.items() if v['wf_pass'] and k != 'v1_baseline'}
    
    if wf_passed:
        best_key = max(wf_passed, key=lambda k: wf_passed[k]['composite'])
        print(f"\n🏆 Best (WF passed): {best_key}")
    else:
        # Pick highest OOS/IS ratio
        v2_results = {k: v for k, v in all_results.items() if k != 'v1_baseline'}
        best_key = max(v2_results, key=lambda k: v2_results[k]['wf_ratio'])
        print(f"\n🏆 Best (highest WF ratio): {best_key}")
    
    best = all_results[best_key]
    print(f"   CAGR: {best['full']['cagr']:.1%}  Sharpe: {best['full']['sharpe']:.2f}  "
          f"MaxDD: {best['full']['max_dd']:.1%}")
    print(f"   IS: {best['is']['sharpe']:.2f}  OOS: {best['oos']['sharpe']:.2f}  "
          f"WF: {best['wf_ratio']:.2f}")
    print(f"   Turnover: {best['avg_turnover']:.1%}/period  "
          f"Composite: {best['composite']:.3f}")
    
    # ─── Holdings Analysis ──────────────────────────────────
    print(f"\n📋 {best_key} — Holdings Analysis")
    hold = best['holdings']
    
    # Most frequent stocks
    all_h = []
    for stocks in hold.values():
        all_h.extend(stocks)
    freq = Counter(all_h).most_common(15)
    total_periods = len(hold)
    print(f"\nTop 15 most selected (total {total_periods} periods):")
    for ticker, count in freq:
        sector = sectors.get(ticker, '?')
        print(f"  {ticker:6s} [{sector:12s}]: {count:3d}/{total_periods} ({count/total_periods:.0%})")
    
    # 2023-2024 holdings
    print(f"\n🔍 2023-2024 Holdings:")
    hot_set = {'NVDA', 'TSLA', 'META', 'AMZN', 'AVGO', 'AMD', 'SMCI', 'PLTR'}
    for ym in sorted(hold.keys()):
        if ym.startswith('2023') or ym.startswith('2024'):
            hot = [s for s in hold[ym] if s in hot_set]
            all_stocks = ', '.join(hold[ym][:5]) + ('...' if len(hold[ym]) > 5 else '')
            print(f"  {ym}: {all_stocks}  {'🔥 ' + ', '.join(hot) if hot else ''}")
    
    # Sector distribution for best
    print(f"\n📊 Sector Distribution:")
    all_sectors = [sectors.get(t, 'Unknown') for t in all_h]
    sec_freq = Counter(all_sectors).most_common()
    for sec, cnt in sec_freq:
        print(f"  {sec:15s}: {cnt:4d} ({cnt/len(all_h):.0%})")
    
    # ─── Save results ──────────────────────────────────────
    output = {
        'comparison': {},
        'best': best_key,
        'spy': m_spy,
    }
    for key, r in all_results.items():
        output['comparison'][key] = {
            'desc': r['desc'],
            'full': r['full'],
            'is': r['is'],
            'oos': r['oos'],
            'wf_ratio': r['wf_ratio'],
            'wf_pass': r['wf_pass'],
            'avg_turnover': r['avg_turnover'],
            'composite': r['composite'],
            'holdings': r['holdings'],
        }
    
    out_file = BASE / "stocks" / "codebear" / "momentum_v2_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved to {out_file}")
    
    return all_results, best_key

if __name__ == '__main__':
    results, best = main()
