#!/usr/bin/env python3
"""
Stock Momentum v3 Round 2 — Focused Optimization
代码熊 🐻

Round 1 findings:
- v3e (trend filter): Composite 1.161 but WF 0.66 (FAIL by 0.04)
- v3b (sector rotation): Composite 1.079, WF 0.72 (PASS)
- Key insight: SMA50 trend filter is the biggest innovation (MaxDD -14.5%)
- Key insight: DD protection hurts OOS → need to soften it

Round 2 strategies:
- v3g: Sector rotation + trend filter (combine v3b + v3e best elements)
- v3h: v3e with NO DD protection (pure trend filter)
- v3i: v3e with softer bear handling (more invested in bear)
- v3j: v3b + trend filter + momentum-weighted (not inv-vol)
- v3k: v2d + trend filter only (minimal change, maximum impact)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


def load_csv(filepath):
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    for col in ['Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_all_data(tickers):
    close_dict, volume_dict = {}, {}
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
        except:
            pass
    return pd.DataFrame(close_dict), pd.DataFrame(volume_dict)


def precompute_signals(close_df, volume_df=None):
    ret_1m = close_df / close_df.shift(22) - 1
    ret_3m = close_df / close_df.shift(63) - 1
    ret_6m = close_df / close_df.shift(126) - 1
    ret_12m = close_df / close_df.shift(252) - 1
    log_ret = np.log(close_df / close_df.shift(1))
    vol_30d = log_ret.rolling(30).std() * np.sqrt(252)
    spy_close = close_df['SPY'] if 'SPY' in close_df.columns else None
    spy_sma200 = spy_close.rolling(200).mean() if spy_close is not None else None
    sma_50 = close_df.rolling(50).mean()
    sma_200 = close_df.rolling(200).mean()
    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'vol_30d': vol_30d,
        'spy_sma200': spy_sma200, 'spy_close': spy_close,
        'sma_50': sma_50, 'sma_200': sma_200,
        'close': close_df,
    }


def get_regime(signals, date):
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid = spy_sma.loc[:date].dropna()
    spy_close = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_close) == 0:
        return 'bull'
    return 'bull' if spy_close.iloc[-1] > valid.iloc[-1] else 'bear'


def make_stock_df(signals, date, trend_filter='none'):
    """Create stock dataframe with all signals. trend_filter: 'none', 'sma50', 'sma200', 'both'"""
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[-1]

    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)

    cols = {
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
    }

    if trend_filter in ('sma50', 'both'):
        cols['sma50'] = signals['sma_50'].loc[idx]
    if trend_filter in ('sma200', 'both'):
        cols['sma200'] = signals['sma_200'].loc[idx]

    df = pd.DataFrame(cols)
    df = df.dropna(subset=['momentum'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]

    if trend_filter == 'sma50':
        df = df.dropna(subset=['sma50'])
        df = df[df['price'] > df['sma50']]
    elif trend_filter == 'sma200':
        df = df.dropna(subset=['sma200'])
        df = df[df['price'] > df['sma200']]
    elif trend_filter == 'both':
        df = df.dropna(subset=['sma50', 'sma200'])
        df = df[(df['price'] > df['sma50']) & (df['price'] > df['sma200'])]

    return df


# ============================================================
# V2D BASELINE
# ============================================================
def select_v2d(signals, sectors, date, prev_holdings, **kw):
    df = make_stock_df(signals, date, trend_filter='none')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    df = df.sort_values('momentum', ascending=False)
    if regime == 'bull':
        top_n, max_sec, cash = 12, 3, 0.0
    else:
        top_n, max_sec, cash = 8, 2, 0.20
    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < max_sec:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= top_n:
            break
    if not selected:
        return {}
    inv_vols = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3G: Sector Rotation + Trend Filter (v3b + v3e best)
# ============================================================
def select_v3g(signals, sectors, date, prev_holdings, **kw):
    """
    v3g: Two-stage sector rotation WITH SMA50 trend filter.
    Stage 1: Filter stocks above SMA50
    Stage 2: Rank sectors by avg momentum of filtered stocks
    Stage 3: Top 4 sectors in bull, top 3 in bear
    Stage 4: Within each, pick top 3 stocks
    """
    df = make_stock_df(signals, date, trend_filter='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))

    # Sector momentum from filtered stocks
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)

    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        stocks_per_sector = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        stocks_per_sector = 3
        cash = 0.15

    # Holdover bonus
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:stocks_per_sector].tolist())

    if not selected:
        return {}
    inv_vols = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3H: Pure Trend Filter (v3e without DD protection)
# ============================================================
def select_v3h(signals, sectors, date, prev_holdings, **kw):
    """
    v3h: v2d + SMA50 trend filter, NO drawdown protection.
    The DD protection in v3e may have hurt OOS by being too cautious.
    Hypothesis: pure trend filter is enough to reduce MaxDD.
    """
    df = make_stock_df(signals, date, trend_filter='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    df = df.sort_values('momentum', ascending=False)

    if regime == 'bull':
        top_n, max_sec, cash = 12, 3, 0.0
    else:
        top_n, max_sec, cash = 8, 2, 0.20

    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < max_sec:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= top_n:
            break
    if not selected:
        return {}
    inv_vols = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3I: Trend Filter + Softer Bear (more invested in bear)
# ============================================================
def select_v3i(signals, sectors, date, prev_holdings, **kw):
    """
    v3i: SMA50 trend filter + very soft bear (90% invested in bear).
    Hypothesis: trend filter already removes weak stocks, 
    so we don't need much cash in bear — let trend filter do the work.
    """
    df = make_stock_df(signals, date, trend_filter='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    df = df.sort_values('momentum', ascending=False)

    if regime == 'bull':
        top_n, max_sec, cash = 12, 3, 0.0
    else:
        # Very soft bear: only 10% cash, let trend filter handle risk
        top_n, max_sec, cash = 10, 2, 0.10

    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < max_sec:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= top_n:
            break
    if not selected:
        return {}
    inv_vols = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3J: Sector Rotation + Trend + Momentum Weight (not inv-vol)
# ============================================================
def select_v3j(signals, sectors, date, prev_holdings, **kw):
    """
    v3j: Like v3g but with momentum-proportional weighting instead of inv-vol.
    More weight to high-momentum stocks → higher CAGR potential.
    """
    df = make_stock_df(signals, date, trend_filter='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)

    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        stocks_per_sector = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        stocks_per_sector = 3
        cash = 0.15

    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:stocks_per_sector].tolist())

    if not selected:
        return {}
    # Momentum-proportional weighting (shifted so min = 0.1)
    mom_vals = {t: max(df.loc[t, 'momentum'], 0.01) for t in selected}
    total = sum(mom_vals.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in mom_vals.items()}


# ============================================================
# V3K: v2d + SMA50 trend only (minimal change, max impact)
# ============================================================
def select_v3k(signals, sectors, date, prev_holdings, **kw):
    """
    v3k: EXACTLY v2d but add SMA50 trend filter.
    This isolates the impact of the trend filter from all other changes.
    """
    df = make_stock_df(signals, date, trend_filter='sma50')
    if df.empty:
        # Fallback to no trend filter
        df = make_stock_df(signals, date, trend_filter='none')
        if df.empty:
            return {}
    regime = get_regime(signals, date)
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    df = df.sort_values('momentum', ascending=False)

    if regime == 'bull':
        top_n, max_sec, cash = 12, 3, 0.0
    else:
        top_n, max_sec, cash = 8, 2, 0.20

    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < max_sec:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= top_n:
            break
    if not selected:
        return {}
    inv_vols = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3L: v3k + sector rotation (combine best of both)
# ============================================================
def select_v3l(signals, sectors, date, prev_holdings, **kw):
    """
    v3l: Trend filter + sector diversification (max 3 in bull, 2 in bear)
         + slightly more aggressive sector max in bull (to capture hot sectors).
    Hybrid: not full sector rotation like v3b, but tighter sector caps.
    """
    df = make_stock_df(signals, date, trend_filter='sma50')
    if df.empty:
        df = make_stock_df(signals, date, trend_filter='none')
        if df.empty:
            return {}
    regime = get_regime(signals, date)
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    df = df.sort_values('momentum', ascending=False)

    if regime == 'bull':
        top_n, max_sec, cash = 14, 3, 0.0
    else:
        top_n, max_sec, cash = 9, 2, 0.15

    selected = []
    sector_count = Counter()
    for t in df.index:
        sec = sectors.get(t, 'Unknown')
        if sector_count[sec] < max_sec:
            selected.append(t)
            sector_count[sec] += 1
        if len(selected) >= top_n:
            break
    if not selected:
        return {}
    inv_vols = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(close_df, signals, sectors, select_fn, start='2015-01-01', end='2025-12-31',
                 cost_per_trade=0.0015):
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    portfolio_values, portfolio_dates = [], []
    holdings_history = {}
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    peak_value = 1.0

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]
        portfolio_dd = (current_value / peak_value) - 1 if peak_value > 0 else 0
        new_weights = select_fn(signals, sectors, date, prev_holdings, portfolio_dd=portfolio_dd)
        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = new_weights.copy()
        prev_holdings = set(new_weights.keys())
        holdings_history[date.strftime('%Y-%m')] = list(new_weights.keys())

        port_ret = 0.0
        for t, w in new_weights.items():
            try:
                series = close_df[t].loc[date:next_date].dropna()
                if len(series) >= 2:
                    port_ret += (series.iloc[-1] / series.iloc[0] - 1) * w
            except:
                pass
        port_ret -= turnover * cost_per_trade * 2
        current_value *= (1 + port_ret)
        peak_value = max(peak_value, current_value)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)

    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    return equity, holdings_history, np.mean(turnover_list) if turnover_list else 0


def compute_metrics(equity, name="Strategy"):
    if len(equity) < 2:
        return {'name': name, 'cagr': 0, 'max_dd': 0, 'sharpe': 0, 'calmar': 0}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return {'name': name, 'cagr': 0, 'max_dd': 0, 'sharpe': 0, 'calmar': 0}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    monthly = equity.pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    dd = (equity - equity.cummax()) / equity.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'name': name, 'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar}


def composite_score(cagr, sharpe, calmar):
    return sharpe * 0.4 + calmar * 0.4 + cagr * 0.2


def main():
    print("=" * 70)
    print("🐻 Stock Momentum v3 Round 2 — Focused Optimization")
    print("=" * 70)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df, volume_df = load_all_data(tickers + ['SPY'])
    sectors_file = CACHE / "sp500_sectors.json"
    sectors = json.load(open(sectors_file)) if sectors_file.exists() else {}

    print(f"Loaded {len(close_df.columns)} stocks")
    signals = precompute_signals(close_df)

    strategies = {
        'v2d_base': select_v2d,
        'v3g_sec+trend': select_v3g,
        'v3h_pure_trend': select_v3h,
        'v3i_soft_bear': select_v3i,
        'v3j_sec+mom_wt': select_v3j,
        'v3k_v2d+trend': select_v3k,
        'v3l_trend+wide': select_v3l,
    }

    results = {}
    for name, fn in strategies.items():
        print(f"\nRunning {name}...")
        eq_full, hold_full, to_full = run_backtest(close_df, signals, sectors, fn)
        m_full = compute_metrics(eq_full, name)
        eq_is, _, _ = run_backtest(close_df, signals, sectors, fn, '2015-01-01', '2020-12-31')
        eq_oos, _, _ = run_backtest(close_df, signals, sectors, fn, '2021-01-01', '2025-12-31')
        m_is = compute_metrics(eq_is, f"{name}_IS")
        m_oos = compute_metrics(eq_oos, f"{name}_OOS")
        wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
        comp = composite_score(m_full['cagr'], m_full['sharpe'], m_full['calmar'])
        results[name] = {
            'full': m_full, 'is': m_is, 'oos': m_oos,
            'wf': wf, 'composite': comp, 'turnover': to_full,
            'holdings': hold_full, 'equity': eq_full,
        }
        wf_pass = "✅" if wf >= 0.70 else "❌"
        print(f"  CAGR: {m_full['cagr']:.1%}  Sharpe: {m_full['sharpe']:.2f}  "
              f"MaxDD: {m_full['max_dd']:.1%}  Calmar: {m_full['calmar']:.2f}")
        print(f"  IS: {m_is['sharpe']:.2f}  OOS: {m_oos['sharpe']:.2f}  WF: {wf:.2f} {wf_pass}")
        print(f"  Turnover: {to_full:.1%}  Composite: {comp:.3f}")

    # SPY benchmark
    spy_eq = close_df['SPY'].loc['2015-01-01':'2025-12-31'].dropna()
    spy_eq = spy_eq / spy_eq.iloc[0]
    spy_m = compute_metrics(spy_eq, "SPY")

    print("\n" + "=" * 115)
    print("📊 ROUND 2 COMPARISON TABLE")
    print("=" * 115)
    header = f"{'Version':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'T/O':>6} {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>5} {'Pass':>5} {'Composite':>10}"
    print(header)
    print("-" * 115)

    for name, r in sorted(results.items(), key=lambda x: -x[1]['composite']):
        m = r['full']
        wf_pass = "✅" if r['wf'] >= 0.70 else "❌"
        print(f"{name:<18} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} {m['sharpe']:>7.2f} "
              f"{m['calmar']:>7.2f} {r['turnover']:>5.1%} {r['is']['sharpe']:>6.2f} "
              f"{r['oos']['sharpe']:>7.2f} {r['wf']:>5.2f} {wf_pass:>5} {r['composite']:>10.3f}")

    print(f"{'SPY':<18} {spy_m['cagr']:>6.1%} {spy_m['max_dd']:>7.1%} {spy_m['sharpe']:>7.2f} "
          f"{spy_m['calmar']:>7.2f} {'—':>6} {'—':>6} {'—':>7} {'—':>5} {'—':>5} {'—':>10}")

    # Analysis
    wf_passing = {k: v for k, v in results.items() if v['wf'] >= 0.70}
    if wf_passing:
        best_wf = max(wf_passing.items(), key=lambda x: x[1]['composite'])
        print(f"\n🏆 Best WF-passing: {best_wf[0]} (Composite: {best_wf[1]['composite']:.3f})")
    best_overall = max(results.items(), key=lambda x: x[1]['composite'])
    print(f"🥇 Best overall: {best_overall[0]} (Composite: {best_overall[1]['composite']:.3f})")

    # Improvement over v2d
    v2d_comp = results['v2d_base']['composite']
    for name, r in sorted(results.items(), key=lambda x: -x[1]['composite']):
        if name != 'v2d_base':
            delta = r['composite'] - v2d_comp
            print(f"  {name}: Δ Composite = {delta:+.3f}")

    return results


if __name__ == '__main__':
    results = main()
