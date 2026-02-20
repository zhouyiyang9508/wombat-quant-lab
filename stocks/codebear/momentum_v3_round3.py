#!/usr/bin/env python3
"""
Stock Momentum v3 Round 3 — v3g Refinement
代码熊 🐻

v3g is the champion: Composite 1.171, WF 0.99 ✅
Now we fine-tune v3g to push even higher.

Variants:
- v3g_base: Original v3g (baseline for Round 3)
- v3m: v3g + 5 sectors (more diversified)
- v3n: v3g + momentum-weighted (blend: 50% inv-vol + 50% mom)
- v3o: v3g + dynamic bear (fewer stocks, more cash in bear)
- v3p: v3g + SMA200 trend (stricter trend filter)
- v3q: v3g + 4 stocks per sector (wider net)
- v3r: v3g + holdover bonus + hysteresis (sector sticky)
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
    close_dict = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                close_dict[t] = df['Close'].dropna()
        except:
            pass
    return pd.DataFrame(close_dict)


def precompute_signals(close_df):
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


def make_stock_df(signals, date, trend='sma50'):
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[-1]
    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)
    df = pd.DataFrame({
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
        'sma50': signals['sma_50'].loc[idx],
        'sma200': signals['sma_200'].loc[idx],
    }).dropna(subset=['momentum'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    if trend == 'sma50':
        df = df.dropna(subset=['sma50'])
        df = df[df['price'] > df['sma50']]
    elif trend == 'sma200':
        df = df.dropna(subset=['sma200'])
        df = df[df['price'] > df['sma200']]
    elif trend == 'both':
        df = df.dropna(subset=['sma50', 'sma200'])
        df = df[(df['price'] > df['sma50']) & (df['price'] > df['sma200'])]
    return df


def inv_vol_weights(df, selected, cash):
    inv_vols = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


def blended_weights(df, selected, cash, mom_share=0.50):
    """Blend of inverse-vol and momentum-proportional weights."""
    if not selected:
        return {}
    # Inverse vol part
    iv = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    iv_total = sum(iv.values())
    iv_w = {t: v / iv_total for t, v in iv.items()}
    # Momentum part (shifted to positive)
    mom_min = min(df.loc[t, 'momentum'] for t in selected)
    shift = max(-mom_min + 0.01, 0)
    mw = {t: df.loc[t, 'momentum'] + shift for t in selected}
    mw_total = sum(mw.values())
    mw_w = {t: v / mw_total for t, v in mw.items()}
    # Blend
    invested = 1.0 - cash
    return {t: ((1 - mom_share) * iv_w[t] + mom_share * mw_w[t]) * invested for t in selected}


# ============================================================
# V3G BASE (Round 2 champion)
# ============================================================
def select_v3g(signals, sectors, date, prev_holdings, prev_sectors=None, **kw):
    df = make_stock_df(signals, date, trend='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)
    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 3
        cash = 0.15
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())
    if not selected:
        return {}
    return inv_vol_weights(df, selected, cash)


# ============================================================
# V3M: 5 sectors (more diversified)
# ============================================================
def select_v3m(signals, sectors, date, prev_holdings, **kw):
    df = make_stock_df(signals, date, trend='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)
    if regime == 'bull':
        top_sectors = sector_mom.head(5).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 2
        cash = 0.15
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())
    if not selected:
        return {}
    return inv_vol_weights(df, selected, cash)


# ============================================================
# V3N: Blended weights (50% inv-vol + 50% momentum)
# ============================================================
def select_v3n(signals, sectors, date, prev_holdings, **kw):
    df = make_stock_df(signals, date, trend='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)
    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 3
        cash = 0.15
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())
    if not selected:
        return {}
    return blended_weights(df, selected, cash, mom_share=0.50)


# ============================================================
# V3O: More aggressive bear handling
# ============================================================
def select_v3o(signals, sectors, date, prev_holdings, **kw):
    df = make_stock_df(signals, date, trend='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)
    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        # More aggressive bear: 2 sectors, 3 stocks each, 25% cash
        top_sectors = sector_mom.head(2).index.tolist()
        sps = 3
        cash = 0.25
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())
    if not selected:
        return {}
    return inv_vol_weights(df, selected, cash)


# ============================================================
# V3P: SMA50+SMA200 dual trend filter (stricter)
# ============================================================
def select_v3p(signals, sectors, date, prev_holdings, **kw):
    df = make_stock_df(signals, date, trend='both')
    if len(df) < 5:
        df = make_stock_df(signals, date, trend='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)
    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 3
        cash = 0.15
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())
    if not selected:
        return {}
    return inv_vol_weights(df, selected, cash)


# ============================================================
# V3Q: 4 stocks per sector (wider net)
# ============================================================
def select_v3q(signals, sectors, date, prev_holdings, **kw):
    df = make_stock_df(signals, date, trend='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)
    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 4  # wider
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 3
        cash = 0.15
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())
    if not selected:
        return {}
    return inv_vol_weights(df, selected, cash)


# ============================================================
# V3R: Sector sticky + holdover bonus (reduce turnover)
# ============================================================
def select_v3r(signals, sectors, date, prev_holdings, prev_sectors=None, **kw):
    """Add sector stickiness: prev-month sectors get +5% bonus in ranking."""
    df = make_stock_df(signals, date, trend='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.04  # slightly higher holdover
    sector_mom = df.groupby('sector')['momentum'].mean()
    # Sector stickiness
    if prev_sectors:
        for sec in prev_sectors:
            if sec in sector_mom.index:
                sector_mom[sec] += 0.02
    sector_mom = sector_mom.sort_values(ascending=False)
    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 3
        cash = 0.15
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())
    if not selected:
        return {}
    return inv_vol_weights(df, selected, cash)


# ============================================================
# V3S: v3g + 30% momentum weight blend + tighter bear
# ============================================================
def select_v3s(signals, sectors, date, prev_holdings, **kw):
    df = make_stock_df(signals, date, trend='sma50')
    if df.empty:
        return {}
    regime = get_regime(signals, date)
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)
    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 2
        cash = 0.20
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())
    if not selected:
        return {}
    return blended_weights(df, selected, cash, mom_share=0.30)


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
    prev_sectors = set()
    current_value = 1.0

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]
        new_weights = select_fn(signals, sectors, date, prev_holdings, prev_sectors=prev_sectors)
        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = new_weights.copy()
        prev_holdings = set(new_weights.keys())
        prev_sectors = set(sectors.get(t, 'Unknown') for t in prev_holdings)
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
    print("🐻 Stock Momentum v3 Round 3 — v3g Refinement")
    print("=" * 70)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors_file = CACHE / "sp500_sectors.json"
    sectors = json.load(open(sectors_file)) if sectors_file.exists() else {}

    print(f"Loaded {len(close_df.columns)} stocks")
    signals = precompute_signals(close_df)

    strategies = {
        'v3g_base': select_v3g,
        'v3m_5sec': select_v3m,
        'v3n_blend': select_v3n,
        'v3o_aggbear': select_v3o,
        'v3p_dual_sma': select_v3p,
        'v3q_4per': select_v3q,
        'v3r_sticky': select_v3r,
        'v3s_blend30': select_v3s,
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
            'holdings': hold_full,
        }
        wf_pass = "✅" if wf >= 0.70 else "❌"
        print(f"  CAGR: {m_full['cagr']:.1%}  Sharpe: {m_full['sharpe']:.2f}  "
              f"MaxDD: {m_full['max_dd']:.1%}  Calmar: {m_full['calmar']:.2f}")
        print(f"  IS: {m_is['sharpe']:.2f}  OOS: {m_oos['sharpe']:.2f}  WF: {wf:.2f} {wf_pass}")
        print(f"  Turnover: {to_full:.1%}  Composite: {comp:.3f}")

    print("\n" + "=" * 115)
    print("📊 ROUND 3 COMPARISON TABLE")
    print("=" * 115)
    header = f"{'Version':<16} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'T/O':>6} {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>5} {'Pass':>5} {'Composite':>10}"
    print(header)
    print("-" * 115)

    for name, r in sorted(results.items(), key=lambda x: -x[1]['composite']):
        m = r['full']
        wf_pass = "✅" if r['wf'] >= 0.70 else "❌"
        best = " ⭐" if r['composite'] == max(x['composite'] for x in results.values() if x['wf'] >= 0.70) and r['wf'] >= 0.70 else ""
        print(f"{name:<16} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} {m['sharpe']:>7.2f} "
              f"{m['calmar']:>7.2f} {r['turnover']:>5.1%} {r['is']['sharpe']:>6.2f} "
              f"{r['oos']['sharpe']:>7.2f} {r['wf']:>5.2f} {wf_pass:>5} {r['composite']:>10.3f}{best}")

    wf_passing = {k: v for k, v in results.items() if v['wf'] >= 0.70}
    if wf_passing:
        best_wf = max(wf_passing.items(), key=lambda x: x[1]['composite'])
        print(f"\n🏆 Best WF-passing: {best_wf[0]} (Composite: {best_wf[1]['composite']:.3f})")
        bm = best_wf[1]['full']
        print(f"   CAGR: {bm['cagr']:.1%}  Sharpe: {bm['sharpe']:.2f}  MaxDD: {bm['max_dd']:.1%}  Calmar: {bm['calmar']:.2f}")
        print(f"   WF: {best_wf[1]['wf']:.2f}  Turnover: {best_wf[1]['turnover']:.1%}")

    # Holdings analysis
    if wf_passing:
        bn, br = best_wf
        print(f"\n📋 {bn} Holdings (2023-2024):")
        hot = {'NVDA', 'TSLA', 'META', 'AVGO', 'AMD', 'SMCI', 'PLTR'}
        for ym in sorted(br['holdings'].keys()):
            if ym.startswith('2023') or ym.startswith('2024'):
                h = br['holdings'][ym]
                hotties = [s for s in h if s in hot]
                hotstr = f" 🔥{','.join(hotties)}" if hotties else ""
                print(f"  {ym}: {', '.join(h[:12])}{hotstr}")

    return results


if __name__ == '__main__':
    results = main()
