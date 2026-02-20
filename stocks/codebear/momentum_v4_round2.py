#!/usr/bin/env python3
"""
v4d DD Responsive — Round 2: Parameter tuning & Hybrid v4f
代码熊 🐻
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

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
    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'vol_30d': vol_30d, 'spy_sma200': spy_sma200, 'spy_close': spy_close,
        'sma_50': sma_50, 'close': close_df,
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


def select_stocks_core(signals, sectors, date, prev_holdings, regime_override=None):
    """Core stock selection (v3b logic). Returns (weights, regime)."""
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}, 'bull'
    idx = idx[-1]
    regime = regime_override or get_regime(signals, date)

    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'momentum': mom, 'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx], 'price': close.loc[idx],
        'sma50': signals['sma_50'].loc[idx],
    }).dropna(subset=['momentum', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    df = df[df['price'] > df['sma50']]
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
        return {}, regime

    mom_share = 0.30
    iv = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    iv_total = sum(iv.values())
    iv_w = {t: v / iv_total for t, v in iv.items()}
    mom_min = min(df.loc[t, 'momentum'] for t in selected)
    shift = max(-mom_min + 0.01, 0)
    mw = {t: df.loc[t, 'momentum'] + shift for t in selected}
    mw_total = sum(mw.values())
    mw_w = {t: v / mw_total for t, v in mw.items()}

    invested = 1.0 - cash
    weights = {t: ((1 - mom_share) * iv_w[t] + mom_share * mw_w[t]) * invested for t in selected}
    return weights, regime


def add_gld(weights, gld_frac):
    """Scale stock weights down and add GLD allocation."""
    if gld_frac <= 0 or not weights:
        return weights
    total_w = sum(weights.values())
    stock_frac = 1.0 - gld_frac
    new = {t: (w / total_w) * stock_frac for t, w in weights.items()}
    new['GLD'] = gld_frac
    return new


def run_backtest(close_df, signals, sectors, gld_prices,
                 variant_name, dd_params=None,
                 start='2015-01-01', end='2025-12-31',
                 cost_per_trade=0.0015):
    """
    variant_name: 'v4d_orig' | 'v4d_tight' | 'v4d_3tier' | 'v4d_soft' | 'v4d_aggr' | 'v4f'
    dd_params: dict with dd thresholds -> gld allocations
    """
    close_range = close_df.loc[start:end].dropna(how='all')
    gld_range = gld_prices.loc[start:end].dropna()
    month_ends = close_range.resample('ME').last().index

    portfolio_values, portfolio_dates = [], []
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    peak_value = 1.0

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]

        dd = (current_value - peak_value) / peak_value if peak_value > 0 else 0

        weights, regime = select_stocks_core(signals, sectors, date, prev_holdings)

        # Determine GLD allocation based on variant
        gld_alloc = 0.0

        if variant_name == 'v4f':
            # Hybrid: bear regime base + DD responsive
            base_gld = 0.20 if regime == 'bear' else 0.0
            if dd < -0.15:
                dd_gld = 0.30
            elif dd < -0.10:
                dd_gld = 0.10
            else:
                dd_gld = 0.0
            gld_alloc = min(base_gld + dd_gld, 0.50)
            # In bear: no cash (all invested in stocks+GLD)
            if regime == 'bear':
                weights, _ = select_stocks_core(signals, sectors, date, prev_holdings, regime_override='bear_no_cash')
                # Re-select but allocate to stocks (no 20% cash), cash goes to GLD
                weights_bull, _ = select_stocks_core(signals, sectors, date, prev_holdings, regime_override='bull')
                weights = weights_bull  # Use full-invested weights
        elif dd_params:
            # Sort thresholds descending (most aggressive first)
            sorted_thresholds = sorted(dd_params.items(), key=lambda x: x[0])
            for threshold, alloc in sorted_thresholds:
                if dd < threshold:
                    gld_alloc = alloc
            # In v4d variants, bear regime keeps its 20% cash; GLD is on top of DD
            # Actually in v4d, GLD replaces cash and then some
            # Original v4d: just replace the stock weights with GLD
        else:
            pass  # No GLD

        new_weights = add_gld(weights, gld_alloc)

        # Turnover
        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = new_weights.copy()
        prev_holdings = set(k for k in new_weights.keys() if k != 'GLD')

        # Returns
        port_ret = 0.0
        for t, w in new_weights.items():
            if t == 'GLD':
                s = gld_range.loc[date:next_date].dropna()
                if len(s) >= 2:
                    port_ret += (s.iloc[-1] / s.iloc[0] - 1) * w
            else:
                if t in close_df.columns:
                    s = close_df[t].loc[date:next_date].dropna()
                    if len(s) >= 2:
                        port_ret += (s.iloc[-1] / s.iloc[0] - 1) * w

        port_ret -= turnover * cost_per_trade * 2
        current_value *= (1 + port_ret)
        if current_value > peak_value:
            peak_value = current_value
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)

    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    return equity, np.mean(turnover_list) if turnover_list else 0


def compute_metrics(equity, name="Strategy"):
    if len(equity) < 3:
        return {'name': name, 'cagr': 0, 'max_dd': 0, 'sharpe': 0, 'calmar': 0}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years < 0.5:
        return {'name': name, 'cagr': 0, 'max_dd': 0, 'sharpe': 0, 'calmar': 0}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    monthly = equity.pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    dd = (equity - equity.cummax()) / equity.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'name': name, 'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar}


def main():
    print("=" * 70)
    print("🐻 v4d Round 2 — Parameter Sensitivity & Hybrid v4f")
    print("=" * 70)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)
    gld_prices = load_csv(CACHE / "GLD.csv")['Close'].dropna()

    variants = {
        'v4d_orig':  {-0.10: 0.30, -0.15: 0.50},
        'v4d_tight': {-0.08: 0.25, -0.13: 0.45},
        'v4d_3tier': {-0.08: 0.20, -0.13: 0.35, -0.18: 0.50},
        'v4d_soft':  {-0.10: 0.20, -0.15: 0.35},
        'v4d_aggr':  {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60},
        'v4f_hybrid': None,  # Special handling
    }

    results = {}
    for name, dd_params in variants.items():
        print(f"\n🔄 {name}...")
        vname = 'v4f' if name == 'v4f_hybrid' else name

        eq_full, to = run_backtest(close_df, signals, sectors, gld_prices,
                                    vname, dd_params)
        eq_is, _ = run_backtest(close_df, signals, sectors, gld_prices,
                                 vname, dd_params, '2015-01-01', '2020-12-31')
        eq_oos, _ = run_backtest(close_df, signals, sectors, gld_prices,
                                  vname, dd_params, '2021-01-01', '2025-12-31')

        m = compute_metrics(eq_full)
        m_is = compute_metrics(eq_is)
        m_oos = compute_metrics(eq_oos)
        wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
        comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2

        results[name] = {'m': m, 'is': m_is, 'oos': m_oos, 'wf': wf, 'comp': comp}
        print(f"  CAGR: {m['cagr']:.1%}  Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_dd']:.1%}  "
              f"Calmar: {m['calmar']:.2f}  Comp: {comp:.3f}  WF: {wf:.2f} {'✅' if wf >= 0.70 else '❌'}")

    # Summary
    print("\n" + "=" * 95)
    print(f"{'Name':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'IS':>6} {'OOS':>6} {'WF':>6} {'Comp':>8}")
    print("-" * 95)
    for name in sorted(results.keys(), key=lambda x: results[x]['comp'], reverse=True):
        r = results[name]
        m = r['m']
        wf_flag = "✅" if r['wf'] >= 0.70 else "❌"
        print(f"{name:<18} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} {m['sharpe']:>7.2f} "
              f"{m['calmar']:>7.2f} {r['is']['sharpe']:>6.2f} {r['oos']['sharpe']:>6.2f} "
              f"{r['wf']:>5.2f}{wf_flag} {r['comp']:>8.3f}")

    best = max(results.items(), key=lambda x: x[1]['comp'] if x[1]['wf'] >= 0.70 else -999)
    print(f"\n🏆 Best: {best[0]} (Composite: {best[1]['comp']:.3f})")


if __name__ == '__main__':
    main()
