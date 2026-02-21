#!/usr/bin/env python3
"""
动量轮动 v5c — v4d 精准升级版：Cash→智能安全资产 + 优化对冲选择
代码熊 🐻

核心理念: 在 v4d 基础上做最小化、精准的改进

v4d 的问题:
1. 熊市时 20% 现金: 现金收益为0，而 TLT/SHY 至少有利息收益
2. DD对冲固定用GLD: 某些时期 TLT 比 GLD 更有效（如2018/2020衰退）

v5c 的改进:
1. 熊市现金→智能安全资产: 根据3月动量选 TLT/SHY/GLD 中最优的
2. DD对冲→动态选 GLD or TLT: 根据3月动量选最优避险资产
3. 股票选择保持 v3b 完全不变（已验证的最优配置）
4. 新增 IEF (中期国债) 作为候选避险资产

理论收益:
  熊市占比约20%时间 × 20% cash 仓位 × TLT预期6-8%回报 ≈ 额外0.2-0.3% CAGR
  DD对冲优化: 2020年TLT在COVID暴跌初期大涨(+20%)，比GLD(+5%)有效得多
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
    spy_c = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_c) == 0:
        return 'bull'
    return 'bull' if spy_c.iloc[-1] > valid.iloc[-1] else 'bear'


def get_best_safe_asset(safe_prices, date, lookback=63):
    """
    Pick best of {TLT, IEF, SHY, GLD} by 3-month momentum.
    Returns ticker with highest positive momentum, or 'SHY' as default.
    """
    moms = {}
    for ticker, prices in safe_prices.items():
        avail = prices.loc[:date].dropna()
        if len(avail) < lookback + 1:
            moms[ticker] = -999
        else:
            moms[ticker] = avail.iloc[-1] / avail.iloc[-lookback - 1] - 1
    best = max(moms.items(), key=lambda x: x[1])
    return best[0] if best[1] > -0.01 else 'SHY'  # Use SHY if all negative


# ============================================================
# v3b stock selection — IDENTICAL to proven champion
# ============================================================
def select_stocks_v3b(signals, sectors, date, prev_holdings):
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}, 'bull', 0.0
    idx = idx[-1]
    regime = get_regime(signals, date)

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
        cash_frac = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 2
        cash_frac = 0.20  # Bear: 20% defensive allocation (cash → will be replaced by safe asset)

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())

    if not selected:
        return {}, regime, cash_frac

    mom_share = 0.30
    iv = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    iv_total = sum(iv.values())
    iv_w = {t: v / iv_total for t, v in iv.items()}

    mom_min = min(df.loc[t, 'momentum'] for t in selected)
    shift = max(-mom_min + 0.01, 0)
    mw = {t: df.loc[t, 'momentum'] + shift for t in selected}
    mw_total = sum(mw.values())
    mw_w = {t: v / mw_total for t, v in mw.items()}

    invested = 1.0 - cash_frac
    weights = {t: ((1 - mom_share) * iv_w[t] + mom_share * mw_w[t]) * invested for t in selected}
    return weights, regime, cash_frac


# ============================================================
# v5c: Main selection with smart safe asset
# ============================================================
def select_v5c(signals, sectors, date, prev_holdings, safe_prices, dd, dd_params):
    """
    v3b stock selection + smart safe asset replacement:
    1. Bear mode 20% cash → best of TLT/IEF/SHY/GLD
    2. DD-responsive hedge → best of GLD/TLT
    All on top of each other (capped at 60% total safe assets).
    """
    result = select_stocks_v3b(signals, sectors, date, prev_holdings)
    weights, regime, cash_frac = result

    # Determine DD-responsive hedge
    gld_alloc = 0.0
    sorted_thresholds = sorted(dd_params.items(), key=lambda x: x[0])
    for threshold, alloc in sorted_thresholds:
        if dd < threshold:
            gld_alloc = alloc

    # Find best safe asset for bear cash replacement
    safe_ticker = get_best_safe_asset(safe_prices, date) if cash_frac > 0 or gld_alloc > 0 else 'SHY'

    # Cap total safe asset allocation
    total_safe = min(cash_frac + gld_alloc, 0.60)
    stock_frac = 1.0 - total_safe

    if total_safe <= 0 or not weights:
        return weights

    total_w = sum(weights.values())
    if total_w <= 0:
        return weights

    new_weights = {t: (w / total_w) * stock_frac for t, w in weights.items()}

    # Bear cash → best safe asset
    if cash_frac > 0:
        new_weights[safe_ticker] = new_weights.get(safe_ticker, 0) + cash_frac * (total_safe / (cash_frac + gld_alloc))

    # DD hedge → best safe asset (might be same or different; we consolidate)
    if gld_alloc > 0:
        # For DD hedge: prefer GLD during stock market crashes (negative correlation)
        gld_mom = -999
        tlt_mom = -999
        if 'GLD' in safe_prices:
            avail = safe_prices['GLD'].loc[:date].dropna()
            if len(avail) >= 64:
                gld_mom = avail.iloc[-1] / avail.iloc[-64] - 1
        if 'TLT' in safe_prices:
            avail = safe_prices['TLT'].loc[:date].dropna()
            if len(avail) >= 64:
                tlt_mom = avail.iloc[-1] / avail.iloc[-64] - 1
        dd_hedge_ticker = 'GLD' if gld_mom >= tlt_mom else 'TLT'
        new_weights[dd_hedge_ticker] = new_weights.get(dd_hedge_ticker, 0) + gld_alloc * (total_safe / (cash_frac + gld_alloc))

    return new_weights


def run_backtest(close_df, signals, sectors, safe_prices,
                 dd_params=None,
                 start='2015-01-01', end='2025-12-31',
                 cost_per_trade=0.0015):
    if dd_params is None:
        dd_params = {-0.08: 0.30, -0.15: 0.50}  # v4d defaults

    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index

    portfolio_values, portfolio_dates = [], []
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    peak_value = 1.0

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]

        dd = (current_value - peak_value) / peak_value if peak_value > 0 else 0
        new_weights = select_v5c(signals, sectors, date, prev_holdings, safe_prices, dd, dd_params)

        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = new_weights.copy()
        prev_holdings = set(k for k in new_weights.keys() if k not in safe_prices.keys())

        port_ret = 0.0
        for t, w in new_weights.items():
            if t in safe_prices:
                s = safe_prices[t].loc[date:next_date].dropna()
                if len(s) >= 2:
                    port_ret += (s.iloc[-1] / s.iloc[0] - 1) * w
            elif t in close_df.columns:
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


def run_v4d_baseline(close_df, signals, sectors, gld_prices,
                     dd_params=None,
                     start='2015-01-01', end='2025-12-31',
                     cost_per_trade=0.0015):
    """Run v4d exact baseline for fair comparison."""
    if dd_params is None:
        dd_params = {-0.08: 0.30, -0.15: 0.50}

    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    portfolio_values, portfolio_dates = [], []
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    peak_value = 1.0

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]
        dd = (current_value - peak_value) / peak_value if peak_value > 0 else 0

        result3 = select_stocks_v3b(signals, sectors, date, prev_holdings)
        weights, regime, cash_frac = result3

        # DD-responsive GLD (original v4d logic)
        gld_alloc = 0.0
        sorted_t = sorted(dd_params.items(), key=lambda x: x[0])
        for threshold, alloc in sorted_t:
            if dd < threshold:
                gld_alloc = alloc

        if gld_alloc > 0 and weights:
            total_w = sum(weights.values())
            stock_frac = 1.0 - gld_alloc
            weights = {t: (w / total_w) * stock_frac for t, w in weights.items()}
            weights['GLD'] = gld_alloc

        all_t = set(list(weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = weights.copy()
        prev_holdings = set(k for k in weights.keys() if k != 'GLD')

        port_ret = 0.0
        for t, w in weights.items():
            if t == 'GLD':
                s = gld_prices.loc[date:next_date].dropna()
                if len(s) >= 2:
                    port_ret += (s.iloc[-1] / s.iloc[0] - 1) * w
            elif t in close_df.columns:
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
    print("🐻 动量轮动 v5c — v4d 精准升级：Cash→智能安全资产 + 双避险选择")
    print("=" * 70)

    print("\n📊 Loading data...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)

    safe_prices = {
        'GLD': load_csv(CACHE / "GLD.csv")['Close'].dropna(),
        'TLT': load_csv(CACHE / "TLT.csv")['Close'].dropna(),
        'IEF': load_csv(CACHE / "IEF.csv")['Close'].dropna(),
        'SHY': load_csv(CACHE / "SHY.csv")['Close'].dropna(),
    }
    gld_prices = safe_prices['GLD']
    print(f"  Loaded {len(close_df.columns)} tickers")
    for t, p in safe_prices.items():
        print(f"  {t}: {len(p)} days")

    # DD params (same as v4d champion)
    dd_params = {-0.08: 0.30, -0.15: 0.50}

    # Run v4d baseline for fair comparison
    print("\n🔄 Running v4d baseline (2015-2025)...")
    eq_v4d, _ = run_v4d_baseline(close_df, signals, sectors, gld_prices, dd_params)
    eq_v4d_is, _ = run_v4d_baseline(close_df, signals, sectors, gld_prices, dd_params,
                                     start='2015-01-01', end='2020-12-31')
    eq_v4d_oos, _ = run_v4d_baseline(close_df, signals, sectors, gld_prices, dd_params,
                                      start='2021-01-01', end='2025-12-31')
    m_v4d = compute_metrics(eq_v4d, 'v4d')
    m_v4d_is = compute_metrics(eq_v4d_is)
    m_v4d_oos = compute_metrics(eq_v4d_oos)
    wf_v4d = m_v4d_oos['sharpe'] / m_v4d_is['sharpe'] if m_v4d_is['sharpe'] != 0 else 0
    comp_v4d = m_v4d['sharpe'] * 0.4 + m_v4d['calmar'] * 0.4 + m_v4d['cagr'] * 0.2
    print(f"  v4d: CAGR {m_v4d['cagr']:.1%}, Sharpe {m_v4d['sharpe']:.2f}, MaxDD {m_v4d['max_dd']:.1%}, "
          f"Calmar {m_v4d['calmar']:.2f}, Comp {comp_v4d:.3f}, WF {wf_v4d:.2f}")

    # Run v5c
    print("\n🔄 Running v5c full (2015-2025)...")
    eq_full, to = run_backtest(close_df, signals, sectors, safe_prices, dd_params)
    print(f"  Done. Avg turnover: {to:.2%}")

    print("🔄 Running v5c IS (2015-2020)...")
    eq_is, _ = run_backtest(close_df, signals, sectors, safe_prices, dd_params,
                             start='2015-01-01', end='2020-12-31')

    print("🔄 Running v5c OOS (2021-2025)...")
    eq_oos, _ = run_backtest(close_df, signals, sectors, safe_prices, dd_params,
                              start='2021-01-01', end='2025-12-31')

    m = compute_metrics(eq_full, 'v5c')
    m_is = compute_metrics(eq_is)
    m_oos = compute_metrics(eq_oos)
    wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
    comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2

    print("\n" + "=" * 80)
    print("📊 COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<20} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS':>6} {'OOS':>6} {'WF':>6} {'Comp':>8}")
    print("-" * 80)
    wf_flag_v4d = "✅" if wf_v4d >= 0.70 else "❌"
    wf_flag = "✅" if wf >= 0.70 else "❌"
    print(f"{'v4d (baseline)':<20} {m_v4d['cagr']:>6.1%} {m_v4d['max_dd']:>7.1%} "
          f"{m_v4d['sharpe']:>8.2f} {m_v4d['calmar']:>8.2f} "
          f"{m_v4d_is['sharpe']:>6.2f} {m_v4d_oos['sharpe']:>6.2f} "
          f"{wf_v4d:>5.2f}{wf_flag_v4d} {comp_v4d:>8.3f}")
    print(f"{'v5c (improved)':<20} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
          f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
          f"{m_is['sharpe']:>6.2f} {m_oos['sharpe']:>6.2f} "
          f"{wf:>5.2f}{wf_flag} {comp:>8.3f}")

    print(f"\n  Delta: CAGR {m['cagr']-m_v4d['cagr']:+.1%}, "
          f"MaxDD {m['max_dd']-m_v4d['max_dd']:+.1%}, "
          f"Sharpe {m['sharpe']-m_v4d['sharpe']:+.2f}, "
          f"Calmar {m['calmar']-m_v4d['calmar']:+.2f}, "
          f"Composite {comp-comp_v4d:+.3f}")

    if comp > 1.8 or m['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】 Composite > 1.8 or Sharpe > 2.0! 🚨🚨🚨")
    elif comp > comp_v4d:
        print(f"\n✅ v5c improves over v4d! Composite {comp:.3f} > {comp_v4d:.3f}")
    else:
        print(f"\n⚠️ v5c does not improve over v4d (Composite {comp:.3f} < {comp_v4d:.3f})")

    # Save results
    results_file = Path(__file__).parent / "momentum_v5c_results.json"
    results_data = {
        'v4d': {'cagr': float(m_v4d['cagr']), 'max_dd': float(m_v4d['max_dd']),
                'sharpe': float(m_v4d['sharpe']), 'calmar': float(m_v4d['calmar']),
                'wf': float(wf_v4d), 'composite': float(comp_v4d)},
        'v5c': {'cagr': float(m['cagr']), 'max_dd': float(m['max_dd']),
                'sharpe': float(m['sharpe']), 'calmar': float(m['calmar']),
                'wf': float(wf), 'composite': float(comp),
                'is_sharpe': float(m_is['sharpe']), 'oos_sharpe': float(m_oos['sharpe'])},
        'delta': {'cagr': float(m['cagr']-m_v4d['cagr']),
                  'sharpe': float(m['sharpe']-m_v4d['sharpe']),
                  'composite': float(comp-comp_v4d)},
    }
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

    return m, comp, wf


if __name__ == '__main__':
    main()
