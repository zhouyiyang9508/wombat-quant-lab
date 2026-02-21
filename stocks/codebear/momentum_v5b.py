#!/usr/bin/env python3
"""
动量轮动 v5b — Multi-Regime Adaptive + Concentrated Quality + TLT Smart Hedge
代码熊 🐻

核心创新 vs v4d:
1. 三档市场 Regime（强牛/弱牛/熊）based on SPY slope + vol
2. 更集中的持仓：每行业 top-2 股票（6-8 只），增加alpha
3. 动量时间权重自适应：根据市场状态调整权重
4. 智能对冲：牛市不对冲，弱牛轻对冲(TLT 15%)，熊市重对冲(GLD 40%)
5. 波动率调整：高波动期缩减仓位（VIX-proxy based）

市场 Regime 判断（三档）:
  - 强牛: SPY > SMA200 AND SPY_20d_vol < 0.18 AND SPY > SMA50
  - 弱牛: SPY > SMA200 但不满足强牛条件
  - 熊市: SPY < SMA200

各档策略:
  强牛: top-4行业 × top-2支股 = 8持仓, 0%对冲, 攻击型
  弱牛: top-4行业 × top-2支股 = 8持仓, 15% TLT, 平衡型
  熊市: top-3行业 × top-2支股 = 6持仓, 40% GLD, 防守型

动量公式:
  强牛: 30% 1m + 40% 3m + 30% 6m (短期信号更重要)
  弱牛: 20% 1m + 35% 3m + 30% 6m + 15% 12m
  熊市: 10% 1m + 25% 3m + 35% 6m + 30% 12m (长期信号更重要)

附加DD保护:
  DD > -10%: 额外+10% GLD
  DD > -18%: 额外+20% GLD (on top of regime-based allocation)
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
    """All signals with proper shift to avoid lookahead."""
    ret_1m  = close_df.shift(1) / close_df.shift(23) - 1
    ret_3m  = close_df.shift(1) / close_df.shift(64) - 1
    ret_6m  = close_df.shift(1) / close_df.shift(127) - 1
    ret_12m = close_df.shift(1) / close_df.shift(253) - 1

    log_ret = np.log(close_df / close_df.shift(1))
    vol_20d = log_ret.rolling(20).std() * np.sqrt(252)   # Short-term vol (VIX-proxy)
    vol_60d = log_ret.rolling(60).std() * np.sqrt(252)   # Longer-term vol

    spy_close = close_df['SPY'] if 'SPY' in close_df.columns else None
    spy_sma200 = spy_close.rolling(200).mean() if spy_close is not None else None
    spy_sma50  = spy_close.rolling(50).mean()  if spy_close is not None else None
    sma_50 = close_df.rolling(50).mean()

    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'vol_20d': vol_20d, 'vol_60d': vol_60d,
        'spy_sma200': spy_sma200, 'spy_sma50': spy_sma50,
        'spy_close': spy_close, 'sma_50': sma_50, 'close': close_df,
    }


def get_regime(signals, date):
    """Three-state regime: strong_bull / weak_bull / bear"""
    spy_c = signals['spy_close']
    spy_sma200 = signals['spy_sma200']
    spy_sma50 = signals['spy_sma50']
    spy_vol20 = signals['vol_20d']['SPY'] if 'SPY' in signals['vol_20d'].columns else None

    if spy_c is None:
        return 'strong_bull'

    spy_now = spy_c.loc[:date].dropna()
    sma200_now = spy_sma200.loc[:date].dropna()
    sma50_now = spy_sma50.loc[:date].dropna()

    if len(spy_now) == 0 or len(sma200_now) == 0:
        return 'strong_bull'

    spy_val = spy_now.iloc[-1]
    sma200_val = sma200_now.iloc[-1]

    if spy_val < sma200_val:
        return 'bear'

    # Above SMA200 → check for strong bull
    sma50_val = sma50_now.iloc[-1] if len(sma50_now) > 0 else sma200_val
    vol_now = 0.20
    if spy_vol20 is not None:
        vol_series = spy_vol20.loc[:date].dropna()
        if len(vol_series) > 0:
            vol_now = vol_series.iloc[-1]

    if spy_val > sma50_val and vol_now < 0.18:
        return 'strong_bull'
    else:
        return 'weak_bull'


def get_momentum_weights(regime):
    """Return (w1m, w3m, w6m, w12m) based on regime."""
    if regime == 'strong_bull':
        return (0.30, 0.40, 0.30, 0.00)
    elif regime == 'weak_bull':
        return (0.20, 0.35, 0.30, 0.15)
    else:  # bear
        return (0.10, 0.25, 0.35, 0.30)


def select_stocks_v5b(signals, sectors, date, prev_holdings, regime):
    """
    Concentrated quality momentum stock selection.
    Returns weights_dict.
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]

    w1, w3, w6, w12 = get_momentum_weights(regime)
    mom = (w1 * signals['ret_1m'].loc[idx] +
           w3 * signals['ret_3m'].loc[idx] +
           w6 * signals['ret_6m'].loc[idx] +
           w12 * signals['ret_12m'].loc[idx])

    df = pd.DataFrame({
        'momentum': mom,
        'ret_1m': signals['ret_1m'].loc[idx],
        'ret_3m': signals['ret_3m'].loc[idx],
        'ret_6m': signals['ret_6m'].loc[idx],
        'vol_60d': signals['vol_60d'].loc[idx],
        'price': close.loc[idx],
        'sma50': signals['sma_50'].loc[idx],
    }).dropna(subset=['momentum', 'sma50', 'vol_60d'])

    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[df['vol_60d'] < 0.65]
    df = df[df['price'] > df['sma50']]  # SMA50 trend filter

    # Positive 6m momentum required
    df = df[df['ret_6m'] > 0]

    # In bear regime: also require positive 3m momentum
    if regime == 'bear':
        df = df[df['ret_3m'] > 0]

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))

    # Continuity bonus
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.025

    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)

    # More concentrated: top 2 per sector
    if regime == 'strong_bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 2  # Top 2 per sector = 8 stocks
    elif regime == 'weak_bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 2  # 8 stocks
    else:  # bear
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 2  # 6 stocks

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())

    if not selected:
        return {}

    # Pure inverse-volatility weighting (more stable)
    iv = {t: 1.0 / max(df.loc[t, 'vol_60d'], 0.10) for t in selected}
    iv_total = sum(iv.values())
    weights = {t: iv[t] / iv_total for t in selected}

    return weights


def compute_regime_hedge(regime, dd):
    """
    Determine hedge allocation based on regime + DD.
    Returns (hedge_ticker, fraction)
    """
    # Base allocation from regime
    if regime == 'strong_bull':
        base_ticker, base_frac = None, 0.0
    elif regime == 'weak_bull':
        base_ticker, base_frac = 'TLT', 0.15
    else:  # bear
        base_ticker, base_frac = 'GLD', 0.40

    # DD overlay (additional GLD)
    dd_frac = 0.0
    if dd < -0.18:
        dd_frac = 0.20
    elif dd < -0.10:
        dd_frac = 0.10

    total_frac = min(base_frac + dd_frac, 0.60)  # Cap at 60%

    # If base is TLT but DD triggered, mix TLT + GLD
    if base_ticker == 'TLT' and dd_frac > 0:
        return [('TLT', base_frac), ('GLD', dd_frac)]
    elif base_ticker is not None:
        return [(base_ticker, total_frac)]
    elif dd_frac > 0:
        return [('GLD', dd_frac)]
    else:
        return []


def apply_hedges(weights, hedge_list):
    """Apply hedge allocations to portfolio."""
    if not hedge_list or not weights:
        return weights
    total_hedge = sum(f for _, f in hedge_list)
    if total_hedge <= 0:
        return weights
    total_hedge = min(total_hedge, 0.60)
    stock_frac = 1.0 - total_hedge
    total_w = sum(weights.values())
    if total_w <= 0:
        return weights
    new_weights = {t: (w / total_w) * stock_frac for t, w in weights.items()}
    for ticker, frac in hedge_list:
        new_weights[ticker] = new_weights.get(ticker, 0) + (frac / total_hedge) * total_hedge
    return new_weights


def run_backtest(close_df, signals, sectors, gld_prices, tlt_prices,
                 start='2015-01-01', end='2025-12-31', cost_per_trade=0.0015):
    """Run v5b backtest."""
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
        regime = get_regime(signals, date)

        # Stock selection
        weights = select_stocks_v5b(signals, sectors, date, prev_holdings, regime)

        # Apply hedges
        hedge_list = compute_regime_hedge(regime, dd)
        weights = apply_hedges(weights, hedge_list)

        # Turnover
        all_t = set(list(weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = weights.copy()
        prev_holdings = set(k for k in weights.keys() if k not in ('GLD', 'TLT', 'SHY'))

        # Returns
        port_ret = 0.0
        hedge_prices = {'GLD': gld_prices, 'TLT': tlt_prices}

        for t, w in weights.items():
            if t in hedge_prices:
                s = hedge_prices[t].loc[date:next_date].dropna()
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
    print("🐻 动量轮动 v5b — Multi-Regime + Concentrated Quality + Smart Hedge")
    print("=" * 70)

    print("\n📊 Loading data...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)
    gld_prices = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    tlt_prices = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    print(f"  Loaded {len(close_df.columns)} tickers")

    # Run backtests
    print("\n🔄 Running v5b full (2015-2025)...")
    eq_full, to = run_backtest(close_df, signals, sectors, gld_prices, tlt_prices)
    print(f"  Done. Avg turnover: {to:.2%}")

    print("🔄 Running v5b IS (2015-2020)...")
    eq_is, _ = run_backtest(close_df, signals, sectors, gld_prices, tlt_prices,
                             start='2015-01-01', end='2020-12-31')

    print("🔄 Running v5b OOS (2021-2025)...")
    eq_oos, _ = run_backtest(close_df, signals, sectors, gld_prices, tlt_prices,
                              start='2021-01-01', end='2025-12-31')

    m = compute_metrics(eq_full, 'v5b')
    m_is = compute_metrics(eq_is, 'v5b IS')
    m_oos = compute_metrics(eq_oos, 'v5b OOS')
    wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
    comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2

    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"  CAGR:      {m['cagr']:.1%}")
    print(f"  MaxDD:     {m['max_dd']:.1%}")
    print(f"  Sharpe:    {m['sharpe']:.2f}")
    print(f"  Calmar:    {m['calmar']:.2f}")
    print(f"  IS Sharpe: {m_is['sharpe']:.2f}")
    print(f"  OOS Sharpe:{m_oos['sharpe']:.2f}")
    print(f"  WF ratio:  {wf:.2f} {'✅' if wf >= 0.70 else '❌'}")
    print(f"  Composite: {comp:.3f}")

    # vs v4d benchmark
    print("\n📊 vs v4d (champion: Composite 1.356, Sharpe 1.45, MaxDD -15.0%, CAGR 27.1%)")
    print(f"  CAGR:      27.1% → {m['cagr']:.1%}  ({m['cagr']-0.271:+.1%})")
    print(f"  MaxDD:     -15.0% → {m['max_dd']:.1%}  ({m['max_dd']-(-0.150):+.1%})")
    print(f"  Sharpe:    1.45 → {m['sharpe']:.2f}  ({m['sharpe']-1.45:+.2f})")
    print(f"  Calmar:    1.81 → {m['calmar']:.2f}  ({m['calmar']-1.81:+.2f})")
    print(f"  Composite: 1.356 → {comp:.3f}  ({comp-1.356:+.3f})")

    if comp > 1.8 or m['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】 Composite > 1.8 or Sharpe > 2.0! 🚨🚨🚨")
    elif comp > 1.356:
        print(f"\n✅ Improvement over v4d! Composite {comp:.3f} > 1.356")
    else:
        print(f"\n⚠️ No improvement over v4d (Composite {comp:.3f} < 1.356)")

    # Save results
    results_file = Path(__file__).parent / "momentum_v5b_results.json"
    results_data = {
        'strategy': 'v5b Multi-Regime + Concentrated Quality + Smart Hedge',
        'full': {k: float(v) for k, v in m.items() if k != 'name'},
        'is': {k: float(v) for k, v in m_is.items() if k != 'name'},
        'oos': {k: float(v) for k, v in m_oos.items() if k != 'name'},
        'wf': float(wf),
        'composite': float(comp),
        'turnover': float(to),
    }
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

    return m, comp, wf


if __name__ == '__main__':
    main()
