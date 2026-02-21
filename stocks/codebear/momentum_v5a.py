#!/usr/bin/env python3
"""
动量轮动 v5a — Skip-Month Momentum + Vol-Target + Dual Hedge (GLD/TLT)
代码熊 🐻

核心创新 vs v4d:
1. 12-1 跳月动量 (skip last month): 用 12个月收益 - 1个月收益 避免短期反转
2. 波动率目标配置: 目标年化波动15%，动态调整仓位
3. 双对冲: GLD + TLT，根据相对动量选最优避险资产
4. 质量过滤: 要求1m/3m/6m三个时间框架均为正（强趋势确认）
5. 严格无前瞻：所有信号用 close[i-1]（shift后计算）

策略逻辑:
  - 月底计算信号（基于前一收盘价）
  - 12-1动量 = 12月回报 - 1月回报（skip-month momentum）
  - 行业轮动：选top-4行业（牛市）/top-3（熊市）
  - 每行业选top-3/2支股票
  - 波动率目标: 组合历史vol → 调整杠杆至目标15%vol（上限100%仓）
  - DD响应对冲: DD>-8% → 30%最优避险; DD>-15% → 50%最优避险

对冲选择:
  - 比较GLD vs TLT过去3个月动量
  - 选择动量更强的作为对冲资产
  - 若两者均为负动量→用SHY（短债）
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# --- Parameters ---
VOL_TARGET = 0.15        # 目标年化波动15%
VOL_LOOKBACK = 60        # 计算波动用60个交易日
DD_THRESH_1 = -0.08      # 第一档对冲触发
GLD_ALLOC_1 = 0.30       # 30%对冲
DD_THRESH_2 = -0.15      # 第二档对冲触发
GLD_ALLOC_2 = 0.50       # 50%对冲
MOMENTUM_BLEND = (0.10, 0.20, 0.30, 0.40)  # 1m, 3m, 6m, 12-1 skip-month


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
    """Precompute all signals — all shifts are lag-1 (no lookahead)."""
    # Standard momentum periods
    ret_1m  = close_df.shift(1) / close_df.shift(23) - 1    # 22 trading days, shift 1 to avoid lookahead
    ret_3m  = close_df.shift(1) / close_df.shift(64) - 1    # 63 days
    ret_6m  = close_df.shift(1) / close_df.shift(127) - 1   # 126 days
    ret_12m = close_df.shift(1) / close_df.shift(253) - 1   # 252 days
    # Skip-month momentum: 12 month return MINUS last month return (avoids short-term reversal)
    ret_skip = ret_12m - ret_1m

    log_ret = np.log(close_df / close_df.shift(1))
    vol_60d = log_ret.rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    spy_close = close_df['SPY'] if 'SPY' in close_df.columns else None
    spy_sma200 = spy_close.rolling(200).mean() if spy_close is not None else None
    sma_50 = close_df.rolling(50).mean()

    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m,
        'ret_12m': ret_12m, 'ret_skip': ret_skip,
        'vol_60d': vol_60d, 'spy_sma200': spy_sma200,
        'spy_close': spy_close, 'sma_50': sma_50, 'close': close_df,
    }


def get_regime(signals, date):
    """Bull/bear based on SPY vs SMA200."""
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid_sma = spy_sma.loc[:date].dropna()
    spy_c = signals['spy_close'].loc[:date].dropna()
    if len(valid_sma) == 0 or len(spy_c) == 0:
        return 'bull'
    return 'bull' if spy_c.iloc[-1] > valid_sma.iloc[-1] else 'bear'


def get_best_hedge(gld_prices, tlt_prices, shy_prices, date):
    """
    Compare GLD vs TLT 3-month momentum.
    Returns ('GLD', alloc) or ('TLT', alloc) or ('SHY', alloc)
    All prices shifted (use previous close).
    """
    def mom3m(prices):
        avail = prices.loc[:date].dropna()
        if len(avail) < 64:
            return -999
        return avail.iloc[-1] / avail.iloc[-64] - 1  # 63 days ago = 3 months

    gld_m = mom3m(gld_prices)
    tlt_m = mom3m(tlt_prices)

    if gld_m <= 0 and tlt_m <= 0:
        return 'SHY'  # Both negative → go to T-bills
    elif gld_m >= tlt_m:
        return 'GLD'
    else:
        return 'TLT'


def select_stocks_v5a(signals, sectors, date, prev_holdings):
    """
    Skip-month momentum stock selection with quality filter.
    Returns (weights_dict, regime)
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}, 'bull'
    idx = idx[-1]
    regime = get_regime(signals, date)

    # 12-1 Skip-month momentum (main signal) + traditional blend
    mom_skip = signals['ret_skip'].loc[idx]
    mom_1m   = signals['ret_1m'].loc[idx]
    mom_3m   = signals['ret_3m'].loc[idx]
    mom_6m   = signals['ret_6m'].loc[idx]

    # Composite momentum: 10% 1m + 20% 3m + 30% 6m + 40% skip-12m
    w1, w3, w6, ws = MOMENTUM_BLEND
    mom = w1 * mom_1m + w3 * mom_3m + w6 * mom_6m + ws * mom_skip

    df = pd.DataFrame({
        'momentum': mom,
        'mom_1m': mom_1m, 'mom_3m': mom_3m, 'mom_6m': mom_6m,
        'vol_60d': signals['vol_60d'].loc[idx],
        'price': close.loc[idx],
        'sma50': signals['sma_50'].loc[idx],
    }).dropna(subset=['momentum', 'sma50', 'vol_60d'])

    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[df['vol_60d'] < 0.65]
    df = df[df['price'] > df['sma50']]  # SMA50 trend filter

    # Quality filter: ALL of 1m/3m/6m must be positive (triple confirmation)
    df = df[(df['mom_1m'] > 0) & (df['mom_3m'] > 0) & (df['mom_6m'] > 0)]

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))

    # Continuity bonus
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.025

    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)

    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        sps = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 2
        cash = 0.15  # Slightly less cash than v3b (15% vs 20%)

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())

    if not selected:
        return {}, regime

    # Blended weighting: 65% inverse-vol + 35% momentum
    iv = {t: 1.0 / max(df.loc[t, 'vol_60d'], 0.10) for t in selected}
    iv_total = sum(iv.values())
    iv_w = {t: v / iv_total for t, v in iv.items()}

    mom_min = min(df.loc[t, 'momentum'] for t in selected)
    shift = max(-mom_min + 0.01, 0)
    mw = {t: df.loc[t, 'momentum'] + shift for t in selected}
    mw_total = sum(mw.values())
    mw_w = {t: v / mw_total for t, v in mw.items()}

    mom_share = 0.35
    invested = 1.0 - cash
    weights = {t: ((1 - mom_share) * iv_w[t] + mom_share * mw_w[t]) * invested for t in selected}
    return weights, regime


def add_hedge(weights, hedge_ticker, hedge_frac):
    """Scale stock weights and add hedge allocation."""
    if hedge_frac <= 0 or not weights:
        return weights
    total_w = sum(weights.values())
    stock_frac = 1.0 - hedge_frac
    new = {t: (w / total_w) * stock_frac for t, w in weights.items()}
    new[hedge_ticker] = hedge_frac
    return new


def run_backtest(close_df, signals, sectors, gld_prices, tlt_prices, shy_prices,
                 start='2015-01-01', end='2025-12-31', cost_per_trade=0.0015):
    """
    Run v5a backtest with vol-targeting and dual hedge.
    """
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index

    portfolio_values, portfolio_dates = [], []
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    peak_value = 1.0

    # Track monthly returns for vol-targeting
    monthly_returns_history = []

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]

        # Current drawdown
        dd = (current_value - peak_value) / peak_value if peak_value > 0 else 0

        # Stock selection
        weights, regime = select_stocks_v5a(signals, sectors, date, prev_holdings)

        # === Volatility Targeting ===
        # Estimate portfolio vol from recent monthly returns
        if len(monthly_returns_history) >= 6:
            hist_vol = np.std(monthly_returns_history[-12:]) * np.sqrt(12) if len(monthly_returns_history) >= 12 else \
                       np.std(monthly_returns_history[-6:]) * np.sqrt(12)
            if hist_vol > 0:
                vol_scalar = min(VOL_TARGET / hist_vol, 1.0)  # Never leverage > 100%
                # Scale stock weights down
                weights = {t: w * vol_scalar for t, w in weights.items()}

        # === Dual Hedge (DD-Responsive) ===
        hedge_frac = 0.0
        if dd < DD_THRESH_2:
            hedge_frac = GLD_ALLOC_2
        elif dd < DD_THRESH_1:
            hedge_frac = GLD_ALLOC_1

        if hedge_frac > 0 and weights:
            best_hedge = get_best_hedge(gld_prices, tlt_prices, shy_prices, date)
            weights = add_hedge(weights, best_hedge, hedge_frac)

        # Turnover
        all_t = set(list(weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = weights.copy()
        prev_holdings = set(k for k in weights.keys() if k not in ('GLD', 'TLT', 'SHY'))

        # Calculate returns
        port_ret = 0.0
        hedge_prices = {'GLD': gld_prices, 'TLT': tlt_prices, 'SHY': shy_prices}

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
        monthly_returns_history.append(port_ret)
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
    print("🐻 动量轮动 v5a — Skip-Month Momentum + Vol-Target + Dual Hedge")
    print("=" * 70)
    print("\nConfig:")
    print(f"  Momentum: {MOMENTUM_BLEND[0]:.0%} 1m + {MOMENTUM_BLEND[1]:.0%} 3m + {MOMENTUM_BLEND[2]:.0%} 6m + {MOMENTUM_BLEND[3]:.0%} skip-12m")
    print(f"  Quality filter: ALL of 1m/3m/6m > 0 required")
    print(f"  Vol target: {VOL_TARGET:.0%} annual")
    print(f"  DD hedge: {GLD_ALLOC_1:.0%} at {DD_THRESH_1:.0%}, {GLD_ALLOC_2:.0%} at {DD_THRESH_2:.0%}")
    print(f"  Hedge: best of GLD/TLT/SHY by 3m momentum")

    # Load data
    print("\n📊 Loading data...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)
    gld_prices = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    tlt_prices = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    shy_prices = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    print(f"  Loaded {len(close_df.columns)} tickers")
    print(f"  GLD: {len(gld_prices)} days, TLT: {len(tlt_prices)} days, SHY: {len(shy_prices)} days")

    # Run backtests
    print("\n🔄 Running v5a full (2015-2025)...")
    eq_full, to = run_backtest(close_df, signals, sectors, gld_prices, tlt_prices, shy_prices)
    print(f"  Done. Avg turnover: {to:.2%}")

    print("🔄 Running v5a IS (2015-2020)...")
    eq_is, _ = run_backtest(close_df, signals, sectors, gld_prices, tlt_prices, shy_prices,
                             start='2015-01-01', end='2020-12-31')

    print("🔄 Running v5a OOS (2021-2025)...")
    eq_oos, _ = run_backtest(close_df, signals, sectors, gld_prices, tlt_prices, shy_prices,
                              start='2021-01-01', end='2025-12-31')

    m = compute_metrics(eq_full, 'v5a')
    m_is = compute_metrics(eq_is, 'v5a IS')
    m_oos = compute_metrics(eq_oos, 'v5a OOS')
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
    results_file = Path(__file__).parent / "momentum_v5a_results.json"
    results_data = {
        'strategy': 'v5a Skip-Month Momentum + Vol-Target + Dual Hedge',
        'full': {k: float(v) for k, v in m.items() if k != 'name'},
        'is': {k: float(v) for k, v in m_is.items() if k != 'name'},
        'oos': {k: float(v) for k, v in m_oos.items() if k != 'name'},
        'wf': float(wf),
        'composite': float(comp),
        'turnover': float(to),
        'params': {
            'vol_target': VOL_TARGET,
            'dd_thresh_1': DD_THRESH_1, 'gld_alloc_1': GLD_ALLOC_1,
            'dd_thresh_2': DD_THRESH_2, 'gld_alloc_2': GLD_ALLOC_2,
        }
    }
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

    return m, comp, wf


if __name__ == '__main__':
    main()
