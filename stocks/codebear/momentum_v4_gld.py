#!/usr/bin/env python3
"""
动量轮动 v4 — Stock v3b + GLD 对冲变种
代码熊 🐻

基于 v3b (Composite 1.173) 添加 GLD 对冲，目标：
- Composite > 1.20
- MaxDD < -15%
- WF ≥ 0.70

变种:
  v4a: 熊市 GLD 对冲 (SPY < SMA200*0.95 → 50% GLD)
  v4c: DualMom (Stock vs GLD 6月动量对比)
  v4d: DD Responsive (回撤响应式 GLD 配置)
  v4e: Soft Bear + GLD (熊市20%清仓→GLD)
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


def load_gld():
    """Load GLD price data"""
    f = CACHE / "GLD.csv"
    df = load_csv(f)
    return df['Close'].dropna()


def load_spy():
    """Load SPY price data"""
    f = CACHE / "SPY.csv"
    df = load_csv(f)
    return df['Close'].dropna()


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
    """Standard regime: SPY vs SMA200"""
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid = spy_sma.loc[:date].dropna()
    spy_close = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_close) == 0:
        return 'bull'
    return 'bull' if spy_close.iloc[-1] > valid.iloc[-1] else 'bear'


def get_deep_bear(signals, date, threshold=0.95):
    """Deep bear: SPY < SMA200 * threshold"""
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return False
    valid = spy_sma.loc[:date].dropna()
    spy_close = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_close) == 0:
        return False
    return spy_close.iloc[-1] < valid.iloc[-1] * threshold


def select_stocks_v3b(signals, sectors, date, prev_holdings):
    """Original v3b stock selection — returns (weights_dict, regime)"""
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}, 'bull'
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


# ============================================================
# v4a: 熊市 GLD 对冲 (Deep Bear → 50% GLD)
# ============================================================
def select_v4a(signals, sectors, date, prev_holdings):
    """
    Deep bear (SPY < SMA200*0.95): 50% top-5 stocks + 50% GLD
    Normal bear (SPY < SMA200): 标准v3b bear模式
    Bull: 标准v3b
    """
    weights, regime = select_stocks_v3b(signals, sectors, date, prev_holdings)
    deep_bear = get_deep_bear(signals, date, 0.95)

    if deep_bear and weights:
        # Keep top 5 stocks by weight, allocate 50% to stocks, 50% to GLD
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        total_w = sum(w for _, w in sorted_w)
        stock_alloc = 0.50
        gld_alloc = 0.50
        new_weights = {t: (w / total_w) * stock_alloc for t, w in sorted_w}
        new_weights['GLD'] = gld_alloc
        return new_weights
    return weights


# ============================================================
# v4c: DualMom (Stock portfolio vs GLD 动量对比)
# ============================================================
def select_v4c(signals, sectors, date, prev_holdings, gld_prices, spy_prices):
    """
    Compare 6-month momentum of stock portfolio vs GLD.
    If GLD momentum > stock momentum → 50% stock + 50% GLD
    Otherwise → 100% stock v3b
    """
    weights, regime = select_stocks_v3b(signals, sectors, date, prev_holdings)

    # Get GLD 6-month return
    gld_available = gld_prices.loc[:date].dropna()
    if len(gld_available) < 126:
        return weights

    gld_ret_6m = gld_available.iloc[-1] / gld_available.iloc[-126] - 1

    # Get SPY 6-month return (proxy for stock momentum)
    spy_available = spy_prices.loc[:date].dropna()
    if len(spy_available) < 126:
        return weights

    spy_ret_6m = spy_available.iloc[-1] / spy_available.iloc[-126] - 1

    if gld_ret_6m > spy_ret_6m and weights:
        # GLD momentum stronger → shift 50% to GLD
        stock_alloc = 0.50
        gld_alloc = 0.50
        total_w = sum(weights.values())
        if total_w > 0:
            new_weights = {t: (w / total_w) * stock_alloc for t, w in weights.items()}
            new_weights['GLD'] = gld_alloc
            return new_weights
    return weights


# ============================================================
# v4d: DD Responsive (回撤响应式 GLD)
# ============================================================
class DDResponsiveTracker:
    def __init__(self):
        self.peak_value = 1.0
        self.current_value = 1.0

    def update(self, monthly_return):
        self.current_value *= (1 + monthly_return)
        if self.current_value > self.peak_value:
            self.peak_value = self.current_value

    def get_drawdown(self):
        if self.peak_value <= 0:
            return 0
        return (self.current_value - self.peak_value) / self.peak_value


def select_v4d(signals, sectors, date, prev_holdings, dd_tracker):
    """
    Normal: 100% stock v3b
    DD > -10%: 30% GLD + 70% stock
    DD > -15%: 50% GLD + 50% stock
    """
    weights, regime = select_stocks_v3b(signals, sectors, date, prev_holdings)
    dd = dd_tracker.get_drawdown()

    gld_alloc = 0.0
    if dd < -0.15:
        gld_alloc = 0.50
    elif dd < -0.10:
        gld_alloc = 0.30

    if gld_alloc > 0 and weights:
        stock_alloc = 1.0 - gld_alloc
        total_w = sum(weights.values())
        if total_w > 0:
            new_weights = {t: (w / total_w) * stock_alloc for t, w in weights.items()}
            new_weights['GLD'] = gld_alloc
            return new_weights
    return weights


# ============================================================
# v4e: Soft Bear + GLD (熊市20%清仓→GLD而非现金)
# ============================================================
def select_v4e(signals, sectors, date, prev_holdings):
    """
    Bull: 100% v3b
    Bear: v3b's 20% cash → 20% GLD instead
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
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
        cash = 0.0
        gld_alloc = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        sps = 2
        cash = 0.0  # No cash, use GLD instead
        gld_alloc = 0.20

    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        selected.extend(sec_df.index[:sps].tolist())

    if not selected:
        return {}

    mom_share = 0.30
    iv = {t: 1.0 / max(df.loc[t, 'vol_30d'], 0.10) for t in selected}
    iv_total = sum(iv.values())
    iv_w = {t: v / iv_total for t, v in iv.items()}

    mom_min = min(df.loc[t, 'momentum'] for t in selected)
    shift = max(-mom_min + 0.01, 0)
    mw = {t: df.loc[t, 'momentum'] + shift for t in selected}
    mw_total = sum(mw.values())
    mw_w = {t: v / mw_total for t, v in mw.items()}

    invested = 1.0 - gld_alloc
    weights = {t: ((1 - mom_share) * iv_w[t] + mom_share * mw_w[t]) * invested for t in selected}
    if gld_alloc > 0:
        weights['GLD'] = gld_alloc
    return weights


# ============================================================
# v4a2: 更柔和的熊市GLD (SPY < SMA200 → 30% GLD)
# ============================================================
def select_v4a2(signals, sectors, date, prev_holdings):
    """
    Bear (SPY < SMA200): 30% GLD + 70% top stocks (no cash)
    Deep Bear (SPY < SMA200*0.95): 50% GLD + 50% top-5 stocks
    Bull: 100% v3b
    """
    weights, regime = select_stocks_v3b(signals, sectors, date, prev_holdings)
    deep_bear = get_deep_bear(signals, date, 0.95)

    if not weights:
        return weights

    if deep_bear:
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        total_w = sum(w for _, w in sorted_w)
        stock_alloc = 0.50
        new_weights = {t: (w / total_w) * stock_alloc for t, w in sorted_w}
        new_weights['GLD'] = 0.50
        return new_weights
    elif regime == 'bear':
        total_w = sum(weights.values())
        stock_alloc = 0.70
        new_weights = {t: (w / total_w) * stock_alloc for t, w in weights.items()}
        new_weights['GLD'] = 0.30
        return new_weights

    return weights


# ============================================================
# Backtest Engine (handles GLD in portfolio)
# ============================================================
def run_backtest_v4(close_df, signals, sectors, gld_prices, spy_prices,
                    variant='v4a', start='2015-01-01', end='2025-12-31',
                    cost_per_trade=0.0015):
    """
    Generic backtest for v4 variants. Handles GLD as a special ticker.
    """
    close_range = close_df.loc[start:end].dropna(how='all')
    gld_range = gld_prices.loc[start:end].dropna()
    month_ends = close_range.resample('ME').last().index
    portfolio_values, portfolio_dates = [], []
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0
    dd_tracker = DDResponsiveTracker()

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]

        # Select stocks based on variant
        if variant == 'v4a':
            new_weights = select_v4a(signals, sectors, date, prev_holdings)
        elif variant == 'v4a2':
            new_weights = select_v4a2(signals, sectors, date, prev_holdings)
        elif variant == 'v4c':
            new_weights = select_v4c(signals, sectors, date, prev_holdings, gld_prices, spy_prices)
        elif variant == 'v4d':
            new_weights = select_v4d(signals, sectors, date, prev_holdings, dd_tracker)
        elif variant == 'v4e':
            new_weights = select_v4e(signals, sectors, date, prev_holdings)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Turnover
        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = new_weights.copy()
        prev_holdings = set(k for k in new_weights.keys() if k != 'GLD')

        # Calculate returns
        port_ret = 0.0
        for t, w in new_weights.items():
            if t == 'GLD':
                gld_slice = gld_range.loc[date:next_date].dropna()
                if len(gld_slice) >= 2:
                    port_ret += (gld_slice.iloc[-1] / gld_slice.iloc[0] - 1) * w
            else:
                stock_slice = close_df[t].loc[date:next_date].dropna() if t in close_df.columns else pd.Series(dtype=float)
                if len(stock_slice) >= 2:
                    port_ret += (stock_slice.iloc[-1] / stock_slice.iloc[0] - 1) * w

        port_ret -= turnover * cost_per_trade * 2
        current_value *= (1 + port_ret)
        dd_tracker.update(port_ret)
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


def run_v3b_backtest(close_df, signals, sectors, start='2015-01-01', end='2025-12-31',
                     cost_per_trade=0.0015):
    """Run original v3b for comparison"""
    close_range = close_df.loc[start:end].dropna(how='all')
    month_ends = close_range.resample('ME').last().index
    portfolio_values, portfolio_dates = [], []
    turnover_list = []
    prev_weights, prev_holdings = {}, set()
    current_value = 1.0

    for i in range(len(month_ends) - 1):
        date, next_date = month_ends[i], month_ends[i + 1]
        new_weights, _ = select_stocks_v3b(signals, sectors, date, prev_holdings)

        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)
        prev_weights = new_weights.copy()
        prev_holdings = set(new_weights.keys())

        port_ret = sum(
            (close_df[t].loc[date:next_date].dropna().iloc[-1] /
             close_df[t].loc[date:next_date].dropna().iloc[0] - 1) * w
            for t, w in new_weights.items()
            if t in close_df.columns and len(close_df[t].loc[date:next_date].dropna()) >= 2
        )
        port_ret -= turnover * cost_per_trade * 2
        current_value *= (1 + port_ret)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)

    equity = pd.Series(portfolio_values, index=pd.DatetimeIndex(portfolio_dates))
    return equity, np.mean(turnover_list) if turnover_list else 0


def analyze_2022_drawdown(equity, name):
    """Analyze performance during 2022 bear market"""
    eq_2022 = equity.loc['2022-01-01':'2022-12-31']
    if len(eq_2022) < 3:
        return {}
    # Include some before for drawdown context
    eq_extended = equity.loc['2021-11-01':'2023-06-30']
    if len(eq_extended) < 3:
        return {}
    dd = (eq_extended - eq_extended.cummax()) / eq_extended.cummax()
    max_dd_2022 = dd.loc['2022-01-01':'2022-12-31'].min()

    # 2022 return
    eq_2022_only = equity.loc['2022-01-01':'2022-12-31']
    if len(eq_2022_only) >= 2:
        ret_2022 = eq_2022_only.iloc[-1] / eq_2022_only.iloc[0] - 1
    else:
        ret_2022 = 0

    # Recovery: find when equity exceeds pre-2022 peak
    pre_peak = equity.loc[:'2022-01-01'].max()
    recovery_dates = equity.loc['2022-01-01':][equity.loc['2022-01-01':] >= pre_peak]
    recovery_date = recovery_dates.index[0] if len(recovery_dates) > 0 else None
    recovery_months = (recovery_date - pd.Timestamp('2022-01-01')).days / 30.0 if recovery_date else None

    return {
        'name': name,
        'max_dd_2022': max_dd_2022,
        'ret_2022': ret_2022,
        'recovery_months': recovery_months,
    }


def main():
    print("=" * 70)
    print("🐻 动量轮动 v4 — Stock v3b + GLD 对冲变种")
    print("=" * 70)

    # Load data
    print("\n📊 Loading data...")
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_all_data(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    signals = precompute_signals(close_df)
    gld_prices = load_gld()
    spy_prices = load_spy()
    print(f"  Loaded {len(close_df.columns)} tickers, GLD {len(gld_prices)} days, SPY {len(spy_prices)} days")

    # Run all variants
    variants = ['v3b', 'v4a', 'v4a2', 'v4c', 'v4d', 'v4e']
    variant_names = {
        'v3b': 'v3b (baseline)',
        'v4a': 'v4a DeepBear GLD',
        'v4a2': 'v4a2 Bear+GLD',
        'v4c': 'v4c DualMom',
        'v4d': 'v4d DD Responsive',
        'v4e': 'v4e SoftBear+GLD',
    }

    results = {}
    for var in variants:
        print(f"\n🔄 Running {variant_names[var]}...")

        if var == 'v3b':
            eq_full, to = run_v3b_backtest(close_df, signals, sectors)
            eq_is, _ = run_v3b_backtest(close_df, signals, sectors, '2015-01-01', '2020-12-31')
            eq_oos, _ = run_v3b_backtest(close_df, signals, sectors, '2021-01-01', '2025-12-31')
        else:
            eq_full, to = run_backtest_v4(close_df, signals, sectors, gld_prices, spy_prices,
                                           variant=var)
            eq_is, _ = run_backtest_v4(close_df, signals, sectors, gld_prices, spy_prices,
                                        variant=var, start='2015-01-01', end='2020-12-31')
            eq_oos, _ = run_backtest_v4(close_df, signals, sectors, gld_prices, spy_prices,
                                         variant=var, start='2021-01-01', end='2025-12-31')

        m = compute_metrics(eq_full, var)
        m_is = compute_metrics(eq_is, f"{var} IS")
        m_oos = compute_metrics(eq_oos, f"{var} OOS")
        wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] != 0 else 0
        comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2

        dd_2022 = analyze_2022_drawdown(eq_full, var)

        results[var] = {
            'metrics': m, 'is': m_is, 'oos': m_oos,
            'wf': wf, 'composite': comp, 'turnover': to,
            'equity': eq_full, 'dd_2022': dd_2022,
        }

        print(f"  CAGR: {m['cagr']:.1%}  Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_dd']:.1%}  "
              f"Calmar: {m['calmar']:.2f}  Composite: {comp:.3f}")
        print(f"  IS: {m_is['sharpe']:.2f}  OOS: {m_oos['sharpe']:.2f}  WF: {wf:.2f} "
              f"{'✅' if wf >= 0.70 else '❌'}")
        if dd_2022:
            print(f"  2022: MaxDD {dd_2022.get('max_dd_2022', 0):.1%}  "
                  f"Return {dd_2022.get('ret_2022', 0):.1%}  "
                  f"Recovery {dd_2022.get('recovery_months', '?')} months")

    # ============================================================
    # Summary table
    # ============================================================
    print("\n" + "=" * 100)
    print("📊 COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Version':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8} {'DD22':>7}")
    print("-" * 100)

    best_comp = -999
    best_var = None
    for var in variants:
        r = results[var]
        m = r['metrics']
        wf_str = f"{r['wf']:.2f}" + (" ✅" if r['wf'] >= 0.70 else " ❌")
        dd22 = r['dd_2022'].get('max_dd_2022', 0) if r['dd_2022'] else 0
        print(f"{variant_names[var]:<22} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f} "
              f"{r['is']['sharpe']:>7.2f} {r['oos']['sharpe']:>7.2f} "
              f"{wf_str:>8} {r['composite']:>8.3f} {dd22:>6.1%}")

        # Find best (must pass WF)
        if var != 'v3b' and r['wf'] >= 0.70 and r['composite'] > best_comp:
            best_comp = r['composite']
            best_var = var

    print("-" * 100)

    # ============================================================
    # Best variant analysis
    # ============================================================
    if best_var:
        print(f"\n🏆 BEST VARIANT: {variant_names[best_var]}")
        r = results[best_var]
        r_base = results['v3b']
        m = r['metrics']
        mb = r_base['metrics']

        print(f"\n  vs v3b:")
        print(f"    CAGR:      {mb['cagr']:.1%} → {m['cagr']:.1%} ({m['cagr']-mb['cagr']:+.1%})")
        print(f"    MaxDD:     {mb['max_dd']:.1%} → {m['max_dd']:.1%} ({m['max_dd']-mb['max_dd']:+.1%})")
        print(f"    Sharpe:    {mb['sharpe']:.2f} → {m['sharpe']:.2f} ({m['sharpe']-mb['sharpe']:+.2f})")
        print(f"    Calmar:    {mb['calmar']:.2f} → {m['calmar']:.2f} ({m['calmar']-mb['calmar']:+.2f})")
        print(f"    Composite: {r_base['composite']:.3f} → {r['composite']:.3f} ({r['composite']-r_base['composite']:+.3f})")
        print(f"    WF:        {r_base['wf']:.2f} → {r['wf']:.2f}")

        if r['dd_2022'] and r_base['dd_2022']:
            print(f"\n  2022 Bear Market:")
            print(f"    MaxDD:     {r_base['dd_2022'].get('max_dd_2022',0):.1%} → {r['dd_2022'].get('max_dd_2022',0):.1%}")
            print(f"    Return:    {r_base['dd_2022'].get('ret_2022',0):.1%} → {r['dd_2022'].get('ret_2022',0):.1%}")
            rec_base = r_base['dd_2022'].get('recovery_months')
            rec_new = r['dd_2022'].get('recovery_months')
            print(f"    Recovery:  {rec_base:.0f}m → {rec_new:.0f}m" if rec_base and rec_new else f"    Recovery:  {rec_base} → {rec_new}")
    else:
        print("\n⚠️ No v4 variant passed WF threshold (≥ 0.70) with better Composite than v3b")
        # Find best even without beating v3b
        best_wf_var = None
        best_wf_comp = -999
        for var in variants:
            if var != 'v3b' and results[var]['wf'] >= 0.70 and results[var]['composite'] > best_wf_comp:
                best_wf_comp = results[var]['composite']
                best_wf_var = var
        if best_wf_var:
            print(f"  Best WF-passing variant: {variant_names[best_wf_var]} (Composite: {best_wf_comp:.3f})")

    # Save results
    results_json = {}
    for var in variants:
        r = results[var]
        results_json[var] = {
            'name': variant_names[var],
            'cagr': float(r['metrics']['cagr']),
            'max_dd': float(r['metrics']['max_dd']),
            'sharpe': float(r['metrics']['sharpe']),
            'calmar': float(r['metrics']['calmar']),
            'is_sharpe': float(r['is']['sharpe']),
            'oos_sharpe': float(r['oos']['sharpe']),
            'wf': float(r['wf']),
            'composite': float(r['composite']),
            'turnover': float(r['turnover']),
            'dd_2022': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                       for k, v in r['dd_2022'].items()} if r['dd_2022'] else {},
        }

    results_file = Path(__file__).parent / "momentum_v4_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

    return results


if __name__ == '__main__':
    main()
