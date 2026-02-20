#!/usr/bin/env python3
"""
Stock Momentum v3 — 全面优化探索
代码熊 🐻

目标: 突破 v2d 的 Composite 1.013
方向:
  v3a: Academic Momentum (12-1 skip-month)
  v3b: Two-Stage Sector Rotation (sector first, then stocks)
  v3c: Triple Regime + Adaptive Sizing (3 regimes + drawdown protection)
  v3d: Momentum × Low-Vol Composite Score (dual factor)
  v3e: Trend Filter + Portfolio DD Protection (individual + portfolio level)
  v3f: Combined Best — cherry-pick the best innovations from a-e
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
    """Precompute ALL signals needed for all v3 variants."""
    # Standard returns
    ret_1m = close_df / close_df.shift(22) - 1
    ret_3m = close_df / close_df.shift(63) - 1
    ret_6m = close_df / close_df.shift(126) - 1
    ret_12m = close_df / close_df.shift(252) - 1

    # Skip-month returns (skip last 22 days) — academic standard
    ret_6m_skip = close_df.shift(22) / close_df.shift(126) - 1
    ret_12m_skip = close_df.shift(22) / close_df.shift(252) - 1

    # Volatility
    log_ret = np.log(close_df / close_df.shift(1))
    vol_30d = log_ret.rolling(30).std() * np.sqrt(252)
    vol_60d = log_ret.rolling(60).std() * np.sqrt(252)

    # SPY signals
    spy_close = close_df['SPY'] if 'SPY' in close_df.columns else None
    spy_sma200 = spy_close.rolling(200).mean() if spy_close is not None else None
    spy_sma50 = spy_close.rolling(50).mean() if spy_close is not None else None

    # SPY realized vol (VIX proxy)
    spy_vol_20 = log_ret['SPY'].rolling(20).std() * np.sqrt(252) if 'SPY' in close_df.columns else None

    # Individual stock SMAs
    sma_50 = close_df.rolling(50).mean()
    sma_200 = close_df.rolling(200).mean()

    # Volume signals (if volume data available)
    vol_ma_20 = None
    vol_ma_50 = None
    if volume_df is not None and len(volume_df.columns) > 0:
        vol_ma_20 = volume_df.rolling(20).mean()
        vol_ma_50 = volume_df.rolling(50).mean()

    return {
        'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
        'ret_6m_skip': ret_6m_skip, 'ret_12m_skip': ret_12m_skip,
        'vol_30d': vol_30d, 'vol_60d': vol_60d,
        'spy_sma200': spy_sma200, 'spy_sma50': spy_sma50,
        'spy_close': spy_close, 'spy_vol_20': spy_vol_20,
        'sma_50': sma_50, 'sma_200': sma_200,
        'close': close_df,
        'vol_ma_20': vol_ma_20, 'vol_ma_50': vol_ma_50,
    }


def get_regime_v2d(signals, date):
    """Original v2d binary regime."""
    spy_sma = signals['spy_sma200']
    if spy_sma is None:
        return 'bull'
    valid = spy_sma.loc[:date].dropna()
    spy_close = signals['spy_close'].loc[:date].dropna()
    if len(valid) == 0 or len(spy_close) == 0:
        return 'bull'
    return 'bull' if spy_close.iloc[-1] > valid.iloc[-1] else 'bear'


def get_regime_triple(signals, date):
    """Triple regime: strong_bull / weak_bull / bear."""
    spy_sma200 = signals['spy_sma200']
    spy_sma50 = signals['spy_sma50']
    if spy_sma200 is None:
        return 'strong_bull'
    sma200 = spy_sma200.loc[:date].dropna()
    sma50 = spy_sma50.loc[:date].dropna()
    spy = signals['spy_close'].loc[:date].dropna()
    if len(sma200) == 0 or len(sma50) == 0 or len(spy) == 0:
        return 'strong_bull'
    price = spy.iloc[-1]
    s200 = sma200.iloc[-1]
    s50 = sma50.iloc[-1]
    if price > s200 and s50 > s200:
        return 'strong_bull'
    elif price > s200 or s50 > s200:
        return 'weak_bull'
    else:
        return 'bear'


# ============================================================
# V2D BASELINE (reference)
# ============================================================
def select_v2d(signals, sectors, date, prev_holdings):
    """Original v2d strategy (baseline)."""
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime_v2d(signals, date)

    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
    }).dropna(subset=['momentum'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
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
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3A: Academic Skip-Month Momentum
# ============================================================
def select_v3a(signals, sectors, date, prev_holdings):
    """
    v3a: Academic skip-month momentum.
    - Use 12-1 and 6-1 momentum (skip last month) — Jegadeesh & Titman standard
    - Blended: 0.50 × ret_12_1_skip + 0.50 × ret_6_1_skip
    - Everything else same as v2d
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime_v2d(signals, date)

    # Skip-month momentum (academic standard)
    mom = (signals['ret_12m_skip'].loc[idx] * 0.50 +
           signals['ret_6m_skip'].loc[idx] * 0.50)

    df = pd.DataFrame({
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
    }).dropna(subset=['momentum'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
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
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3B: Two-Stage Sector Rotation
# ============================================================
def select_v3b(signals, sectors, date, prev_holdings):
    """
    v3b: Two-stage sector rotation.
    Stage 1: Rank sectors by average momentum of members → top 4
    Stage 2: Within each top sector, pick top 3 stocks
    Total: 12 in bull, 8 in bear (2 sectors × 4 stocks)
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime_v2d(signals, date)

    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
    }).dropna(subset=['momentum'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]

    # Assign sectors
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))

    # Stage 1: Rank sectors by average momentum
    sector_mom = df.groupby('sector')['momentum'].mean().sort_values(ascending=False)

    if regime == 'bull':
        top_sectors = sector_mom.head(4).index.tolist()
        stocks_per_sector = 3
        cash = 0.0
    else:
        top_sectors = sector_mom.head(3).index.tolist()
        stocks_per_sector = 3
        cash = 0.20

    # Stage 2: Within each top sector, pick top stocks
    selected = []
    for sec in top_sectors:
        sec_df = df[df['sector'] == sec].sort_values('momentum', ascending=False)
        for t in sec_df.index[:stocks_per_sector]:
            if t in prev_holdings:
                pass  # holdover bonus implicit in sector selection
            selected.append(t)

    if not selected:
        return {}
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3C: Triple Regime + Adaptive Sizing + DD Protection
# ============================================================
def select_v3c(signals, sectors, date, prev_holdings, portfolio_dd=0.0):
    """
    v3c: Triple regime with drawdown protection.
    - Strong bull: 15 stocks, max 4/sector, 100%
    - Weak bull: 10 stocks, max 3/sector, 90%
    - Bear: 6 stocks, max 2/sector, 65%
    - DD protection: if DD > -8%, further reduce to 50%
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime_triple(signals, date)

    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
    }).dropna(subset=['momentum'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03

    df = df.sort_values('momentum', ascending=False)

    if regime == 'strong_bull':
        top_n, max_sec, cash = 15, 4, 0.0
    elif regime == 'weak_bull':
        top_n, max_sec, cash = 10, 3, 0.10
    else:
        top_n, max_sec, cash = 6, 2, 0.35

    # DD protection overlay
    if portfolio_dd < -0.08:
        dd_mult = max(0.5, 1.0 + (portfolio_dd + 0.08) * 5)  # linear reduce
        invested_base = 1.0 - cash
        cash = 1.0 - invested_base * dd_mult

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
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = max(1.0 - cash, 0.0)
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3D: Momentum × Low-Vol Composite Score
# ============================================================
def select_v3d(signals, sectors, date, prev_holdings):
    """
    v3d: Dual-factor composite — momentum × low-volatility.
    - Rank by momentum (higher = better percentile)
    - Rank by inverse volatility (lower vol = better percentile)
    - Combined = 0.60 × mom_pct + 0.40 × inv_vol_pct
    - Select top 12/8 by combined score
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime_v2d(signals, date)

    mom = (signals['ret_1m'].loc[idx] * 0.20 +
           signals['ret_3m'].loc[idx] * 0.40 +
           signals['ret_6m'].loc[idx] * 0.30 +
           signals['ret_12m'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
    }).dropna(subset=['momentum'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]

    if len(df) < 5:
        return {}

    # Percentile ranks
    df['mom_pct'] = df['momentum'].rank(pct=True)
    df['vol_pct'] = (1.0 / df['vol_30d']).rank(pct=True)  # inverse vol → higher = lower vol

    # Holdover bonus
    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'mom_pct'] = min(df.loc[t, 'mom_pct'] + 0.05, 1.0)

    # Combined score
    df['score'] = df['mom_pct'] * 0.60 + df['vol_pct'] * 0.40
    df = df.sort_values('score', ascending=False)

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
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = 1.0 - cash
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3E: Trend Filter + Portfolio DD Protection
# ============================================================
def select_v3e(signals, sectors, date, prev_holdings, portfolio_dd=0.0):
    """
    v3e: Individual trend filter + portfolio DD protection.
    - Stock must be above its 50-day SMA (short-term trend intact)
    - Stock must be above its 200-day SMA (long-term trend intact)
    - Portfolio DD > -10%: reduce invested to 50%
    - Portfolio DD > -15%: reduce invested to 30%
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime_v2d(signals, date)

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
    }).dropna(subset=['momentum', 'sma50', 'sma200'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]

    # Core filters
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]

    # Trend filters: price above both 50 and 200 day SMA
    df = df[(df['price'] > df['sma50']) & (df['price'] > df['sma200'])]

    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'momentum'] += 0.03

    df = df.sort_values('momentum', ascending=False)

    if regime == 'bull':
        top_n, max_sec, cash = 12, 3, 0.0
    else:
        top_n, max_sec, cash = 8, 2, 0.20

    # Portfolio DD protection
    if portfolio_dd < -0.10:
        cash = max(cash, 0.50)
    elif portfolio_dd < -0.15:
        cash = max(cash, 0.70)

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
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = max(1.0 - cash, 0.0)
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# V3F: Combined Best — All innovations merged
# ============================================================
def select_v3f(signals, sectors, date, prev_holdings, portfolio_dd=0.0):
    """
    v3f: Combined best innovations from v3a-v3e.
    - Skip-month momentum (v3a): 0.50 × ret_12_1 + 0.30 × ret_6_1 + 0.20 × ret_3m
    - Low-vol factor (v3d): combined score 0.55 × mom_pct + 0.35 × vol_pct + 0.10 × holdover
    - Trend filter (v3e): price > SMA50
    - Triple regime (v3c): strong_bull / weak_bull / bear
    - Sector rotation (v3b): max sectors limited
    - DD protection (v3c/v3e): gentle drawdown overlay
    """
    close = signals['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    idx = idx[-1]
    regime = get_regime_triple(signals, date)

    # Skip-month blended momentum (academic)
    mom = (signals['ret_12m_skip'].loc[idx] * 0.50 +
           signals['ret_6m_skip'].loc[idx] * 0.30 +
           signals['ret_3m'].loc[idx] * 0.20)

    df = pd.DataFrame({
        'momentum': mom,
        'abs_6m': signals['ret_6m'].loc[idx],
        'vol_30d': signals['vol_30d'].loc[idx],
        'price': close.loc[idx],
        'sma50': signals['sma_50'].loc[idx],
    }).dropna(subset=['momentum', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]

    # Core filters + trend filter
    df = df[(df['abs_6m'] > 0) & (df['vol_30d'] < 0.65)]
    df = df[df['price'] > df['sma50']]

    if len(df) < 5:
        # Fallback: relax trend filter
        df_orig = pd.DataFrame({
            'momentum': mom,
            'abs_6m': signals['ret_6m'].loc[idx],
            'vol_30d': signals['vol_30d'].loc[idx],
            'price': close.loc[idx],
        }).dropna(subset=['momentum'])
        df_orig = df_orig[(df_orig['price'] >= 5) & (df_orig.index != 'SPY')]
        df_orig = df_orig[(df_orig['abs_6m'] > 0) & (df_orig['vol_30d'] < 0.65)]
        if len(df_orig) >= 5:
            df = df_orig

    # Dual-factor score
    df['mom_pct'] = df['momentum'].rank(pct=True)
    df['vol_pct'] = (1.0 / df['vol_30d'].clip(lower=0.10)).rank(pct=True)

    for t in df.index:
        if t in prev_holdings:
            df.loc[t, 'mom_pct'] = min(df.loc[t, 'mom_pct'] + 0.05, 1.0)

    df['score'] = df['mom_pct'] * 0.55 + df['vol_pct'] * 0.35
    df = df.sort_values('score', ascending=False)

    if regime == 'strong_bull':
        top_n, max_sec, cash = 14, 4, 0.0
    elif regime == 'weak_bull':
        top_n, max_sec, cash = 10, 3, 0.10
    else:
        top_n, max_sec, cash = 7, 2, 0.25

    # Gentle DD protection
    if portfolio_dd < -0.10:
        extra_cash = min((-portfolio_dd - 0.10) * 3, 0.30)
        cash = min(cash + extra_cash, 0.60)

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
    inv_vols = {}
    for t in selected:
        vol = max(df.loc[t, 'vol_30d'], 0.10)
        inv_vols[t] = 1.0 / vol
    total = sum(inv_vols.values())
    invested = max(1.0 - cash, 0.0)
    return {t: (v / total) * invested for t, v in inv_vols.items()}


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(close_df, signals, sectors, select_fn, start='2015-01-01', end='2025-12-31',
                 cost_per_trade=0.0015, needs_dd=False):
    """Monthly backtest with optional DD tracking."""
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

        # Portfolio DD for strategies that need it
        portfolio_dd = (current_value / peak_value) - 1 if peak_value > 0 else 0

        if needs_dd:
            new_weights = select_fn(signals, sectors, date, prev_holdings, portfolio_dd=portfolio_dd)
        else:
            new_weights = select_fn(signals, sectors, date, prev_holdings)

        # Turnover
        all_t = set(list(new_weights.keys()) + list(prev_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_t) / 2
        turnover_list.append(turnover)

        prev_weights = new_weights.copy()
        prev_holdings = set(new_weights.keys())
        holdings_history[date.strftime('%Y-%m')] = list(new_weights.keys())

        # Period return
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
    """Simple composite = Sharpe×0.4 + Calmar×0.4 + CAGR×0.2"""
    return sharpe * 0.4 + calmar * 0.4 + cagr * 0.2


def main():
    print("=" * 70)
    print("🐻 Stock Momentum v3 — Comprehensive Optimization Exploration")
    print("=" * 70)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df, volume_df = load_all_data(tickers + ['SPY'])
    sectors_file = CACHE / "sp500_sectors.json"
    sectors = json.load(open(sectors_file)) if sectors_file.exists() else {}

    print(f"Loaded {len(close_df.columns)} stocks, {len(close_df)} days")
    signals = precompute_signals(close_df, volume_df)

    # Define all strategies
    strategies = {
        'v2d_base': {'fn': select_v2d, 'dd': False},
        'v3a_skip': {'fn': select_v3a, 'dd': False},
        'v3b_sector': {'fn': select_v3b, 'dd': False},
        'v3c_triple': {'fn': select_v3c, 'dd': True},
        'v3d_lowvol': {'fn': select_v3d, 'dd': False},
        'v3e_trend': {'fn': select_v3e, 'dd': True},
        'v3f_combo': {'fn': select_v3f, 'dd': True},
    }

    results = {}
    for name, cfg in strategies.items():
        print(f"\nRunning {name}...")

        # Full period
        eq_full, hold_full, to_full = run_backtest(
            close_df, signals, sectors, cfg['fn'],
            '2015-01-01', '2025-12-31', needs_dd=cfg['dd'])
        m_full = compute_metrics(eq_full, name)

        # IS period (2015-2020)
        eq_is, _, _ = run_backtest(
            close_df, signals, sectors, cfg['fn'],
            '2015-01-01', '2020-12-31', needs_dd=cfg['dd'])
        m_is = compute_metrics(eq_is, f"{name}_IS")

        # OOS period (2021-2025)
        eq_oos, _, _ = run_backtest(
            close_df, signals, sectors, cfg['fn'],
            '2021-01-01', '2025-12-31', needs_dd=cfg['dd'])
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

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n" + "=" * 110)
    print("📊 COMPARISON TABLE")
    print("=" * 110)
    header = f"{'Version':<15} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'T/O':>6} {'IS_Sh':>6} {'OOS_Sh':>7} {'WF':>5} {'WF_Pass':>8} {'Composite':>10}"
    print(header)
    print("-" * 110)

    for name, r in sorted(results.items(), key=lambda x: -x[1]['composite']):
        m = r['full']
        wf_pass = "✅" if r['wf'] >= 0.70 else "❌"
        is_best = " ⭐" if r['composite'] == max(x['composite'] for x in results.values()) else ""
        print(f"{name:<15} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} {m['sharpe']:>7.2f} "
              f"{r['turnover']:>5.1%} {r['is']['sharpe']:>6.2f} {r['oos']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f} {wf_pass:>8} {r['composite']:>10.3f}{is_best}")

    print(f"{'SPY':<15} {spy_m['cagr']:>6.1%} {spy_m['max_dd']:>7.1%} {spy_m['sharpe']:>7.2f} "
          f"{'—':>6} {'—':>6} {'—':>7} {'—':>5} {'—':>8} {'—':>10}")

    # Find best WF-passing strategy
    wf_passing = {k: v for k, v in results.items() if v['wf'] >= 0.70}
    if wf_passing:
        best_wf = max(wf_passing.items(), key=lambda x: x[1]['composite'])
        print(f"\n🏆 Best WF-passing: {best_wf[0]} (Composite: {best_wf[1]['composite']:.3f})")
    best_overall = max(results.items(), key=lambda x: x[1]['composite'])
    print(f"🥇 Best overall: {best_overall[0]} (Composite: {best_overall[1]['composite']:.3f})")

    # Holdings analysis for best strategy
    best_name, best_r = best_overall
    print(f"\n📋 {best_name} Holdings (2023-2024):")
    hot = {'NVDA', 'TSLA', 'META', 'AVGO', 'AMD', 'SMCI', 'PLTR'}
    for ym in sorted(best_r['holdings'].keys()):
        if ym.startswith('2023') or ym.startswith('2024'):
            h = best_r['holdings'][ym]
            hotties = [s for s in h if s in hot]
            hotstr = f" 🔥{','.join(hotties)}" if hotties else ""
            print(f"  {ym}: {', '.join(h[:10])}{hotstr}")

    return results


if __name__ == '__main__':
    results = main()
