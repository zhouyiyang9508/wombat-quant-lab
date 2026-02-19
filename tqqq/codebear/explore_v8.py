#!/usr/bin/env python3
"""
TQQQ Strategy Exploration — v8 候选方向
Author: 代码熊 🐻
Date: 2026-02-19

探索方向：
  A. 自适应滞后带（ATR-based）
  B. 多时间框架确认（周线 SMA50）
  C. 连续仓位函数（sigmoid RSI → 仓位）
  D. 止盈机制（ATH 回撤止盈）
  E. Vol-of-Vol 信号
  F. 组合策略

所有策略与 v5 做公平 lump sum $10,000 对比。
"""

import pandas as pd
import numpy as np
import sys, os

# ── Data ──────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tqqq_daily.csv')

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
    return df[['Close']].dropna()

# ── Common Indicators ─────────────────────────────────────────────

def compute_sma(prices, window):
    return prices.rolling(window).mean()

def compute_rsi(prices, period=10):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    """ATR using Close only (no High/Low for some strategies)."""
    # Use High/Low if available, else approximate from Close
    if 'High' in df.columns and 'Low' in df.columns:
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
    else:
        # Approximate: use absolute daily change
        tr = df['Close'].diff().abs()
    return tr.rolling(period).mean()

def compute_weekly_ret(prices, period=5):
    return prices.pct_change(period)

def compute_realized_vol(prices, window=20):
    """Annualized realized volatility."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)

def compute_vol_of_vol(prices, vol_window=20, vov_window=20):
    """Volatility of volatility."""
    vol = compute_realized_vol(prices, vol_window)
    return vol.rolling(vov_window).std()

def compute_weekly_sma(prices, period=50):
    """Resample to weekly, compute SMA, then forward-fill to daily."""
    weekly = prices.resample('W-FRI').last().dropna()
    weekly_sma = weekly.rolling(period).mean()
    # Forward fill to daily
    return weekly_sma.reindex(prices.index, method='ffill')

# ── Metrics ───────────────────────────────────────────────────────

def calc_metrics(portfolio_values, initial_capital=10000):
    pv = pd.Series(portfolio_values) if not isinstance(portfolio_values, pd.Series) else portfolio_values
    final_val = pv.iloc[-1]
    start_val = pv.iloc[0]
    n_days = len(pv)
    years = n_days / 252.0

    cagr = (final_val / start_val) ** (1 / years) - 1

    peak = pv.cummax()
    dd = (pv - peak) / peak
    max_dd = dd.min()

    dr = pv.pct_change().dropna()
    rf = 0.045 / 252
    excess = dr - rf
    sharpe = (excess.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
    downside = dr[dr < 0].std()
    sortino = (excess.mean() / downside) * np.sqrt(252) if downside > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'final_val': final_val,
        'cagr': cagr,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
    }

# ── V5 Baseline ──────────────────────────────────────────────────

def run_v5(data, initial_capital=10000):
    """Exact replica of beast_v5.py logic."""
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)

    cash = initial_capital
    shares = 0.0
    in_bear = False
    portfolio_values = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]

        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90:
                in_bear = True
            elif in_bear and price > sma * 1.05:
                in_bear = False

        if not in_bear:
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            else:
                target_pct = 1.00
        else:
            if not pd.isna(rsi) and rsi < 20:
                target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12:
                target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30:
                target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65:
                target_pct = 0.30
            else:
                curr_val = cash + shares * price
                target_pct = (shares * price) / curr_val if curr_val > 0 else 0.30
                target_pct = max(target_pct, 0.30)

        curr_val = cash + shares * price
        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY A: Adaptive Hysteresis Bands (ATR-based)
# ══════════════════════════════════════════════════════════════════

def run_strategy_a(data, initial_capital=10000,
                   atr_period=60, bear_mult=1.5, bull_mult=1.0,
                   base_bear=0.90, base_bull=1.05):
    """
    Instead of fixed 90%/105% bands, adapt based on ATR.
    Higher volatility → wider bands (less whipsaw).
    Lower volatility → tighter bands (faster reaction).
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    atr = compute_atr(data, atr_period)

    # Normalize ATR as % of price
    atr_pct = atr / prices

    # Median ATR% for centering
    atr_median = atr_pct.rolling(252).median()

    cash = initial_capital
    shares = 0.0
    in_bear = False
    portfolio_values = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]
        atr_p = atr_pct.iloc[i]
        atr_m = atr_median.iloc[i]

        # Adaptive bands
        if i >= 252 and not pd.isna(sma) and not pd.isna(atr_p) and not pd.isna(atr_m) and atr_m > 0:
            vol_ratio = atr_p / atr_m  # >1 = higher vol than normal
            # Wider bands when vol is high, tighter when low
            bear_band = base_bear - (vol_ratio - 1) * bear_mult * 0.05  # e.g. 0.90 → 0.85 at 2x vol
            bull_band = base_bull + (vol_ratio - 1) * bull_mult * 0.03  # e.g. 1.05 → 1.08 at 2x vol
            bear_band = np.clip(bear_band, 0.80, 0.95)
            bull_band = np.clip(bull_band, 1.02, 1.15)
        else:
            bear_band = base_bear
            bull_band = base_bull

        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * bear_band:
                in_bear = True
            elif in_bear and price > sma * bull_band:
                in_bear = False

        # Same position logic as v5
        if not in_bear:
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            else:
                target_pct = 1.00
        else:
            if not pd.isna(rsi) and rsi < 20:
                target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12:
                target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30:
                target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65:
                target_pct = 0.30
            else:
                curr_val = cash + shares * price
                target_pct = (shares * price) / curr_val if curr_val > 0 else 0.30
                target_pct = max(target_pct, 0.30)

        curr_val = cash + shares * price
        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY B: Multi-Timeframe Confirmation (Weekly SMA50)
# ══════════════════════════════════════════════════════════════════

def run_strategy_b(data, initial_capital=10000):
    """
    Add weekly SMA50 as regime confirmation.
    Only enter bear when BOTH daily SMA200 band triggers AND weekly SMA50 confirms downtrend.
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    weekly_sma50 = compute_weekly_sma(prices, 50)

    cash = initial_capital
    shares = 0.0
    in_bear = False
    portfolio_values = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]
        wsma = weekly_sma50.iloc[i]

        if i >= 250 and not pd.isna(sma) and not pd.isna(wsma):
            # Enter bear: need both daily band AND weekly confirmation
            if not in_bear and price < sma * 0.90 and price < wsma:
                in_bear = True
            # Exit bear: either daily band recovery OR strong weekly trend
            elif in_bear and price > sma * 1.05:
                in_bear = False
        elif i >= 200 and not pd.isna(sma):
            # Fallback to v5 logic before weekly data available
            if not in_bear and price < sma * 0.90:
                in_bear = True
            elif in_bear and price > sma * 1.05:
                in_bear = False

        if not in_bear:
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            else:
                target_pct = 1.00
        else:
            if not pd.isna(rsi) and rsi < 20:
                target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12:
                target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30:
                target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65:
                target_pct = 0.30
            else:
                curr_val = cash + shares * price
                target_pct = (shares * price) / curr_val if curr_val > 0 else 0.30
                target_pct = max(target_pct, 0.30)

        curr_val = cash + shares * price
        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY C: Continuous Position Sizing (Sigmoid RSI → Position)
# ══════════════════════════════════════════════════════════════════

def sigmoid(x, center, steepness):
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

def run_strategy_c(data, initial_capital=10000,
                   bear_floor=0.30, bear_ceiling=0.85,
                   rsi_center=35, rsi_steepness=-0.15):
    """
    In bear regime, instead of discrete 30%/60%/80%, use sigmoid:
    position = floor + (ceiling - floor) * sigmoid(RSI, center, steepness)
    Low RSI → high position (panic buy), high RSI → floor.
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)

    cash = initial_capital
    shares = 0.0
    in_bear = False
    portfolio_values = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]

        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90:
                in_bear = True
            elif in_bear and price > sma * 1.05:
                in_bear = False

        if not in_bear:
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            else:
                target_pct = 1.00
        else:
            if not pd.isna(rsi):
                # Sigmoid: low RSI → high position, high RSI → floor
                sig_val = sigmoid(rsi, rsi_center, rsi_steepness)
                target_pct = bear_floor + (bear_ceiling - bear_floor) * sig_val
                # Also boost on crash weeks
                if not pd.isna(wret) and wret < -0.12:
                    target_pct = max(target_pct, 0.80)
            else:
                target_pct = bear_floor

        curr_val = cash + shares * price
        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY D: Profit-Taking Mechanism (ATH Drawdown)
# ══════════════════════════════════════════════════════════════════

def run_strategy_d(data, initial_capital=10000,
                   take_profit_dd=-0.10, trim_to=0.70, recovery_pct=0.95):
    """
    In bull regime, if portfolio drops >10% from ATH, trim to 70%.
    Restore to 100% when portfolio recovers to 95% of ATH.
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)

    cash = initial_capital
    shares = 0.0
    in_bear = False
    trimmed = False  # profit-take state
    portfolio_values = []
    ath = initial_capital

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]

        curr_val = cash + shares * price
        ath = max(ath, curr_val)

        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90:
                in_bear = True
                trimmed = False
            elif in_bear and price > sma * 1.05:
                in_bear = False
                trimmed = False

        if not in_bear:
            # Profit-taking logic
            pv_dd = (curr_val - ath) / ath
            if not trimmed and pv_dd < take_profit_dd:
                target_pct = trim_to
                trimmed = True
            elif trimmed and curr_val > ath * recovery_pct:
                target_pct = 1.00
                trimmed = False
            elif trimmed:
                target_pct = trim_to
            else:
                if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                    target_pct = 0.80
                else:
                    target_pct = 1.00
        else:
            if not pd.isna(rsi) and rsi < 20:
                target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12:
                target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30:
                target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65:
                target_pct = 0.30
            else:
                target_pct = (shares * price) / curr_val if curr_val > 0 else 0.30
                target_pct = max(target_pct, 0.30)

        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY E: Vol-of-Vol Signal
# ══════════════════════════════════════════════════════════════════

def run_strategy_e(data, initial_capital=10000,
                   vov_threshold_high=0.08, vov_threshold_low=0.03):
    """
    When vol-of-vol is high (regime uncertainty), reduce position.
    When vol-of-vol is low (stable regime), trust the regime signal more.
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    vov = compute_vol_of_vol(prices, 20, 20)

    cash = initial_capital
    shares = 0.0
    in_bear = False
    portfolio_values = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]
        v = vov.iloc[i]

        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90:
                in_bear = True
            elif in_bear and price > sma * 1.05:
                in_bear = False

        # Base target same as v5
        if not in_bear:
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.80
            else:
                target_pct = 1.00
        else:
            if not pd.isna(rsi) and rsi < 20:
                target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12:
                target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30:
                target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65:
                target_pct = 0.30
            else:
                curr_val = cash + shares * price
                target_pct = (shares * price) / curr_val if curr_val > 0 else 0.30
                target_pct = max(target_pct, 0.30)

        # VoV overlay: scale down when VoV is very high
        if not pd.isna(v):
            if v > vov_threshold_high:
                # High regime uncertainty → reduce by 15-25%
                scale = max(0.75, 1.0 - (v - vov_threshold_high) * 3)
                target_pct *= scale
            elif v < vov_threshold_low and not in_bear:
                # Very stable → boost slightly (max 100%)
                target_pct = min(1.0, target_pct * 1.05)

        curr_val = cash + shares * price
        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY F: Combined Best Ideas
# ══════════════════════════════════════════════════════════════════

def run_strategy_f(data, initial_capital=10000):
    """
    Combine: adaptive bands + sigmoid position + crash boost.
    Keep it rule-based, no ML.
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    atr = compute_atr(data, 60)
    atr_pct = atr / prices
    atr_median = atr_pct.rolling(252).median()
    vol = compute_realized_vol(prices, 20)

    cash = initial_capital
    shares = 0.0
    in_bear = False
    portfolio_values = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]
        atr_p = atr_pct.iloc[i]
        atr_m = atr_median.iloc[i]
        v = vol.iloc[i]

        # Adaptive bands
        if i >= 252 and not pd.isna(sma) and not pd.isna(atr_p) and not pd.isna(atr_m) and atr_m > 0:
            vol_ratio = atr_p / atr_m
            bear_band = np.clip(0.90 - (vol_ratio - 1) * 0.075, 0.82, 0.94)
            bull_band = np.clip(1.05 + (vol_ratio - 1) * 0.04, 1.03, 1.12)
        else:
            bear_band = 0.90
            bull_band = 1.05

        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * bear_band:
                in_bear = True
            elif in_bear and price > sma * bull_band:
                in_bear = False

        if not in_bear:
            # Bull: slight vol-aware trim
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.75
            elif not pd.isna(v) and v > 0.80:
                # Extremely high vol in bull → slight caution
                target_pct = 0.90
            else:
                target_pct = 1.00
        else:
            # Bear: sigmoid position
            if not pd.isna(rsi):
                sig_val = sigmoid(rsi, 35, -0.15)
                target_pct = 0.30 + 0.55 * sig_val
                # Crash boost
                if not pd.isna(wret) and wret < -0.12:
                    target_pct = max(target_pct, 0.80)
                if not pd.isna(wret) and wret < -0.20:
                    target_pct = max(target_pct, 0.90)
            else:
                target_pct = 0.30

        curr_val = cash + shares * price
        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY G: Aggressive Bear Floor Boost + Tighter Euphoria Trim
# ══════════════════════════════════════════════════════════════════

def run_strategy_g(data, initial_capital=10000, bear_floor=0.40):
    """
    v5 variant with higher bear floor (40%) and more aggressive euphoria trim.
    The BEST_MODEL shows 50% floor has higher CAGR (41.4%) but slightly worse MaxDD.
    Try 40% as a sweet spot.
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)

    cash = initial_capital
    shares = 0.0
    in_bear = False
    portfolio_values = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]

        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90:
                in_bear = True
            elif in_bear and price > sma * 1.05:
                in_bear = False

        if not in_bear:
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                target_pct = 0.75
            else:
                target_pct = 1.00
        else:
            if not pd.isna(rsi) and rsi < 20:
                target_pct = 0.85
            elif not pd.isna(wret) and wret < -0.12:
                target_pct = 0.75
            elif not pd.isna(rsi) and rsi < 30:
                target_pct = 0.65
            elif not pd.isna(rsi) and rsi > 65:
                target_pct = bear_floor
            else:
                curr_val = cash + shares * price
                target_pct = (shares * price) / curr_val if curr_val > 0 else bear_floor
                target_pct = max(target_pct, bear_floor)

        curr_val = cash + shares * price
        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# STRATEGY H: Momentum Overlay (use price momentum to adjust allocation)
# ══════════════════════════════════════════════════════════════════

def run_strategy_h(data, initial_capital=10000):
    """
    In bull, if 1-month momentum is strongly negative, preemptively reduce.
    In bear, if 1-month momentum is strongly positive, preemptively increase.
    """
    prices = data['Close']
    sma200 = compute_sma(prices, 200)
    rsi10 = compute_rsi(prices, 10)
    weekly_ret = compute_weekly_ret(prices, 5)
    monthly_ret = prices.pct_change(21)

    cash = initial_capital
    shares = 0.0
    in_bear = False
    portfolio_values = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        sma = sma200.iloc[i]
        rsi = rsi10.iloc[i]
        wret = weekly_ret.iloc[i]
        mret = monthly_ret.iloc[i]

        if i >= 200 and not pd.isna(sma):
            if not in_bear and price < sma * 0.90:
                in_bear = True
            elif in_bear and price > sma * 1.05:
                in_bear = False

        if not in_bear:
            base = 1.00
            if not pd.isna(rsi) and rsi > 80 and not pd.isna(wret) and wret > 0.15:
                base = 0.80
            # Momentum overlay in bull
            if not pd.isna(mret) and mret < -0.15:
                base = min(base, 0.70)  # Pre-emptive defense
            target_pct = base
        else:
            if not pd.isna(rsi) and rsi < 20:
                target_pct = 0.80
            elif not pd.isna(wret) and wret < -0.12:
                target_pct = 0.70
            elif not pd.isna(rsi) and rsi < 30:
                target_pct = 0.60
            elif not pd.isna(rsi) and rsi > 65:
                target_pct = 0.30
            else:
                curr_val = cash + shares * price
                target_pct = (shares * price) / curr_val if curr_val > 0 else 0.30
                target_pct = max(target_pct, 0.30)
            # Momentum boost in bear
            if not pd.isna(mret) and mret > 0.15:
                target_pct = min(target_pct + 0.15, 0.85)

        curr_val = cash + shares * price
        target_equity = curr_val * target_pct
        diff = target_equity - shares * price

        if diff > 0 and cash > 0:
            buy = min(diff, cash)
            shares += buy / price
            cash -= buy
        elif diff < 0:
            sell = abs(diff)
            shares -= sell / price
            cash += sell

        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=prices.index)


# ══════════════════════════════════════════════════════════════════
# MAIN: Run All and Compare
# ══════════════════════════════════════════════════════════════════

def main():
    data = load_data()
    print(f"📊 Data: {data.index[0].date()} → {data.index[-1].date()}, {len(data)} rows")
    print(f"{'='*90}")

    strategies = {
        'V5 Baseline':         lambda: run_v5(data),
        'A: Adaptive Bands':   lambda: run_strategy_a(data),
        'B: Multi-TF Confirm': lambda: run_strategy_b(data),
        'C: Sigmoid Position': lambda: run_strategy_c(data),
        'D: Profit-Taking':    lambda: run_strategy_d(data),
        'E: Vol-of-Vol':       lambda: run_strategy_e(data),
        'F: Combined':         lambda: run_strategy_f(data),
        'G: Higher Floor(40%)':lambda: run_strategy_g(data),
        'H: Momentum Overlay': lambda: run_strategy_h(data),
    }

    results = {}
    for name, fn in strategies.items():
        try:
            pv = fn()
            m = calc_metrics(pv)
            results[name] = m
        except Exception as e:
            print(f"❌ {name}: {e}")
            import traceback; traceback.print_exc()

    # Print comparison table
    print(f"\n{'Strategy':<24} {'Final Val':>12} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Sortino':>8}")
    print('-' * 90)

    v5_metrics = results.get('V5 Baseline', {})

    for name, m in results.items():
        marker = ' ★' if name == 'V5 Baseline' else ''
        print(f"{name:<24} ${m['final_val']:>10,.0f} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['sortino']:>7.2f}{marker}")

    # Compare each vs v5
    if v5_metrics:
        print(f"\n{'='*90}")
        print("📊 vs V5 比较（需 ≥2 指标显著优于 v5 才能胜出）:")
        print(f"{'='*90}")

        for name, m in results.items():
            if name == 'V5 Baseline':
                continue
            wins = 0
            details = []
            # CAGR: significantly better = +2% absolute
            cagr_diff = m['cagr'] - v5_metrics['cagr']
            if cagr_diff > 0.02:
                wins += 1
                details.append(f"CAGR +{cagr_diff*100:.1f}%")
            elif cagr_diff < -0.02:
                details.append(f"CAGR {cagr_diff*100:.1f}%")

            # MaxDD: significantly better = less negative by >3%
            dd_diff = m['max_dd'] - v5_metrics['max_dd']  # less negative = better
            if dd_diff > 0.03:
                wins += 1
                details.append(f"MaxDD +{dd_diff*100:.1f}%")
            elif dd_diff < -0.03:
                details.append(f"MaxDD {dd_diff*100:.1f}%")

            # Sharpe: significantly better = +0.05
            sh_diff = m['sharpe'] - v5_metrics['sharpe']
            if sh_diff > 0.05:
                wins += 1
                details.append(f"Sharpe +{sh_diff:.2f}")
            elif sh_diff < -0.05:
                details.append(f"Sharpe {sh_diff:.2f}")

            # Calmar: significantly better = +0.05
            cal_diff = m['calmar'] - v5_metrics['calmar']
            if cal_diff > 0.05:
                wins += 1
                details.append(f"Calmar +{cal_diff:.2f}")
            elif cal_diff < -0.05:
                details.append(f"Calmar {cal_diff:.2f}")

            verdict = "✅ WINS" if wins >= 2 else "❌ 不足"
            print(f"  {name:<22} → {verdict} (胜出指标: {wins}/4) [{', '.join(details)}]")

    # Also run parameter sweeps for the most promising strategies
    print(f"\n{'='*90}")
    print("🔬 参数扫描 — Strategy A (Adaptive Bands)")
    print(f"{'='*90}")
    print(f"{'ATR_period':<12} {'bear_mult':<12} {'bull_mult':<12} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    for atr_p in [30, 60, 90]:
        for bm in [1.0, 1.5, 2.0]:
            for blm in [0.5, 1.0, 1.5]:
                pv = run_strategy_a(data, atr_period=atr_p, bear_mult=bm, bull_mult=blm)
                m = calc_metrics(pv)
                print(f"{atr_p:<12} {bm:<12} {blm:<12} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")

    print(f"\n{'='*90}")
    print("🔬 参数扫描 — Strategy C (Sigmoid Position)")
    print(f"{'='*90}")
    print(f"{'floor':<8} {'ceiling':<10} {'center':<10} {'steep':<10} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    for floor in [0.25, 0.30, 0.35, 0.40]:
        for ceiling in [0.75, 0.85, 0.95]:
            for center in [30, 35, 40]:
                for steep in [-0.10, -0.15, -0.20]:
                    pv = run_strategy_c(data, bear_floor=floor, bear_ceiling=ceiling,
                                        rsi_center=center, rsi_steepness=steep)
                    m = calc_metrics(pv)
                    # Only print if CAGR > v5
                    if m['cagr'] > v5_metrics.get('cagr', 0):
                        print(f"{floor:<8} {ceiling:<10} {center:<10} {steep:<10} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")

    print(f"\n{'='*90}")
    print("🔬 参数扫描 — Strategy G (Bear Floor)")
    print(f"{'='*90}")
    print(f"{'floor':<8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    for floor in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        pv = run_strategy_g(data, bear_floor=floor)
        m = calc_metrics(pv)
        print(f"{floor:<8} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")

    print(f"\n{'='*90}")
    print("🔬 参数扫描 — Strategy D (Profit-Taking)")
    print(f"{'='*90}")
    print(f"{'tp_dd':<10} {'trim_to':<10} {'recovery':<10} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    for tp_dd in [-0.08, -0.10, -0.12, -0.15]:
        for trim_to in [0.60, 0.70, 0.80]:
            for recovery in [0.93, 0.95, 0.97]:
                pv = run_strategy_d(data, take_profit_dd=tp_dd, trim_to=trim_to, recovery_pct=recovery)
                m = calc_metrics(pv)
                print(f"{tp_dd:<10} {trim_to:<10} {recovery:<10} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")


if __name__ == '__main__':
    main()
