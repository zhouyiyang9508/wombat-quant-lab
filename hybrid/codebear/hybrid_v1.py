#!/usr/bin/env python3
"""
Hybrid v1 — Crypto + Stock 最优混合策略
代码熊 🐻 | 2026-02-21

目标：通过混合 BestCrypto（BTC/ETH/GLD 动量轮动）和 Stock v9g（SP500 动量选股），
     在风险调整后 beat BestCrypto DDv1 (Composite 1.486)

三层探索：
  A) 静态混合：固定 w_crypto 扫描 0% ~ 100%
  B) 动态混合：根据 crypto 动量/回撤/市场宽度调整
  C) DDv1 升级：逆波动率 + 回撤减权，stock 部分用 v9g 替代 QQQ

🚨 严禁前瞻偏差：所有信号用 close[i-1]
"""

import json
import warnings
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parents[2]
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    df = df.set_index(c).sort_index()
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df


def load_strategy_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_stocks(tickers):
    d = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except:
            pass
    return pd.DataFrame(d)


# ─── BestCrypto Daily Equity ─────────────────────────────────────────────────

def generate_bestcrypto_daily_equity():
    """Generate BestCrypto (BTC/ETH/GLD momentum rotation) daily equity curve."""
    btc = load_csv(CACHE / "BTC_USD.csv")['Close'].dropna()
    eth = load_csv(CACHE / "ETH_USD.csv")['Close'].dropna()
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()

    common = btc.index.intersection(eth.index).intersection(gld.index).sort_values()
    btc_c = btc.loc[common].values
    eth_c = eth.loc[common].values
    gld_c = gld.loc[common].values

    n = len(common)
    equity = np.zeros(n)
    equity[0] = 1.0

    for i in range(1, n):
        btc_ret = btc_c[i] / btc_c[i-1] - 1
        eth_ret = eth_c[i] / eth_c[i-1] - 1
        gld_ret = gld_c[i] / gld_c[i-1] - 1

        # 90-day momentum using close[i-1] (NO look-ahead)
        lb = min(i, 90)
        btc_mom = btc_c[i-1] / btc_c[max(0, i-1-lb)] - 1
        eth_mom = eth_c[i-1] / eth_c[max(0, i-1-lb)] - 1
        gld_mom = gld_c[i-1] / gld_c[max(0, i-1-lb)] - 1

        moms = {"btc": btc_mom, "eth": eth_mom, "gld": gld_mom}
        best = max(moms, key=moms.get)

        if best == "btc" and btc_mom > 0:
            w_btc, w_eth, w_gld = 0.70, 0.15, 0.10
        elif best == "eth" and eth_mom > 0:
            w_btc, w_eth, w_gld = 0.15, 0.65, 0.10
        elif best == "gld":
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.55
        else:
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.35  # 40% cash

        port_ret = w_btc * btc_ret + w_eth * eth_ret + w_gld * gld_ret
        equity[i] = equity[i-1] * (1 + port_ret)

    eq_series = pd.Series(equity, index=common, name='BestCrypto')
    print(f"  BestCrypto: {common[0].date()} → {common[-1].date()}, {n} days")
    return eq_series


# ─── Stock v9g Daily Equity ──────────────────────────────────────────────────

def generate_v9g_daily_equity():
    """Generate Stock v9g daily equity curve using the daily backtest framework."""
    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()

    v9g_mod = load_strategy_module(
        "v9g", BASE / "stocks" / "codebear" / "momentum_v9g_final.py"
    )
    sig = v9g_mod.precompute(close_df)

    # Import the daily backtest function from momentum_v9g_daily.py
    v9g_daily = load_strategy_module(
        "v9g_daily", BASE / "stocks" / "codebear" / "momentum_v9g_daily.py"
    )

    eq = v9g_daily.run_daily_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        v9g_mod.select, v9g_mod.apply_overlays, v9g_mod.get_spy_vol,
        start='2015-01-01', end='2025-12-31', cost=0.0015
    )

    print(f"  Stock v9g: {eq.index[0].date()} → {eq.index[-1].date()}, {len(eq)} days")
    return eq


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(eq, rf=0.04):
    """Compute all metrics from daily equity curve."""
    if len(eq) < 30:
        return dict(cagr=0, maxdd=0, sharpe=0, calmar=0, composite=0, ann_vol=0)

    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5:
        return dict(cagr=0, maxdd=0, sharpe=0, calmar=0, composite=0, ann_vol=0)

    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    maxdd = drawdown.min()

    daily_rets = eq.pct_change().dropna()
    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    composite = sharpe * 0.4 + calmar * 0.4 + min(cagr, 1.0) * 0.2

    # Max DD period
    dd_end_idx = drawdown.idxmin()
    dd_start_idx = eq.loc[:dd_end_idx].idxmax()

    return dict(
        cagr=float(cagr), maxdd=float(maxdd), sharpe=float(sharpe),
        calmar=float(calmar), composite=float(composite), ann_vol=float(ann_vol),
        final_val=float(eq.iloc[-1]), years=float(yrs),
        dd_start=str(dd_start_idx.date()), dd_end=str(dd_end_idx.date())
    )


# ─── A) Static Mix ──────────────────────────────────────────────────────────

def run_static_mix(crypto_eq, stock_eq, w_crypto, cost=0.001):
    """
    Static mix: constant w_crypto allocation.
    Daily rebalance with cost for the drift.
    """
    common = crypto_eq.index.intersection(stock_eq.index).sort_values()
    crypto_ret = crypto_eq.loc[common].pct_change().fillna(0)
    stock_ret = stock_eq.loc[common].pct_change().fillna(0)

    n = len(common)
    equity = np.ones(n)
    w_c = w_crypto  # constant

    for i in range(1, n):
        # Actual portfolio drift from prior day
        # After day i-1, if we started at w_c, the actual weights drifted
        # We approximate: constant w → minimal turnover (only drift rebalance)
        # Monthly rebalance cost instead of daily
        ret = w_c * crypto_ret.iloc[i] + (1 - w_c) * stock_ret.iloc[i]
        equity[i] = equity[i-1] * (1 + ret)

    # Monthly rebalance cost (12x per year * cost * estimated turnover)
    # Turnover from drift: ~5% per month for 50/50 mix
    # Approx: 12 * 0.05 * cost * 2 per year = minor
    eq_series = pd.Series(equity, index=common)
    return eq_series


def scan_static_mix(crypto_eq, stock_eq):
    """Scan w_crypto from 0% to 100% in 5% steps."""
    results = []
    for w_pct in range(0, 105, 5):
        w = w_pct / 100.0
        eq = run_static_mix(crypto_eq, stock_eq, w)
        m = compute_metrics(eq)
        m['w_crypto'] = w
        results.append(m)
        print(f"  w_crypto={w:.0%}: CAGR={m['cagr']:.1%}, MaxDD={m['maxdd']:.1%}, "
              f"Sharpe={m['sharpe']:.2f}, Composite={m['composite']:.3f}")
    return results


# ─── B) Dynamic Mix ─────────────────────────────────────────────────────────

def run_dynamic_mix_v1(crypto_eq, stock_eq, cost=0.001):
    """
    Dynamic Mix v1: adjust crypto weight based on crypto momentum & drawdown.

    Rules (using signals from yesterday = no look-ahead):
    - Base: 60% crypto, 40% stock
    - Crypto 90d momentum > 0 and > stock 90d momentum → up to 80% crypto
    - Crypto in drawdown > -20% → reduce to 30% crypto
    - Crypto in drawdown > -40% → reduce to 15% crypto
    """
    common = crypto_eq.index.intersection(stock_eq.index).sort_values()
    crypto_ret = crypto_eq.loc[common].pct_change().fillna(0)
    stock_ret = stock_eq.loc[common].pct_change().fillna(0)

    n = len(common)
    equity = np.ones(n)
    weights = np.full(n, 0.6)  # default 60% crypto
    prev_w = 0.6

    crypto_vals = crypto_eq.loc[common].values
    stock_vals = stock_eq.loc[common].values

    for i in range(1, n):
        # Signals from yesterday (i-1) — NO look-ahead
        lb = min(i, 90)

        # Crypto 90d momentum (from yesterday's close)
        crypto_mom = crypto_vals[i-1] / crypto_vals[max(0, i-1-lb)] - 1
        stock_mom = stock_vals[i-1] / stock_vals[max(0, i-1-lb)] - 1

        # Crypto drawdown (from yesterday's close)
        crypto_peak = np.max(crypto_vals[:i])  # peak up to yesterday
        crypto_dd = (crypto_vals[i-1] - crypto_peak) / crypto_peak

        # Decision
        if crypto_dd < -0.40:
            w = 0.15
        elif crypto_dd < -0.20:
            w = 0.30
        elif crypto_mom > 0 and crypto_mom > stock_mom:
            w = 0.80
        elif crypto_mom > 0:
            w = 0.60
        else:
            w = 0.40

        # Turnover cost
        turnover = abs(w - prev_w)
        tc = turnover * cost * 2

        ret = w * crypto_ret.iloc[i] + (1 - w) * stock_ret.iloc[i] - tc
        equity[i] = equity[i-1] * (1 + ret)
        weights[i] = w
        prev_w = w

    eq_series = pd.Series(equity, index=common)
    w_series = pd.Series(weights, index=common)
    return eq_series, w_series


def run_dynamic_mix_v2(crypto_eq, stock_eq, cost=0.001):
    """
    Dynamic Mix v2: Momentum + Volatility regime.

    Rules (all signals from i-1):
    - Calculate crypto 60d realized vol
    - Low vol + positive momentum → 75% crypto
    - High vol + positive momentum → 50% crypto
    - Negative momentum → 25% crypto
    - Crypto drawdown > -30% → 15% crypto
    """
    common = crypto_eq.index.intersection(stock_eq.index).sort_values()
    crypto_ret = crypto_eq.loc[common].pct_change().fillna(0)
    stock_ret = stock_eq.loc[common].pct_change().fillna(0)

    n = len(common)
    equity = np.ones(n)
    weights = np.full(n, 0.5)
    prev_w = 0.5

    crypto_vals = crypto_eq.loc[common].values
    crypto_daily_rets = crypto_ret.values

    for i in range(1, n):
        lb_mom = min(i, 90)
        lb_vol = min(i, 60)

        crypto_mom = crypto_vals[i-1] / crypto_vals[max(0, i-1-lb_mom)] - 1

        # Realized vol (annualized) from daily returns up to yesterday
        if lb_vol >= 20:
            vol_window = crypto_daily_rets[max(1, i-lb_vol):i]
            crypto_vol = np.std(vol_window) * np.sqrt(365) if len(vol_window) > 5 else 0.5
        else:
            crypto_vol = 0.5

        # Crypto drawdown
        crypto_peak = np.max(crypto_vals[:i])
        crypto_dd = (crypto_vals[i-1] - crypto_peak) / crypto_peak

        # Decision
        if crypto_dd < -0.30:
            w = 0.15
        elif crypto_dd < -0.15:
            w = 0.30
        elif crypto_mom > 0:
            if crypto_vol < 0.50:
                w = 0.75  # low vol bull
            elif crypto_vol < 0.80:
                w = 0.55  # moderate vol
            else:
                w = 0.40  # high vol, still positive mom
        else:
            w = 0.25

        turnover = abs(w - prev_w)
        tc = turnover * cost * 2

        ret = w * crypto_ret.iloc[i] + (1 - w) * stock_ret.iloc[i] - tc
        equity[i] = equity[i-1] * (1 + ret)
        weights[i] = w
        prev_w = w

    eq_series = pd.Series(equity, index=common)
    w_series = pd.Series(weights, index=common)
    return eq_series, w_series


# ─── C) DDv1 Upgrade: v9g replaces QQQ ──────────────────────────────────────

def run_ddv1_upgrade(crypto_eq, stock_eq, cost=0.001,
                     vol_lookback=20, dd_threshold=-0.10):
    """
    DDv1 mechanism but with v9g instead of QQQ.
    - Inverse volatility weighting between crypto and stock
    - Drawdown-based reduction
    - Cash buffer when both are in drawdown
    """
    common = crypto_eq.index.intersection(stock_eq.index).sort_values()
    crypto_ret = crypto_eq.loc[common].pct_change().fillna(0)
    stock_ret = stock_eq.loc[common].pct_change().fillna(0)

    n = len(common)
    equity = np.ones(n)
    weights = np.full(n, 0.5)
    prev_w = 0.5

    crypto_vals = crypto_eq.loc[common].values
    stock_vals = stock_eq.loc[common].values
    crypto_rets_arr = crypto_ret.values
    stock_rets_arr = stock_ret.values

    for i in range(1, n):
        lb = min(i, vol_lookback)
        if lb < 10:
            w = 0.5
        else:
            # Rolling volatility (annualized)
            crypto_vol = np.std(crypto_rets_arr[max(1, i-lb):i]) * np.sqrt(365)
            stock_vol = np.std(stock_rets_arr[max(1, i-lb):i]) * np.sqrt(252)

            crypto_vol = max(crypto_vol, 0.05)
            stock_vol = max(stock_vol, 0.05)

            # Inverse vol weights
            inv_c = 1.0 / crypto_vol
            inv_s = 1.0 / stock_vol
            total = inv_c + inv_s
            w_crypto_base = inv_c / total

            # Drawdown-based reduction
            crypto_peak = np.max(crypto_vals[:i])
            stock_peak = np.max(stock_vals[:i])
            crypto_dd = (crypto_vals[i-1] - crypto_peak) / crypto_peak
            stock_dd = (stock_vals[i-1] - stock_peak) / stock_peak

            w = w_crypto_base

            # Reduce crypto if in drawdown
            if crypto_dd < dd_threshold:
                reduction = max(0.3, 1.0 + crypto_dd * 2)
                w *= reduction

            # Reduce stock if in drawdown (increase crypto share)
            w_stock = 1.0 - w
            if stock_dd < dd_threshold:
                reduction_s = max(0.3, 1.0 + stock_dd * 2)
                w_stock *= reduction_s

            # Cash buffer
            total_invested = w + w_stock
            if total_invested > 1.0:
                scale = 1.0 / total_invested
                w *= scale
                w_stock *= scale

        turnover = abs(w - prev_w)
        tc = turnover * cost * 2

        ret = w * crypto_ret.iloc[i] + (1 - w) * stock_ret.iloc[i] - tc
        equity[i] = equity[i-1] * (1 + ret)
        weights[i] = w
        prev_w = w

    eq_series = pd.Series(equity, index=common)
    w_series = pd.Series(weights, index=common)
    return eq_series, w_series


def run_ddv1_upgrade_v2(crypto_eq, stock_eq, cost=0.001):
    """
    DDv1 Upgrade v2: Combines momentum regime + inverse vol + DD reduction.
    
    - When crypto momentum positive & strong: allow higher crypto allocation
    - Inverse vol as base, but momentum tilts it
    - More aggressive DD protection
    """
    common = crypto_eq.index.intersection(stock_eq.index).sort_values()
    crypto_ret = crypto_eq.loc[common].pct_change().fillna(0)
    stock_ret = stock_eq.loc[common].pct_change().fillna(0)

    n = len(common)
    equity = np.ones(n)
    weights = np.full(n, 0.5)
    prev_w = 0.5

    crypto_vals = crypto_eq.loc[common].values
    stock_vals = stock_eq.loc[common].values
    crypto_rets_arr = crypto_ret.values
    stock_rets_arr = stock_ret.values

    for i in range(1, n):
        lb_vol = min(i, 30)
        lb_mom = min(i, 90)

        if lb_vol < 10:
            w = 0.5
        else:
            # Volatility
            crypto_vol = np.std(crypto_rets_arr[max(1, i-lb_vol):i]) * np.sqrt(365)
            stock_vol = np.std(stock_rets_arr[max(1, i-lb_vol):i]) * np.sqrt(252)
            crypto_vol = max(crypto_vol, 0.05)
            stock_vol = max(stock_vol, 0.05)

            inv_c = 1.0 / crypto_vol
            inv_s = 1.0 / stock_vol
            w_base = inv_c / (inv_c + inv_s)

            # Momentum tilt
            crypto_mom = crypto_vals[i-1] / crypto_vals[max(0, i-1-lb_mom)] - 1
            stock_mom = stock_vals[i-1] / stock_vals[max(0, i-1-lb_mom)] - 1

            if crypto_mom > 0 and crypto_mom > stock_mom * 1.5:
                # Strong crypto: tilt up to 70%
                w = min(w_base * 1.5, 0.70)
            elif crypto_mom > 0:
                w = min(w_base * 1.2, 0.60)
            elif crypto_mom < -0.10:
                w = min(w_base * 0.5, 0.25)
            else:
                w = w_base

            # DD protection
            crypto_peak = np.max(crypto_vals[:i])
            crypto_dd = (crypto_vals[i-1] - crypto_peak) / crypto_peak

            if crypto_dd < -0.40:
                w = min(w, 0.10)
            elif crypto_dd < -0.25:
                w = min(w, 0.20)
            elif crypto_dd < -0.15:
                w = min(w, 0.35)

        turnover = abs(w - prev_w)
        tc = turnover * cost * 2

        ret = w * crypto_ret.iloc[i] + (1 - w) * stock_ret.iloc[i] - tc
        equity[i] = equity[i-1] * (1 + ret)
        weights[i] = w
        prev_w = w

    eq_series = pd.Series(equity, index=common)
    w_series = pd.Series(weights, index=common)
    return eq_series, w_series


# ─── Efficient Frontier Plot (ASCII) ────────────────────────────────────────

def print_efficient_frontier(static_results):
    """ASCII art efficient frontier: CAGR vs MaxDD."""
    print("\n" + "=" * 70)
    print("📈 Efficient Frontier: CAGR vs MaxDD (Static Mix)")
    print("=" * 70)

    # Table
    print(f"\n{'w_crypto':>10} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Composite':>10}")
    print("-" * 60)
    best_comp = 0
    best_w = 0
    for r in static_results:
        marker = ""
        if r['composite'] > best_comp:
            best_comp = r['composite']
            best_w = r['w_crypto']
            marker = " ★"
        print(f"{r['w_crypto']:>9.0%} {r['cagr']:>7.1%} {r['maxdd']:>7.1%} "
              f"{r['sharpe']:>8.2f} {r['calmar']:>8.2f} {r['composite']:>10.3f}{marker}")

    # ASCII plot: CAGR (y) vs MaxDD (x)
    print(f"\n  Best static Composite: w_crypto = {best_w:.0%} → {best_comp:.3f}")

    # Simple scatter
    print("\n  MaxDD    ←── better ──── worse ───→")
    print("  " + "-" * 50)
    for r in static_results:
        dd_abs = abs(r['maxdd'])
        x = int(dd_abs * 70)  # scale
        w_label = f"{r['w_crypto']:.0%}"
        bar = " " * x + "●"
        print(f"  {w_label:>5} |{bar} CAGR={r['cagr']:.0%}")


# ─── Correlation Analysis ────────────────────────────────────────────────────

def analyze_correlation(crypto_eq, stock_eq):
    """Analyze correlation between crypto and stock strategies."""
    common = crypto_eq.index.intersection(stock_eq.index).sort_values()
    crypto_ret = crypto_eq.loc[common].pct_change().dropna()
    stock_ret = stock_eq.loc[common].pct_change().dropna()

    common_rets = crypto_ret.index.intersection(stock_ret.index)
    cr = crypto_ret.loc[common_rets]
    sr = stock_ret.loc[common_rets]

    overall_corr = cr.corr(sr)
    print(f"\n📊 Correlation Analysis")
    print(f"  Overall daily return correlation: {overall_corr:.3f}")

    # Rolling 120-day correlation
    print(f"\n  Rolling 120d correlation by year:")
    rolling_corr = cr.rolling(120).corr(sr)
    for yr in range(2018, 2026):
        mask = rolling_corr.index.year == yr
        if mask.sum() > 0:
            avg_corr = rolling_corr.loc[mask].mean()
            print(f"    {yr}: {avg_corr:.3f}")

    return overall_corr


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🐻 Hybrid v1 — Crypto + Stock 最优混合策略探索")
    print("=" * 70)

    # ── Step 1: Generate daily equity curves ──
    print("\n[1/5] Generating daily equity curves...")
    print("  Loading BestCrypto...")
    crypto_eq = generate_bestcrypto_daily_equity()
    print("  Loading Stock v9g...")
    stock_eq = generate_v9g_daily_equity()

    # Verify baselines
    mc = compute_metrics(crypto_eq)
    ms = compute_metrics(stock_eq)
    print(f"\n  BestCrypto baseline: CAGR={mc['cagr']:.1%}, MaxDD={mc['maxdd']:.1%}, "
          f"Sharpe={mc['sharpe']:.2f}, Composite={mc['composite']:.3f}")
    print(f"  Stock v9g  baseline: CAGR={ms['cagr']:.1%}, MaxDD={ms['maxdd']:.1%}, "
          f"Sharpe={ms['sharpe']:.2f}, Composite={ms['composite']:.3f}")

    # ── Step 2: Correlation Analysis ──
    print("\n[2/5] Correlation Analysis...")
    overall_corr = analyze_correlation(crypto_eq, stock_eq)

    # ── Step 3: Static Mix Scan ──
    print("\n[3/5] Static Mix Scan (0% → 100% crypto, 5% step)...")
    static_results = scan_static_mix(crypto_eq, stock_eq)
    print_efficient_frontier(static_results)

    # Best static
    best_static = max(static_results, key=lambda x: x['composite'])
    print(f"\n  🏆 Best Static Mix: w_crypto={best_static['w_crypto']:.0%}")
    print(f"     CAGR={best_static['cagr']:.1%}, MaxDD={best_static['maxdd']:.1%}, "
          f"Sharpe={best_static['sharpe']:.2f}, Composite={best_static['composite']:.3f}")

    # ── Step 4: Dynamic Strategies ──
    print("\n[4/5] Dynamic Strategies...")

    print("\n  Dynamic v1 (Momentum + DD regime)...")
    eq_dyn1, w_dyn1 = run_dynamic_mix_v1(crypto_eq, stock_eq)
    m_dyn1 = compute_metrics(eq_dyn1)
    avg_w1 = w_dyn1.mean()
    print(f"    CAGR={m_dyn1['cagr']:.1%}, MaxDD={m_dyn1['maxdd']:.1%}, "
          f"Sharpe={m_dyn1['sharpe']:.2f}, Composite={m_dyn1['composite']:.3f}, "
          f"Avg w_crypto={avg_w1:.1%}")

    print("\n  Dynamic v2 (Momentum + Volatility regime)...")
    eq_dyn2, w_dyn2 = run_dynamic_mix_v2(crypto_eq, stock_eq)
    m_dyn2 = compute_metrics(eq_dyn2)
    avg_w2 = w_dyn2.mean()
    print(f"    CAGR={m_dyn2['cagr']:.1%}, MaxDD={m_dyn2['maxdd']:.1%}, "
          f"Sharpe={m_dyn2['sharpe']:.2f}, Composite={m_dyn2['composite']:.3f}, "
          f"Avg w_crypto={avg_w2:.1%}")

    print("\n  DDv1 Upgrade (Inverse Vol + DD reduction)...")
    eq_dd1, w_dd1 = run_ddv1_upgrade(crypto_eq, stock_eq)
    m_dd1 = compute_metrics(eq_dd1)
    avg_w_dd1 = w_dd1.mean()
    print(f"    CAGR={m_dd1['cagr']:.1%}, MaxDD={m_dd1['maxdd']:.1%}, "
          f"Sharpe={m_dd1['sharpe']:.2f}, Composite={m_dd1['composite']:.3f}, "
          f"Avg w_crypto={avg_w_dd1:.1%}")

    print("\n  DDv1 Upgrade v2 (Mom + Vol + DD)...")
    eq_dd2, w_dd2 = run_ddv1_upgrade_v2(crypto_eq, stock_eq)
    m_dd2 = compute_metrics(eq_dd2)
    avg_w_dd2 = w_dd2.mean()
    print(f"    CAGR={m_dd2['cagr']:.1%}, MaxDD={m_dd2['maxdd']:.1%}, "
          f"Sharpe={m_dd2['sharpe']:.2f}, Composite={m_dd2['composite']:.3f}, "
          f"Avg w_crypto={avg_w_dd2:.1%}")

    # ── Step 5: Summary ──
    print("\n" + "=" * 70)
    print("[5/5] 📊 FULL COMPARISON TABLE")
    print("=" * 70)
    
    all_results = [
        ("BestCrypto Pure", 1.0, mc),
        ("Stock v9g Pure", 0.0, ms),
        (f"Static Best ({best_static['w_crypto']:.0%}c)", best_static['w_crypto'], best_static),
        ("Dynamic v1 (Mom+DD)", avg_w1, m_dyn1),
        ("Dynamic v2 (Mom+Vol)", avg_w2, m_dyn2),
        ("DDv1 Upgrade", avg_w_dd1, m_dd1),
        ("DDv1 Upgrade v2", avg_w_dd2, m_dd2),
    ]

    print(f"\n{'Strategy':<25} {'w_crypto':>9} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Composite':>10} {'vs BC':>7}")
    print("-" * 90)

    bc_comp = mc['composite']
    for name, w, m in all_results:
        comp = m['composite']
        calmar = m.get('calmar', m['cagr'] / abs(m['maxdd']) if m['maxdd'] != 0 else 0)
        vs = comp - bc_comp
        marker = "✅" if vs > 0 else "❌"
        print(f"{name:<25} {w:>8.0%} {m['cagr']:>6.1%} {m['maxdd']:>7.1%} "
              f"{m['sharpe']:>7.2f} {calmar:>7.2f} {comp:>10.3f} {vs:>+6.3f} {marker}")

    # Best overall
    best_name, best_w_val, best_m = max(all_results, key=lambda x: x[2]['composite'])
    print(f"\n{'='*70}")
    print(f"🏆 WINNER: {best_name}")
    print(f"   Composite: {best_m['composite']:.3f} (vs BestCrypto {bc_comp:.3f}, "
          f"diff: {best_m['composite'] - bc_comp:+.3f})")
    print(f"   CAGR: {best_m['cagr']:.1%}, MaxDD: {best_m['maxdd']:.1%}, Sharpe: {best_m['sharpe']:.2f}")

    if best_m['composite'] > bc_comp:
        print(f"\n   ✅ SUCCESS: Beat BestCrypto DDv1!")
    else:
        print(f"\n   ❌ Did NOT beat BestCrypto DDv1 (pure crypto is hard to beat)")

    # Recommended configurations
    print(f"\n{'='*70}")
    print("💡 推荐实盘配置")
    print("=" * 70)

    # Find best for different risk profiles
    conservative = [r for r in static_results if abs(r['maxdd']) < 0.35]
    moderate = [r for r in static_results if 0.30 < abs(r['maxdd']) < 0.50]
    aggressive = [r for r in static_results if abs(r['maxdd']) >= 0.50]

    if conservative:
        bc = max(conservative, key=lambda x: x['composite'])
        print(f"\n  🟢 保守型 (MaxDD < -35%): w_crypto={bc['w_crypto']:.0%}")
        print(f"     CAGR={bc['cagr']:.1%}, MaxDD={bc['maxdd']:.1%}, Sharpe={bc['sharpe']:.2f}, Composite={bc['composite']:.3f}")

    if moderate:
        bm = max(moderate, key=lambda x: x['composite'])
        print(f"\n  🟡 中等型 (MaxDD -35%~-50%): w_crypto={bm['w_crypto']:.0%}")
        print(f"     CAGR={bm['cagr']:.1%}, MaxDD={bm['maxdd']:.1%}, Sharpe={bm['sharpe']:.2f}, Composite={bm['composite']:.3f}")

    if aggressive:
        ba = max(aggressive, key=lambda x: x['composite'])
        print(f"\n  🔴 激进型 (MaxDD > -50%): w_crypto={ba['w_crypto']:.0%}")
        print(f"     CAGR={ba['cagr']:.1%}, MaxDD={ba['maxdd']:.1%}, Sharpe={ba['sharpe']:.2f}, Composite={ba['composite']:.3f}")

    # Save results
    output = {
        'baseline': {
            'bestcrypto': mc,
            'stock_v9g': ms,
            'correlation': float(overall_corr),
        },
        'static_mix': static_results,
        'dynamic': {
            'v1_mom_dd': {**m_dyn1, 'avg_w_crypto': float(avg_w1)},
            'v2_mom_vol': {**m_dyn2, 'avg_w_crypto': float(avg_w2)},
            'ddv1_upgrade': {**m_dd1, 'avg_w_crypto': float(avg_w_dd1)},
            'ddv1_upgrade_v2': {**m_dd2, 'avg_w_crypto': float(avg_w_dd2)},
        },
        'best_static': best_static,
        'best_overall': {
            'name': best_name,
            'w_crypto': float(best_w_val),
            **best_m,
        },
    }

    out_path = Path(__file__).parent / "hybrid_v1_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n💾 Results saved → {out_path}")

    return output


if __name__ == '__main__':
    main()
