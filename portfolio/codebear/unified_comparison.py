"""
Unified Comparison — All Crypto Strategies + Portfolio Combinations
统一口径对比：相同数据范围、年化方式、指标计算

对比对象：
1. BestCrypto (原版，crypto_dual_momentum.py)
2. CryptoDualMom (原版)
3. ETH2 (新版，beast_eth2_cryptodm.py)
4. BTC v7f (baseline)
5. 各自 + Stock v3b 的 Portfolio 组合

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "btc/codebear"))

from crypto_dual_momentum import (
    load_data, strategy_best_crypto, strategy_crypto_dual_mom,
    strategy_btc_only_v7f, strategy_crypto_vol_weighted
)

# ── metrics (unified: 365.25 annualization, matching original) ──

def calc_metrics(equity, rf=0.04):
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    years = n / 365.25

    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = dd.min()

    ann_ret = np.mean(returns) * 365.25
    ann_vol = np.std(returns) * np.sqrt(365.25)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0

    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar,
            "final_value": equity[-1], "ann_vol": ann_vol, "years": years}

def composite_score(m):
    return m["sharpe"] * 0.4 + m["calmar"] * 0.4 + min(m["cagr"], 1.0) * 0.2

def walk_forward(equity, split=0.6):
    n = len(equity)
    s = int(n * split)
    m_is = calc_metrics(equity[:s+1])
    m_oos = calc_metrics(equity[s:])
    wfr = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
    return m_is, m_oos, wfr

def equity_to_returns(equity):
    """Convert equity curve to daily returns array."""
    r = np.diff(equity) / equity[:-1]
    r[~np.isfinite(r)] = 0
    return r

# ── portfolio helpers ──

def portfolio_risk_parity(r1, r2, lookback=252):
    n = len(r1)
    port = np.zeros(n)
    for i in range(lookback, n):
        v1 = np.std(r1[i-lookback:i]) * np.sqrt(365.25)
        v2 = np.std(r2[i-lookback:i]) * np.sqrt(365.25)
        inv1 = 1 / max(v1, 0.01)
        inv2 = 1 / max(v2, 0.01)
        w1 = inv1 / (inv1 + inv2)
        port[i] = w1 * r1[i] + (1 - w1) * r2[i]
    port[:lookback] = 0.5 * r1[:lookback] + 0.5 * r2[:lookback]
    return port

def portfolio_dd_responsive(r1, r2, gld_r):
    n = len(r1)
    port = np.zeros(n)
    eq = np.ones(n + 1)
    for i in range(n):
        peak = eq[:i+1].max()
        dd = (eq[i] - peak) / peak if peak > 0 else 0
        if dd < -0.15:
            port[i] = 0.20 * r1[i] + 0.30 * r2[i] + 0.40 * gld_r[i]
        elif dd < -0.10:
            port[i] = 0.25 * r1[i] + 0.40 * r2[i] + 0.25 * gld_r[i]
        elif dd < -0.05:
            port[i] = 0.30 * r1[i] + 0.50 * r2[i] + 0.15 * gld_r[i]
        else:
            port[i] = 0.40 * r1[i] + 0.60 * r2[i]
        eq[i+1] = eq[i] * (1 + port[i])
    return port

def portfolio_fixed(r1, r2, w1):
    return w1 * r1 + (1 - w1) * r2

def returns_to_equity(r, initial=10000):
    eq = np.zeros(len(r) + 1)
    eq[0] = initial
    for i, ret in enumerate(r):
        eq[i+1] = eq[i] * (1 + ret)
    return eq

# ── main ──

def main():
    # Load raw price data
    btc_df, eth_df, gld_df, dates = load_data()
    btc_c = btc_df["Close"].values
    eth_c = eth_df["Close"].values
    gld_c = gld_df["Close"].values

    # Load stock v3b returns
    stock_ret_df = pd.read_csv(BASE / "stocks/codebear/v3b_daily_returns.csv",
                               parse_dates=["Date"]).set_index("Date")

    # Run crypto strategies (full date range: 2015-08 → 2026-02)
    eq_bestcrypto = strategy_best_crypto(btc_c, eth_c, gld_c, dates)
    eq_cryptodm   = strategy_crypto_dual_mom(btc_c, eth_c, gld_c, dates)
    eq_v7f        = strategy_btc_only_v7f(btc_c, gld_c, dates)
    eq_volvgt     = strategy_crypto_vol_weighted(btc_c, eth_c, gld_c, dates)

    crypto_strategies = {
        "BestCrypto":    eq_bestcrypto,
        "CryptoDualMom": eq_cryptodm,
        "CryptoVolWgt":  eq_volvgt,
        "BTC v7f":       eq_v7f,
    }

    # ═══════════════════════════════════════════════
    print("=" * 110)
    print("PART 1: SINGLE CRYPTO STRATEGIES (Full Range)")
    print(f"Data: {dates[0].date()} → {dates[-1].date()} ({len(dates)} days)")
    print("=" * 110)

    print(f"\n{'Strategy':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Comp':>7}"
          f" {'IS_Sh':>7} {'OOS_Sh':>7} {'WF_r':>6} {'Pass':>4}")
    print("-" * 100)

    for name, eq in crypto_strategies.items():
        m = calc_metrics(eq)
        m_is, m_oos, wfr = walk_forward(eq)
        p = "✅" if wfr >= 0.70 else "❌"
        s = composite_score(m)
        print(f"{name:<18} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}%"
              f" {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {s:>7.3f}"
              f" {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")

    # ═══════════════════════════════════════════════
    # PART 2: Portfolio combos (need common dates with stock)
    # ═══════════════════════════════════════════════

    # Build crypto equity on same dates index
    crypto_dates_series = pd.Series(range(len(dates)), index=dates)

    # GLD returns on crypto dates
    gld_rets_full = np.diff(gld_c) / gld_c[:-1]
    gld_rets_full = np.insert(gld_rets_full, 0, 0)

    # Find common dates between crypto and stock
    common = dates.intersection(stock_ret_df.index).sort_values()
    print(f"\n{'='*110}")
    print(f"PART 2: PORTFOLIO COMBINATIONS (Crypto + Stock v3b)")
    print(f"Common range: {common[0].date()} → {common[-1].date()} ({len(common)} days)")
    print(f"{'='*110}")

    # Get stock returns on common dates
    stock_r = stock_ret_df.loc[common, "Return"].values

    # Get GLD returns on common dates
    gld_price_series = pd.Series(gld_c, index=dates)
    gld_common = gld_price_series.loc[common].values
    gld_r = np.diff(gld_common) / gld_common[:-1]
    gld_r = np.insert(gld_r, 0, 0)

    # Get crypto equity on common dates → returns
    crypto_returns = {}
    for name, eq in crypto_strategies.items():
        eq_series = pd.Series(eq, index=dates)
        eq_common = eq_series.loc[common].values
        r = np.diff(eq_common) / eq_common[:-1]
        r = np.insert(r, 0, 0)
        r[~np.isfinite(r)] = 0
        crypto_returns[name] = r

    # Correlations
    print(f"\n📊 Correlations with Stock v3b:")
    for name, r in crypto_returns.items():
        c = np.corrcoef(r[1:], stock_r[1:])[0, 1]
        print(f"   {name:<18}: {c:.3f}")

    # Portfolio results header
    print(f"\n{'Strategy':<35} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Comp':>7}"
          f" {'IS_Sh':>7} {'OOS_Sh':>7} {'WF_r':>6} {'Pass':>4}")
    print("-" * 115)

    # Solo baselines on common range
    for name, r in [("Stock v3b Solo", stock_r)] + [(n, cr) for n, cr in crypto_returns.items()]:
        eq = returns_to_equity(r)
        m = calc_metrics(eq)
        m_is, m_oos, wfr = walk_forward(eq)
        p = "✅" if wfr >= 0.70 else "❌"
        s = composite_score(m)
        print(f"{name:<35} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}%"
              f" {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {s:>7.3f}"
              f" {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")

    print("-" * 115)

    # Test each crypto strategy combined with Stock
    all_portfolios = {}

    for crypto_name, crypto_r in crypto_returns.items():
        combos = {}

        # Fixed weights
        for w in [0.40, 0.50, 0.60]:
            pname = f"{crypto_name} Fixed_{int(w*100)}/{int((1-w)*100)}"
            combos[pname] = portfolio_fixed(crypto_r, stock_r, w)

        # Risk Parity
        for lb in [126, 252]:
            pname = f"{crypto_name} RP_{lb}d"
            combos[pname] = portfolio_risk_parity(crypto_r, stock_r, lb)

        # DD Responsive
        pname = f"{crypto_name} DD_Resp"
        combos[pname] = portfolio_dd_responsive(crypto_r, stock_r, gld_r)

        all_portfolios.update(combos)

    # Sort by composite and print
    scored = []
    for name, r in all_portfolios.items():
        eq = returns_to_equity(r)
        m = calc_metrics(eq)
        m_is, m_oos, wfr = walk_forward(eq)
        s = composite_score(m)
        scored.append((name, m, m_is, m_oos, wfr, s))

    scored.sort(key=lambda x: x[5], reverse=True)

    for name, m, m_is, m_oos, wfr, s in scored:
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"{name:<35} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}%"
              f" {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {s:>7.3f}"
              f" {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")

    # ═══════════════════════════════════════════════
    # PART 3: Multi-split WF for top 5
    # ═══════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PART 3: WALK-FORWARD MULTI-SPLIT (Top 5)")
    print(f"{'='*110}")

    splits = [0.50, 0.55, 0.60, 0.65, 0.70]
    top5 = scored[:5]

    for name, m, _, _, _, s in top5:
        r = all_portfolios[name]
        eq = returns_to_equity(r)
        ratios = []
        for sp in splits:
            _, _, wfr = walk_forward(eq, sp)
            ratios.append(wfr)
        avg = np.mean(ratios)
        detail = " | ".join([f"{sp:.0%}:{r:.2f}" for sp, r in zip(splits, ratios)])
        passed = sum(1 for r in ratios if r >= 0.70)
        print(f"  {name:<35}: {detail} | avg={avg:.2f} ({passed}/{len(splits)} pass)")

    # ═══════════════════════════════════════════════
    # PART 4: Recent 3-month performance
    # ═══════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PART 4: RECENT 3-MONTH PERFORMANCE")
    print(f"{'='*110}")

    cutoff = common[-1] - pd.DateOffset(months=3)
    mask = common >= cutoff

    print(f"\n{'Strategy':<35} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8}")
    print("-" * 65)

    # Stock solo
    eq = returns_to_equity(stock_r[mask])
    m = calc_metrics(eq)
    print(f"{'Stock v3b Solo':<35} {m['cagr']*100:>7.1f}% {m['sharpe']:>7.2f} {m['maxdd']*100:>7.1f}%")

    # Top 5 portfolios
    for name, _, _, _, _, s in top5:
        r = all_portfolios[name]
        eq = returns_to_equity(r[mask])
        m = calc_metrics(eq)
        print(f"{name:<35} {m['cagr']*100:>7.1f}% {m['sharpe']:>7.2f} {m['maxdd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════
    # PART 5: Final Summary
    # ═══════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("🏆 FINAL RANKINGS (统一口径)")
    print(f"{'='*110}")

    # Combine solo crypto (full range) + portfolio combos (common range)
    print("\n📌 Solo Crypto (Full Range 2015-2026):")
    for name, eq in sorted(crypto_strategies.items(),
                           key=lambda x: composite_score(calc_metrics(x[1])), reverse=True):
        m = calc_metrics(eq)
        s = composite_score(m)
        print(f"   {name:<18}: Composite {s:.3f} | CAGR {m['cagr']*100:.1f}% | MaxDD {m['maxdd']*100:.1f}% | Sharpe {m['sharpe']:.2f}")

    print(f"\n📌 Top 5 Portfolio Combos (Common Range {common[0].date()} → {common[-1].date()}):")
    for i, (name, m, m_is, m_oos, wfr, s) in enumerate(top5, 1):
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"   {i}. {name:<35}: Composite {s:.3f} | CAGR {m['cagr']*100:.1f}% | MaxDD {m['maxdd']*100:.1f}% | Sharpe {m['sharpe']:.2f} | WF {wfr:.2f} {p}")

    print(f"\n{'='*110}")

if __name__ == "__main__":
    main()
