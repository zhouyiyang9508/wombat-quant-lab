"""
FINAL Unified Comparison — All strategies on same footing
统一口径对比，包含所有历史最佳策略

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "btc/codebear"))
sys.path.insert(0, str(BASE / "portfolio/codebear"))

from crypto_dual_momentum import (
    load_data, strategy_best_crypto, strategy_crypto_dual_mom,
    strategy_btc_only_v7f
)

# ── unified metrics (365.25 annualization) ──

def calc_metrics(equity, rf=0.04, ann_factor=252):
    """ann_factor: 252 for trading days, 365.25 for calendar days."""
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    years = n / ann_factor
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = dd.min()
    ann_ret = np.mean(returns) * ann_factor
    ann_vol = np.std(returns) * np.sqrt(ann_factor)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar,
            "ann_vol": ann_vol, "years": years}

def composite_score(m):
    return m["sharpe"] * 0.4 + m["calmar"] * 0.4 + min(m["cagr"], 1.0) * 0.2

def walk_forward(equity, split=0.6, ann_factor=252):
    n = len(equity)
    s = int(n * split)
    m_is = calc_metrics(equity[:s+1], ann_factor=ann_factor)
    m_oos = calc_metrics(equity[s:], ann_factor=ann_factor)
    wfr = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
    return m_is, m_oos, wfr

def returns_to_equity(r, initial=10000):
    eq = np.zeros(len(r) + 1)
    eq[0] = initial
    for i, ret in enumerate(r):
        eq[i+1] = eq[i] * (1 + ret)
    return eq

# ── portfolio methods ──

def portfolio_fixed(r1, r2, w1):
    return w1 * r1 + (1 - w1) * r2

def portfolio_rp(r1, r2, lookback=252):
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

def portfolio_dd_responsive_v1(crypto_r, stock_r, lookback=20, dd_threshold=-0.10):
    """Original DD Responsive from portfolio_v2_analysis.py
    逆波动率基础 + 单策略回撤时减权（不加GLD，允许现金仓位）"""
    n = len(crypto_r)
    port_r = np.zeros(n)
    
    crypto_cum = np.cumprod(1 + crypto_r)
    stock_cum = np.cumprod(1 + stock_r)
    
    for i in range(lookback, n):
        # Current drawdown of each strategy
        crypto_peak = crypto_cum[:i+1].max()
        stock_peak = stock_cum[:i+1].max()
        crypto_dd = (crypto_cum[i] - crypto_peak) / crypto_peak
        stock_dd = (stock_cum[i] - stock_peak) / stock_peak
        
        # Base: inverse vol
        crypto_vol = np.std(crypto_r[i-lookback:i]) * np.sqrt(252)
        stock_vol = np.std(stock_r[i-lookback:i]) * np.sqrt(252)
        crypto_vol = max(crypto_vol, 0.01)
        stock_vol = max(stock_vol, 0.01)
        
        inv_c = 1.0 / crypto_vol
        inv_s = 1.0 / stock_vol
        total = inv_c + inv_s
        w_c = inv_c / total
        w_s = inv_s / total
        
        # If in deep drawdown, reduce weight (allow cash)
        if crypto_dd < dd_threshold:
            reduction = max(0.3, 1.0 + crypto_dd * 2)
            w_c *= reduction
        if stock_dd < dd_threshold:
            reduction = max(0.3, 1.0 + stock_dd * 2)
            w_s *= reduction
        
        port_r[i] = w_c * crypto_r[i] + w_s * stock_r[i]
    
    port_r[:lookback] = 0.5 * crypto_r[:lookback] + 0.5 * stock_r[:lookback]
    return port_r

def portfolio_dd_responsive_v2(crypto_r, stock_r, gld_r):
    """DD Responsive v2: 组合回撤时切换到 GLD"""
    n = len(crypto_r)
    port_r = np.zeros(n)
    eq = np.ones(n + 1)
    for i in range(n):
        peak = eq[:i+1].max()
        dd = (eq[i] - peak) / peak if peak > 0 else 0
        if dd < -0.15:
            port_r[i] = 0.20 * crypto_r[i] + 0.30 * stock_r[i] + 0.40 * gld_r[i]
        elif dd < -0.10:
            port_r[i] = 0.25 * crypto_r[i] + 0.40 * stock_r[i] + 0.25 * gld_r[i]
        elif dd < -0.05:
            port_r[i] = 0.30 * crypto_r[i] + 0.50 * stock_r[i] + 0.15 * gld_r[i]
        else:
            port_r[i] = 0.40 * crypto_r[i] + 0.60 * stock_r[i]
        eq[i+1] = eq[i] * (1 + port_r[i])
    return port_r


def main():
    # Load raw data
    btc_df, eth_df, gld_df, dates = load_data()
    btc_c = btc_df["Close"].values
    eth_c = eth_df["Close"].values
    gld_c = gld_df["Close"].values

    # Load stock v3b
    stock_ret_df = pd.read_csv(BASE / "stocks/codebear/v3b_daily_returns.csv",
                               parse_dates=["Date"]).set_index("Date")

    # Crypto strategies (full range)
    eq_bestcrypto = strategy_best_crypto(btc_c, eth_c, gld_c, dates)
    eq_cryptodm   = strategy_crypto_dual_mom(btc_c, eth_c, gld_c, dates)
    eq_v7f        = strategy_btc_only_v7f(btc_c, gld_c, dates)

    # Common dates for portfolio
    common = dates.intersection(stock_ret_df.index).sort_values()
    stock_r = stock_ret_df.loc[common, "Return"].values

    # GLD returns
    gld_series = pd.Series(gld_c, index=dates)
    gld_common = gld_series.loc[common].values
    gld_r = np.diff(gld_common) / gld_common[:-1]
    gld_r = np.insert(gld_r, 0, 0)

    # Crypto returns on common dates
    def get_returns(eq):
        eq_s = pd.Series(eq, index=dates)
        eq_c = eq_s.loc[common].values
        r = np.diff(eq_c) / eq_c[:-1]
        r = np.insert(r, 0, 0)
        r[~np.isfinite(r)] = 0
        return r

    bestcrypto_r = get_returns(eq_bestcrypto)
    cryptodm_r = get_returns(eq_cryptodm)
    v7f_r = get_returns(eq_v7f)

    # Portfolio data is on trading days → use 252 annualization
    ANN = 252

    print("=" * 115)
    print("FINAL UNIFIED COMPARISON — ALL STRATEGIES (252-day annualization)")
    print(f"Portfolio range: {common[0].date()} → {common[-1].date()} ({len(common)} trading days)")
    print("=" * 115)

    # Correlations
    print(f"\n📊 Correlations with Stock v3b:")
    for name, r in [("BestCrypto", bestcrypto_r), ("CryptoDualMom", cryptodm_r), ("BTC v7f", v7f_r)]:
        c = np.corrcoef(r[1:], stock_r[1:])[0, 1]
        print(f"   {name:<18}: {c:.3f}")

    # Build ALL portfolio combinations
    all_portfolios = {}

    for crypto_name, crypto_r in [("BestCrypto", bestcrypto_r), ("CryptoDualMom", cryptodm_r), ("BTC v7f", v7f_r)]:
        # Fixed
        for w in [0.40, 0.50, 0.60]:
            pname = f"{crypto_name} Fix_{int(w*100)}/{int((1-w)*100)}"
            all_portfolios[pname] = portfolio_fixed(crypto_r, stock_r, w)

        # RP
        for lb in [126, 252]:
            pname = f"{crypto_name} RP_{lb}d"
            all_portfolios[pname] = portfolio_rp(crypto_r, stock_r, lb)

        # DD Responsive v1 (original — 逆vol + 策略DD减权)
        pname = f"{crypto_name} DDv1"
        all_portfolios[pname] = portfolio_dd_responsive_v1(crypto_r, stock_r)

        # DD Responsive v2 (GLD hedge on portfolio DD)
        pname = f"{crypto_name} DDv2_GLD"
        all_portfolios[pname] = portfolio_dd_responsive_v2(crypto_r, stock_r, gld_r)

    # Score all
    scored = []
    for name, r in all_portfolios.items():
        eq = returns_to_equity(r)
        m = calc_metrics(eq, ann_factor=ANN)
        m_is, m_oos, wfr = walk_forward(eq, ann_factor=ANN)
        s = composite_score(m)
        scored.append((name, m, m_is, m_oos, wfr, s))

    scored.sort(key=lambda x: x[5], reverse=True)

    # Solo baselines
    print(f"\n{'Strategy':<32} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Comp':>7}"
          f" {'IS_Sh':>7} {'OOS_Sh':>7} {'WF':>6} {'Pass':>4}")
    print("-" * 115)

    for name, r in [("Stock v3b Solo", stock_r), ("BestCrypto Solo", bestcrypto_r),
                     ("CryptoDualMom Solo", cryptodm_r), ("BTC v7f Solo", v7f_r)]:
        eq = returns_to_equity(r)
        m = calc_metrics(eq, ann_factor=ANN)
        m_is, m_oos, wfr = walk_forward(eq, ann_factor=ANN)
        s = composite_score(m)
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"{name:<32} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}%"
              f" {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {s:>7.3f}"
              f" {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")

    print("-" * 115)

    # All portfolio combos (top 20)
    for name, m, m_is, m_oos, wfr, s in scored[:20]:
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"{name:<32} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}%"
              f" {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {s:>7.3f}"
              f" {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wfr:>6.2f} {p:>4}")

    # Multi-split WF for top 5
    print(f"\n{'='*115}")
    print("WALK-FORWARD MULTI-SPLIT (Top 5)")
    print(f"{'='*115}")

    splits = [0.50, 0.55, 0.60, 0.65, 0.70]
    for name, m, _, _, _, s in scored[:5]:
        r = all_portfolios[name]
        eq = returns_to_equity(r)
        ratios = []
        for sp in splits:
            _, _, wfr = walk_forward(eq, sp, ann_factor=ANN)
            ratios.append(wfr)
        avg = np.mean(ratios)
        passed = sum(1 for r in ratios if r >= 0.70)
        detail = " | ".join([f"{sp:.0%}:{r:.2f}" for sp, r in zip(splits, ratios)])
        print(f"  {name:<32}: {detail} | avg={avg:.2f} ({passed}/{len(splits)} pass)")

    # Final ranking
    print(f"\n{'='*115}")
    print("🏆 FINAL UNIFIED RANKINGS")
    print(f"{'='*115}\n")
    for i, (name, m, m_is, m_oos, wfr, s) in enumerate(scored[:10], 1):
        p = "✅" if wfr >= 0.70 else "❌"
        print(f"  {i:>2}. {name:<32} Comp={s:.3f} | CAGR {m['cagr']*100:.1f}% | "
              f"MaxDD {m['maxdd']*100:.1f}% | Sharpe {m['sharpe']:.2f} | WF {wfr:.2f} {p}")

    print(f"\n{'='*115}")

if __name__ == "__main__":
    main()
