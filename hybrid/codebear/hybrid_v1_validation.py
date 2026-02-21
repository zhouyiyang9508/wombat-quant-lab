#!/usr/bin/env python3
"""
Hybrid v1 Validation — Walk-Forward + Finer Granularity + Year-by-Year
代码熊 🐻 | 2026-02-21

Validates the hybrid v1 findings:
1. Finer static mix scan (1% step around optimal zone)
2. IS/OOS walk-forward for top strategies
3. Year-by-year breakdown
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Import from hybrid_v1
import sys
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_v1 import (
    generate_bestcrypto_daily_equity,
    generate_v9g_daily_equity,
    compute_metrics,
    run_static_mix,
    run_dynamic_mix_v1,
    run_dynamic_mix_v2,
    run_ddv1_upgrade_v2,
)


def walk_forward_split(crypto_eq, stock_eq, strategy_fn, split_date='2021-01-01'):
    """Run IS/OOS split for a strategy."""
    common = crypto_eq.index.intersection(stock_eq.index).sort_values()
    split = pd.Timestamp(split_date)

    is_crypto = crypto_eq.loc[common[common < split]]
    is_stock = stock_eq.loc[common[common < split]]
    oos_crypto = crypto_eq.loc[common[common >= split]]
    oos_stock = stock_eq.loc[common[common >= split]]

    return strategy_fn(is_crypto, is_stock), strategy_fn(oos_crypto, oos_stock)


def yearly_breakdown(eq):
    """Compute metrics per year."""
    results = {}
    for yr in range(eq.index[0].year, eq.index[-1].year + 1):
        mask = eq.index.year == yr
        if mask.sum() < 30:
            continue
        yr_eq = eq.loc[mask]
        # Normalize to start at first value
        yr_eq = yr_eq / yr_eq.iloc[0]
        ret = yr_eq.iloc[-1] / yr_eq.iloc[0] - 1
        peak = yr_eq.cummax()
        dd = ((yr_eq - peak) / peak).min()
        daily_rets = yr_eq.pct_change().dropna()
        vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 10 else 0
        results[yr] = {'return': float(ret), 'maxdd': float(dd), 'vol': float(vol)}
    return results


def main():
    print("=" * 70)
    print("🐻 Hybrid v1 — Validation & Robustness Check")
    print("=" * 70)

    # Generate curves
    print("\n[1/4] Loading equity curves...")
    crypto_eq = generate_bestcrypto_daily_equity()
    stock_eq = generate_v9g_daily_equity()

    # ── Finer static scan ──
    print("\n[2/4] Fine-grained static scan (30%~50% crypto, 1% step)...")
    fine_results = []
    for w_pct in range(25, 56):
        w = w_pct / 100.0
        eq = run_static_mix(crypto_eq, stock_eq, w)
        m = compute_metrics(eq)
        m['w_crypto'] = w
        fine_results.append(m)

    # Find optimal
    best = max(fine_results, key=lambda x: x['composite'])
    print(f"\n  📊 Fine-grained optimal: w_crypto = {best['w_crypto']:.0%}")
    print(f"     CAGR={best['cagr']:.1%}, MaxDD={best['maxdd']:.1%}, "
          f"Sharpe={best['sharpe']:.2f}, Composite={best['composite']:.3f}")

    # Show top-5
    top5 = sorted(fine_results, key=lambda x: -x['composite'])[:5]
    print(f"\n  Top-5 static mixes:")
    for r in top5:
        print(f"    w_crypto={r['w_crypto']:.0%}: CAGR={r['cagr']:.1%}, MaxDD={r['maxdd']:.1%}, "
              f"Sharpe={r['sharpe']:.2f}, Composite={r['composite']:.3f}")

    # ── Walk-Forward Validation ──
    print("\n[3/4] Walk-Forward Validation (IS: 2015-2020, OOS: 2021+)...")

    strategies = {
        f"Static {best['w_crypto']:.0%}c": lambda c, s: run_static_mix(c, s, best['w_crypto']),
        "Static 40%c": lambda c, s: run_static_mix(c, s, 0.40),
        "Static 35%c": lambda c, s: run_static_mix(c, s, 0.35),
        "Dynamic v1": lambda c, s: run_dynamic_mix_v1(c, s)[0],
        "Dynamic v2": lambda c, s: run_dynamic_mix_v2(c, s)[0],
        "DDv1 Upgrade v2": lambda c, s: run_ddv1_upgrade_v2(c, s)[0],
        "BestCrypto Pure": lambda c, s: c,
    }

    print(f"\n{'Strategy':<22} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'WF Ratio':>9} {'IS Comp':>8} {'OOS Comp':>9}")
    print("-" * 75)

    for name, fn in strategies.items():
        try:
            eq_is, eq_oos = walk_forward_split(crypto_eq, stock_eq, fn)
            m_is = compute_metrics(eq_is)
            m_oos = compute_metrics(eq_oos)
            wf = m_oos['sharpe'] / m_is['sharpe'] if m_is['sharpe'] > 0 else 0
            wf_mark = "✅" if wf >= 0.7 else ("⚠️" if wf >= 0.5 else "❌")
            print(f"{name:<22} {m_is['sharpe']:>10.2f} {m_oos['sharpe']:>11.2f} "
                  f"{wf:>8.2f} {wf_mark} {m_is['composite']:>7.3f} {m_oos['composite']:>9.3f}")
        except Exception as e:
            print(f"{name:<22} ERROR: {e}")

    # ── Year-by-Year Breakdown ──
    print("\n[4/4] Year-by-Year Breakdown...")

    eq_best_static = run_static_mix(crypto_eq, stock_eq, best['w_crypto'])
    eq_dyn1 = run_dynamic_mix_v1(crypto_eq, stock_eq)[0]

    common = crypto_eq.index.intersection(stock_eq.index).sort_values()
    crypto_aligned = crypto_eq.loc[common]
    stock_aligned = stock_eq.loc[common]

    yr_crypto = yearly_breakdown(crypto_aligned)
    yr_stock = yearly_breakdown(stock_aligned)
    yr_static = yearly_breakdown(eq_best_static)
    yr_dyn1 = yearly_breakdown(eq_dyn1)

    print(f"\n{'Year':>6} {'BestCrypto':>12} {'Stock v9g':>12} {'Static Best':>12} {'Dynamic v1':>12}")
    print("-" * 58)
    for yr in sorted(set(list(yr_crypto.keys()) + list(yr_stock.keys()))):
        cr = yr_crypto.get(yr, {}).get('return', float('nan'))
        sr = yr_stock.get(yr, {}).get('return', float('nan'))
        st = yr_static.get(yr, {}).get('return', float('nan'))
        dy = yr_dyn1.get(yr, {}).get('return', float('nan'))
        print(f"  {yr}  {cr:>+10.1%}  {sr:>+10.1%}  {st:>+10.1%}  {dy:>+10.1%}")

    print(f"\n{'Year':>6} {'BC MaxDD':>10} {'v9g MaxDD':>10} {'Static DD':>10} {'Dyn v1 DD':>10}")
    print("-" * 48)
    for yr in sorted(set(list(yr_crypto.keys()) + list(yr_stock.keys()))):
        cr = yr_crypto.get(yr, {}).get('maxdd', float('nan'))
        sr = yr_stock.get(yr, {}).get('maxdd', float('nan'))
        st = yr_static.get(yr, {}).get('maxdd', float('nan'))
        dy = yr_dyn1.get(yr, {}).get('maxdd', float('nan'))
        print(f"  {yr}  {cr:>9.1%}  {sr:>9.1%}  {st:>9.1%}  {dy:>9.1%}")

    # Save validation results
    output = {
        'fine_grained_optimal': best,
        'top5_static': top5,
        'fine_results': fine_results,
    }
    out_path = Path(__file__).parent / "hybrid_v1_validation_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n💾 Validation results → {out_path}")


if __name__ == '__main__':
    main()
