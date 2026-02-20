"""
BestCrypto Robustness Test — Exclude ETH best year analysis
Author: 代码熊 🐻 | 2026-02-20
"""
import pandas as pd
import numpy as np
from pathlib import Path
from crypto_dual_momentum import load_data, calc_metrics, composite_score, strategy_best_crypto

def strategy_best_crypto_exclude_eth_year(btc_close, eth_close, gld_close, dates, exclude_year):
    """BestCrypto but force ETH out in exclude_year (only pick BTC or GLD)."""
    n = len(btc_close)
    equity = np.zeros(n)
    equity[0] = 10000.0
    
    for i in range(1, n):
        btc_ret = btc_close[i] / btc_close[i-1] - 1
        eth_ret = eth_close[i] / eth_close[i-1] - 1
        gld_ret = gld_close[i] / gld_close[i-1] - 1
        
        lb = min(i, 90)
        btc_mom = btc_close[i] / btc_close[max(0, i-lb)] - 1
        eth_mom = eth_close[i] / eth_close[max(0, i-lb)] - 1
        gld_mom = gld_close[i] / gld_close[max(0, i-lb)] - 1
        
        yr = dates[i].year
        
        if yr == exclude_year:
            # Only BTC vs GLD
            moms = {"btc": btc_mom, "gld": gld_mom}
        else:
            moms = {"btc": btc_mom, "eth": eth_mom, "gld": gld_mom}
        
        best = max(moms, key=moms.get)
        
        if best == "btc" and btc_mom > 0:
            w_btc, w_eth, w_gld = 0.70, 0.15, 0.10
        elif best == "eth" and eth_mom > 0:
            w_btc, w_eth, w_gld = 0.15, 0.65, 0.10
        elif best == "gld":
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.55
        else:
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.35
        
        # In exclude year, zero out ETH
        if yr == exclude_year:
            if best == "eth":
                # Fallback: pick btc or gld
                if btc_mom > gld_mom and btc_mom > 0:
                    w_btc, w_eth, w_gld = 0.70, 0.0, 0.10
                else:
                    w_btc, w_eth, w_gld = 0.15, 0.0, 0.55
            else:
                w_eth = 0.0
        
        port_ret = w_btc * btc_ret + w_eth * eth_ret + w_gld * gld_ret
        equity[i] = equity[i-1] * (1 + port_ret)
    
    return equity

def strategy_best_crypto_skip_year(btc_close, eth_close, gld_close, dates, skip_year):
    """BestCrypto but skip an entire year (hold cash = 0% return)."""
    n = len(btc_close)
    equity = np.zeros(n)
    equity[0] = 10000.0
    
    for i in range(1, n):
        yr = dates[i].year
        if yr == skip_year:
            equity[i] = equity[i-1]  # flat
            continue
            
        btc_ret = btc_close[i] / btc_close[i-1] - 1
        eth_ret = eth_close[i] / eth_close[i-1] - 1
        gld_ret = gld_close[i] / gld_close[i-1] - 1
        
        lb = min(i, 90)
        btc_mom = btc_close[i] / btc_close[max(0, i-lb)] - 1
        eth_mom = eth_close[i] / eth_close[max(0, i-lb)] - 1
        gld_mom = gld_close[i] / gld_close[max(0, i-lb)] - 1
        
        moms = {"btc": btc_mom, "eth": eth_mom, "gld": gld_mom}
        best = max(moms, key=moms.get)
        
        if best == "btc" and btc_mom > 0:
            w_btc, w_eth, w_gld = 0.70, 0.15, 0.10
        elif best == "eth" and eth_mom > 0:
            w_btc, w_eth, w_gld = 0.15, 0.65, 0.10
        elif best == "gld":
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.55
        else:
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.35
        
        port_ret = w_btc * btc_ret + w_eth * eth_ret + w_gld * gld_ret
        equity[i] = equity[i-1] * (1 + port_ret)
    
    return equity

def main():
    btc, eth, gld, dates = load_data()
    
    # Filter to 2018+ for consistency with STRATEGY_REPORT
    mask = dates >= pd.Timestamp("2018-01-01")
    btc_c = btc["Close"].values[mask]
    eth_c = eth["Close"].values[mask]
    gld_c = gld["Close"].values[mask]
    dates_f = dates[mask]
    
    print(f"Period: {dates_f[0].date()} → {dates_f[-1].date()} ({len(dates_f)} days)\n")
    
    # === Table 1: Yearly returns ===
    print("=" * 70)
    print("TABLE 1: Yearly Asset Returns")
    print("=" * 70)
    print(f"{'Year':<6} {'ETH':>10} {'BTC':>10} {'GLD':>10}")
    print("-" * 40)
    
    yearly_eth = {}
    for yr in range(2018, 2026):
        yr_mask = np.array([d.year == yr for d in dates_f])
        if yr_mask.sum() < 10:
            continue
        idx = np.where(yr_mask)[0]
        eth_yr = eth_c[idx[-1]] / eth_c[idx[0]] - 1
        btc_yr = btc_c[idx[-1]] / btc_c[idx[0]] - 1
        gld_yr = gld_c[idx[-1]] / gld_c[idx[0]] - 1
        yearly_eth[yr] = eth_yr
        print(f"{yr:<6} {eth_yr*100:>+9.1f}% {btc_yr*100:>+9.1f}% {gld_yr*100:>+9.1f}%")
    
    best_eth_year = max(yearly_eth, key=yearly_eth.get)
    print(f"\n🔥 ETH best year: {best_eth_year} ({yearly_eth[best_eth_year]*100:+.1f}%)")
    
    # === Baseline: Full BestCrypto ===
    eq_full = strategy_best_crypto(btc_c, eth_c, gld_c, dates_f)
    m_full = calc_metrics(eq_full)
    cs_full = composite_score(m_full)
    
    # Apply 0.5% annual cost
    cost_factor = (1 - 0.005) ** m_full["years"]
    
    print(f"\n{'=' * 70}")
    print("TABLE 2: Exclude ETH in Best Year")
    print(f"{'=' * 70}")
    
    # === Scheme 1: Exclude ETH in best year ===
    eq_excl = strategy_best_crypto_exclude_eth_year(btc_c, eth_c, gld_c, dates_f, best_eth_year)
    m_excl = calc_metrics(eq_excl)
    cs_excl = composite_score(m_excl)
    
    print(f"{'Version':<25} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Composite':>10}")
    print("-" * 70)
    print(f"{'Full BestCrypto':<25} {m_full['cagr']*100:>7.1f}% {m_full['maxdd']*100:>7.1f}% {m_full['sharpe']:>7.2f} {m_full['calmar']:>7.2f} {cs_full:>10.3f}")
    print(f"{'Excl ETH '+str(best_eth_year):<25} {m_excl['cagr']*100:>7.1f}% {m_excl['maxdd']*100:>7.1f}% {m_excl['sharpe']:>7.2f} {m_excl['calmar']:>7.2f} {cs_excl:>10.3f}")
    print(f"{'Delta':<25} {(m_excl['cagr']-m_full['cagr'])*100:>+7.1f}% {(m_excl['maxdd']-m_full['maxdd'])*100:>+7.1f}% {m_excl['sharpe']-m_full['sharpe']:>+7.2f} {m_excl['calmar']-m_full['calmar']:>+7.2f} {cs_excl-cs_full:>+10.3f}")
    
    # Also exclude both 2020 and 2021
    eq_excl2 = strategy_best_crypto_exclude_eth_year(btc_c, eth_c, gld_c, dates_f, 2020)
    # Chain: also exclude 2021
    eq_excl_both = strategy_best_crypto_exclude_eth_years(btc_c, eth_c, gld_c, dates_f, [2020, 2021])
    m_excl_both = calc_metrics(eq_excl_both)
    cs_excl_both = composite_score(m_excl_both)
    print(f"{'Excl ETH 2020+2021':<25} {m_excl_both['cagr']*100:>7.1f}% {m_excl_both['maxdd']*100:>7.1f}% {m_excl_both['sharpe']:>7.2f} {m_excl_both['calmar']:>7.2f} {cs_excl_both:>10.3f}")
    
    # === Scheme 3: Leave-one-year-out ===
    print(f"\n{'=' * 70}")
    print("TABLE 3: Leave-One-Year-Out Sensitivity")
    print(f"{'=' * 70}")
    print(f"{'Skip Year':<12} {'CAGR':>8} {'Sharpe':>7} {'Composite':>10} {'Impact':>10}")
    print("-" * 50)
    
    for yr in range(2018, 2026):
        eq_skip = strategy_best_crypto_skip_year(btc_c, eth_c, gld_c, dates_f, yr)
        m_skip = calc_metrics(eq_skip)
        cs_skip = composite_score(m_skip)
        impact = cs_skip - cs_full
        bar = "📉" if impact < -0.3 else ("📈" if impact > 0.1 else "—")
        print(f"Skip {yr:<8} {m_skip['cagr']*100:>7.1f}% {m_skip['sharpe']:>7.2f} {cs_skip:>10.3f} {impact:>+9.3f} {bar}")
    
    # === Key conclusions ===
    print(f"\n{'=' * 70}")
    print("KEY CONCLUSIONS")
    print(f"{'=' * 70}")
    print(f"1. ETH best year: {best_eth_year} ({yearly_eth[best_eth_year]*100:+.1f}%)")
    print(f"2. Full CAGR: {m_full['cagr']*100:.1f}% → Excl ETH {best_eth_year}: {m_excl['cagr']*100:.1f}% (Δ {(m_excl['cagr']-m_full['cagr'])*100:+.1f}%)")
    print(f"3. Full Sharpe: {m_full['sharpe']:.2f} → Excl: {m_excl['sharpe']:.2f} (still {'> 1.5 ✅' if m_excl['sharpe'] > 1.5 else '< 1.5 ❌'})")
    print(f"4. Full Composite: {cs_full:.3f} → Excl: {cs_excl:.3f} (still {'> 1.5 ✅' if cs_excl > 1.5 else '< 1.5 ❌'})")
    print(f"5. Stock v4d Composite = 1.356 → BestCrypto excl still {'beats ✅' if cs_excl > 1.356 else 'loses ❌'}")
    
    return m_full, m_excl, cs_full, cs_excl, best_eth_year, yearly_eth

def strategy_best_crypto_exclude_eth_years(btc_close, eth_close, gld_close, dates, exclude_years):
    """BestCrypto but force ETH out in multiple years."""
    n = len(btc_close)
    equity = np.zeros(n)
    equity[0] = 10000.0
    
    for i in range(1, n):
        btc_ret = btc_close[i] / btc_close[i-1] - 1
        eth_ret = eth_close[i] / eth_close[i-1] - 1
        gld_ret = gld_close[i] / gld_close[i-1] - 1
        
        lb = min(i, 90)
        btc_mom = btc_close[i] / btc_close[max(0, i-lb)] - 1
        eth_mom = eth_close[i] / eth_close[max(0, i-lb)] - 1
        gld_mom = gld_close[i] / gld_close[max(0, i-lb)] - 1
        
        yr = dates[i].year
        exclude = yr in exclude_years
        
        if exclude:
            moms = {"btc": btc_mom, "gld": gld_mom}
        else:
            moms = {"btc": btc_mom, "eth": eth_mom, "gld": gld_mom}
        
        best = max(moms, key=moms.get)
        
        if best == "btc" and btc_mom > 0:
            w_btc, w_eth, w_gld = 0.70, 0.0 if exclude else 0.15, 0.10
        elif best == "eth" and eth_mom > 0 and not exclude:
            w_btc, w_eth, w_gld = 0.15, 0.65, 0.10
        elif best == "gld":
            w_btc, w_eth, w_gld = 0.15, 0.0 if exclude else 0.10, 0.55
        else:
            w_btc, w_eth, w_gld = 0.15, 0.0 if exclude else 0.10, 0.35
        
        port_ret = w_btc * btc_ret + w_eth * eth_ret + w_gld * gld_ret
        equity[i] = equity[i-1] * (1 + port_ret)
    
    return equity

if __name__ == "__main__":
    main()
