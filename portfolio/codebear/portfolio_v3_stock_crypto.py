"""
Portfolio v3: Stock v4d + BestCrypto Combination
=================================================
Combines the two highest-scoring strategy families:
  - Stock v4d (DD Responsive GLD Hedge): Composite 1.350, WF ✅
  - BestCrypto (BTC+ETH+GLD rotation): Composite 2.281, WF ✅

Key insight: Stock momentum and crypto momentum have very different drivers.
Stock v4d trades S&P 500 sector rotation; BestCrypto trades BTC/ETH/GLD.
Correlation should be low → massive diversification benefit.

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parents[2] / "data_cache"
STOCK_RETURNS = Path(__file__).resolve().parents[2] / "stocks" / "codebear" / "v3b_daily_returns.csv"

# ─── Helper functions ───

def calc_metrics(equity, rf=0.04):
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    years = n / 252  # trading days for mixed portfolio
    
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = dd.min()
    
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    
    win_rate = np.mean(returns > 0) * 100
    
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar,
            "final_value": equity[-1], "ann_vol": ann_vol, "years": years, "win_rate": win_rate}

def composite_score(m):
    return m["sharpe"] * 0.4 + m["calmar"] * 0.4 + min(m["cagr"], 1.0) * 0.2

def walk_forward(equity, dates, split=0.6, rf=0.04):
    n = len(equity)
    split_idx = int(n * split)
    is_m = calc_metrics(equity[:split_idx+1], rf)
    oos_m = calc_metrics(equity[split_idx:], rf)
    ratio = oos_m["sharpe"] / is_m["sharpe"] if is_m["sharpe"] > 0 else 0
    return is_m, oos_m, ratio

# ─── BestCrypto strategy (from crypto_dual_momentum.py) ───

def run_bestcrypto(lookback=90):
    btc = pd.read_csv(DATA_DIR / "BTC_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    eth = pd.read_csv(DATA_DIR / "ETH_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    gld = pd.read_csv(DATA_DIR / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    
    common = btc.index.intersection(eth.index).intersection(gld.index).sort_values()
    btc, eth, gld = btc[common], eth[common], gld[common]
    
    btc_mom = btc.pct_change(lookback)
    eth_mom = eth.pct_change(lookback)
    gld_mom = gld.pct_change(lookback)
    
    btc_ret = btc.pct_change()
    eth_ret = eth.pct_change()
    gld_ret = gld.pct_change()
    
    equity = [10000.0]
    dates_out = []
    daily_returns = []
    
    for i in range(lookback + 1, len(common)):
        bm, em, gm = btc_mom.iloc[i-1], eth_mom.iloc[i-1], gld_mom.iloc[i-1]
        
        if pd.isna(bm) or pd.isna(em) or pd.isna(gm):
            continue
        
        best = max(bm, em, gm)
        if best == bm and bm > 0:
            w_btc, w_eth, w_gld = 0.70, 0.15, 0.10
        elif best == em and em > 0:
            w_btc, w_eth, w_gld = 0.15, 0.65, 0.10
        elif best == gm:
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.55
        else:  # all negative
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.35
        
        day_ret = w_btc * btc_ret.iloc[i] + w_eth * eth_ret.iloc[i] + w_gld * gld_ret.iloc[i]
        if np.isnan(day_ret):
            day_ret = 0
        
        equity.append(equity[-1] * (1 + day_ret))
        dates_out.append(common[i])
        daily_returns.append(day_ret)
    
    return pd.Series(daily_returns, index=dates_out, name="BestCrypto"), np.array(equity[1:]), dates_out

# ─── Stock v4d returns (pre-computed) ───

def load_stock_v4d_returns():
    """Load Stock v3b daily returns and apply DD Responsive GLD overlay."""
    stock = pd.read_csv(STOCK_RETURNS, parse_dates=["Date"]).set_index("Date")
    gld = pd.read_csv(DATA_DIR / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    gld_ret = gld["Close"].pct_change()
    
    # Align
    common = stock.index.intersection(gld_ret.index).sort_values()
    stock_ret = stock.loc[common, "Return"]
    gld_ret = gld_ret[common]
    
    # Apply DD Responsive GLD overlay (v4d_aggr params)
    equity = [10000.0]
    v4d_returns = []
    
    for i in range(len(common)):
        # Calculate current drawdown from peak
        peak = max(equity)
        dd = (equity[-1] - peak) / peak if peak > 0 else 0
        
        # DD Responsive GLD allocation
        if dd < -0.18:
            gld_w = 0.60
        elif dd < -0.12:
            gld_w = 0.50
        elif dd < -0.08:
            gld_w = 0.30
        else:
            gld_w = 0.0
        
        sr = stock_ret.iloc[i] if not np.isnan(stock_ret.iloc[i]) else 0
        gr = gld_ret.iloc[i] if not np.isnan(gld_ret.iloc[i]) else 0
        
        day_ret = (1 - gld_w) * sr + gld_w * gr
        v4d_returns.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    
    return pd.Series(v4d_returns, index=common, name="StockV4d"), np.array(equity[1:])

# ─── Portfolio combination strategies ───

def combine_fixed(stock_ret, crypto_ret, w_stock=0.5):
    """Fixed weight combination."""
    common = stock_ret.index.intersection(crypto_ret.index).sort_values()
    s = stock_ret[common].fillna(0)
    c = crypto_ret[common].fillna(0)
    combined = w_stock * s + (1 - w_stock) * c
    equity = (1 + combined).cumprod() * 10000
    return combined, equity.values, common

def combine_risk_parity(stock_ret, crypto_ret, lookback=126):
    """Inverse volatility weighting."""
    common = stock_ret.index.intersection(crypto_ret.index).sort_values()
    s = stock_ret[common].fillna(0)
    c = crypto_ret[common].fillna(0)
    
    combined_ret = []
    equity = [10000.0]
    
    for i in range(lookback, len(common)):
        s_vol = s.iloc[i-lookback:i].std() * np.sqrt(252)
        c_vol = c.iloc[i-lookback:i].std() * np.sqrt(252)
        
        if s_vol + c_vol == 0:
            ws, wc = 0.5, 0.5
        else:
            ws = (1/s_vol) / (1/s_vol + 1/c_vol) if s_vol > 0 else 0.9
            wc = 1 - ws
        
        day_ret = ws * s.iloc[i] + wc * c.iloc[i]
        combined_ret.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    
    dates = common[lookback:]
    return pd.Series(combined_ret, index=dates), np.array(equity[1:]), dates

def combine_dd_responsive(stock_ret, crypto_ret, lookback=126):
    """DD-responsive: increase stock allocation during crypto drawdowns and vice versa."""
    common = stock_ret.index.intersection(crypto_ret.index).sort_values()
    s = stock_ret[common].fillna(0)
    c = crypto_ret[common].fillna(0)
    
    combined_ret = []
    equity = [10000.0]
    s_eq = [10000.0]
    c_eq = [10000.0]
    
    for i in range(lookback, len(common)):
        s_eq.append(s_eq[-1] * (1 + s.iloc[i-1]))
        c_eq.append(c_eq[-1] * (1 + c.iloc[i-1]))
        
        # Check drawdowns of each sub-strategy
        s_peak = max(s_eq)
        c_peak = max(c_eq)
        s_dd = (s_eq[-1] - s_peak) / s_peak
        c_dd = (c_eq[-1] - c_peak) / c_peak
        
        # Base: risk parity
        s_vol = s.iloc[i-lookback:i].std() * np.sqrt(252)
        c_vol = c.iloc[i-lookback:i].std() * np.sqrt(252)
        if s_vol + c_vol == 0:
            ws, wc = 0.5, 0.5
        else:
            ws = (1/s_vol) / (1/s_vol + 1/c_vol) if s_vol > 0 else 0.9
            wc = 1 - ws
        
        # DD adjustment: reduce allocation to strategy in deep drawdown
        if c_dd < -0.20:
            wc = max(wc * 0.5, 0.15)
            ws = 1 - wc
        elif c_dd < -0.10:
            wc = max(wc * 0.75, 0.20)
            ws = 1 - wc
            
        if s_dd < -0.15:
            ws = max(ws * 0.5, 0.15)
            wc = 1 - ws
        elif s_dd < -0.08:
            ws = max(ws * 0.75, 0.20)
            wc = 1 - ws
        
        day_ret = ws * s.iloc[i] + wc * c.iloc[i]
        combined_ret.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    
    dates = common[lookback:]
    return pd.Series(combined_ret, index=dates), np.array(equity[1:]), dates

def combine_momentum_switch(stock_ret, crypto_ret, lookback=63):
    """Allocate more to the strategy with better recent momentum."""
    common = stock_ret.index.intersection(crypto_ret.index).sort_values()
    s = stock_ret[common].fillna(0)
    c = crypto_ret[common].fillna(0)
    
    combined_ret = []
    equity = [10000.0]
    
    for i in range(lookback, len(common)):
        s_mom = s.iloc[i-lookback:i].sum()
        c_mom = c.iloc[i-lookback:i].sum()
        
        if s_mom > c_mom and s_mom > 0:
            ws, wc = 0.65, 0.35
        elif c_mom > s_mom and c_mom > 0:
            ws, wc = 0.35, 0.65
        elif s_mom > 0 or c_mom > 0:
            ws, wc = 0.50, 0.50
        else:
            ws, wc = 0.40, 0.40  # 20% cash
        
        day_ret = ws * s.iloc[i] + wc * c.iloc[i]
        combined_ret.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    
    dates = common[lookback:]
    return pd.Series(combined_ret, index=dates), np.array(equity[1:]), dates

def combine_target_vol(stock_ret, crypto_ret, target_vol=0.25, lookback=63):
    """Target portfolio volatility, scale leverage between strategies."""
    common = stock_ret.index.intersection(crypto_ret.index).sort_values()
    s = stock_ret[common].fillna(0)
    c = crypto_ret[common].fillna(0)
    
    combined_ret = []
    equity = [10000.0]
    
    for i in range(lookback, len(common)):
        # First do risk parity
        s_vol = s.iloc[i-lookback:i].std() * np.sqrt(252)
        c_vol = c.iloc[i-lookback:i].std() * np.sqrt(252)
        
        if s_vol + c_vol == 0:
            ws, wc = 0.5, 0.5
        else:
            ws = (1/s_vol) / (1/s_vol + 1/c_vol) if s_vol > 0 else 0.9
            wc = 1 - ws
        
        # Portfolio vol estimate
        s_c_corr = np.corrcoef(s.iloc[i-lookback:i], c.iloc[i-lookback:i])[0, 1]
        if np.isnan(s_c_corr):
            s_c_corr = 0.2
        port_vol = np.sqrt((ws*s_vol)**2 + (wc*c_vol)**2 + 2*ws*wc*s_vol*c_vol*s_c_corr)
        
        # Scale to target
        if port_vol > 0:
            scale = min(target_vol / port_vol, 1.5)  # cap at 1.5x
        else:
            scale = 1.0
        
        day_ret = scale * (ws * s.iloc[i] + wc * c.iloc[i])
        combined_ret.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    
    dates = common[lookback:]
    return pd.Series(combined_ret, index=dates), np.array(equity[1:]), dates

# ─── Main ───

def main():
    print("=" * 80)
    print("Portfolio v3: Stock v4d + BestCrypto Combination")
    print("=" * 80)
    
    # Load strategies
    print("\n📊 Loading strategies...")
    stock_ret, stock_eq = load_stock_v4d_returns()
    crypto_ret, crypto_eq, crypto_dates = run_bestcrypto(lookback=90)
    
    # Check overlap
    common = stock_ret.index.intersection(crypto_ret.index).sort_values()
    print(f"\nStock v4d: {stock_ret.index[0].date()} → {stock_ret.index[-1].date()}")
    print(f"BestCrypto: {crypto_ret.index[0].date()} → {crypto_ret.index[-1].date()}")
    print(f"Overlap: {common[0].date()} → {common[-1].date()} ({len(common)} days)")
    
    # Correlation
    s_aligned = stock_ret[common].fillna(0)
    c_aligned = crypto_ret[common].fillna(0)
    corr = np.corrcoef(s_aligned, c_aligned)[0, 1]
    print(f"\n📈 Correlation (Stock v4d vs BestCrypto): {corr:.3f}")
    
    # Run combinations
    strategies = {}
    
    # Fixed weights
    for ws in [0.3, 0.4, 0.5, 0.6, 0.7]:
        name = f"Fixed {int(ws*100)}/{int((1-ws)*100)}"
        ret, eq, dates = combine_fixed(stock_ret, crypto_ret, w_stock=ws)
        strategies[name] = (ret, eq, dates)
    
    # Risk parity
    for lb in [63, 126]:
        name = f"RiskParity {lb}d"
        ret, eq, dates = combine_risk_parity(stock_ret, crypto_ret, lookback=lb)
        strategies[name] = (ret, eq, dates)
    
    # DD Responsive
    for lb in [63, 126]:
        name = f"DD_Responsive {lb}d"
        ret, eq, dates = combine_dd_responsive(stock_ret, crypto_ret, lookback=lb)
        strategies[name] = (ret, eq, dates)
    
    # Momentum switch
    for lb in [42, 63, 126]:
        name = f"MomSwitch {lb}d"
        ret, eq, dates = combine_momentum_switch(stock_ret, crypto_ret, lookback=lb)
        strategies[name] = (ret, eq, dates)
    
    # Target vol
    for tv in [0.20, 0.25, 0.30]:
        name = f"TargetVol {int(tv*100)}%"
        ret, eq, dates = combine_target_vol(stock_ret, crypto_ret, target_vol=tv, lookback=63)
        strategies[name] = (ret, eq, dates)
    
    # Solo strategies for comparison (aligned to common period)
    stock_common = stock_ret[common].fillna(0)
    crypto_common = crypto_ret[common].fillna(0)
    stock_solo_eq = (1 + stock_common).cumprod().values * 10000
    crypto_solo_eq = (1 + crypto_common).cumprod().values * 10000
    
    # Print results
    print(f"\n{'='*120}")
    print(f"{'Strategy':<25} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>8} {'WinRate':>8} {'IS Sh':>7} {'OOS Sh':>8} {'WF50':>6} {'WF60':>6} {'Score':>7}")
    print(f"{'='*120}")
    
    results = []
    
    # Solo baselines (use common period only)
    for name, eq in [("Stock v4d Solo", stock_solo_eq), ("BestCrypto Solo", crypto_solo_eq)]:
        m = calc_metrics(eq)
        sc = composite_score(m)
        is_m, oos_m, wf60 = walk_forward(eq, common, split=0.6)
        _, _, wf50 = walk_forward(eq, common, split=0.5)
        print(f"  {name:<23} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>8.2f} {m['win_rate']:>7.1f}% {is_m['sharpe']:>7.2f} {oos_m['sharpe']:>7.2f} {wf50:>6.2f} {wf60:>6.2f} {sc:>7.3f}")
        results.append({"name": name, **m, "composite": sc, "wf50": wf50, "wf60": wf60, 
                        "is_sharpe": is_m["sharpe"], "oos_sharpe": oos_m["sharpe"]})
    
    print(f"  {'-'*118}")
    
    for name, (ret, eq, dates) in sorted(strategies.items()):
        m = calc_metrics(eq)
        sc = composite_score(m)
        is_m, oos_m, wf60 = walk_forward(eq, dates, split=0.6)
        _, _, wf50 = walk_forward(eq, dates, split=0.5)
        wf50_pass = "✅" if wf50 >= 0.70 else "❌"
        wf60_pass = "✅" if wf60 >= 0.70 else "❌"
        print(f"  {name:<23} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>8.2f} {m['win_rate']:>7.1f}% {is_m['sharpe']:>7.2f} {oos_m['sharpe']:>7.2f} {wf50:>6.2f} {wf60:>6.2f} {sc:>7.3f}")
        results.append({"name": name, **m, "composite": sc, "wf50": wf50, "wf60": wf60,
                        "is_sharpe": is_m["sharpe"], "oos_sharpe": oos_m["sharpe"]})
    
    # Find best
    best = max(results[2:], key=lambda x: x["composite"])  # skip solos
    print(f"\n🏆 Best Composite: {best['name']} — Score {best['composite']:.3f}")
    print(f"   CAGR {best['cagr']*100:.1f}% | MaxDD {best['maxdd']*100:.1f}% | Sharpe {best['sharpe']:.2f} | Calmar {best['calmar']:.2f}")
    print(f"   WF 50/50: {best['wf50']:.2f} | WF 60/40: {best['wf60']:.2f}")
    
    # Best WF-passing
    wf_pass = [r for r in results[2:] if r["wf60"] >= 0.70]
    if wf_pass:
        best_wf = max(wf_pass, key=lambda x: x["composite"])
        print(f"\n🏆 Best WF-Passing (60/40): {best_wf['name']} — Score {best_wf['composite']:.3f}")
        print(f"   CAGR {best_wf['cagr']*100:.1f}% | MaxDD {best_wf['maxdd']*100:.1f}% | Sharpe {best_wf['sharpe']:.2f} | Calmar {best_wf['calmar']:.2f}")
    
    wf50_pass = [r for r in results[2:] if r["wf50"] >= 0.70]
    if wf50_pass:
        best_wf50 = max(wf50_pass, key=lambda x: x["composite"])
        print(f"\n🏆 Best WF-Passing (50/50): {best_wf50['name']} — Score {best_wf50['composite']:.3f}")
    
    # Yearly analysis for top strategy
    print(f"\n📅 Yearly Performance: {best['name']}")
    best_name = best['name']
    best_ret, best_eq, best_dates = strategies[best_name]
    
    yearly = best_ret.groupby(best_ret.index.year)
    stock_yearly = stock_common.groupby(stock_common.index.year)
    crypto_yearly = crypto_common.groupby(crypto_common.index.year)
    
    print(f"  {'Year':<6} {'Combo':>8} {'Stock':>8} {'Crypto':>8}")
    for year in sorted(yearly.groups.keys()):
        cy = yearly.get_group(year).sum() * 100 if year in yearly.groups else 0
        sy = stock_yearly.get_group(year).sum() * 100 if year in stock_yearly.groups else 0
        cry = crypto_yearly.get_group(year).sum() * 100 if year in crypto_yearly.groups else 0
        print(f"  {year:<6} {cy:>7.1f}% {sy:>7.1f}% {cry:>7.1f}%")
    
    # Save results
    save_results = []
    for r in results:
        sr = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}
        save_results.append(sr)
    
    out_path = Path(__file__).parent / "portfolio_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {out_path}")

if __name__ == "__main__":
    main()
