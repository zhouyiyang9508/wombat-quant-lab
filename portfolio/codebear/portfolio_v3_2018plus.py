"""
Portfolio v3 Round 2: Stock v4d + BestCrypto (2018+)
=====================================================
Restrict to 2018+ where both strategies are well-validated.
Add regime-aware and adaptive combinations.

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parents[2] / "data_cache"
STOCK_RETURNS = Path(__file__).resolve().parents[2] / "stocks" / "codebear" / "v3b_daily_returns.csv"

def calc_metrics(equity, rf=0.04):
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    years = n / 252
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

def walk_forward_multi(equity, dates, splits=[0.5, 0.55, 0.6, 0.65]):
    results = {}
    for sp in splits:
        n = len(equity)
        si = int(n * sp)
        is_m = calc_metrics(equity[:si+1])
        oos_m = calc_metrics(equity[si:])
        ratio = oos_m["sharpe"] / is_m["sharpe"] if is_m["sharpe"] > 0 else 0
        results[f"wf{int(sp*100)}"] = ratio
        if sp == 0.6:
            results["is_sharpe"] = is_m["sharpe"]
            results["oos_sharpe"] = oos_m["sharpe"]
    return results

# ─── Load data ───

def load_stock_v4d():
    stock = pd.read_csv(STOCK_RETURNS, parse_dates=["Date"]).set_index("Date")
    gld = pd.read_csv(DATA_DIR / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    gld_ret = gld["Close"].pct_change()
    common = stock.index.intersection(gld_ret.index).sort_values()
    stock_ret = stock.loc[common, "Return"]
    gld_r = gld_ret[common]
    
    equity = [10000.0]
    v4d_returns = []
    for i in range(len(common)):
        peak = max(equity)
        dd = (equity[-1] - peak) / peak if peak > 0 else 0
        gld_w = 0.60 if dd < -0.18 else 0.50 if dd < -0.12 else 0.30 if dd < -0.08 else 0.0
        sr = stock_ret.iloc[i] if not np.isnan(stock_ret.iloc[i]) else 0
        gr = gld_r.iloc[i] if not np.isnan(gld_r.iloc[i]) else 0
        day_ret = (1 - gld_w) * sr + gld_w * gr
        v4d_returns.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    return pd.Series(v4d_returns, index=common, name="StockV4d")

def load_bestcrypto(lookback=90):
    btc = pd.read_csv(DATA_DIR / "BTC_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    eth = pd.read_csv(DATA_DIR / "ETH_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    gld = pd.read_csv(DATA_DIR / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    common = btc.index.intersection(eth.index).intersection(gld.index).sort_values()
    btc, eth, gld_p = btc[common], eth[common], gld[common]
    
    btc_mom = btc.pct_change(lookback)
    eth_mom = eth.pct_change(lookback)
    gld_mom = gld_p.pct_change(lookback)
    btc_ret = btc.pct_change()
    eth_ret = eth.pct_change()
    gld_ret = gld_p.pct_change()
    
    daily_returns = {}
    for i in range(lookback + 1, len(common)):
        bm, em, gm = btc_mom.iloc[i-1], eth_mom.iloc[i-1], gld_mom.iloc[i-1]
        if pd.isna(bm) or pd.isna(em) or pd.isna(gm):
            continue
        best = max(bm, em, gm)
        if best == bm and bm > 0:
            w = (0.70, 0.15, 0.10)
        elif best == em and em > 0:
            w = (0.15, 0.65, 0.10)
        elif best == gm:
            w = (0.15, 0.10, 0.55)
        else:
            w = (0.15, 0.10, 0.35)
        day_ret = w[0]*btc_ret.iloc[i] + w[1]*eth_ret.iloc[i] + w[2]*gld_ret.iloc[i]
        daily_returns[common[i]] = day_ret if not np.isnan(day_ret) else 0
    
    return pd.Series(daily_returns, name="BestCrypto")

# ─── Combination methods ───

def combine(stock_ret, crypto_ret, method, start_date="2018-01-01", **kwargs):
    common = stock_ret.index.intersection(crypto_ret.index)
    common = common[common >= start_date].sort_values()
    s = stock_ret[common].fillna(0)
    c = crypto_ret[common].fillna(0)
    
    lb = kwargs.get("lookback", 63)
    
    equity = [10000.0]
    rets = []
    
    # Pre-compute rolling stats
    for i in range(lb, len(common)):
        si, ci = s.iloc[i], c.iloc[i]
        
        if method == "fixed":
            ws = kwargs.get("w_stock", 0.5)
            wc = 1 - ws
        
        elif method == "risk_parity":
            sv = s.iloc[i-lb:i].std() * np.sqrt(252)
            cv = c.iloc[i-lb:i].std() * np.sqrt(252)
            if sv + cv == 0 or sv == 0 or cv == 0:
                ws, wc = 0.5, 0.5
            else:
                ws = (1/sv) / (1/sv + 1/cv)
                wc = 1 - ws
        
        elif method == "adaptive_rp":
            # Risk parity + tilt toward recent winner
            sv = s.iloc[i-lb:i].std() * np.sqrt(252)
            cv = c.iloc[i-lb:i].std() * np.sqrt(252)
            if sv + cv == 0 or sv == 0 or cv == 0:
                ws, wc = 0.5, 0.5
            else:
                ws = (1/sv) / (1/sv + 1/cv)
                wc = 1 - ws
            # Momentum tilt
            s_mom = s.iloc[i-lb:i].sum()
            c_mom = c.iloc[i-lb:i].sum()
            tilt = kwargs.get("tilt", 0.15)
            if s_mom > c_mom:
                ws = min(ws + tilt, 0.85)
                wc = 1 - ws
            elif c_mom > s_mom:
                wc = min(wc + tilt, 0.85)
                ws = 1 - wc
        
        elif method == "dd_responsive":
            # Base: risk parity
            sv = s.iloc[i-lb:i].std() * np.sqrt(252)
            cv = c.iloc[i-lb:i].std() * np.sqrt(252)
            if sv + cv == 0 or sv == 0 or cv == 0:
                ws, wc = 0.5, 0.5
            else:
                ws = (1/sv) / (1/sv + 1/cv)
                wc = 1 - ws
            # Portfolio DD
            peak = max(equity)
            dd = (equity[-1] - peak) / peak
            # In deep DD, reduce both and hold more cash
            if dd < -0.15:
                scale = 0.6
            elif dd < -0.10:
                scale = 0.75
            elif dd < -0.05:
                scale = 0.9
            else:
                scale = 1.0
            ws *= scale
            wc *= scale
        
        elif method == "correlation_aware":
            # When corr rises, reduce total exposure
            sv = s.iloc[i-lb:i].std() * np.sqrt(252)
            cv = c.iloc[i-lb:i].std() * np.sqrt(252)
            if sv + cv == 0 or sv == 0 or cv == 0:
                ws, wc = 0.5, 0.5
            else:
                ws = (1/sv) / (1/sv + 1/cv)
                wc = 1 - ws
            corr = np.corrcoef(s.iloc[i-lb:i], c.iloc[i-lb:i])[0, 1]
            if np.isnan(corr):
                corr = 0.2
            # High corr = less diversification benefit = reduce exposure
            if corr > 0.5:
                scale = 0.7
            elif corr > 0.3:
                scale = 0.85
            else:
                scale = 1.0
            ws *= scale
            wc *= scale
        
        elif method == "kelly_inspired":
            # Simplified Kelly: allocate based on Sharpe ratio of each
            sv = s.iloc[i-lb:i].std() * np.sqrt(252)
            cv = c.iloc[i-lb:i].std() * np.sqrt(252)
            sm = s.iloc[i-lb:i].mean() * 252
            cm = c.iloc[i-lb:i].mean() * 252
            
            s_sharpe = (sm - 0.04) / sv if sv > 0 else 0
            c_sharpe = (cm - 0.04) / cv if cv > 0 else 0
            
            # Kelly fraction ≈ Sharpe / vol for each
            s_kelly = max(s_sharpe / (sv * np.sqrt(252)) if sv > 0 else 0, 0)
            c_kelly = max(c_sharpe / (cv * np.sqrt(252)) if cv > 0 else 0, 0)
            
            total = s_kelly + c_kelly
            if total > 0:
                ws = s_kelly / total
                wc = c_kelly / total
                # Cap total at 1.0
                if ws + wc > 1.0:
                    ws, wc = ws / (ws + wc), wc / (ws + wc)
            else:
                ws, wc = 0.3, 0.3  # 40% cash
        
        else:
            ws, wc = 0.5, 0.5
        
        day_ret = ws * si + wc * ci
        rets.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
    
    dates = common[lb:]
    return pd.Series(rets, index=dates), np.array(equity[1:]), dates

def main():
    print("=" * 80)
    print("Portfolio v3 Round 2: Stock v4d + BestCrypto (2018+)")
    print("=" * 80)
    
    stock_ret = load_stock_v4d()
    crypto_ret = load_bestcrypto(90)
    
    start = "2018-01-01"
    common = stock_ret.index.intersection(crypto_ret.index)
    common = common[common >= start].sort_values()
    print(f"\nPeriod: {common[0].date()} → {common[-1].date()} ({len(common)} days)")
    
    # Correlation
    s_a = stock_ret[common].fillna(0)
    c_a = crypto_ret[common].fillna(0)
    print(f"Correlation: {np.corrcoef(s_a, c_a)[0, 1]:.3f}")
    
    configs = [
        ("Stock v4d Solo", "fixed", {"w_stock": 1.0}),
        ("BestCrypto Solo", "fixed", {"w_stock": 0.0}),
        ("Fixed 70/30", "fixed", {"w_stock": 0.70}),
        ("Fixed 60/40", "fixed", {"w_stock": 0.60}),
        ("Fixed 50/50", "fixed", {"w_stock": 0.50}),
        ("Fixed 40/60", "fixed", {"w_stock": 0.40}),
        ("RiskParity 63d", "risk_parity", {"lookback": 63}),
        ("RiskParity 126d", "risk_parity", {"lookback": 126}),
        ("AdaptRP 63d t10", "adaptive_rp", {"lookback": 63, "tilt": 0.10}),
        ("AdaptRP 63d t15", "adaptive_rp", {"lookback": 63, "tilt": 0.15}),
        ("AdaptRP 126d t15", "adaptive_rp", {"lookback": 126, "tilt": 0.15}),
        ("DD_Resp 63d", "dd_responsive", {"lookback": 63}),
        ("DD_Resp 126d", "dd_responsive", {"lookback": 126}),
        ("CorrAware 63d", "correlation_aware", {"lookback": 63}),
        ("CorrAware 126d", "correlation_aware", {"lookback": 126}),
        ("Kelly 63d", "kelly_inspired", {"lookback": 63}),
        ("Kelly 126d", "kelly_inspired", {"lookback": 126}),
    ]
    
    results = []
    
    print(f"\n{'Strategy':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>8} {'IS Sh':>7} {'OOS Sh':>8} {'WF50':>7} {'WF60':>7} {'Score':>7}")
    print("=" * 105)
    
    for name, method, kwargs in configs:
        ret, eq, dates = combine(stock_ret, crypto_ret, method, start_date=start, **kwargs)
        m = calc_metrics(eq)
        sc = composite_score(m)
        wf = walk_forward_multi(eq, dates)
        
        wf50_sym = "✅" if wf.get("wf50", 0) >= 0.70 else ""
        wf60_sym = "✅" if wf.get("wf60", 0) >= 0.70 else ""
        
        print(f"  {name:<20} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>8.2f} {wf.get('is_sharpe',0):>7.2f} {wf.get('oos_sharpe',0):>7.2f} {wf.get('wf50',0):>5.2f}{wf50_sym:>2} {wf.get('wf60',0):>5.2f}{wf60_sym:>2} {sc:>7.3f}")
        
        results.append({"name": name, **m, "composite": sc, **wf})
    
    # Best overall
    combos = [r for r in results if "Solo" not in r["name"]]
    best = max(combos, key=lambda x: x["composite"])
    print(f"\n🏆 Best Composite: {best['name']} — Score {best['composite']:.3f}")
    print(f"   CAGR {best['cagr']*100:.1f}% | MaxDD {best['maxdd']*100:.1f}% | Sharpe {best['sharpe']:.2f} | Calmar {best['calmar']:.2f}")
    
    # Best WF passing
    for wf_key, label in [("wf60", "60/40"), ("wf50", "50/50")]:
        passing = [r for r in combos if r.get(wf_key, 0) >= 0.70]
        if passing:
            b = max(passing, key=lambda x: x["composite"])
            print(f"\n🏆 Best WF {label}: {b['name']} — Score {b['composite']:.3f} (WF={b[wf_key]:.2f})")
            print(f"   CAGR {b['cagr']*100:.1f}% | MaxDD {b['maxdd']*100:.1f}% | Sharpe {b['sharpe']:.2f} | Calmar {b['calmar']:.2f}")
    
    # Yearly for best
    best_name = best["name"]
    for name, method, kwargs in configs:
        if name == best_name:
            ret, eq, dates = combine(stock_ret, crypto_ret, method, start_date=start, **kwargs)
            s_common = stock_ret[dates].fillna(0)
            c_common = crypto_ret[dates].fillna(0)
            
            print(f"\n📅 Yearly: {best_name}")
            print(f"  {'Year':<6} {'Combo':>8} {'Stock':>8} {'Crypto':>8}")
            for year in sorted(ret.groupby(ret.index.year).groups.keys()):
                cy = ret[ret.index.year == year].sum() * 100
                sy = s_common[s_common.index.year == year].sum() * 100
                cry = c_common[c_common.index.year == year].sum() * 100
                print(f"  {year:<6} {cy:>7.1f}% {sy:>7.1f}% {cry:>7.1f}%")
            break
    
    # Save
    out = Path(__file__).parent / "portfolio_v3_2018_results.json"
    save = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in results]
    with open(out, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n💾 Saved to {out}")

if __name__ == "__main__":
    main()
