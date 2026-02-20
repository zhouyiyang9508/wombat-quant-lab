"""
Portfolio v3 Robustness: Parameter sensitivity for top strategies
Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data_cache"
STOCK_RETURNS = Path(__file__).resolve().parents[2] / "stocks" / "codebear" / "v3b_daily_returns.csv"

def calc_metrics(equity, rf=0.04):
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    n = len(returns); years = n / 252
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak; maxdd = dd.min()
    ann_ret = np.mean(returns) * 252; ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar}

def composite_score(m):
    return m["sharpe"] * 0.4 + m["calmar"] * 0.4 + min(m["cagr"], 1.0) * 0.2

def walk_forward(equity, split=0.6):
    n = len(equity); si = int(n * split)
    is_m = calc_metrics(equity[:si+1]); oos_m = calc_metrics(equity[si:])
    return oos_m["sharpe"] / is_m["sharpe"] if is_m["sharpe"] > 0 else 0

def load_stock_v4d():
    stock = pd.read_csv(STOCK_RETURNS, parse_dates=["Date"]).set_index("Date")
    gld = pd.read_csv(DATA_DIR / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    gld_ret = gld["Close"].pct_change()
    common = stock.index.intersection(gld_ret.index).sort_values()
    stock_ret = stock.loc[common, "Return"]; gld_r = gld_ret[common]
    equity = [10000.0]; v4d_returns = []
    for i in range(len(common)):
        peak = max(equity); dd = (equity[-1] - peak) / peak if peak > 0 else 0
        gld_w = 0.60 if dd < -0.18 else 0.50 if dd < -0.12 else 0.30 if dd < -0.08 else 0.0
        sr = stock_ret.iloc[i] if not np.isnan(stock_ret.iloc[i]) else 0
        gr = gld_r.iloc[i] if not np.isnan(gld_r.iloc[i]) else 0
        v4d_returns.append((1 - gld_w) * sr + gld_w * gr)
        equity.append(equity[-1] * (1 + v4d_returns[-1]))
    return pd.Series(v4d_returns, index=common)

def load_bestcrypto(lookback=90):
    btc = pd.read_csv(DATA_DIR / "BTC_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    eth = pd.read_csv(DATA_DIR / "ETH_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    gld = pd.read_csv(DATA_DIR / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"]
    common = btc.index.intersection(eth.index).intersection(gld.index).sort_values()
    btc, eth, gld_p = btc[common], eth[common], gld[common]
    btc_mom = btc.pct_change(lookback); eth_mom = eth.pct_change(lookback); gld_mom = gld_p.pct_change(lookback)
    btc_ret = btc.pct_change(); eth_ret = eth.pct_change(); gld_ret = gld_p.pct_change()
    daily = {}
    for i in range(lookback + 1, len(common)):
        bm, em, gm = btc_mom.iloc[i-1], eth_mom.iloc[i-1], gld_mom.iloc[i-1]
        if pd.isna(bm) or pd.isna(em) or pd.isna(gm): continue
        best = max(bm, em, gm)
        if best == bm and bm > 0: w = (0.70, 0.15, 0.10)
        elif best == em and em > 0: w = (0.15, 0.65, 0.10)
        elif best == gm: w = (0.15, 0.10, 0.55)
        else: w = (0.15, 0.10, 0.35)
        r = w[0]*btc_ret.iloc[i] + w[1]*eth_ret.iloc[i] + w[2]*gld_ret.iloc[i]
        daily[common[i]] = r if not np.isnan(r) else 0
    return pd.Series(daily)

def run_dd_resp(s, c, start, lb, dd1, dd2, dd3, s1, s2, s3):
    """DD Responsive with configurable thresholds."""
    common = s.index.intersection(c.index)
    common = common[common >= start].sort_values()
    sr, cr = s[common].fillna(0), c[common].fillna(0)
    equity = [10000.0]
    for i in range(lb, len(common)):
        sv = sr.iloc[i-lb:i].std() * np.sqrt(252)
        cv = cr.iloc[i-lb:i].std() * np.sqrt(252)
        if sv + cv == 0 or sv == 0 or cv == 0: ws, wc = 0.5, 0.5
        else: ws = (1/sv)/(1/sv+1/cv); wc = 1-ws
        peak = max(equity); dd = (equity[-1]-peak)/peak
        if dd < dd3: scale = s3
        elif dd < dd2: scale = s2
        elif dd < dd1: scale = s1
        else: scale = 1.0
        ws *= scale; wc *= scale
        equity.append(equity[-1]*(1+ws*sr.iloc[i]+wc*cr.iloc[i]))
    return np.array(equity[1:])

def run_fixed(s, c, start, lb, w_stock):
    common = s.index.intersection(c.index)
    common = common[common >= start].sort_values()
    sr, cr = s[common].fillna(0), c[common].fillna(0)
    equity = [10000.0]
    for i in range(lb, len(common)):
        equity.append(equity[-1]*(1+w_stock*sr.iloc[i]+(1-w_stock)*cr.iloc[i]))
    return np.array(equity[1:])

def run_rp(s, c, start, lb):
    common = s.index.intersection(c.index)
    common = common[common >= start].sort_values()
    sr, cr = s[common].fillna(0), c[common].fillna(0)
    equity = [10000.0]
    for i in range(lb, len(common)):
        sv = sr.iloc[i-lb:i].std()*np.sqrt(252)
        cv = cr.iloc[i-lb:i].std()*np.sqrt(252)
        if sv+cv==0 or sv==0 or cv==0: ws=0.5
        else: ws=(1/sv)/(1/sv+1/cv)
        equity.append(equity[-1]*(1+ws*sr.iloc[i]+(1-ws)*cr.iloc[i]))
    return np.array(equity[1:])

def main():
    print("="*80)
    print("Portfolio v3 Robustness Testing")
    print("="*80)
    
    stock = load_stock_v4d()
    crypto = load_bestcrypto(90)
    start = "2018-01-01"
    
    # 1. Fixed weight sensitivity
    print("\n📊 Fixed Weight Sensitivity")
    print(f"  {'W_Stock':>8} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>8} {'WF60':>6} {'Score':>7}")
    for ws in np.arange(0.30, 0.81, 0.05):
        eq = run_fixed(stock, crypto, start, 63, ws)
        m = calc_metrics(eq); sc = composite_score(m); wf = walk_forward(eq, 0.6)
        print(f"  {ws:>7.0%} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>8.2f} {wf:>6.2f} {sc:>7.3f}")
    
    # 2. Risk Parity lookback sensitivity
    print("\n📊 Risk Parity Lookback Sensitivity")
    print(f"  {'Lookback':>10} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>8} {'WF60':>6} {'Score':>7}")
    scores_rp = []
    for lb in [21, 42, 63, 84, 126, 189, 252]:
        eq = run_rp(stock, crypto, start, lb)
        m = calc_metrics(eq); sc = composite_score(m); wf = walk_forward(eq, 0.6)
        scores_rp.append(sc)
        print(f"  {lb:>8}d {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>8.2f} {wf:>6.2f} {sc:>7.3f}")
    print(f"  Score spread: {max(scores_rp)-min(scores_rp):.3f}")
    
    # 3. DD Responsive sensitivity
    print("\n📊 DD Responsive Parameter Sensitivity")
    print(f"  {'Params':>30} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>8} {'WF60':>6} {'Score':>7}")
    scores_dd = []
    dd_configs = [
        # (dd1, dd2, dd3, s1, s2, s3)
        (-0.03, -0.08, -0.12, 0.90, 0.75, 0.60),
        (-0.05, -0.10, -0.15, 0.90, 0.75, 0.60),
        (-0.05, -0.10, -0.15, 0.85, 0.70, 0.55),
        (-0.05, -0.12, -0.18, 0.90, 0.75, 0.60),
        (-0.08, -0.15, -0.20, 0.90, 0.75, 0.60),
        (-0.08, -0.15, -0.20, 0.85, 0.70, 0.55),
        (-0.10, -0.18, -0.25, 0.90, 0.75, 0.60),
        # No DD response (baseline)
        (-0.50, -0.50, -0.50, 1.0, 1.0, 1.0),
    ]
    for dd1, dd2, dd3, s1, s2, s3 in dd_configs:
        eq = run_dd_resp(stock, crypto, start, 126, dd1, dd2, dd3, s1, s2, s3)
        m = calc_metrics(eq); sc = composite_score(m); wf = walk_forward(eq, 0.6)
        scores_dd.append(sc)
        label = f"{dd1:.0%}/{dd2:.0%}/{dd3:.0%} s{s1}/{s2}/{s3}"
        print(f"  {label:>30} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>8.2f} {wf:>6.2f} {sc:>7.3f}")
    print(f"  Score spread: {max(scores_dd)-min(scores_dd):.3f}")
    
    # 4. BestCrypto lookback sensitivity (combined with Fixed 70/30)
    print("\n📊 BestCrypto Lookback Sensitivity (with Fixed 70/30 portfolio)")
    scores_lk = []
    for lk in [30, 60, 90, 120, 150, 180]:
        cr = load_bestcrypto(lk)
        eq = run_fixed(stock, cr, start, 63, 0.70)
        m = calc_metrics(eq); sc = composite_score(m); wf = walk_forward(eq, 0.6)
        scores_lk.append(sc)
        print(f"  {lk:>4}d: CAGR {m['cagr']*100:>6.1f}% MaxDD {m['maxdd']*100:>7.1f}% Sharpe {m['sharpe']:>5.2f} Calmar {m['calmar']:>5.2f} WF60 {wf:.2f} Score {sc:.3f}")
    print(f"  Score spread: {max(scores_lk)-min(scores_lk):.3f}")
    
    # 5. Transaction costs sensitivity (Fixed 70/30)
    print("\n📊 Transaction Cost Sensitivity (Fixed 70/30)")
    common = stock.index.intersection(crypto.index)
    common = common[common >= start].sort_values()
    sr = stock[common].fillna(0)
    cr = crypto[common].fillna(0)
    combined = 0.70 * sr + 0.30 * cr
    
    for cost_bps in [0, 5, 10, 15, 20, 30]:
        # Estimate monthly rebalance cost
        # ~12 rebalances/year, each costing cost_bps * turnover
        # Average turnover ~10% per rebalance (conservative estimate for monthly)
        monthly_cost = cost_bps * 0.0001 * 0.10 / 21  # daily cost
        adj_combined = combined - monthly_cost
        eq = (1 + adj_combined).cumprod().values * 10000
        eq = eq[63:]  # skip warmup
        m = calc_metrics(eq)
        print(f"  {cost_bps:>3}bps: CAGR {m['cagr']*100:>6.1f}% Sharpe {m['sharpe']:.2f} Calmar {m['calmar']:.2f}")
    
    print("\n✅ Robustness testing complete")

if __name__ == "__main__":
    main()
