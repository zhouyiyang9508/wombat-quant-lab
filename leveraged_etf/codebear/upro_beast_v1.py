"""
UPRO Beast v1 — 3x S&P500 Trend Following + GLD Crisis Hedge
Based on TQQQ v9g logic adapted for UPRO's characteristics.

UPRO vs TQQQ: S&P500 is less volatile than Nasdaq-100, so:
- Narrower hysteresis bands (S&P trends are smoother)
- Lower bear floor (S&P bottoms are less extreme)
- GLD hedge in bear mode

Variants:
  v1a: TQQQ v9g clone (105/93, floor 25%)
  v1b: Narrower bands (103/95, floor 20%) — S&P is smoother
  v1c: Wide bands (107/90, floor 30%)
  v1d: Best params + no GLD (cash only, for comparison)
  v1e: Adaptive vol-scaled bands

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data_cache"

def load_data():
    upro = pd.read_csv(DATA_DIR / "UPRO.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    gld = pd.read_csv(DATA_DIR / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    # Align dates
    common = upro.index.intersection(gld.index)
    return upro.loc[common], gld.loc[common]

def run_beast(upro_prices, gld_prices, params, cost_per_side=0.0005, slippage=0.001):
    """Run UPRO Beast strategy with GLD hedge."""
    bull_enter = params["bull_enter"]
    bear_enter = params["bear_enter"]
    bear_floor = params["bear_floor"]
    sma_period = params.get("sma_period", 200)
    use_gld = params.get("use_gld", True)
    
    close = upro_prices["Close"].values
    gld_close = gld_prices["Close"].values
    dates = upro_prices.index
    n = len(close)
    
    sma = pd.Series(close, index=dates).rolling(sma_period).mean().values
    rsi_period = 10
    
    # Compute RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0).astype(float)
    loss = np.where(delta < 0, -delta, 0).astype(float)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False).mean().values
    rsi = np.where(avg_loss == 0, 100, 100 - 100 / (1 + avg_gain / avg_loss))
    
    # Weekly returns
    weekly_ret = np.zeros(n)
    for i in range(5, n):
        weekly_ret[i] = (close[i] / close[i-5]) - 1
    
    # Simulation
    equity = np.zeros(n)
    equity[0] = 10000.0
    positions_upro = np.zeros(n)  # fraction in UPRO
    positions_gld = np.zeros(n)   # fraction in GLD
    regime = np.zeros(n, dtype=int)  # 1=bull, -1=bear
    regime[0] = 1
    
    for i in range(1, n):
        if np.isnan(sma[i]):
            equity[i] = equity[i-1]
            regime[i] = regime[i-1]
            continue
        
        # Regime detection with hysteresis
        prev_regime = regime[i-1]
        if prev_regime == 1:
            if close[i] < sma[i] * bear_enter:
                regime[i] = -1
            else:
                regime[i] = 1
        else:
            if close[i] > sma[i] * bull_enter:
                regime[i] = 1
            else:
                regime[i] = -1
        
        # Position sizing
        if regime[i] == 1:
            # Bull mode
            if rsi[i] > 80 and weekly_ret[i] > 0.12:
                upro_pct = 0.80
            else:
                upro_pct = 1.0
            gld_pct = 0.0
        else:
            # Bear mode
            upro_pct = bear_floor
            gld_pct = (1.0 - bear_floor) * 0.85 if use_gld else 0.0  # 85% of remainder to GLD
            
            if rsi[i] < 20 or weekly_ret[i] < -0.10:
                upro_pct = 0.75
                gld_pct = 0.15 if use_gld else 0.0
            elif rsi[i] < 30:
                upro_pct = 0.55
                gld_pct = 0.30 if use_gld else 0.0
            elif rsi[i] > 65:
                upro_pct = bear_floor
                gld_pct = (1.0 - bear_floor) * 0.85 if use_gld else 0.0
        
        # Returns
        upro_ret = (close[i] / close[i-1]) - 1
        gld_ret = (gld_close[i] / gld_close[i-1]) - 1
        
        # Trading costs
        prev_upro = positions_upro[i-1]
        prev_gld = positions_gld[i-1]
        turnover = abs(upro_pct - prev_upro) + abs(gld_pct - prev_gld)
        tc = turnover * (cost_per_side + slippage)
        
        port_ret = upro_pct * upro_ret + gld_pct * gld_ret - tc
        equity[i] = equity[i-1] * (1 + port_ret)
        positions_upro[i] = upro_pct
        positions_gld[i] = gld_pct
    
    return equity, dates, regime

def calc_metrics(equity, dates, rf=0.04):
    """Calculate performance metrics."""
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]
    
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = dd.min()
    
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    
    win_rate = np.mean(returns > 0)
    
    return {
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar,
        "final_value": equity[-1], "ann_vol": ann_vol, "win_rate": win_rate,
        "years": years
    }

def walk_forward(upro_prices, gld_prices, params, split=0.6):
    """Walk-forward validation."""
    n = len(upro_prices)
    split_idx = int(n * split)
    
    # In-sample
    eq_is, dates_is, _ = run_beast(upro_prices.iloc[:split_idx], gld_prices.iloc[:split_idx], params)
    m_is = calc_metrics(eq_is, dates_is)
    
    # Out-of-sample
    eq_oos, dates_oos, _ = run_beast(upro_prices.iloc[split_idx:], gld_prices.iloc[split_idx:], params)
    m_oos = calc_metrics(eq_oos, dates_oos)
    
    return m_is, m_oos

def run_buyhold(upro_prices):
    close = upro_prices["Close"].values
    dates = upro_prices.index
    equity = 10000.0 * close / close[0]
    return equity, dates

def composite_score(m):
    return m["sharpe"] * 0.4 + m["calmar"] * 0.4 + min(m["cagr"], 1.0) * 0.2

def main():
    upro, gld = load_data()
    print(f"UPRO data: {upro.index[0].date()} → {upro.index[-1].date()} ({len(upro)} days)")
    print(f"GLD data: {gld.index[0].date()} → {gld.index[-1].date()}")
    print()
    
    # Buy & Hold
    eq_bh, dates_bh = run_buyhold(upro)
    m_bh = calc_metrics(eq_bh, dates_bh)
    
    variants = {
        "v1a_clone":    {"bull_enter": 1.05, "bear_enter": 0.93, "bear_floor": 0.25, "use_gld": True},
        "v1b_narrow":   {"bull_enter": 1.03, "bear_enter": 0.95, "bear_floor": 0.20, "use_gld": True},
        "v1c_wide":     {"bull_enter": 1.07, "bear_enter": 0.90, "bear_floor": 0.30, "use_gld": True},
        "v1d_no_gld":   {"bull_enter": 1.05, "bear_enter": 0.93, "bear_floor": 0.25, "use_gld": False},
        "v1e_mid":      {"bull_enter": 1.04, "bear_enter": 0.94, "bear_floor": 0.22, "use_gld": True},
        "v1f_tight":    {"bull_enter": 1.03, "bear_enter": 0.96, "bear_floor": 0.18, "use_gld": True},
    }
    
    print(f"{'Version':<16} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Final$':>12} {'IS_Sh':>7} {'OOS_Sh':>7} {'WF':>5} {'Score':>7}")
    print("-" * 100)
    
    # B&H
    print(f"{'UPRO B&H':<16} {m_bh['cagr']*100:>6.1f}% {m_bh['maxdd']*100:>7.1f}% {m_bh['sharpe']:>7.2f} {m_bh['calmar']:>7.2f} {m_bh['final_value']:>11,.0f} {'':>7} {'':>7} {'':>5} {composite_score(m_bh):>7.3f}")
    print("-" * 100)
    
    results = {}
    for name, params in variants.items():
        eq, dates, reg = run_beast(upro, gld, params)
        m = calc_metrics(eq, dates)
        m_is, m_oos = walk_forward(upro, gld, params)
        
        wf_ratio = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
        wf_pass = "✅" if wf_ratio >= 0.70 else "❌"
        score = composite_score(m)
        
        results[name] = {"metrics": m, "is": m_is, "oos": m_oos, "wf_ratio": wf_ratio, "score": score}
        
        print(f"{name:<16} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['final_value']:>11,.0f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {wf_pass:>5} {score:>7.3f}")
    
    # Find best
    best_name = max(results, key=lambda k: results[k]["score"])
    best = results[best_name]
    print(f"\n🏆 Best: {best_name} (Composite {best['score']:.3f})")
    
    # Parameter robustness for best variant
    print("\n=== Parameter Robustness (best variant) ===")
    best_params = variants[best_name].copy()
    
    for param_name, test_range in [
        ("bull_enter", [1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08]),
        ("bear_enter", [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96]),
        ("bear_floor", [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]),
    ]:
        scores = []
        print(f"\n{param_name}:")
        for val in test_range:
            p = best_params.copy()
            p[param_name] = val
            eq, dates, _ = run_beast(upro, gld, p)
            m = calc_metrics(eq, dates)
            s = composite_score(m)
            scores.append(s)
            marker = " ←" if val == best_params[param_name] else ""
            print(f"  {val:.2f}: Score={s:.3f} Sharpe={m['sharpe']:.2f} Calmar={m['calmar']:.2f}{marker}")
        print(f"  Spread: {max(scores)-min(scores):.3f}")

if __name__ == "__main__":
    main()
