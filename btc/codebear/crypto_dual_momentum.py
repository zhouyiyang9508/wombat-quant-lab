"""
Crypto Dual Momentum — BTC + ETH + GLD rotation
Extends BTC v7f logic to include ETH for crypto diversification.

Hypothesis: BTC and ETH have different momentum cycles (ETH led in 2017, 2021 DeFi summer).
A rotation strategy between them + GLD hedge could improve risk-adjusted returns.

Author: 代码熊 🐻 | 2026-02-20
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data_cache"

def download_data():
    """Download ETH data if not present."""
    eth_path = DATA_DIR / "ETH_USD.csv"
    if not eth_path.exists():
        try:
            import yfinance as yf
            print("Downloading ETH-USD data...")
            eth = yf.download("ETH-USD", start="2015-01-01", end="2026-02-19", auto_adjust=True)
            eth = eth.reset_index()
            eth = eth[["Date", "Open", "High", "Low", "Close", "Volume"]]
            eth.to_csv(eth_path, index=False)
            print(f"Saved ETH data: {len(eth)} rows")
        except Exception as e:
            print(f"yfinance failed: {e}, trying stooq...")
            eth = pd.read_csv(f"https://stooq.com/q/d/l/?s=eth-usd&i=d")
            eth.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            eth["Date"] = pd.to_datetime(eth["Date"])
            eth = eth.sort_values("Date")
            eth.to_csv(eth_path, index=False)
            print(f"Saved ETH data: {len(eth)} rows")
    return eth_path

def load_data():
    download_data()
    
    btc = pd.read_csv(DATA_DIR / "BTC_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    eth = pd.read_csv(DATA_DIR / "ETH_USD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    gld = pd.read_csv(DATA_DIR / "GLD.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    
    # Common dates (ETH starts ~2017-11 on some sources, or 2015-08)
    common = btc.index.intersection(eth.index).intersection(gld.index)
    common = common.sort_values()
    
    print(f"BTC: {btc.index[0].date()} → {btc.index[-1].date()}")
    print(f"ETH: {eth.index[0].date()} → {eth.index[-1].date()}")
    print(f"Common: {common[0].date()} → {common[-1].date()} ({len(common)} days)")
    
    return btc.loc[common], eth.loc[common], gld.loc[common], common

def calc_momentum(prices, lookback_days):
    """Calculate momentum (total return over lookback)."""
    return prices / prices.shift(lookback_days) - 1

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
    return m_is, m_oos

def strategy_btc_only_v7f(btc_close, gld_close, dates):
    """BTC v7f DualMom (BTC vs GLD) - baseline."""
    n = len(btc_close)
    equity = np.zeros(n)
    equity[0] = 10000.0
    
    sma200 = pd.Series(btc_close).rolling(200).mean().values
    
    # Halving dates
    halving_dates = [pd.Timestamp("2016-07-09"), pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-19")]
    
    def months_since_halving(d):
        for h in reversed(halving_dates):
            if d >= h:
                return (d - h).days / 30.44
        return 999
    
    for i in range(1, n):
        if np.isnan(sma200[i]):
            equity[i] = equity[i-1]
            continue
        
        btc_ret = btc_close[i] / btc_close[i-1] - 1
        gld_ret = gld_close[i] / gld_close[i-1] - 1
        
        # 3M and 6M momentum
        lb3 = min(i, 63)
        lb6 = min(i, 126)
        btc_mom3 = btc_close[i] / btc_close[max(0, i-lb3)] - 1
        btc_mom6 = btc_close[i] / btc_close[max(0, i-lb6)] - 1
        gld_mom3 = gld_close[i] / gld_close[max(0, i-lb3)] - 1
        gld_mom6 = gld_close[i] / gld_close[max(0, i-lb6)] - 1
        
        btc_mom = 0.5 * btc_mom3 + 0.5 * btc_mom6
        gld_mom = 0.5 * gld_mom3 + 0.5 * gld_mom6
        
        # Mayer multiple
        mayer = btc_close[i] / sma200[i]
        
        # Halving
        msh = months_since_halving(dates[i])
        
        # Allocation logic (from v7f)
        btc_pos = btc_mom > 0
        gld_pos = gld_mom > 0
        
        if btc_pos and gld_pos:
            if btc_mom > gld_mom:
                w_btc, w_gld = 0.80, 0.15
            else:
                w_btc, w_gld = 0.50, 0.40
        elif btc_pos and not gld_pos:
            w_btc, w_gld = 0.85, 0.05
        elif not btc_pos and gld_pos:
            w_btc, w_gld = 0.25, 0.50
        else:
            w_btc, w_gld = 0.20, 0.30
        
        # Halving override
        if msh < 18:
            w_btc = max(w_btc, 0.50)
        
        # Mayer bubble protection
        if mayer > 3.5:
            w_btc = min(w_btc, 0.35)
        elif mayer > 2.4:
            w_btc = min(w_btc, 0.60)
        
        port_ret = w_btc * btc_ret + w_gld * gld_ret
        equity[i] = equity[i-1] * (1 + port_ret)
    
    return equity

def strategy_crypto_dual_mom(btc_close, eth_close, gld_close, dates):
    """
    Crypto Dual Momentum: BTC vs ETH vs GLD rotation.
    - Compare BTC and ETH momentum, allocate more to the stronger
    - Use GLD as hedge when both crypto are negative
    """
    n = len(btc_close)
    equity = np.zeros(n)
    equity[0] = 10000.0
    
    for i in range(1, n):
        btc_ret = btc_close[i] / btc_close[i-1] - 1
        eth_ret = eth_close[i] / eth_close[i-1] - 1
        gld_ret = gld_close[i] / gld_close[i-1] - 1
        
        lb3 = min(i, 63)
        lb6 = min(i, 126)
        
        btc_mom = 0.5 * (btc_close[i]/btc_close[max(0,i-lb3)]-1) + 0.5 * (btc_close[i]/btc_close[max(0,i-lb6)]-1)
        eth_mom = 0.5 * (eth_close[i]/eth_close[max(0,i-lb3)]-1) + 0.5 * (eth_close[i]/eth_close[max(0,i-lb6)]-1)
        gld_mom = 0.5 * (gld_close[i]/gld_close[max(0,i-lb3)]-1) + 0.5 * (gld_close[i]/gld_close[max(0,i-lb6)]-1)
        
        crypto_pos = [m > 0 for m in [btc_mom, eth_mom]]
        
        if all(crypto_pos):
            # Both crypto positive — allocate by relative strength
            if btc_mom > eth_mom:
                w_btc, w_eth, w_gld = 0.55, 0.30, 0.10
            else:
                w_btc, w_eth, w_gld = 0.30, 0.50, 0.10
        elif crypto_pos[0] and not crypto_pos[1]:
            # Only BTC positive
            w_btc, w_eth, w_gld = 0.65, 0.10, 0.20
        elif not crypto_pos[0] and crypto_pos[1]:
            # Only ETH positive
            w_btc, w_eth, w_gld = 0.15, 0.55, 0.20
        else:
            # Both negative — defensive
            if gld_mom > 0:
                w_btc, w_eth, w_gld = 0.15, 0.10, 0.55
            else:
                w_btc, w_eth, w_gld = 0.15, 0.10, 0.30  # 45% cash
        
        port_ret = w_btc * btc_ret + w_eth * eth_ret + w_gld * gld_ret
        equity[i] = equity[i-1] * (1 + port_ret)
    
    return equity

def strategy_crypto_vol_weighted(btc_close, eth_close, gld_close, dates):
    """Inverse-vol weighted BTC + ETH + GLD momentum."""
    n = len(btc_close)
    equity = np.zeros(n)
    equity[0] = 10000.0
    
    btc_rets = np.diff(btc_close, prepend=btc_close[0]) / np.roll(btc_close, 1)
    eth_rets = np.diff(eth_close, prepend=eth_close[0]) / np.roll(eth_close, 1)
    
    for i in range(1, n):
        btc_ret = btc_close[i] / btc_close[i-1] - 1
        eth_ret = eth_close[i] / eth_close[i-1] - 1
        gld_ret = gld_close[i] / gld_close[i-1] - 1
        
        lb = min(i, 60)
        if lb < 20:
            equity[i] = equity[i-1] * (1 + 0.33*btc_ret + 0.33*eth_ret + 0.33*gld_ret)
            continue
        
        vol_b = np.std(btc_rets[max(1,i-lb):i]) * np.sqrt(365)
        vol_e = np.std(eth_rets[max(1,i-lb):i]) * np.sqrt(365)
        
        # Momentum
        lb6 = min(i, 126)
        btc_mom = btc_close[i] / btc_close[max(0, i-lb6)] - 1
        eth_mom = eth_close[i] / eth_close[max(0, i-lb6)] - 1
        
        # Inverse vol weights for crypto portion
        inv_b = 1/max(vol_b, 0.1)
        inv_e = 1/max(vol_e, 0.1)
        
        # Only allocate to positive momentum crypto
        if btc_mom > 0 and eth_mom > 0:
            total_inv = inv_b + inv_e
            crypto_pct = 0.85
            w_btc = crypto_pct * inv_b / total_inv
            w_eth = crypto_pct * inv_e / total_inv
            w_gld = 0.10
        elif btc_mom > 0:
            w_btc, w_eth, w_gld = 0.60, 0.0, 0.25
        elif eth_mom > 0:
            w_btc, w_eth, w_gld = 0.0, 0.55, 0.25
        else:
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.50
        
        port_ret = w_btc * btc_ret + w_eth * eth_ret + w_gld * gld_ret
        equity[i] = equity[i-1] * (1 + port_ret)
    
    return equity

def strategy_best_crypto(btc_close, eth_close, gld_close, dates):
    """Pick the single best crypto by momentum, hedge with GLD."""
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
        
        # Best among all three
        moms = {"btc": btc_mom, "eth": eth_mom, "gld": gld_mom}
        best = max(moms, key=moms.get)
        
        if best == "btc" and btc_mom > 0:
            w_btc, w_eth, w_gld = 0.70, 0.15, 0.10
        elif best == "eth" and eth_mom > 0:
            w_btc, w_eth, w_gld = 0.15, 0.65, 0.10
        elif best == "gld":
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.55
        else:
            w_btc, w_eth, w_gld = 0.15, 0.10, 0.35  # cash
        
        port_ret = w_btc * btc_ret + w_eth * eth_ret + w_gld * gld_ret
        equity[i] = equity[i-1] * (1 + port_ret)
    
    return equity

def main():
    btc, eth, gld, dates = load_data()
    
    btc_c = btc["Close"].values
    eth_c = eth["Close"].values
    gld_c = gld["Close"].values
    
    # BTC-ETH correlation
    btc_rets = np.diff(btc_c) / btc_c[:-1]
    eth_rets = np.diff(eth_c) / eth_c[:-1]
    corr = np.corrcoef(btc_rets, eth_rets)[0, 1]
    print(f"\nBTC-ETH daily return correlation: {corr:.3f}")
    
    # Buy & Hold baselines
    eq_btc_bh = 10000 * btc_c / btc_c[0]
    eq_eth_bh = 10000 * eth_c / eth_c[0]
    
    strategies = {
        "BTC B&H": eq_btc_bh,
        "ETH B&H": eq_eth_bh,
        "BTC v7f (solo)": strategy_btc_only_v7f(btc_c, gld_c, dates),
        "CryptoDualMom": strategy_crypto_dual_mom(btc_c, eth_c, gld_c, dates),
        "CryptoVolWgt": strategy_crypto_vol_weighted(btc_c, eth_c, gld_c, dates),
        "BestCrypto": strategy_best_crypto(btc_c, eth_c, gld_c, dates),
    }
    
    print(f"\n{'Strategy':<18} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Final$':>12} {'IS_Sh':>7} {'OOS_Sh':>7} {'WF':>5} {'Score':>7}")
    print("-" * 105)
    
    for name, eq in strategies.items():
        m = calc_metrics(eq)
        m_is, m_oos = walk_forward(eq)
        wfr = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] > 0 else 0
        p = "✅" if wfr >= 0.70 else "❌"
        s = composite_score(m)
        print(f"{name:<18} {m['cagr']*100:>6.1f}% {m['maxdd']*100:>7.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['final_value']:>11,.0f} {m_is['sharpe']:>7.2f} {m_oos['sharpe']:>7.2f} {p:>5} {s:>7.3f}")
    
    # Rolling correlation BTC-ETH
    print("\n=== BTC-ETH Rolling Correlation (120d) ===")
    for yr in range(2018, 2026):
        mask = np.array([d.year == yr for d in dates[1:]])
        if mask.sum() < 30:
            continue
        c = np.corrcoef(btc_rets[mask], eth_rets[mask])[0, 1]
        print(f"  {yr}: {c:.3f}")

if __name__ == "__main__":
    main()
