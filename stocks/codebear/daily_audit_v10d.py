#!/usr/bin/env python3
"""
v10d Final 日频审计 — 矢量化计算
代码熊 🐻

方法:
  1. 预计算所有资产的日频收益率矩阵
  2. 月频rebalance, 记录每月持仓
  3. 在每月持仓期间, 用日频收益矩阵的加权和追踪净值
  4. 计算真实日频MaxDD
"""

import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v10d", Path(__file__).parent / "momentum_v10d_final.py")
v10d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v10d)


def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c]); df = df.set_index(c).sort_index()
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df


def load_stocks(tickers):
    d = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500: continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except: pass
    return pd.DataFrame(d)


def main():
    print("=" * 72)
    print("🔍 v10d Final 日频审计 — 真实MaxDD")
    print("=" * 72)

    print("  Loading data...")
    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    ief_p  = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    sig = v10d.precompute(close_df)

    # Build combined price matrix (stocks + hedges)
    hedge_df = pd.DataFrame({
        'GLD': gld_p, 'GDX': gdx_p, 'GDXJ': gdxj_p,
        'SHY': shy_p, 'TLT': tlt_p, 'IEF': ief_p
    })
    all_prices = pd.concat([close_df, hedge_df], axis=1)
    all_prices = all_prices.loc['2014-01-01':'2025-12-31'].ffill()
    daily_rets = all_prices.pct_change()
    print(f"  Price matrix: {all_prices.shape[0]} days × {all_prices.shape[1]} assets")

    HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'SHY'}

    # Step 1: Monthly signals
    start, end = '2015-01-01', '2025-12-31'
    rng = close_df.loc[start:end].dropna(how='all')
    month_ends = rng.resample('ME').last().index

    print(f"  Computing {len(month_ends)-1} monthly selections...")
    periods = []  # (dt, ndt, weights_dict, turnover_cost)
    prev_w, prev_h = {}, set()
    port_returns = []; val = 1.0; peak = 1.0
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(month_ends) - 1):
        dt, ndt = month_ends[i], month_ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15
        if len(port_returns) >= 3:
            pv = np.std(port_returns[-3:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        w, _ = v10d.select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, ief_p)
        w, shy_boost = v10d.apply_overlays(w, spy_vol, dd, pv)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2

        # Add SHY as explicit cash
        invested = sum(w.values())
        cash = max(1.0 - invested, 0.0) + shy_boost
        if cash > 0: w['SHY'] = w.get('SHY', 0) + cash

        periods.append((dt, ndt, dict(w), float(to)))
        prev_w = {k: v for k, v in w.items() if k != 'SHY'}
        prev_h = {k for k in prev_w if k not in HEDGE_KEYS}

        # Monthly return (for val/peak tracking)
        ret = 0.0
        for t, wt in w.items():
            p = {'GLD': gld_p, 'GDX': gdx_p, 'GDXJ': gdxj_p,
                 'SHY': shy_p, 'TLT': tlt_p, 'IEF': ief_p}.get(t,
                close_df[t] if t in close_df.columns else None)
            if p is None: continue
            s = p.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt
        ret -= to * 0.0015 * 2
        val *= (1 + ret); peak = max(peak, val)
        port_returns.append(ret)

    # Step 2: Vectorized daily NAV
    print("  Computing daily NAV (vectorized)...")
    nav_series = [1.0]
    nav_dates  = [month_ends[0]]
    port_val   = 1.0

    for dt, ndt, weights, turnover in periods:
        # Deduct rebalancing cost once
        port_val *= (1.0 - turnover * 0.0015 * 2)

        # Get daily dates in this period (trading days in (dt, ndt])
        mask = (daily_rets.index > dt) & (daily_rets.index <= ndt)
        period_rets = daily_rets.loc[mask]
        if len(period_rets) == 0: continue

        # Portfolio daily return = sum of weight × asset daily return
        assets = [t for t in weights if t in daily_rets.columns]
        if not assets:
            # Cash-only period
            for day in period_rets.index:
                port_val = port_val  # unchanged
                nav_series.append(port_val); nav_dates.append(day)
            continue

        w_vec = np.array([weights.get(t, 0) for t in assets])
        ret_mat = period_rets[assets].fillna(0).values  # shape: (n_days, n_assets)
        port_daily = ret_mat @ w_vec  # shape: (n_days,)

        for j, day in enumerate(period_rets.index):
            port_val *= (1 + port_daily[j])
            nav_series.append(port_val); nav_dates.append(day)

    nav = pd.Series(nav_series, index=pd.DatetimeIndex(nav_dates))
    nav = nav[~nav.index.duplicated(keep='first')].sort_index()
    nav = nav.dropna()

    # Metrics
    dd_series = (nav - nav.cummax()) / nav.cummax()
    max_dd = float(dd_series.min())
    trough_idx = int(dd_series.argmin())
    peak_idx = int(nav.iloc[:max(trough_idx,1)].argmax())

    mo_nav = nav.resample('ME').last()
    mo_ret = mo_nav.pct_change().dropna()
    sharpe = float(mo_ret.mean() / mo_ret.std() * np.sqrt(12)) if mo_ret.std() > 0 else 0
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1/yrs) - 1)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly_reported = -0.100

    print("\n" + "=" * 72)
    print("📊 v10d 日频审计结果")
    print("=" * 72)
    print(f"  月频报告 MaxDD:  {monthly_reported:.1%}")
    print(f"  日频真实 MaxDD: {max_dd:.2%}")
    print(f"  低估倍数:        {abs(max_dd)/abs(monthly_reported):.2f}x")
    print(f"  峰值日期:        {nav.index[peak_idx].date()}")
    print(f"  谷底日期:        {nav.index[trough_idx].date()}")
    print(f"  日频 CAGR:      {cagr:.1%}")
    print(f"  日频 Sharpe:    {sharpe:.2f}")
    print(f"  日频 Calmar:    {calmar:.2f}")

    print(f"\n  历史对比:")
    print(f"  v9f/v9g: 月频-14.9% → 日频-26.5% (1.78x)")
    print(f"  v9i/v9j: 月频-10.3% → 日频约-18%  (估1.75x)")
    print(f"  v10d:    月频-10.0% → 日频{max_dd:.1%}  ({abs(max_dd)/0.10:.2f}x)")

    target = -0.25
    if max_dd > target:
        print(f"\n✅ 日频MaxDD {max_dd:.1%} < 25% 目标: PASS")
    else:
        print(f"\n⚠️ 日频MaxDD {max_dd:.1%} 超过25%目标！")

    out = {
        'monthly_reported': monthly_reported,
        'daily_max_dd': max_dd,
        'ratio': abs(max_dd) / abs(monthly_reported),
        'peak_date': str(nav.index[peak_idx].date()),
        'trough_date': str(nav.index[trough_idx].date()),
        'cagr': cagr, 'sharpe': sharpe, 'calmar': calmar,
    }
    jf = Path(__file__).parent / "daily_audit_v10d_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Saved → {jf}")
    return out


if __name__ == '__main__':
    main()
