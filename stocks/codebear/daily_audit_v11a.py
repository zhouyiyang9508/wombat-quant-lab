#!/usr/bin/env python3
"""
v11a Master 日频审计 — 真实 MaxDD 验证
代码熊 🐻 | 2026-02-22

方法（与 daily_audit_v10d.py 完全一致）：
  1. 导入 v11a 模块，获取月频持仓逻辑
  2. 构建所有资产的日频收益矩阵
  3. 月频调仓，日频追踪净值
  4. 计算真实日频 MaxDD

v11a 相比 v10d 的新增资产：
  - XLV, XLP, XLU（防御行业桥 ETF，软牛期触发）
  - IEF（已有，自适应债券）

预期结论（来自 Round 5 估算）：
  月频 MaxDD = -9.5%
  推算日频 MaxDD ≈ -22.6%（= -9.5% × 2.38x）
  目标 < 25% → 预期 ✅ PASS
"""

import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# 导入 v11a 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v11a", Path(__file__).parent / "momentum_v11a_master.py")
v11a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v11a)


def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    df = df.set_index(c).sort_index()
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
    print("🔍 v11a Master 日频审计 — 真实 MaxDD 验证")
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

    # 防御行业 ETF（v11a 新增）
    xlv_p = load_csv(CACHE / "XLV.csv")['Close'].dropna()
    xlp_p = load_csv(CACHE / "XLP.csv")['Close'].dropna()
    xlu_p = load_csv(CACHE / "XLU.csv")['Close'].dropna()
    def_prices = {'XLV': xlv_p, 'XLP': xlp_p, 'XLU': xlu_p}

    sig = v11a.precompute(close_df)

    # 构建完整价格矩阵（股票 + 所有对冲 ETF）
    hedge_df = pd.DataFrame({
        'GLD': gld_p, 'GDX': gdx_p, 'GDXJ': gdxj_p,
        'SHY': shy_p, 'TLT': tlt_p, 'IEF': ief_p,
        'XLV': xlv_p, 'XLP': xlp_p, 'XLU': xlu_p,
    })
    all_prices = pd.concat([close_df, hedge_df], axis=1)
    all_prices = all_prices.loc['2014-01-01':'2025-12-31'].ffill()
    daily_rets = all_prices.pct_change()
    print(f"  Price matrix: {all_prices.shape[0]} days × {all_prices.shape[1]} assets")

    HEDGE_KEYS = v11a.HEDGE_KEYS  # {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'XLV', 'XLP', 'XLU'}

    # ── Step 1: 月频调仓，记录每期持仓 ──────────────────────────────────────
    print("  Computing monthly rebalances (v11a logic)...")
    rng = close_df.loc['2015-01-01':'2025-12-31'].dropna(how='all')
    month_ends = rng.resample('ME').last().index

    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None
    periods = []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []

    for i in range(len(month_ends) - 1):
        dt, ndt = month_ends[i], month_ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0

        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15

        # SPY 1m 收益（用于 v9m 软对冲触发）
        spy_1m = v11a.get_spy_1m(sig, dt)

        # 组合波动率（月频收益，用于 vol targeting）
        if len(port_returns) >= v11a.VOL_LOOKBACK:
            pv = np.std(port_returns[-v11a.VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        # v11a 选股（含三大创新层）
        w, reg, bond_t = v11a.select(sig, sectors, dt, prev_h,
                                      gld_p, gdx_p, tlt_p, ief_p, def_prices)
        w, shy_boost = v11a.apply_overlays(w, spy_vol, dd, pv, spy_1m)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2

        # 显式加入 SHY 现金
        invested = sum(w.values())
        cash = max(1.0 - invested, 0.0) + shy_boost
        if cash > 0: w['SHY'] = w.get('SHY', 0) + cash

        periods.append((dt, ndt, dict(w), float(to)))
        prev_w = {k: v for k, v in w.items() if k != 'SHY'}
        prev_h = {k for k in prev_w if k not in HEDGE_KEYS}

        # 月度收益（用于 vol targeting）
        ret = 0.0
        hedge_map = {'GLD': gld_p, 'GDX': gdx_p, 'GDXJ': gdxj_p,
                     'SHY': shy_p, 'TLT': tlt_p, 'IEF': ief_p,
                     'XLV': xlv_p, 'XLP': xlp_p, 'XLU': xlu_p}
        for t, wt in w.items():
            p = hedge_map.get(t, close_df[t] if t in close_df.columns else None)
            if p is None: continue
            s = p.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1] / s.iloc[0] - 1) * wt
        ret -= to * 0.0015 * 2
        val *= (1 + ret); peak = max(peak, val)
        port_returns.append(ret)

    print(f"  ✓ {len(periods)} 个月频调仓期已处理")

    # ── Step 2: 日频净值追踪（向量化）───────────────────────────────────────
    print("  Computing daily NAV (vectorized)...")
    nav_series = [1.0]
    nav_dates  = [month_ends[0]]
    port_val   = 1.0

    for dt, ndt, weights, turnover in periods:
        port_val *= (1.0 - turnover * 0.0015 * 2)

        mask = (daily_rets.index > dt) & (daily_rets.index <= ndt)
        period_rets = daily_rets.loc[mask]
        if len(period_rets) == 0: continue

        assets = [t for t in weights if t in daily_rets.columns]
        if not assets:
            for day in period_rets.index:
                nav_series.append(port_val); nav_dates.append(day)
            continue

        w_vec   = np.array([weights.get(t, 0) for t in assets])
        ret_mat = period_rets[assets].fillna(0).values
        port_daily = ret_mat @ w_vec

        for j, day in enumerate(period_rets.index):
            port_val *= (1 + port_daily[j])
            nav_series.append(port_val); nav_dates.append(day)

    nav = pd.Series(nav_series, index=pd.DatetimeIndex(nav_dates))
    nav = nav[~nav.index.duplicated(keep='first')].sort_index().dropna()

    # ── 计算指标 ────────────────────────────────────────────────────────────
    dd_series = (nav - nav.cummax()) / nav.cummax()
    max_dd    = float(dd_series.min())
    trough_idx = int(dd_series.argmin())
    peak_idx   = int(nav.iloc[:max(trough_idx, 1)].argmax())

    mo_nav = nav.resample('ME').last()
    mo_ret = mo_nav.pct_change().dropna()
    sharpe = float(mo_ret.mean() / mo_ret.std() * np.sqrt(12)) if mo_ret.std() > 0 else 0
    yrs  = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    monthly_reported = -0.095  # v11a 月频报告值

    print("\n" + "=" * 72)
    print("📊 v11a Master 日频审计结果")
    print("=" * 72)
    print(f"  月频报告 MaxDD:   {monthly_reported:.1%}")
    print(f"  日频真实 MaxDD:  {max_dd:.2%}")
    print(f"  低估倍数:         {abs(max_dd)/abs(monthly_reported):.2f}x")
    print(f"  峰值日期:         {nav.index[peak_idx].date()}")
    print(f"  谷底日期:         {nav.index[trough_idx].date()}")
    print(f"  日频 CAGR:       {cagr:.1%}")
    print(f"  日频 Sharpe:     {sharpe:.2f}  (月度计算)")
    print(f"  日频 Calmar:     {calmar:.2f}")
    print(f"\n  历史低估倍数对比:")
    print(f"    v9f/v9g: 月频-14.9% → 日频-26.5% (1.78x)")
    print(f"    v10d:    月频-10.0% → 日频-23.8% (2.38x)")
    print(f"    v11a:    月频{monthly_reported:.1%} → 日频{max_dd:.1%}  ({abs(max_dd)/abs(monthly_reported):.2f}x)")

    target = -0.25
    pass_flag = max_dd > target
    print(f"\n  目标: 日频 MaxDD < 25%")
    if pass_flag:
        print(f"  ✅ PASS：{max_dd:.2%} > {target:.0%}")
    else:
        print(f"  ❌ FAIL：{max_dd:.2%} < {target:.0%}！")

    # 保存结果
    out = {
        'monthly_reported': monthly_reported,
        'daily_max_dd': max_dd,
        'ratio': abs(max_dd) / abs(monthly_reported),
        'peak_date': str(nav.index[peak_idx].date()),
        'trough_date': str(nav.index[trough_idx].date()),
        'daily_cagr': cagr,
        'daily_sharpe': sharpe,
        'daily_calmar': calmar,
        'pass_25pct': pass_flag,
    }
    jf = Path(__file__).parent / "daily_audit_v11a_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Saved → {jf}")
    return out


if __name__ == '__main__':
    main()
