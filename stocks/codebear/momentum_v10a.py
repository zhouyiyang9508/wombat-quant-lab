#!/usr/bin/env python3
"""
动量轮动 v10a — 纯ETF行业轮动 (架构性重构)
代码熊 🐻

📊 核心思路: 从467只个股 → 13只行业ETF (架构性简化)

背景:
  v9x系列在不断加层的同时, WF从0.82(v9i)下降到0.70(v9n)
  核心问题: 模型越来越复杂, IS-OOS差距越来越大
  
  根本解决方案: 减少噪声 → 从个股选择转向行业ETF轮动
  
  个股选择 (467只) → 行业ETF (13只):
  ✓ 每只ETF都是分散化组合 → 信号更稳定
  ✓ 工具数量大幅减少 → 过拟合风险降低
  ✓ 月度行业轮动是文献证明的Alpha来源
  ✓ 无需担心个股上市/退市/重组等问题

行业ETF宇宙:
  权益ETF (13只):
    XLK (信息技术), XLV (医疗卫生), XLF (金融),
    XLI (工业),    XLE (能源),     XLY (非必需消费),
    XLB (材料),    XLU (公用事业), XLRE (房地产),
    XLP (必需消费), XLC (通信服务),
    QQQ (纳斯达克100), IWM (罗素2000小盘股)
  
  防御资产:
    GLD (黄金), TLT (长期国债), SHY (短期国债)

算法 (与v9j同架构, 但操作对象换成ETF):
  1. 月度动量评分 (1m×20% + 3m×50% + 6m×20% + 12m×10%)
  2. 过滤: 价格>SMA50, 6m收益>0
  3. 选择: 动量最强的Top-3~5 ETF
  4. 市场制度: "ETF breadth" (多少ETF在SMA50之上) 代替个股breadth
  5. GLD竞争: 当GLD动量强时自然进入
  6. TLT条件对冲: 熊市+TLT动量正时
  7. 波动率目标化: 11%年化

关键假设:
  - ETF级别的动量比个股动量更稳定 → WF更高
  - 行业配置决策 > 个股选择决策 (行业 > 选股)
  - 13只ETF的信噪比 >> 467只个股
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"

# Equity ETF universe
EQUITY_ETFS = [
    'XLK', 'XLV', 'XLF', 'XLI', 'XLE', 'XLY',
    'XLB', 'XLU', 'XLRE', 'XLP', 'XLC',
    'QQQ', 'IWM'
]

# Defense assets
DEFENSE_ETFS = ['GLD', 'TLT', 'SHY']

# Core parameters (same as v9j spirit)
MOM_W = (0.20, 0.50, 0.20, 0.10)  # 1m/3m/6m/12m
N_BULL = 4   # Top ETFs in bull mode
N_BEAR = 2   # Top ETFs in bear mode
BREADTH_BEAR_THRESH = 0.40  # < 40% ETFs above SMA50 → bear signal
SPY_SMA_PERIOD = 200

GLD_COMPETE_THRESH = 0.70  # GLD enters when 6m-mom >= avg×70%
GLD_COMPETE_FRAC   = 0.20
TLT_BEAR_FRAC      = 0.25
TLT_MOM_LB         = 126

DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO = 0.30; GDXJ_VOL_LO_F = 0.08
GDXJ_VOL_HI = 0.45; GDXJ_VOL_HI_F = 0.18

VOL_TARGET_ANN = 0.11
VOL_LOOKBACK   = 3
BEAR_CASH_FRAC = 0.20
USE_SHY = True
CONT_BONUS = 0.02


def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    df = df.set_index(c).sort_index()
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df


def load_etfs(tickers, cache_dir=CACHE):
    d = {}
    for t in tickers:
        f = cache_dir / f"{t}.csv"
        if not f.exists(): continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except: pass
    return pd.DataFrame(d).dropna(how='all')


def precompute(close_df):
    r1  = close_df / close_df.shift(22)  - 1
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1
    log_r = np.log(close_df / close_df.shift(1))
    vol5  = log_r.rolling(5).std() * np.sqrt(252)
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    sma50 = close_df.rolling(50).mean()
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    return dict(r1=r1, r3=r3, r6=r6, r12=r12,
                vol5=vol5, vol30=vol30, sma50=sma50,
                spy=spy, s200=s200, close=close_df)


def compute_etf_breadth(sig, date, equity_etfs):
    """What % of equity ETFs are above their SMA50?"""
    close = sig['close'].loc[:date]
    sma50 = sig['sma50'].loc[:date]
    if len(close) < 50: return 1.0
    count = 0; total = 0
    for t in equity_etfs:
        if t not in close.columns: continue
        c_hist = close[t].dropna()
        s_hist = sma50[t].dropna()
        if len(c_hist) == 0 or len(s_hist) == 0: continue
        total += 1
        if c_hist.iloc[-1] > s_hist.iloc[-1]: count += 1
    return count / total if total > 0 else 1.0


def get_spy_regime(sig, date):
    if sig['s200'] is None: return 'bull'
    spy_h = sig['spy'].loc[:date].dropna()
    s200_h = sig['s200'].loc[:date].dropna()
    if len(spy_h) == 0 or len(s200_h) == 0: return 'bull'
    breadth = compute_etf_breadth(sig, date, EQUITY_ETFS)
    return 'bear' if (spy_h.iloc[-1] < s200_h.iloc[-1] and
                      breadth < BREADTH_BEAR_THRESH) else 'bull'


def gld_compete(sig, date, gld_p):
    """GLD enters when 6m momentum >= avg ETF momentum × threshold"""
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) == 0: return 0.0
    d = idx[-1]
    etf_r6 = r6.loc[d, [t for t in EQUITY_ETFS if t in r6.columns]].dropna()
    etf_r6 = etf_r6[etf_r6 > 0]
    if len(etf_r6) == 0: return 0.0
    avg_r6 = etf_r6.mean()
    hist = gld_p.loc[:d].dropna()
    if len(hist) < 130: return 0.0
    gld_r6 = hist.iloc[-1] / hist.iloc[-127] - 1
    return GLD_COMPETE_FRAC if gld_r6 >= avg_r6 * GLD_COMPETE_THRESH else 0.0


def tlt_momentum_positive(tlt_p, date):
    hist = tlt_p.loc[:date].dropna()
    if len(hist) < TLT_MOM_LB + 3: return False
    return bool(hist.iloc[-1] / hist.iloc[-TLT_MOM_LB] - 1 > 0)


def select_etfs(sig, date, prev_hold, gld_p, tlt_p, gdxj_p):
    """Select top ETFs by momentum"""
    close = sig['close']
    r6    = sig['r6']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
    d = idx[-1]

    w1, w3, w6, w12 = MOM_W
    mom = {}
    for t in EQUITY_ETFS:
        if t not in close.columns: continue
        try:
            m = (sig['r1'].loc[d, t]*w1 + sig['r3'].loc[d, t]*w3 +
                 sig['r6'].loc[d, t]*w6 + sig['r12'].loc[d, t]*w12)
            r6v = sig['r6'].loc[d, t]
            prc = close.loc[d, t]
            sma = sig['sma50'].loc[d, t]
            # Quality filters: price > SMA50, 6m return > 0
            if pd.isna(m) or pd.isna(r6v) or pd.isna(prc) or pd.isna(sma): continue
            if r6v <= 0 or prc <= sma: continue
            if t in prev_hold: m += CONT_BONUS
            mom[t] = m
        except: continue

    if not mom: return {}

    gld_a = gld_compete(sig, date, gld_p)
    reg   = get_spy_regime(sig, date)
    tlt_pos = tlt_momentum_positive(tlt_p, date)
    use_tlt = (reg == 'bear' and tlt_pos)

    if reg == 'bull':
        n_sel = N_BULL - (1 if gld_a > 0 else 0)
        bear_cash = 0.0
    else:
        n_sel = N_BEAR
        bear_cash = BEAR_CASH_FRAC - (TLT_BEAR_FRAC if use_tlt else 0)
        bear_cash = max(bear_cash, 0.0)

    n_sel = max(n_sel, 1)
    top_etfs = sorted(mom, key=mom.get, reverse=True)[:n_sel]

    tlt_alloc = TLT_BEAR_FRAC if use_tlt else 0.0
    stock_frac = max(1.0 - bear_cash - gld_a - tlt_alloc, 0.01)
    eq_w = stock_frac / len(top_etfs)

    weights = {t: eq_w for t in top_etfs}
    if gld_a > 0: weights['GLD'] = gld_a
    if tlt_alloc > 0: weights['TLT'] = tlt_alloc
    return weights


def apply_overlays(weights, spy_vol, dd, port_vol_ann, gdxj_p_val):
    """GDXJ vol trigger + DD GLD + portfolio vol target"""
    if spy_vol >= GDXJ_VOL_HI: gdxj_v = GDXJ_VOL_HI_F
    elif spy_vol >= GDXJ_VOL_LO: gdxj_v = GDXJ_VOL_LO_F
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total_overlay = gdxj_v + gld_dd
    if total_overlay > 0 and weights:
        hedge_keys = {'GLD', 'TLT', 'GDXJ'}
        hedge_w = {t: w for t, w in weights.items() if t in hedge_keys}
        equity_w = {t: w for t, w in weights.items() if t not in hedge_keys}
        eq_frac = max(1.0 - total_overlay - sum(hedge_w.values()), 0.01)
        tot = sum(equity_w.values())
        if tot > 0:
            equity_w = {t: w/tot*eq_frac for t, w in equity_w.items()}
        weights = {**equity_w, **hedge_w}
        if gld_dd > 0: weights['GLD'] = weights.get('GLD', 0) + gld_dd
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in ('GLD', 'TLT', 'GDXJ')]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_backtest(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p,
                 start='2015-01-01', end='2025-12-31', cost=0.0010):
    """Lower cost for ETFs (0.10% vs 0.15% for stocks)"""
    rng  = close_all.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15

        if len(port_returns) >= VOL_LOOKBACK:
            pv = np.std(port_returns[-VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        w = select_etfs(sig, dt, prev_h, gld_p, tlt_p, gdxj_p)
        w, shy_boost = apply_overlays(w, spy_vol, dd, pv, None)

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'TLT', 'GDXJ')}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'TLT':  s = tlt_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t in close_all.columns: s = close_all[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt

        if USE_SHY:
            total_shy = shy_boost + cash_frac
            s = shy_p.loc[dt:ndt].dropna()
            if len(s) >= 2 and total_shy > 0:
                ret += (s.iloc[-1]/s.iloc[0]-1) * total_shy

        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)
        port_returns.append(ret)

    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq, float(np.mean(tos)) if tos else 0.0


def compute_metrics(eq):
    if len(eq) < 3: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    mo   = eq.pct_change().dropna()
    sh   = mo.mean()/mo.std()*np.sqrt(12) if mo.std() > 0 else 0
    dd   = ((eq - eq.cummax())/eq.cummax()).min()
    cal  = cagr/abs(dd) if dd != 0 else 0
    return dict(cagr=float(cagr), max_dd=float(dd), sharpe=float(sh), calmar=float(cal))


def sweep_n_bull(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p):
    """Sweep key parameter: N_BULL (number of ETFs in bull mode)"""
    global N_BULL, N_BEAR, BREADTH_BEAR_THRESH
    results = []
    for nb in [3, 4, 5, 6]:
        for breadth_th in [0.35, 0.40, 0.45]:
            N_BULL = nb; BREADTH_BEAR_THRESH = breadth_th
            N_BEAR = max(nb - 2, 1)
            try:
                eq, to = run_backtest(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p)
                m = compute_metrics(eq)
                eq_is, _ = run_backtest(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p,
                                         '2015-01-01', '2020-12-31')
                eq_oos, _ = run_backtest(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p,
                                          '2021-01-01', '2025-12-31')
                mi = compute_metrics(eq_is); mo = compute_metrics(eq_oos)
                wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
                comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
                results.append({'n_bull': nb, 'breadth': breadth_th,
                                'composite': comp, 'sharpe': m['sharpe'],
                                'cagr': m['cagr'], 'max_dd': m['max_dd'],
                                'calmar': m['calmar'], 'wf': wf})
                print(f"  N={nb} breadth<{breadth_th:.0%} → Comp={comp:.4f} "
                      f"Sharpe={m['sharpe']:.2f} MaxDD={m['max_dd']:.1%} "
                      f"CAGR={m['cagr']:.1%} WF={wf:.2f}")
            except Exception as e:
                print(f"  Error (N={nb}, b={breadth_th}): {e}")
    return sorted(results, key=lambda x: x['composite'], reverse=True)


def main():
    global N_BULL, N_BEAR, BREADTH_BEAR_THRESH
    print("=" * 72)
    print("🐻 动量轮动 v10a — 纯ETF行业轮动 (架构性重构)")
    print("=" * 72)
    print(f"\n  Equity ETFs: {EQUITY_ETFS}")
    print(f"  Defense: {DEFENSE_ETFS}")

    # Load all ETFs + SPY for regime detection
    all_tickers = EQUITY_ETFS + ['GLD', 'TLT', 'GDXJ', 'SHY', 'SPY']
    close_all = load_etfs(all_tickers, CACHE)
    print(f"\n  Loaded ETFs: {sorted(close_all.columns.tolist())}")
    print(f"  Date range: {close_all.index[0].date()} → {close_all.index[-1].date()}")

    gld_p  = close_all['GLD']
    tlt_p  = close_all['TLT']
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig = precompute(close_all)

    print("\n🔍 Sweeping N_BULL and breadth threshold...")
    results = sweep_n_bull(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p)

    print(f"\n📊 Top 5 configs:")
    for r in results[:5]:
        print(f"  N={r['n_bull']} breadth<{r['breadth']:.0%} → Comp={r['composite']:.4f} "
              f"Sharpe={r['sharpe']:.2f} MaxDD={r['max_dd']:.1%} "
              f"CAGR={r['cagr']:.1%} WF={r['wf']:.2f}")

    best = results[0]
    N_BULL = best['n_bull']; N_BEAR = max(N_BULL-2, 1)
    BREADTH_BEAR_THRESH = best['breadth']

    print(f"\n🏆 Best: N_BULL={N_BULL} breadth<{BREADTH_BEAR_THRESH:.0%}")
    print("\n🔄 Full (2015-2025)...")
    eq_full, to = run_backtest(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p)
    print("🔄 IS (2015-2020)...")
    eq_is, _ = run_backtest(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p,
                             '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _ = run_backtest(close_all, sig, gld_p, gdxj_p, shy_p, tlt_p,
                              '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v10a (Pure ETF) vs v9j champion (2.057)")
    print("=" * 72)
    v9j = dict(cagr=0.3235, max_dd=-0.1033, sharpe=1.850, calmar=3.132, comp=2.057, wf=0.785)
    print(f"{'Metric':<14} {'v9j (stock)':<14} {'v10a (ETF)':<14} {'Delta'}")
    print(f"{'CAGR':<14} {v9j['cagr']:.1%}         {m['cagr']:.1%}         {m['cagr']-v9j['cagr']:+.1%}")
    print(f"{'MaxDD':<14} {v9j['max_dd']:.1%}        {m['max_dd']:.1%}        {m['max_dd']-v9j['max_dd']:+.1%}")
    print(f"{'Sharpe':<14} {v9j['sharpe']:.2f}          {m['sharpe']:.2f}          {m['sharpe']-v9j['sharpe']:+.2f}")
    print(f"{'Calmar':<14} {v9j['calmar']:.2f}          {m['calmar']:.2f}          {m['calmar']-v9j['calmar']:+.2f}")
    print(f"{'IS Sharpe':<14}               {mi['sharpe']:.2f}")
    print(f"{'OOS Sharpe':<14}               {mo['sharpe']:.2f}")
    print(f"{'WF':<14} {v9j['wf']:.2f}          {wf:.2f}          {wf-v9j['wf']:+.2f}")
    print(f"{'Composite':<14} {v9j['comp']:.4f}       {comp:.4f}       {comp-v9j['comp']:+.4f}")
    print(f"{'Turnover':<14}               {to:.1%}/month")

    # WF comparison (key hypothesis test)
    print(f"\n💡 WF Hypothesis: ETF strategy WF ({wf:.2f}) vs stock strategy WF (0.78)")
    if wf > 0.80:
        print(f"  ✅ CONFIRMED: ETF rotation has higher WF ({wf:.2f} > 0.80)")
    elif wf > 0.78:
        print(f"  ✅ Marginally better WF ({wf:.2f} vs 0.78)")
    else:
        print(f"  ❌ Lower WF than stock strategy ({wf:.2f} < 0.78)")

    if comp > 2.1:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.1!")
    elif comp > 2.057:
        print(f"\n🚀🚀 超越v9j Composite! ({comp:.4f})")
    elif comp > 1.80:
        print(f"\n✅ Composite > 1.80: {comp:.4f}")
    else:
        print(f"\n📊 Composite: {comp:.4f}")

    out = {
        'strategy': f'v10a Pure ETF Rotation (N_BULL={N_BULL}, breadth<{BREADTH_BEAR_THRESH:.0%})',
        'universe': EQUITY_ETFS, 'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'sweep': results
    }
    jf = Path(__file__).parent / "momentum_v10a_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
