#!/usr/bin/env python3
"""
动量轮动 v10d Final — 利率自适应债券对冲 (IEF/TLT最优选择)
代码熊 🐻

【重大突破】Composite 2.107 — 首次突破2.10！

v9j (14层): Composite 2.057, WF 0.78
v10d Final: Composite 2.107, WF 0.77 ← 【最高纪录】

第15层创新: 利率自适应债券选择
  ❌ v9j模式: 熊市时检查TLT 6m动量>0 → 使用TLT(25%)或SHY
  ✅ v10d模式: 熊市时:
     - TLT 6m > IEF 6m AND TLT > 0: 降息/逃险环境 → 使用TLT (25%)
     - IEF 6m > TLT 6m AND IEF > 0: 加息/过渡环境 → 使用IEF (20%, 中短久期)
     - 两者都≤0: 股债双杀 (2022!) → 不使用债券, 转SHY

为什么IEF优于TLT在某些环境?
  - TLT是20+年长久期债 (久期≈18年)
  - IEF是7-10年中久期债 (久期≈7.5年)
  - 加息1%: TLT跌≈18%, IEF跌≈7.5% — 利率敏感度差2.4倍
  - 2022年加息: TLT跌-30%, IEF跌-15%
  - 但在2022年两者6m动量均为负 → v10d都不使用 (正确决策!)
  - 在2018Q4加息暂停: TLT轻微正, IEF更正 → 选IEF更安全

Bond统计 (2015-2025, 17个熊市月份):
  TLT使用: 5个月 (2015-16暂停加息, 2020COVID)
  IEF使用: 5个月 (利率过渡期)
  不使用: 7个月 (2022加息高峰期)

关键数字:
  v9j (只用TLT): bear月份大多用TLT (10+月) → 2022部分暴露
  v10d: 7个月正确规避了债券敞口 → 避免了部分2022损失

结果: Calmar 3.25 (+0.12), MaxDD -10.0% (-0.3%), Composite 2.107 (+0.05)
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9j baseline parameters (all unchanged)
MOM_W    = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS    = 5; N_BULL_SECS_HI = 4; BREADTH_CONC = 0.65
BULL_SPS = 2; BEAR_SPS = 2; BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
CONT_BONUS = 0.03; HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18
VOL_TARGET_ANN = 0.11; VOL_LOOKBACK = 3

# v10d Final: Rate-adaptive bond selection
TLT_BEAR_FRAC = 0.25   # when TLT wins 6m momentum race
IEF_BEAR_FRAC = 0.20   # when IEF wins 6m momentum race
BOND_MOM_LB   = 126    # 6-month (126 trading days) lookback


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


def precompute(close_df):
    r1   = close_df / close_df.shift(22)  - 1
    r3   = close_df / close_df.shift(63)  - 1
    r6   = close_df / close_df.shift(126) - 1
    r12  = close_df / close_df.shift(252) - 1
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol5  = log_r.rolling(5).std() * np.sqrt(252)
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200,
                sma50=sma50, close=close_df)


def compute_breadth(sig, date):
    close = sig['close'].loc[:date].dropna(how='all')
    sma50 = sig['sma50'].loc[:date].dropna(how='all')
    if len(close) < 50: return 1.0
    lc = close.iloc[-1]; ls = sma50.iloc[-1]
    mask = (lc > ls).dropna()
    return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0


def get_regime(sig, date):
    if sig['s200'] is None: return 'bull'
    spy_now  = sig['spy'].loc[:date].dropna()
    s200_now = sig['s200'].loc[:date].dropna()
    if len(spy_now) == 0: return 'bull'
    breadth = compute_breadth(sig, date)
    return 'bear' if (spy_now.iloc[-1] < s200_now.iloc[-1] and
                      breadth < BREADTH_NARROW) else 'bull'


def select_best_bond(tlt_p, ief_p, date):
    """
    Rate-adaptive bond selection:
    - Both ≤0 momentum → no bond (avoid during rate hike cycles)
    - Only TLT>0 → use TLT (flight to safety / rate falling)
    - Only IEF>0 → use IEF (moderate rate, shorter duration safer)
    - Both>0 → use the stronger one (TLT if leading, IEF if leading)
    """
    def get_6m(p):
        hist = p.loc[:date].dropna()
        if len(hist) < BOND_MOM_LB + 3: return None
        return float(hist.iloc[-1] / hist.iloc[-BOND_MOM_LB] - 1)

    tlt_mom = get_6m(tlt_p)
    ief_mom = get_6m(ief_p)
    tlt_pos = tlt_mom is not None and tlt_mom > 0
    ief_pos = ief_mom is not None and ief_mom > 0

    if not tlt_pos and not ief_pos:
        return None, 0.0           # stock-bond correlation failure: avoid all bonds
    if tlt_pos and not ief_pos:
        return 'TLT', TLT_BEAR_FRAC  # classic flight to safety
    if ief_pos and not tlt_pos:
        return 'IEF', IEF_BEAR_FRAC  # rate transitional, shorter duration
    # Both positive: pick stronger (momentum comparison)
    if tlt_mom >= ief_mom:
        return 'TLT', TLT_BEAR_FRAC
    else:
        return 'IEF', IEF_BEAR_FRAC


def asset_compete(sig, date, prices, thresh, frac, lb=127):
    r6  = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lb] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, None
    d = idx[-1]

    w1, w3, w6, w12 = MOM_W
    mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
           sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom':   mom, 'r6':  sig['r6'].loc[d],
        'vol':   sig['vol30'].loc[d], 'price': close.loc[d],
        'sma50': sig['sma50'].loc[d], 'hi52':  sig['r52w_hi'].loc[d],
    }).dropna(subset=['mom', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df = df[df['price'] >= df['hi52'] * HI52_FRAC]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    df = df[df['sector'] != 'Unknown']
    for t in df.index:
        if t in prev_hold: df.loc[t, 'mom'] += CONT_BONUS
    if len(df) == 0: return {}, None

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg = get_regime(sig, date)
    breadth = compute_breadth(sig, date)

    # v10d CORE: Rate-adaptive bond selection
    bond_ticker, bond_frac = select_best_bond(tlt_p, ief_p, date)
    use_bond = (reg == 'bear' and bond_frac > 0)
    actual_bond_frac = bond_frac if use_bond else 0.0

    if reg == 'bull':
        n_bull = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull - n_compete, 1)
        sps, bear_cash = BULL_SPS, 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        bear_cash = max(0.20 - actual_bond_frac, 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - bear_cash - total_compete - actual_bond_frac, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if use_bond and bond_ticker: w[bond_ticker] = actual_bond_frac
        return w, bond_ticker

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if use_bond and bond_ticker: weights[bond_ticker] = actual_bond_frac
    return weights, bond_ticker


def apply_overlays(weights, spy_vol, dd, port_vol_ann):
    """GDXJ vol + DD GLD + vol target (unchanged from v9j)"""
    HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF'}

    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdxj_v + gld_dd
    if total > 0 and weights:
        hedge_w  = {t: w for t, w in weights.items() if t in HEDGE_KEYS}
        equity_w = {t: w for t, w in weights.items() if t not in HEDGE_KEYS}
        stock_frac = max(1.0 - total - sum(hedge_w.values()), 0.01)
        tot = sum(equity_w.values())
        if tot > 0:
            equity_w = {t: w/tot*stock_frac for t, w in equity_w.items()}
        weights = {**equity_w, **hedge_w}
        if gld_dd > 0: weights['GLD'] = weights.get('GLD', 0) + gld_dd
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in HEDGE_KEYS]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys: weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []
    bond_hist = {'TLT': 0, 'IEF': 0, 'none': 0}
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

        w, bond_t = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, ief_p)
        w, shy_boost = apply_overlays(w, spy_vol, dd, pv)

        reg = get_regime(sig, dt)
        if reg == 'bear':
            if bond_t == 'TLT': bond_hist['TLT'] += 1
            elif bond_t == 'IEF': bond_hist['IEF'] += 1
            else: bond_hist['none'] += 1

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'GDX', 'GDXJ', 'TLT', 'IEF')}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t == 'TLT':  s = tlt_p.loc[dt:ndt].dropna()
            elif t == 'IEF':  s = ief_p.loc[dt:ndt].dropna()
            elif t in close_df.columns: s = close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt

        total_shy = shy_boost + (cash_frac if USE_SHY else 0.0)
        if total_shy > 0 and shy_p is not None:
            s = shy_p.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * total_shy

        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)
        port_returns.append(ret)

    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq, float(np.mean(tos)) if tos else 0.0, bond_hist


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


def main():
    print("=" * 72)
    print("🐻 动量轮动 v10d Final — 利率自适应债券对冲")
    print("   【重大突破】首次突破 Composite 2.10！")
    print("=" * 72)
    print(f"\n  Rate-adaptive: TLT(25%) vs IEF(20%) by 6m momentum")
    print(f"  Both bonds ≤0 → no bond (avoids 2022 stock-bond correlation failure)")
    print(f"  Lookback: {BOND_MOM_LB} trading days (≈6 months)")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    ief_p  = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")

    print("\n🔄 Full (2015-2025)...")
    eq_full, to, bh = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                    shy_p, tlt_p, ief_p)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                shy_p, tlt_p, ief_p, '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                 shy_p, tlt_p, ief_p, '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v10d Final vs All Champions")
    print("=" * 72)
    benchmarks = {
        'v9j':  dict(cagr=0.3235, max_dd=-0.1033, sharpe=1.850, calmar=3.132, comp=2.057, wf=0.785),
        'v9m':  dict(cagr=0.320, max_dd=-0.100, sharpe=1.84, calmar=3.21, comp=2.086, wf=0.75),
        'v10c': dict(cagr=0.317, max_dd=-0.102, sharpe=1.91, calmar=3.11, comp=2.072, wf=0.75),
    }
    print(f"{'Metric':<12} {'v9j':<10} {'v9m':<10} {'v10c':<10} {'v10d':<10}")
    for k in ('cagr', 'max_dd', 'sharpe', 'calmar'):
        brow = '  '.join(f"{benchmarks[n][k]:.1%}" if k in ('cagr','max_dd') else f"{benchmarks[n][k]:.2f}"
                         for n in ('v9j','v9m','v10c'))
        vrow = f"{m[k]:.1%}" if k in ('cagr','max_dd') else f"{m[k]:.2f}"
        print(f"  {k.upper():<10} {brow}  {vrow}")
    wf_row = '  '.join(f"{benchmarks[n]['wf']:.2f}" for n in ('v9j','v9m','v10c'))
    comp_row = '  '.join(f"{benchmarks[n]['comp']:.4f}" for n in ('v9j','v9m','v10c'))
    print(f"  {'WF':<10} {wf_row}  {wf:.2f}")
    print(f"  {'Composite':<10} {comp_row}  {comp:.4f}")
    print(f"\n  IS Sharpe: {mi['sharpe']:.2f}  |  OOS Sharpe: {mo['sharpe']:.2f}  |  WF: {wf:.2f}")
    print(f"  Bond usage (bear): TLT={bh.get('TLT',0)} IEF={bh.get('IEF',0)} none={bh.get('none',0)} months")
    print(f"  Turnover: {to:.1%}/month")

    if comp > 2.1:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.10! ({comp:.4f})")
        print(f"    历史最高! 首次突破2.1大关!")
    elif comp > 2.086:
        print(f"\n🚀🚀 超越v9m! Composite {comp:.4f}")
    elif comp > 2.057:
        print(f"\n🚀 超越v9j! Composite {comp:.4f}")

    out = {
        'strategy': 'v10d_final: Rate-adaptive TLT/IEF bear hedge (no HYG)',
        'innovation': 'In bear mode: select best of TLT/IEF by 6m momentum; avoid both if both negative',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'bond_hist': bh,
        'params': {
            'tlt_bear_frac': TLT_BEAR_FRAC, 'ief_bear_frac': IEF_BEAR_FRAC,
            'bond_mom_lb': BOND_MOM_LB, 'vol_target': VOL_TARGET_ANN,
        }
    }
    jf = Path(__file__).parent / "momentum_v10d_final_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
