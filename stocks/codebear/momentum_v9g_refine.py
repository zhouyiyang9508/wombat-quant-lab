#!/usr/bin/env python3
"""
v9g 精调 — 动态集中度阈值 + 自适应权重微调
目标: 在 Composite > 1.75 的同时提升 WF ratio > 0.75

v9g发现:
  C_dynsecs: Comp=1.7589, WF=0.78  (稳健)
  AC:        Comp=1.7648, WF=0.72  (最高)
  ABC_5:     Comp=1.7632, WF=0.72

核心问题: 动态集中度参数优化
  当前: breadth > 0.65 → 4 secs; else 5 secs
  测试: 不同阈值 (0.55, 0.60, 0.65, 0.70, 0.75)
  以及: 强牛市时 3 secs (极度集中)

自适应权重精调:
  当前强牛: (0.28, 0.50, 0.14, 0.08)
  测试: (0.25, 0.52, 0.15, 0.08), (0.30, 0.48, 0.14, 0.08)
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

BASE_MOM_W       = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS      = 5
BULL_SPS         = 2
BEAR_SPS         = 2
BREADTH_NARROW   = 0.45
GLD_AVG_THRESH   = 0.70
GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH   = 0.20
GDX_COMPETE_FRAC = 0.04
CONT_BONUS       = 0.03
HI52_FRAC        = 0.60
USE_SHY          = True
DD_PARAMS        = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30
GDXJ_VOL_LO_FRAC   = 0.08
GDXJ_VOL_HI_THRESH = 0.45
GDXJ_VOL_HI_FRAC   = 0.18


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
    s50   = spy.rolling(50).mean()  if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200, s50=s50,
                sma50=sma50, close=close_df)

def get_spy_vol(sig, date):
    if sig['vol5'] is None or 'SPY' not in sig['vol5'].columns: return 0.15
    v = sig['vol5']['SPY'].loc[:date].dropna()
    return float(v.iloc[-1]) if len(v) > 0 else 0.15

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
    if len(spy_now) == 0 or len(s200_now) == 0: return 'bull'
    return 'bear' if (spy_now.iloc[-1] < s200_now.iloc[-1] and
                      compute_breadth(sig, date) < BREADTH_NARROW) else 'bull'

def get_spy_trend(sig, date):
    if sig['spy'] is None or sig['s50'] is None: return 1.0, 0.5
    spy_now = sig['spy'].loc[:date].dropna()
    s50_now = sig['s50'].loc[:date].dropna()
    if len(spy_now) == 0 or len(s50_now) == 0: return 1.0, 0.5
    ratio   = float(spy_now.iloc[-1]) / float(s50_now.iloc[-1]) if float(s50_now.iloc[-1]) > 0 else 1.0
    breadth = compute_breadth(sig, date)
    return ratio, breadth

def get_adaptive_mom_weights(sig, date, cfg):
    ratio, breadth = get_spy_trend(sig, date)
    bull_hi_w  = cfg.get('bull_hi_w',  (0.28, 0.50, 0.14, 0.08))
    bull_lo_w  = cfg.get('bull_lo_w',  (0.10, 0.38, 0.35, 0.17))
    hi_thresh  = cfg.get('spy_s50_hi', 1.03)
    breadth_hi = cfg.get('breadth_hi', 0.55)

    if ratio > hi_thresh and breadth > breadth_hi:
        return bull_hi_w
    elif ratio < 0.99:
        return bull_lo_w
    else:
        return BASE_MOM_W

def asset_compete(sig, date, prices, thresh, frac, lb=127):
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lb] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0

def select(sig, sectors, date, prev_hold, gld_p, gdx_p, cfg):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
    d = idx[-1]

    if cfg.get('adaptive_mom', False):
        w1, w3, w6, w12 = get_adaptive_mom_weights(sig, date, cfg)
    else:
        w1, w3, w6, w12 = BASE_MOM_W

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
    for t in df.index:
        if t in prev_hold: df.loc[t, 'mom'] += CONT_BONUS
    if len(df) == 0: return {}

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg = get_regime(sig, date)

    # Dynamic sector concentration
    if cfg.get('dynamic_secs', False):
        _, breadth = get_spy_trend(sig, date)
        hi_thresh = cfg.get('breadth_conc_thresh', 0.65)
        n_at_hi   = cfg.get('n_secs_hi', 4)
        if reg == 'bull' and breadth > hi_thresh:
            n_bull_target = n_at_hi
        else:
            n_bull_target = N_BULL_SECS
    else:
        n_bull_target = N_BULL_SECS

    if reg == 'bull':
        n_secs = n_bull_target - n_compete
        sps, cash = BULL_SPS, 0.0
    else:
        n_secs = 3 - n_compete
        sps, cash = BEAR_SPS, 0.20
    n_secs = max(n_secs, 1)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - cash - total_compete, 0.0)
    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        return w

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = gld_a
    if gdx_a > 0: weights['GDX'] = gdx_a
    return weights

def apply_overlays(weights, spy_vol, dd, cfg=None):
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    total = gdxj_v + gld_dd
    if total <= 0 or not weights: return weights
    stock_frac = max(1.0 - total, 0.01)
    tot = sum(weights.values())
    if tot <= 0: return weights
    new = {t: w/tot*stock_frac for t, w in weights.items()}
    if gld_dd > 0: new['GLD'] = new.get('GLD', 0) + gld_dd
    if gdxj_v > 0: new['GDXJ'] = new.get('GDXJ', 0) + gdxj_v
    return new

def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, cfg,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = get_spy_vol(sig, dt)
        w = select(sig, sectors, dt, prev_h, gld_p, gdx_p, cfg)
        w = apply_overlays(w, spy_vol, dd)
        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t,0) - prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD','GDX','GDXJ','IEF')}
        invested = sum(w.values()); cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t in close_df.columns: s = close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt
        if USE_SHY and cash_frac > 0 and shy_p is not None:
            s = shy_p.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * cash_frac
        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)
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


# Refined configs: focus on C threshold tuning
REFINE_CONFIGS = {
    # Dynamic secs threshold sweep
    'C_th60':   {'dynamic_secs': True, 'breadth_conc_thresh': 0.60, 'n_secs_hi': 4, 'adaptive_mom': False},
    'C_th65':   {'dynamic_secs': True, 'breadth_conc_thresh': 0.65, 'n_secs_hi': 4, 'adaptive_mom': False},
    'C_th70':   {'dynamic_secs': True, 'breadth_conc_thresh': 0.70, 'n_secs_hi': 4, 'adaptive_mom': False},
    'C_th75':   {'dynamic_secs': True, 'breadth_conc_thresh': 0.75, 'n_secs_hi': 4, 'adaptive_mom': False},
    'C_th60_3': {'dynamic_secs': True, 'breadth_conc_thresh': 0.60, 'n_secs_hi': 3, 'adaptive_mom': False},
    'C_th65_3': {'dynamic_secs': True, 'breadth_conc_thresh': 0.65, 'n_secs_hi': 3, 'adaptive_mom': False},
    # AC combos with different thresholds
    'AC_th60':  {'dynamic_secs': True, 'breadth_conc_thresh': 0.60, 'n_secs_hi': 4, 'adaptive_mom': True},
    'AC_th70':  {'dynamic_secs': True, 'breadth_conc_thresh': 0.70, 'n_secs_hi': 4, 'adaptive_mom': True},
    'AC_th75':  {'dynamic_secs': True, 'breadth_conc_thresh': 0.75, 'n_secs_hi': 4, 'adaptive_mom': True},
    # v9g base for reference
    'C_th65_ref': {'dynamic_secs': True, 'breadth_conc_thresh': 0.65, 'n_secs_hi': 4, 'adaptive_mom': False},
}

def main():
    print("=" * 70)
    print("🐻 v9g 精调 — 动态集中度阈值优化")
    print("=" * 70)
    print(f"\nBase: v9g AC champion (Composite 1.7648, WF 0.72)")
    print(f"Goal: Find Composite > 1.75 with WF > 0.75\n")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig    = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers\n")

    results = {}
    for name, cfg in REFINE_CONFIGS.items():
        print(f"--- {name} ---", flush=True)
        try:
            eq_f, to = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, cfg)
            eq_is, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, cfg,
                                     '2015-01-01', '2020-12-31')
            eq_oo, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, cfg,
                                     '2021-01-01', '2025-12-31')
            m  = compute_metrics(eq_f)
            mi = compute_metrics(eq_is)
            mo = compute_metrics(eq_oo)
            wf   = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            results[name] = {'full': m, 'is': mi, 'oos': mo, 'wf': wf, 'composite': comp}
            wf_ok = '✅' if wf >= 0.75 else ('⚠️' if wf >= 0.70 else '❌')
            tag   = '🚀🚀' if comp > 1.77 else ('🚀' if comp > 1.70 else '')
            print(f"  Comp={comp:.4f} {tag} | Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} "
                  f"DD={m['max_dd']:.1%} WF={wf:.2f} {wf_ok}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print("📊 RANKINGS (sorted by Composite)")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda x: x[1]['composite'], reverse=True)
    for i, (n, r) in enumerate(ranked):
        m = r['full']
        wf_ok = '✅' if r['wf'] >= 0.75 else ('⚠️' if r['wf'] >= 0.70 else '❌')
        print(f"  #{i+1} {n:18s}: Comp={r['composite']:.4f} | "
              f"Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%} WF={r['wf']:.2f} {wf_ok}")

    # Best with WF >= 0.75
    valid = [(n, r) for n, r in results.items() if r['wf'] >= 0.75]
    if valid:
        best_robust = max(valid, key=lambda x: x[1]['composite'])
        print(f"\n🏆 Best with WF≥0.75: {best_robust[0]} → Composite {best_robust[1]['composite']:.4f}")

    best_all = max(results.items(), key=lambda x: x[1]['composite'])
    print(f"🏆 Best overall: {best_all[0]} → Composite {best_all[1]['composite']:.4f}")

    out = {'results': {n: {'full': r['full'], 'is': r['is'], 'oos': r['oos'],
                           'wf': r['wf'], 'composite': r['composite']} for n, r in results.items()}}
    jf = Path(__file__).parent / "momentum_v9g_refine_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results

if __name__ == '__main__':
    main()
