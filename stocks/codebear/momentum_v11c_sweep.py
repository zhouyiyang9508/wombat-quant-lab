#!/usr/bin/env python3
"""
v11c 参数联合优化扫描 — CONT_BONUS + 防御比例
代码熊 🐻

目标: 找到比v11a更优的参数组合, 特别是提升WF

参数1: CONT_BONUS (持仓延续奖励)
  当前: 0.03 (3%)
  假设: 更高的延续奖励 → 减少换手 → 减少交易成本 → 更稳定
  
参数2: DEFENSIVE_FRAC (防御行业比例)
  当前: 0.15 (15% XLV/XLP/XLU in soft-bull)
  假设: 12% 可能在降低WF代价的同时保持大部分Composite提升

参数3: SPY_SOFT_HI_FRAC (SPY软对冲GLD比例)
  当前: 0.10 (10% GLD when SPY 1m < -7%)
  假设: 8% 可能更保守, WF更好

v11a基准: Composite 2.190, WF 0.742
目标: 找到 Composite > 2.18 AND WF > 0.76 的组合
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# Fixed parameters (v11a base)
MOM_W    = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS    = 5; N_BULL_SECS_HI = 4; BREADTH_CONC = 0.65
BULL_SPS = 2; BEAR_SPS = 2; BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18
VOL_TARGET_ANN = 0.11; VOL_LOOKBACK = 3
TLT_BEAR_FRAC = 0.25; IEF_BEAR_FRAC = 0.20; BOND_MOM_LB = 126
DEFENSIVE_ETFS = ['XLV', 'XLP', 'XLU']
SPY_SOFT_HI_THRESH = -0.07

# Sweep parameters (will be set per run)
CONT_BONUS     = 0.03
DEFENSIVE_FRAC = 0.15
DEFENSIVE_EACH = DEFENSIVE_FRAC / 3
SPY_SOFT_HI_FRAC = 0.10


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
    r1  = close_df / close_df.shift(22)  - 1
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1
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


def get_three_regime(sig, date):
    if sig['s200'] is None: return 'bull_hi', 1.0
    spy_now  = sig['spy'].loc[:date].dropna()
    s200_now = sig['s200'].loc[:date].dropna()
    if len(spy_now) == 0: return 'bull_hi', 1.0
    breadth = compute_breadth(sig, date)
    if spy_now.iloc[-1] < s200_now.iloc[-1] and breadth < BREADTH_NARROW:
        return 'bear', breadth
    elif breadth < BREADTH_CONC:
        return 'soft_bull', breadth
    else:
        return 'bull_hi', breadth


def get_spy_1m(sig, date):
    if sig['r1'] is None or 'SPY' not in sig['r1'].columns: return 0.0
    hist = sig['r1']['SPY'].loc[:date].dropna()
    return float(hist.iloc[-1]) if len(hist) > 0 else 0.0


def select_best_bond(tlt_p, ief_p, date):
    def get_6m(p):
        hist = p.loc[:date].dropna()
        if len(hist) < BOND_MOM_LB + 3: return None
        return float(hist.iloc[-1] / hist.iloc[-BOND_MOM_LB] - 1)
    tlt_mom = get_6m(tlt_p); ief_mom = get_6m(ief_p)
    tlt_pos = tlt_mom is not None and tlt_mom > 0
    ief_pos = ief_mom is not None and ief_mom > 0
    if not tlt_pos and not ief_pos: return None, 0.0
    if tlt_pos and not ief_pos: return 'TLT', TLT_BEAR_FRAC
    if ief_pos and not tlt_pos: return 'IEF', IEF_BEAR_FRAC
    return ('TLT', TLT_BEAR_FRAC) if (tlt_mom or 0) >= (ief_mom or 0) else ('IEF', IEF_BEAR_FRAC)


def asset_compete(sig, date, prices, thresh, frac, lb=127):
    r6  = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].dropna(); stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    return frac if hist.iloc[-1] / hist.iloc[-lb] - 1 >= avg_r6 * thresh else 0.0


HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'XLV', 'XLP', 'XLU'}


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
    d = idx[-1]

    w1, w3, w6, w12 = MOM_W
    mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
           sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom': mom, 'r6': sig['r6'].loc[d],
        'vol': sig['vol30'].loc[d], 'price': close.loc[d],
        'sma50': sig['sma50'].loc[d], 'hi52': sig['r52w_hi'].loc[d],
    }).dropna(subset=['mom', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df = df[df['price'] >= df['hi52'] * HI52_FRAC]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    df = df[df['sector'] != 'Unknown']
    for t in df.index:
        if t in prev_hold: df.loc[t, 'mom'] += CONT_BONUS
    if len(df) == 0: return {}

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)
    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg, breadth = get_three_regime(sig, date)
    bond_ticker, bond_frac = select_best_bond(tlt_p, ief_p, date)
    use_bond = (reg == 'bear' and bond_frac > 0)
    actual_bond_frac = bond_frac if use_bond else 0.0

    if reg == 'soft_bull':
        avail_def = [t for t, p in def_prices.items()
                     if p is not None and len(p.loc[:date].dropna()) > 0]
        def_alloc = {t: DEFENSIVE_FRAC/len(avail_def) for t in avail_def} if avail_def else {}
        def_frac_total = sum(def_alloc.values())
    else:
        def_alloc = {}; def_frac_total = 0.0

    if reg == 'bull_hi':
        n_secs = max(N_BULL_SECS_HI - n_compete, 1); sps = BULL_SPS; bear_cash = 0.0
    elif reg == 'soft_bull':
        n_secs = max(N_BULL_SECS - n_compete, 1); sps = BULL_SPS; bear_cash = 0.0
    else:
        n_secs = max(3 - n_compete, 1); sps = BEAR_SPS
        bear_cash = max(0.20 - actual_bond_frac, 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - bear_cash - total_compete - actual_bond_frac - def_frac_total, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if use_bond and bond_ticker: w[bond_ticker] = actual_bond_frac
        w.update(def_alloc)
        return w

    iv  = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn  = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw  = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if use_bond and bond_ticker: weights[bond_ticker] = actual_bond_frac
    weights.update(def_alloc)
    return weights


def apply_overlays(weights, spy_vol, dd, pv, spy_1m):
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    gld_soft = SPY_SOFT_HI_FRAC if spy_1m <= SPY_SOFT_HI_THRESH else 0.0
    gld_total = max(gld_dd, gld_soft)
    total_overlay = gdxj_v + gld_total
    if total_overlay > 0 and weights:
        hedge_w  = {t: w for t, w in weights.items() if t in HEDGE_KEYS}
        equity_w = {t: w for t, w in weights.items() if t not in HEDGE_KEYS}
        sf = max(1.0 - total_overlay - sum(hedge_w.values()), 0.01)
        tot = sum(equity_w.values())
        if tot > 0: equity_w = {t: w/tot*sf for t, w in equity_w.items()}
        weights = {**equity_w, **hedge_w}
        if gld_total > 0: weights['GLD'] = weights.get('GLD', 0) + gld_total
        if gdxj_v > 0:   weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v
    shy_boost = 0.0
    if pv > 0.01:
        scale = min(VOL_TARGET_ANN / max(pv, 0.01), 1.0)
        if scale < 0.98:
            eq_keys = [t for t in weights if t not in HEDGE_KEYS]
            eq_frac = sum(weights[t] for t in eq_keys)
            if eq_frac > 0:
                for t in eq_keys: weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)
    return weights, shy_boost


def run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
           start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0; pr = []
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        sv = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15
        pv = np.std(pr[-VOL_LOOKBACK:], ddof=1)*np.sqrt(12) if len(pr) >= VOL_LOOKBACK else 0.20
        spy_1m = get_spy_1m(sig, dt)
        w = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, ief_p, def_prices)
        w, shy_boost = apply_overlays(w, sv, dd, pv, spy_1m)
        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t,0) - prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in HEDGE_KEYS}
        invested = sum(w.values()); cash = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            hedge_map = {'GLD': gld_p, 'GDX': gdx_p, 'GDXJ': gdxj_p, 'TLT': tlt_p, 'IEF': ief_p}
            if t in hedge_map:
                s = hedge_map[t]
            elif t in def_prices and def_prices[t] is not None:
                s = def_prices[t]
            elif t in close_df.columns:
                s = close_df[t]
            else:
                continue
            seg = s.loc[dt:ndt].dropna()
            if len(seg) >= 2: ret += (seg.iloc[-1]/seg.iloc[0]-1) * wt
        if shy_boost + cash > 0:
            seg = shy_p.loc[dt:ndt].dropna()
            if len(seg) >= 2: ret += (seg.iloc[-1]/seg.iloc[0]-1) * (shy_boost + cash)
        ret -= to * cost * 2
        val *= (1 + ret); peak = max(peak, val)
        vals.append(val); dates.append(ndt); pr.append(ret)
    return pd.Series(vals, index=pd.DatetimeIndex(dates)), float(np.mean(tos)) if tos else 0


def metrics(eq):
    if len(eq) < 3: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    mo = eq.pct_change().dropna()
    sh = mo.mean()/mo.std()*np.sqrt(12) if mo.std() > 0 else 0
    dd = ((eq - eq.cummax())/eq.cummax()).min()
    return dict(cagr=float(cagr), max_dd=float(dd),
                sharpe=float(sh), calmar=float(cagr/abs(dd)) if dd!=0 else 0)


def main():
    global CONT_BONUS, DEFENSIVE_FRAC, DEFENSIVE_EACH, SPY_SOFT_HI_FRAC
    print("=" * 72)
    print("🐻 v11c 参数联合扫描 — CONT_BONUS + DEFENSIVE_FRAC + SPY_SOFT")
    print("=" * 72)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    ief_p  = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    def_prices = {t: load_csv(CACHE / f"{t}.csv")['Close'].dropna()
                  for t in ['XLV', 'XLP', 'XLU'] if (CACHE/f"{t}.csv").exists()}
    sig = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers")

    # Parameter grid
    cont_bonus_vals   = [0.02, 0.03, 0.04, 0.05]
    def_frac_vals     = [0.10, 0.12, 0.15, 0.18]
    spy_soft_vals     = [0.07, 0.08, 0.10, 0.12]

    print(f"\n🔍 Sweeping CONT_BONUS × DEFENSIVE_FRAC × SPY_SOFT...")
    results = []

    # First: sweep CONT_BONUS with fixed others (v11a defaults)
    print("\n  [Pass 1] CONT_BONUS sweep (def=15%, spy_soft=10%):")
    DEFENSIVE_FRAC = 0.15; SPY_SOFT_HI_FRAC = 0.10
    for cb in cont_bonus_vals:
        CONT_BONUS = cb; DEFENSIVE_EACH = DEFENSIVE_FRAC / 3
        eq, to = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
        eq_is, _ = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                          '2015-01-01', '2020-12-31')
        eq_oos, _ = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                           '2021-01-01', '2025-12-31')
        m = metrics(eq); mi = metrics(eq_is); mo = metrics(eq_oos)
        wf = mo['sharpe']/mi['sharpe'] if mi['sharpe'] > 0 else 0
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
        results.append({'cb': cb, 'df': DEFENSIVE_FRAC, 'sf': SPY_SOFT_HI_FRAC,
                        'composite': comp, 'sharpe': m['sharpe'], 'cagr': m['cagr'],
                        'max_dd': m['max_dd'], 'calmar': m['calmar'], 'wf': wf,
                        'is_sh': mi['sharpe'], 'oos_sh': mo['sharpe']})
        print(f"    CONT={cb:.2f}: Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
              f"WF={wf:.2f} (IS={mi['sharpe']:.2f}/OOS={mo['sharpe']:.2f})")

    # Best CONT_BONUS
    best_cb = sorted([r for r in results if r['df']==0.15], key=lambda x: x['composite'], reverse=True)[0]['cb']
    print(f"\n  Best CONT_BONUS: {best_cb}")

    # Second: sweep DEFENSIVE_FRAC with best CONT_BONUS
    print(f"\n  [Pass 2] DEFENSIVE_FRAC sweep (cb={best_cb}, spy_soft=10%):")
    CONT_BONUS = best_cb; SPY_SOFT_HI_FRAC = 0.10
    for df_val in def_frac_vals:
        DEFENSIVE_FRAC = df_val; DEFENSIVE_EACH = df_val / 3
        eq, to = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
        eq_is, _ = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                          '2015-01-01', '2020-12-31')
        eq_oos, _ = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                           '2021-01-01', '2025-12-31')
        m = metrics(eq); mi = metrics(eq_is); mo = metrics(eq_oos)
        wf = mo['sharpe']/mi['sharpe'] if mi['sharpe'] > 0 else 0
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
        results.append({'cb': best_cb, 'df': df_val, 'sf': SPY_SOFT_HI_FRAC,
                        'composite': comp, 'sharpe': m['sharpe'], 'cagr': m['cagr'],
                        'max_dd': m['max_dd'], 'calmar': m['calmar'], 'wf': wf,
                        'is_sh': mi['sharpe'], 'oos_sh': mo['sharpe']})
        print(f"    DEF={df_val:.2f}: Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
              f"WF={wf:.2f} (IS={mi['sharpe']:.2f}/OOS={mo['sharpe']:.2f})")

    # Best DEFENSIVE_FRAC
    pass2 = [r for r in results if r['cb']==best_cb and r['sf']==0.10]
    best_df = sorted(pass2, key=lambda x: x['composite'], reverse=True)[0]['df']
    print(f"\n  Best DEFENSIVE_FRAC: {best_df}")

    # Third: sweep SPY_SOFT_HI_FRAC with best CONT+DEF
    print(f"\n  [Pass 3] SPY_SOFT_HI_FRAC sweep (cb={best_cb}, def={best_df}):")
    CONT_BONUS = best_cb; DEFENSIVE_FRAC = best_df; DEFENSIVE_EACH = best_df / 3
    for sf_val in spy_soft_vals:
        SPY_SOFT_HI_FRAC = sf_val
        eq, to = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices)
        eq_is, _ = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                          '2015-01-01', '2020-12-31')
        eq_oos, _ = run_bt(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                           '2021-01-01', '2025-12-31')
        m = metrics(eq); mi = metrics(eq_is); mo = metrics(eq_oos)
        wf = mo['sharpe']/mi['sharpe'] if mi['sharpe'] > 0 else 0
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
        results.append({'cb': best_cb, 'df': best_df, 'sf': sf_val,
                        'composite': comp, 'sharpe': m['sharpe'], 'cagr': m['cagr'],
                        'max_dd': m['max_dd'], 'calmar': m['calmar'], 'wf': wf,
                        'is_sh': mi['sharpe'], 'oos_sh': mo['sharpe']})
        print(f"    SPY_SOFT={sf_val:.2f}: Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
              f"WF={wf:.2f} (IS={mi['sharpe']:.2f}/OOS={mo['sharpe']:.2f})")

    print("\n" + "=" * 72)
    print("📊 Top 10 by Composite:")
    top10 = sorted(results, key=lambda x: x['composite'], reverse=True)[:10]
    for r in top10:
        print(f"  cb={r['cb']:.2f} def={r['df']:.2f} sf={r['sf']:.2f} → "
              f"Comp={r['composite']:.4f} Sharpe={r['sharpe']:.2f} "
              f"WF={r['wf']:.2f} MaxDD={r['max_dd']:.1%}")

    print("\n📊 Top 5 by WF (WF > v11a baseline 0.742):")
    high_wf = sorted([r for r in results if r['wf'] > 0.742],
                     key=lambda x: x['composite'], reverse=True)[:5]
    if high_wf:
        for r in high_wf:
            print(f"  cb={r['cb']:.2f} def={r['df']:.2f} sf={r['sf']:.2f} → "
                  f"Comp={r['composite']:.4f} Sharpe={r['sharpe']:.2f} WF={r['wf']:.2f}")
    else:
        print("  (no configs with WF > 0.742)")

    best_overall = top10[0]
    print(f"\n🏆 Best config: cb={best_overall['cb']:.2f} def={best_overall['df']:.2f} "
          f"sf={best_overall['sf']:.2f}")
    print(f"   Composite {best_overall['composite']:.4f} Sharpe {best_overall['sharpe']:.2f} "
          f"WF {best_overall['wf']:.2f}")

    v11a_comp = 2.190
    if best_overall['composite'] > v11a_comp:
        print(f"\n🚀🚀 超越v11a! {best_overall['composite']:.4f} > {v11a_comp}")
    elif best_overall['composite'] > 2.10:
        print(f"\n✅ Composite > 2.10: {best_overall['composite']:.4f}")

    out = {'sweep': results, 'best': best_overall,
           'v11a_baseline': {'composite': 2.190, 'wf': 0.742}}
    Path(Path(__file__).parent / "momentum_v11c_sweep_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n💾 Saved sweep results")
    return best_overall


if __name__ == '__main__':
    main()
