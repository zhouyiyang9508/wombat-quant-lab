#!/usr/bin/env python3
"""
动量轮动 v11b Final — 联合参数优化最优配置
代码熊 🐻 | 2026-02-22

目标: 将所有正交的改进叠加到单一策略

三大独立创新 (互不干扰):
  ① v10d: 利率自适应债券 (熊市时IEF/TLT动量择优,双负→不用债券)
  ② v10b: 防御行业桥接 (软牛市45-65%时+15% XLV/XLP/XLU)
  ③ v9m:  SPY软对冲预警 (月跌>7%时+10% GLD)

正交性分析 (三种触发条件不同):
  ①触发条件: 熊市 (SPY<SMA200 AND breadth<45%)
  ②触发条件: 软牛市 (breadth 45-65%)
  ③触发条件: SPY月跌>7% (不依赖制度)
  
  可能同时触发②+③: 软牛期市场急跌>7%
  → 此时: 15%防御ETF + 10%GLD + 75%个股 (合理)
  
  不可能同时触发①+②: 因为①=熊市(breadth<45%), ②=软牛(breadth45-65%)

设计目标:
  v10d: Composite 2.107, WF 0.77 (当前冠军)
  v11a预期: Composite 2.12-2.15?, WF 0.74-0.76?
  
三制度框架 (from v10b):
  bull_hi: breadth > 65% → 纯个股 (4-5行业×2)
  soft_bull: breadth 45-65% → 个股 + 15% XLV/XLP/XLU
  bear: SPY<SMA200 AND breadth<45% → 现金/IEF/TLT + GLD

严格无前瞻: 所有信号使用历史收盘价
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9j baseline parameters
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

# v10d: Rate-adaptive bond (Bear mode)
TLT_BEAR_FRAC = 0.25; IEF_BEAR_FRAC = 0.20; BOND_MOM_LB = 126

# v10b/v10c: Defensive sector bridge (Soft-bull mode)
DEFENSIVE_ETFS = ['XLV', 'XLP', 'XLU']
DEFENSIVE_FRAC = 0.12
DEFENSIVE_EACH = DEFENSIVE_FRAC / len(DEFENSIVE_ETFS)

# v9m: SPY hi-only soft GLD hedge (month return < -7%)
SPY_SOFT_HI_THRESH = -0.07; SPY_SOFT_HI_FRAC = 0.08


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


def get_three_regime(sig, date):
    """Three-way: bull_hi / soft_bull / bear"""
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
    """v10d: Rate-adaptive bond selection"""
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
    stock_r6 = r6.loc[d].dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lb] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, 'bull_hi', None
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
    if len(df) == 0: return {}, 'bull_hi', None

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg, breadth = get_three_regime(sig, date)

    # v10d: Rate-adaptive bond (bear only)
    bond_ticker, bond_frac = select_best_bond(tlt_p, ief_p, date)
    use_bond = (reg == 'bear' and bond_frac > 0)
    actual_bond_frac = bond_frac if use_bond else 0.0

    # v10b: Defensive ETF bridge (soft-bull only)
    if reg == 'soft_bull':
        avail_def = [t for t, p in def_prices.items()
                     if p is not None and len(p.loc[:date].dropna()) > 0]
        def_alloc = {t: DEFENSIVE_EACH for t in avail_def}
        def_frac = sum(def_alloc.values())
    else:
        avail_def = []; def_alloc = {}; def_frac = 0.0

    if reg == 'bull_hi':
        n_secs = max(N_BULL_SECS_HI - n_compete, 1)
        sps, bear_cash = BULL_SPS, 0.0
    elif reg == 'soft_bull':
        n_secs = max(N_BULL_SECS - n_compete, 1)
        sps, bear_cash = BULL_SPS, 0.0
    else:  # bear
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        bear_cash = max(0.20 - actual_bond_frac, 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - bear_cash - total_compete - actual_bond_frac - def_frac, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if use_bond and bond_ticker: w[bond_ticker] = actual_bond_frac
        w.update(def_alloc)
        return w, reg, bond_ticker

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if use_bond and bond_ticker: weights[bond_ticker] = actual_bond_frac
    weights.update(def_alloc)
    return weights, reg, bond_ticker


HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'XLV', 'XLP', 'XLU'}


def apply_overlays(weights, spy_vol, dd, port_vol_ann, spy_1m_ret):
    """GDXJ + DD GLD + v9m SPY hi-only + vol target"""
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    # v9m: SPY hi-only soft hedge (independent of regime)
    gld_soft = SPY_SOFT_HI_FRAC if spy_1m_ret <= SPY_SOFT_HI_THRESH else 0.0
    gld_total = max(gld_dd, gld_soft)

    total_overlay = gdxj_v + gld_total
    if total_overlay > 0 and weights:
        hedge_w  = {t: w for t, w in weights.items() if t in HEDGE_KEYS}
        equity_w = {t: w for t, w in weights.items() if t not in HEDGE_KEYS}
        stock_frac = max(1.0 - total_overlay - sum(hedge_w.values()), 0.01)
        tot = sum(equity_w.values())
        if tot > 0:
            equity_w = {t: w/tot*stock_frac for t, w in equity_w.items()}
        weights = {**equity_w, **hedge_w}
        if gld_total > 0: weights['GLD'] = weights.get('GLD', 0) + gld_total
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


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                 tlt_p, ief_p, def_prices,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []
    regime_hist = {'bull_hi': 0, 'soft_bull': 0, 'bear': 0}
    bond_hist   = {'TLT': 0, 'IEF': 0, 'none': 0}
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15
        spy_1m = get_spy_1m(sig, dt)

        if len(port_returns) >= VOL_LOOKBACK:
            pv = np.std(port_returns[-VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        w, reg, bond_t = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, ief_p, def_prices)
        w, shy_boost = apply_overlays(w, spy_vol, dd, pv, spy_1m)

        regime_hist[reg] = regime_hist.get(reg, 0) + 1
        if reg == 'bear':
            if bond_t == 'TLT': bond_hist['TLT'] += 1
            elif bond_t == 'IEF': bond_hist['IEF'] += 1
            else: bond_hist['none'] += 1

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in HEDGE_KEYS}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t == 'TLT':  s = tlt_p.loc[dt:ndt].dropna()
            elif t == 'IEF':  s = ief_p.loc[dt:ndt].dropna()
            elif t in def_prices and def_prices[t] is not None:
                s = def_prices[t].loc[dt:ndt].dropna()
            elif t in close_df.columns:
                s = close_df[t].loc[dt:ndt].dropna()
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
    return eq, float(np.mean(tos)) if tos else 0.0, regime_hist, bond_hist


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


def compute_wf3(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices):
    """Three-period WF: IS=2015-2018, OOS1=2019-2021, OOS2=2022-2025"""
    print("  Computing 3-period Walk-Forward...")
    eq_is, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                   tlt_p, ief_p, def_prices, '2015-01-01', '2018-12-31')
    eq_oos1, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                     tlt_p, ief_p, def_prices, '2019-01-01', '2021-12-31')
    eq_oos2, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                     tlt_p, ief_p, def_prices, '2022-01-01', '2025-12-31')
    m_is   = compute_metrics(eq_is)
    m_oos1 = compute_metrics(eq_oos1)
    m_oos2 = compute_metrics(eq_oos2)
    wf3 = min(m_oos1['sharpe'], m_oos2['sharpe']) / m_is['sharpe'] if m_is['sharpe'] > 0 else 0
    print(f"  IS(2015-18) Sharpe={m_is['sharpe']:.2f} | OOS1(19-21)={m_oos1['sharpe']:.2f} | OOS2(22-25)={m_oos2['sharpe']:.2f}")
    print(f"  WF3 = min(OOS1,OOS2)/IS = min({m_oos1['sharpe']:.2f},{m_oos2['sharpe']:.2f}) / {m_is['sharpe']:.2f} = {wf3:.2f}")
    return wf3, m_is, m_oos1, m_oos2


def main():
    print("=" * 72)
    print("🐻 动量轮动 v11a Master — 三大创新叠加 (v10d + v10b + v9m)")
    print("=" * 72)
    print(f"  ① v10d: Adaptive IEF/TLT (熊市智能债券选择)")
    print(f"  ② v10b: Defensive Bridge {DEFENSIVE_FRAC:.0%} XLV/XLP/XLU (软牛市)")
    print(f"  ③ v9m: SPY hi-only {SPY_SOFT_HI_FRAC:.0%}GLD if SPY 1m<{SPY_SOFT_HI_THRESH:.0%}")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    ief_p  = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    def_prices = {}
    for t in DEFENSIVE_ETFS:
        f = CACHE / f"{t}.csv"
        def_prices[t] = load_csv(f)['Close'].dropna() if f.exists() else None
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")
    print(f"  Defensive: {[t for t,p in def_prices.items() if p is not None]}")

    print("\n🔄 Full (2015-2025)...")
    eq_full, to, rh, bh = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                        shy_p, tlt_p, ief_p, def_prices)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                   shy_p, tlt_p, ief_p, def_prices, '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                    shy_p, tlt_p, ief_p, def_prices, '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v11a vs v10d (Champion) vs v10c (Sharpe)")
    print("=" * 72)
    refs = {
        'v10d': dict(cagr=0.325, max_dd=-0.100, sharpe=1.86, calmar=3.25, comp=2.107, wf=0.77),
        'v10c': dict(cagr=0.317, max_dd=-0.102, sharpe=1.91, calmar=3.11, comp=2.072, wf=0.75),
    }
    print(f"{'Metric':<12} {'v10d(champ)':<14} {'v10c(Sharpe)':<14} {'v11a':<12} {'vs v10d'}")
    for k, fmt in [('cagr','%'),('max_dd','%'),('sharpe','f'),('calmar','f')]:
        r1 = f"{refs['v10d'][k]:.1%}" if fmt=='%' else f"{refs['v10d'][k]:.2f}"
        r2 = f"{refs['v10c'][k]:.1%}" if fmt=='%' else f"{refs['v10c'][k]:.2f}"
        rv = f"{m[k]:.1%}" if fmt=='%' else f"{m[k]:.2f}"
        delta = f"{m[k]-refs['v10d'][k]:+.1%}" if fmt=='%' else f"{m[k]-refs['v10d'][k]:+.2f}"
        print(f"  {k.upper():<10} {r1:<14} {r2:<14} {rv:<12} {delta}")
    print(f"  {'IS Sharpe':<10} {'2.02':<14} {'2.11':<14} {mi['sharpe']:.2f}")
    print(f"  {'OOS Sharpe':<10} {'1.56':<14} {'1.57':<14} {mo['sharpe']:.2f}")
    print(f"  {'WF':<10} {refs['v10d']['wf']:<14.2f} {refs['v10c']['wf']:<14.2f} {wf:.2f}         {wf-refs['v10d']['wf']:+.2f}")
    print(f"  {'Composite':<10} {refs['v10d']['comp']:<14.4f} {refs['v10c']['comp']:<14.4f} {comp:.4f}       {comp-refs['v10d']['comp']:+.4f}")
    print(f"\n  Regime: bull_hi={rh['bull_hi']} soft_bull={rh['soft_bull']} bear={rh.get('bear',0)}")
    print(f"  Bear bonds: TLT={bh.get('TLT',0)} IEF={bh.get('IEF',0)} none={bh.get('none',0)} months")
    print(f"  Turnover: {to:.1%}/month")

    # 3-period Walk Forward Validation
    print(f"\n📐 Three-Period Walk Forward Validation:")
    wf3, m_is3, m_oos1, m_oos2 = compute_wf3(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                               shy_p, tlt_p, ief_p, def_prices)

    if comp > 2.15:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.15! ({comp:.4f})")
    elif comp > 2.107:
        print(f"\n🚀🚀 超越v10d! Composite {comp:.4f}")
    elif comp > 2.10:
        print(f"\n🚀 Composite仍>2.10: {comp:.4f}")
    elif comp > 2.057:
        print(f"\n✅ 超越v9j: {comp:.4f}")

    if wf3 >= 0.70:
        print(f"\n🎯 3-Period WF = {wf3:.2f} ≥ 0.70: 强大的样本外一致性!")
    elif wf3 >= 0.60:
        print(f"\n✅ 3-Period WF = {wf3:.2f} ≥ 0.60: 合格")

    out = {
        'strategy': 'v11a Master: v10d + v10b defensive bridge + v9m soft hedge',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'wf3': float(wf3),
        'wf3_periods': {
            'is_2015_18': m_is3, 'oos1_2019_21': m_oos1, 'oos2_2022_25': m_oos2
        },
        'regime_hist': rh, 'bond_hist': bh,
    }
    jf = Path(__file__).parent / "momentum_v11a_master_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf, wf3


if __name__ == '__main__':
    main()
