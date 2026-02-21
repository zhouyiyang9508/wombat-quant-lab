#!/usr/bin/env python3
"""
动量轮动 v9n — 组合优化: 自适应Vol + 预熊软对冲
代码熊 🐻

v9j_final (14层) + v9l (自适应Vol) + v9m (预熊软对冲) = 16层

探索 v9l 和 v9m 的最佳发现组合:
  v9l: bull_hi=14%, normal=11%, defensive=10% → Comp 2.077, WF 0.71
  v9m: lo=-0.03/+8%GLD, hi=-0.07/+15%GLD → Comp 2.086, WF 0.75

联合假设: 两种机制互补
  - v9l: 在强牛时多持仓, 在防御期少持仓 (持仓量控制)
  - v9m: 在SPY急跌时提前加GLD防御 (单月冲击防护)
  - 两种机制解决不同问题 → 预期叠加效果好于单独使用

风险: WF可能进一步下降
策略: 参数扫描以找到Composite和WF的最优平衡点

第16层创新:
  ① GLD竞争 ② Breadth+SPY双确认 ③ 3m主导动量
  ④ 5/4行业×2股 ⑤ 宽度阈值45% ⑥ 52周高点过滤
  ⑦ SHY熊市现金 ⑧ GDXJ波动预警 ⑨ 激进DD GLD
  ⑩ GDX精细竞争 ⑪ GLD自然竞争 ⑫ Vol目标化(v9i)
  ⑬ 条件TLT(v9j) ⑭ 自适应Vol目标(v9l) ⑮ 预熊软对冲(v9m)
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9j baseline parameters (unchanged)
MOM_W    = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS    = 5; N_BULL_SECS_HI = 4; BREADTH_CONC = 0.65
BULL_SPS = 2; BEAR_SPS = 2; BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
CONT_BONUS = 0.03; HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18
TLT_BEAR_FRAC = 0.25; TLT_MOM_LOOKBACK = 126
VOL_LOOKBACK = 3

# v9l: Adaptive vol target
VOL_TARGET_BULL_HI   = 0.14
VOL_TARGET_NORMAL    = 0.11
VOL_TARGET_DEFENSIVE = 0.10

# v9m: Pre-bear soft hedge
SPY_SOFT_THRESH_LO = -0.03; SPY_SOFT_FRAC_LO = 0.08
SPY_SOFT_THRESH_HI = -0.07; SPY_SOFT_FRAC_HI = 0.15


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
    if len(spy_now) == 0 or len(s200_now) == 0: return 'bull'
    breadth = compute_breadth(sig, date)
    return 'bear' if (spy_now.iloc[-1] < s200_now.iloc[-1] and
                      breadth < BREADTH_NARROW) else 'bull'


def get_vol_target(breadth):
    if breadth > BREADTH_CONC:     return VOL_TARGET_BULL_HI
    elif breadth > BREADTH_NARROW: return VOL_TARGET_NORMAL
    else:                           return VOL_TARGET_DEFENSIVE


def get_spy_1m_return(sig, date):
    if sig['r1'] is None or 'SPY' not in sig['r1'].columns: return 0.0
    r1_spy = sig['r1']['SPY']
    hist = r1_spy.loc[:date].dropna()
    if len(hist) == 0: return 0.0
    return float(hist.iloc[-1])


def get_spy_soft_gld(spy_1m_ret):
    if spy_1m_ret <= SPY_SOFT_THRESH_HI: return SPY_SOFT_FRAC_HI
    elif spy_1m_ret <= SPY_SOFT_THRESH_LO: return SPY_SOFT_FRAC_LO
    return 0.0


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


def get_tlt_momentum(tlt_p, date):
    hist = tlt_p.loc[:date].dropna()
    if len(hist) < TLT_MOM_LOOKBACK + 3: return False
    return bool(hist.iloc[-1] / hist.iloc[-TLT_MOM_LOOKBACK] - 1 > 0)


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, False, VOL_TARGET_NORMAL
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
    if len(df) == 0: return {}, False, VOL_TARGET_NORMAL

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg     = get_regime(sig, date)
    breadth = compute_breadth(sig, date)
    vol_target = get_vol_target(breadth)

    tlt_positive = get_tlt_momentum(tlt_p, date)
    use_tlt_bear = (reg == 'bear' and tlt_positive)

    if reg == 'bull':
        n_bull = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull - n_compete, 1)
        sps = BULL_SPS; bear_cash = 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        bear_cash = max(0.20 - (TLT_BEAR_FRAC if use_tlt_bear else 0), 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    tlt_alloc = TLT_BEAR_FRAC if use_tlt_bear else 0.0
    stock_frac = max(1.0 - bear_cash - total_compete - tlt_alloc, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if tlt_alloc > 0: w['TLT'] = tlt_alloc
        return w, use_tlt_bear, vol_target

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if tlt_alloc > 0: weights['TLT'] = tlt_alloc
    return weights, use_tlt_bear, vol_target


def apply_overlays(weights, spy_vol, dd, port_vol_ann, vol_target, spy_1m_ret):
    """v9n: combines v9l adaptive vol + v9m soft hedge"""
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    gld_soft = get_spy_soft_gld(spy_1m_ret)
    gld_total = max(gld_dd, gld_soft)

    total_overlay = gdxj_v + gld_total
    if total_overlay > 0 and weights:
        hedge_keys = {'GLD', 'GDX', 'GDXJ', 'TLT'}
        hedge_w = {t: w for t, w in weights.items() if t in hedge_keys}
        equity_w = {t: w for t, w in weights.items() if t not in hedge_keys}
        stock_frac = max(1.0 - total_overlay - sum(hedge_w.values()), 0.01)
        tot_eq = sum(equity_w.values())
        if tot_eq > 0:
            equity_w = {t: w/tot_eq*stock_frac for t, w in equity_w.items()}
        weights = {**equity_w, **hedge_w}
        if gld_total > 0: weights['GLD'] = weights.get('GLD', 0) + gld_total
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    # Adaptive vol targeting (v9l)
    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(vol_target / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in ('GLD', 'GDX', 'GDXJ', 'TLT')]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] = weights[t] * scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []; tlt_months = 0; soft_months = 0
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15
        spy_1m = get_spy_1m_return(sig, dt)
        if spy_1m <= SPY_SOFT_THRESH_LO: soft_months += 1

        if len(port_returns) >= VOL_LOOKBACK:
            port_vol_mon = np.std(port_returns[-VOL_LOOKBACK:], ddof=1)
            port_vol_ann = port_vol_mon * np.sqrt(12)
        else:
            port_vol_ann = 0.20

        w, use_tlt, vol_target = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p)
        w, shy_boost = apply_overlays(w, spy_vol, dd, port_vol_ann, vol_target, spy_1m)
        if use_tlt: tlt_months += 1

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'GDX', 'GDXJ', 'TLT')}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t == 'TLT':  s = tlt_p.loc[dt:ndt].dropna()
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
    return eq, float(np.mean(tos)), soft_months, tlt_months


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
    global VOL_TARGET_BULL_HI, VOL_TARGET_NORMAL, VOL_TARGET_DEFENSIVE
    global SPY_SOFT_THRESH_LO, SPY_SOFT_FRAC_LO, SPY_SOFT_THRESH_HI, SPY_SOFT_FRAC_HI
    print("=" * 72)
    print("🐻 动量轮动 v9n — 组合: 自适应Vol + 预熊软对冲")
    print("=" * 72)
    print(f"  v9l: bull={VOL_TARGET_BULL_HI:.0%}/norm={VOL_TARGET_NORMAL:.0%}/def={VOL_TARGET_DEFENSIVE:.0%}")
    print(f"  v9m: lo={SPY_SOFT_THRESH_LO:.2f}/+{SPY_SOFT_FRAC_LO:.0%} hi={SPY_SOFT_THRESH_HI:.2f}/+{SPY_SOFT_FRAC_HI:.0%}")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")

    # Quick sweep of combined params
    print("\n🔍 Sweeping v9l+v9m combined parameters...")
    results = []
    configs = [
        # (bull_hi, norm, def, lo_t, lo_f, hi_t, hi_f)
        (0.14, 0.11, 0.10, -0.03, 0.08, -0.07, 0.15),  # best of both
        (0.14, 0.11, 0.10, -0.03, 0.05, -0.07, 0.10),  # moderate combo
        (0.14, 0.11, 0.11, -0.03, 0.08, -0.07, 0.15),  # v9l mild + v9m
        (0.13, 0.11, 0.10, -0.03, 0.08, -0.07, 0.15),  # less bull vol
        (0.11, 0.11, 0.11, -0.03, 0.08, -0.07, 0.15),  # v9m only (control)
        (0.14, 0.11, 0.10, -9.99, 0.00, -0.07, 0.10),  # v9l + hi-only soft
        (0.11, 0.11, 0.11, -9.99, 0.00, -0.07, 0.10),  # hi-only soft (control)
        (0.11, 0.11, 0.11, -9.99, 0.00, -9.99, 0.00),  # v9j (baseline)
    ]
    for bhi, norm, defen, lo_t, lo_f, hi_t, hi_f in configs:
        VOL_TARGET_BULL_HI = bhi; VOL_TARGET_NORMAL = norm; VOL_TARGET_DEFENSIVE = defen
        SPY_SOFT_THRESH_LO = lo_t; SPY_SOFT_FRAC_LO = lo_f
        SPY_SOFT_THRESH_HI = hi_t; SPY_SOFT_FRAC_HI = hi_f
        try:
            eq, to, sm, tm = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p)
            m = compute_metrics(eq)
            eq_is, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                                           '2015-01-01', '2020-12-31')
            eq_oos, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                                            '2021-01-01', '2025-12-31')
            mi = compute_metrics(eq_is); mo = compute_metrics(eq_oos)
            wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            results.append({'bhi': bhi, 'norm': norm, 'defen': defen,
                            'lo_t': lo_t, 'lo_f': lo_f, 'hi_t': hi_t, 'hi_f': hi_f,
                            'composite': comp, 'sharpe': m['sharpe'], 'cagr': m['cagr'],
                            'max_dd': m['max_dd'], 'calmar': m['calmar'], 'wf': wf})
            lo_str = f"{lo_t:.2f}/{lo_f:.2f}" if lo_t > -9 else "off"
            hi_str = f"{hi_t:.2f}/{hi_f:.2f}" if hi_t > -9 else "off"
            print(f"  vol=({bhi:.2f}/{norm:.2f}/{defen:.2f}) spy=(lo={lo_str} hi={hi_str}) "
                  f"→ Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
                  f"MaxDD={m['max_dd']:.1%} WF={wf:.2f}")
        except Exception as e:
            print(f"  Error: {e}")

    results = sorted(results, key=lambda x: x['composite'], reverse=True)
    print(f"\n📊 Top 5 configurations:")
    for r in results[:5]:
        lo_str = f"{r['lo_t']:.2f}/{r['lo_f']:.2f}" if r['lo_t'] > -9 else "off"
        hi_str = f"{r['hi_t']:.2f}/{r['hi_f']:.2f}" if r['hi_t'] > -9 else "off"
        print(f"  vol=({r['bhi']:.2f}/{r['norm']:.2f}/{r['defen']:.2f}) spy=(lo={lo_str} hi={hi_str}) "
              f"→ Comp={r['composite']:.4f} Sharpe={r['sharpe']:.2f} WF={r['wf']:.2f}")

    best = results[0]
    VOL_TARGET_BULL_HI = best['bhi']; VOL_TARGET_NORMAL = best['norm']
    VOL_TARGET_DEFENSIVE = best['defen']
    SPY_SOFT_THRESH_LO = best['lo_t']; SPY_SOFT_FRAC_LO = best['lo_f']
    SPY_SOFT_THRESH_HI = best['hi_t']; SPY_SOFT_FRAC_HI = best['hi_f']

    print("\n🔄 Final backtest with best config...")
    print("🔄 Full (2015-2025)...")
    eq_full, to, sm, tm = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _, _ = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
        '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _, _ = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
        '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v9n vs Benchmarks")
    print("=" * 72)
    print(f"{'Metric':<12} {'v9j(2.057)':<12} {'v9m(2.086)':<12} {'v9n'}")
    print(f"{'CAGR':<12} {'32.3%':<12} {'32.0%':<12} {m['cagr']:.1%}")
    print(f"{'MaxDD':<12} {'-10.3%':<12} {'-10.0%':<12} {m['max_dd']:.1%}")
    print(f"{'Sharpe':<12} {'1.85':<12} {'1.84':<12} {m['sharpe']:.2f}")
    print(f"{'Calmar':<12} {'3.13':<12} {'3.21':<12} {m['calmar']:.2f}")
    print(f"{'IS Sharpe':<12} {'2.00':<12} {'2.03':<12} {mi['sharpe']:.2f}")
    print(f"{'OOS Sharpe':<12} {'1.57':<12} {'1.52':<12} {mo['sharpe']:.2f}")
    print(f"{'WF':<12} {'0.78':<12} {'0.75':<12} {wf:.2f}")
    print(f"{'Composite':<12} {'2.057':<12} {'2.086':<12} {comp:.4f}")

    if comp > 2.1:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.1! ({comp:.4f})")
    elif comp > 2.086:
        print(f"\n🚀🚀 超越v9m! Composite {comp:.4f}")
    elif comp > 2.057:
        print(f"\n🚀 超越v9j冠军! Composite {comp:.4f}")
    elif comp > 2.00:
        print(f"\n✅ Composite > 2.0: {comp:.4f}")

    out = {
        'strategy': f'v9n Combo: adaptive_vol({VOL_TARGET_BULL_HI:.2f}/{VOL_TARGET_NORMAL:.2f}/{VOL_TARGET_DEFENSIVE:.2f}) + soft_hedge({SPY_SOFT_THRESH_LO:.2f}/{SPY_SOFT_FRAC_LO:.2f}+{SPY_SOFT_THRESH_HI:.2f}/{SPY_SOFT_FRAC_HI:.2f})',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'soft_months': sm, 'tlt_months': tm,
        'all_configs': results,
    }
    jf = Path(__file__).parent / "momentum_v9n_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
