#!/usr/bin/env python3
"""
动量轮动 v10c Final — 最优综合 (v9j + v10b防御桥 + v9m软对冲)
代码熊 🐻

📊 综合三大发现:
  v9j (14层): Composite 2.057, WF 0.78 ← 推荐基线
  v10b (防御桥): +0.05 Sharpe, Composite 2.064, WF 0.76
  v9m (软对冲hi-only): +0.008 Composite, WF 0.77

v10c = v9j + 防御行业桥 (15%) + SPY hi-only软对冲 (-7%→+10%GLD)

第16层创新 (防御行业桥 + 软对冲):
  ① GLD竞争 ② Breadth+SPY熊市 ③ 3m主导动量
  ④ 5/4行业×2股 ⑤ 宽度阈值45% ⑥ 52周高点过滤
  ⑦ SHY熊市现金 ⑧ GDXJ波动预警 ⑨ 激进DD GLD
  ⑩ GDX精细竞争 ⑪ GLD自然竞争 ⑫ Vol目标化(11%)
  ⑬ 条件TLT(v9j) ⑭ 防御行业桥(v10b): 软牛期+15%XLV/XLP/XLU
  ⑮ SPY月跌>7%软对冲(v9m hi-only): +10%GLD

预期效果:
  - 软牛期(48%月份): XLV/XLP/XLU降低组合波动 → Sharpe提升
  - 月跌>7%: 提前GLD布防 → MaxDD改善
  - 两种机制解决不同时间尺度的风险
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
TLT_BEAR_FRAC = 0.25; TLT_MOM_LOOKBACK = 126
VOL_TARGET_ANN = 0.11; VOL_LOOKBACK = 3

# v10b: Defensive sector bridge (soft-bull: 45%<breadth≤65%)
DEFENSIVE_ETFS = ['XLV', 'XLP', 'XLU']
DEFENSIVE_FRAC = 0.15   # 15% total = 5% each
DEFENSIVE_EACH = DEFENSIVE_FRAC / len(DEFENSIVE_ETFS)

# v9m hi-only: Pre-bear soft GLD (SPY 1m return < -7%)
SPY_SOFT_HI_THRESH = -0.07
SPY_SOFT_HI_FRAC   = 0.10


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
    """Three-way regime: bull_hi / soft_bull / bear"""
    if sig['s200'] is None: return 'bull_hi', 1.0
    spy_now  = sig['spy'].loc[:date].dropna()
    s200_now = sig['s200'].loc[:date].dropna()
    if len(spy_now) == 0: return 'bull_hi', 1.0
    breadth = compute_breadth(sig, date)
    if spy_now.iloc[-1] < s200_now.iloc[-1] and breadth < BREADTH_NARROW:
        return 'bear', breadth
    elif breadth < BREADTH_CONC:   # 45-65%
        return 'soft_bull', breadth
    else:
        return 'bull_hi', breadth


def get_spy_1m(sig, date):
    if sig['r1'] is None or 'SPY' not in sig['r1'].columns: return 0.0
    hist = sig['r1']['SPY'].loc[:date].dropna()
    return float(hist.iloc[-1]) if len(hist) > 0 else 0.0


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


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, def_prices):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, False
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
    if len(df) == 0: return {}, False

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg, breadth = get_three_regime(sig, date)
    tlt_pos = get_tlt_momentum(tlt_p, date)
    use_tlt_bear = (reg == 'bear' and tlt_pos)

    # Determine defensive ETF allocation
    if reg == 'soft_bull':
        avail_def = [t for t, p in def_prices.items()
                     if p is not None and len(p.loc[:date].dropna()) > 0]
        def_alloc = {t: DEFENSIVE_EACH for t in avail_def}
        def_frac = sum(def_alloc.values())
    else:
        avail_def = []; def_alloc = {}; def_frac = 0.0

    if reg == 'bull_hi':
        n_secs = max(N_BULL_SECS_HI - n_compete, 1)
        sps = BULL_SPS; bear_cash = 0.0
    elif reg == 'soft_bull':
        n_secs = max(N_BULL_SECS - n_compete, 1)
        sps = BULL_SPS; bear_cash = 0.0
    else:  # bear
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        bear_cash = max(0.20 - (TLT_BEAR_FRAC if use_tlt_bear else 0), 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    tlt_alloc = TLT_BEAR_FRAC if use_tlt_bear else 0.0
    stock_frac = max(1.0 - bear_cash - total_compete - tlt_alloc - def_frac, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if tlt_alloc > 0: w['TLT'] = tlt_alloc
        w.update(def_alloc)
        return w, use_tlt_bear

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if tlt_alloc > 0: weights['TLT'] = tlt_alloc
    weights.update(def_alloc)
    return weights, use_tlt_bear


def apply_overlays(weights, spy_vol, dd, port_vol_ann, spy_1m_ret):
    """GDXJ + DD + SPY hi-only soft hedge + vol target"""
    HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'XLV', 'XLP', 'XLU'}

    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    # v9m hi-only soft hedge
    gld_soft = SPY_SOFT_HI_FRAC if spy_1m_ret <= SPY_SOFT_HI_THRESH else 0.0
    gld_total = max(gld_dd, gld_soft)

    total_overlay = gdxj_v + gld_total
    if total_overlay > 0 and weights:
        hedge_w = {t: w for t, w in weights.items() if t in HEDGE_KEYS}
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
                for t in equity_keys:
                    weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                 def_prices, start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []
    HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'XLV', 'XLP', 'XLU'}
    regime_hist = {'bull_hi': 0, 'soft_bull': 0, 'bear': 0}
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

        w, use_tlt = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, def_prices)
        w, shy_boost = apply_overlays(w, spy_vol, dd, pv, spy_1m)

        reg, _ = get_three_regime(sig, dt)
        regime_hist[reg] = regime_hist.get(reg, 0) + 1

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
    return eq, float(np.mean(tos)) if tos else 0.0, regime_hist


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
    print("🐻 动量轮动 v10c Final — 综合最优 (v9j + 防御桥 + 软对冲)")
    print("=" * 72)
    print(f"\n  Defensive Bridge: {DEFENSIVE_ETFS} {DEFENSIVE_FRAC:.0%} in soft-bull (45-65%)")
    print(f"  SPY hi-only: SPY 1m < {SPY_SOFT_HI_THRESH:.0%} → +{SPY_SOFT_HI_FRAC:.0%} GLD")
    print(f"  Vol Target: {VOL_TARGET_ANN:.0%} | TLT Bear: {TLT_BEAR_FRAC:.0%} conditional")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    def_prices = {}
    for t in DEFENSIVE_ETFS:
        f = CACHE / f"{t}.csv"
        if f.exists():
            def_prices[t] = load_csv(f)['Close'].dropna()
        else:
            def_prices[t] = None
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")
    print(f"  Defensive data: {[t for t,p in def_prices.items() if p is not None]}")

    print("\n🔄 Full (2015-2025)...")
    eq_full, to, rh = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, def_prices)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, def_prices,
                                '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, def_prices,
                                 '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v10c Final vs All Benchmarks")
    print("=" * 72)
    print(f"{'Metric':<12} {'v9j':<10} {'v10b':<10} {'v10c':<10} {'vs v9j'}")
    v9j = dict(cagr=0.3235, max_dd=-0.1033, sharpe=1.850, calmar=3.132, comp=2.057, wf=0.785)
    v10b_r = dict(cagr=0.316, max_dd=-0.102, sharpe=1.90, calmar=3.10, comp=2.0638, wf=0.76)
    print(f"{'CAGR':<12} {v9j['cagr']:.1%}     {v10b_r['cagr']:.1%}     {m['cagr']:.1%}     {m['cagr']-v9j['cagr']:+.1%}")
    print(f"{'MaxDD':<12} {v9j['max_dd']:.1%}    {v10b_r['max_dd']:.1%}    {m['max_dd']:.1%}    {m['max_dd']-v9j['max_dd']:+.1%}")
    print(f"{'Sharpe':<12} {v9j['sharpe']:.2f}      {v10b_r['sharpe']:.2f}      {m['sharpe']:.2f}      {m['sharpe']-v9j['sharpe']:+.2f}")
    print(f"{'Calmar':<12} {v9j['calmar']:.2f}      {v10b_r['calmar']:.2f}      {m['calmar']:.2f}      {m['calmar']-v9j['calmar']:+.2f}")
    print(f"{'IS Sharpe':<12}                         {mi['sharpe']:.2f}")
    print(f"{'OOS Sharpe':<12}                         {mo['sharpe']:.2f}")
    print(f"{'WF':<12} {v9j['wf']:.2f}      {v10b_r['wf']:.2f}      {wf:.2f}      {wf-v9j['wf']:+.2f}")
    print(f"{'Composite':<12} {v9j['comp']:.4f}  {v10b_r['comp']:.4f}  {comp:.4f}  {comp-v9j['comp']:+.4f}")
    print(f"\n  Regime: bull_hi={rh['bull_hi']} soft_bull={rh['soft_bull']} bear={rh.get('bear',0)}")
    print(f"  Turnover: {to:.1%}/month")

    if comp > 2.1:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.1! ({comp:.4f})")
    elif comp > 2.086:
        print(f"\n🚀🚀 超越v9m! Composite {comp:.4f}")
    elif comp > 2.057:
        print(f"\n🚀 超越v9j冠军! Composite {comp:.4f}")
    elif comp > 2.00:
        print(f"\n✅ Composite > 2.0: {comp:.4f}")

    out = {
        'strategy': 'v10c_final: v9j + defensive_bridge_15% + spy_hi_soft_10%',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'regime_hist': rh,
        'params': {
            'defensive_frac': DEFENSIVE_FRAC, 'defensive_etfs': DEFENSIVE_ETFS,
            'spy_soft_hi_thresh': SPY_SOFT_HI_THRESH, 'spy_soft_hi_frac': SPY_SOFT_HI_FRAC,
            'tlt_bear_frac': TLT_BEAR_FRAC, 'vol_target': VOL_TARGET_ANN,
        }
    }
    jf = Path(__file__).parent / "momentum_v10c_final_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
