#!/usr/bin/env python3
"""
动量轮动 v9f — GDXJ + GDX阈值精调 + 趋势一致性质量过滤
代码熊 🐻

当前最佳: v9e — Composite 1.617, CAGR 33.3%, Sharpe 1.64, WF 0.88

v9f 三个新方向:

[A] GDXJ 替代/补充 GDX (作为波动率触发对冲)
    GDXJ = VanEck Junior Gold Miners ETF
    vs GDX (大型金矿): 
      GDXJ beta更高 (~1.5-2x vs GLD, 而GDX~1.2-1.7x)
      危机后反弹更猛 (小型矿商弹性大)
      但正常时波动也更大
    测试: GDXJ替代GDX做vol trigger, 或GDX+GDXJ双层对冲
      - 低vol (>30%): GDX (稳健)  
      - 高vol (>45%): GDXJ (极端危机, 弹性更大)

[B] GDX竞争阈值精调 (当前: 30% → 10%)
    核心问题: 30%门槛是否最优?
    太低 → GDX太频繁 → 稀释股票收益
    太高 → GDX太少 → 错过黄金牛市
    测试: 20%, 25%, 30%, 35%, 40%
    同时测试: 10%分配量是否最优

[C] 趋势一致性质量过滤 (Novel!)
    当前选股只看: 6m>0 + price>SMA50 + 52w高点filter
    新增: 过去6个月中正回报月数 ≥ 4
    理由: 避免 "lucky one-month wonder" 
           动量一致的股票更可能持续
    实现: 计算过去6个22天窗口的return, 统计正数个数
    注意: 严格无前瞻 (全部用截至当月末的历史数据)

评估: 能否超越 v9e Composite 1.617?
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9e champion base params
BASE_MOM_W  = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS = 5
BULL_SPS    = 2
BEAR_SPS    = 2
BREADTH_NARROW   = 0.45
GLD_AVG_THRESH   = 0.70
GLD_COMPETE_FRAC = 0.20
CONT_BONUS       = 0.03
HI52_FRAC        = 0.60
USE_SHY          = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}


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
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except:
            pass
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
    # Trend consistency: count positive 22-day periods in past 6 months
    # Each period: r at 22, 44, 66, 88, 110, 132 days
    r_w1 = close_df / close_df.shift(22) - 1
    r_w2 = close_df.shift(22) / close_df.shift(44) - 1
    r_w3 = close_df.shift(44) / close_df.shift(66) - 1
    r_w4 = close_df.shift(66) / close_df.shift(88) - 1
    r_w5 = close_df.shift(88) / close_df.shift(110) - 1
    r_w6 = close_df.shift(110) / close_df.shift(132) - 1
    # Count: how many of last 6 monthly windows were positive
    consistency = (
        (r_w1 > 0).astype(int) + (r_w2 > 0).astype(int) +
        (r_w3 > 0).astype(int) + (r_w4 > 0).astype(int) +
        (r_w5 > 0).astype(int) + (r_w6 > 0).astype(int)
    )
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, consistency=consistency,
                spy=spy, s200=s200, sma50=sma50, close=close_df)


def get_spy_vol(sig, date):
    if 'SPY' not in sig['vol5'].columns: return 0.0
    v = sig['vol5']['SPY'].loc[:date].dropna()
    return float(v.iloc[-1]) if len(v) > 0 else 0.0


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


def asset_competition(sig, date, prices, thresh, frac, lookback=127):
    """Generic momentum competition for GLD/GDX/GDXJ"""
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lookback + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lookback] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0


def select_stocks(sig, sectors, date, prev_hold, gld_p, gdx_p, gdxj_p, params):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
    d = idx[-1]

    w1, w3, w6, w12 = BASE_MOM_W
    mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
           sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom':   mom, 'r6':  sig['r6'].loc[d],
        'vol':   sig['vol30'].loc[d], 'price': close.loc[d],
        'sma50': sig['sma50'].loc[d], 'hi52':  sig['r52w_hi'].loc[d],
        'cons':  sig['consistency'].loc[d],
    }).dropna(subset=['mom', 'sma50'])

    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df = df[df['price'] >= df['hi52'] * HI52_FRAC]

    # Trend consistency filter (if enabled)
    min_cons = params.get('min_consistency', 0)
    if min_cons > 0:
        df = df[df['cons'] >= min_cons]

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += CONT_BONUS

    if len(df) == 0: return {}
    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    # Competing safe assets
    gld_a  = asset_competition(sig, date, gld_p,  GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a  = asset_competition(sig, date, gdx_p,  params.get('gdx_thresh', 0.30), params.get('gdx_frac', 0.10))
    gdxj_a = asset_competition(sig, date, gdxj_p, params.get('gdxj_thresh', 0), params.get('gdxj_frac', 0)) if params.get('gdxj_thresh', 0) > 0 else 0.0

    total_compete = gld_a + gdx_a + gdxj_a
    n_compete = sum([1 for x in [gld_a, gdx_a, gdxj_a] if x > 0])
    reg = get_regime(sig, date)

    if reg == 'bull':
        n_secs = N_BULL_SECS - n_compete
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
        if gdxj_a > 0: w['GDXJ'] = gdxj_a
        return w

    iv  = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}

    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a  > 0: weights['GLD']  = gld_a
    if gdx_a  > 0: weights['GDX']  = gdx_a
    if gdxj_a > 0: weights['GDXJ'] = gdxj_a
    return weights


def apply_overlays(weights, spy_vol, dd, params):
    """Vol-trigger (GDX/GDXJ) + DD-responsive (GLD)"""
    vol_lo = params.get('vol_lo_thresh', 0.30)
    vol_hi = params.get('vol_hi_thresh', 0.45)
    lo_asset  = params.get('lo_asset', 'GDX')
    hi_asset  = params.get('hi_asset', 'GDX')
    lo_frac   = params.get('vol_lo_frac', 0.12)
    hi_frac   = params.get('vol_hi_frac', 0.25)

    if spy_vol >= vol_hi:
        vol_alloc = {hi_asset: hi_frac}
    elif spy_vol >= vol_lo:
        vol_alloc = {lo_asset: lo_frac}
    else:
        vol_alloc = {}

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    # Total
    gdx_v   = vol_alloc.get('GDX', 0)
    gdxj_v  = vol_alloc.get('GDXJ', 0)
    gld_tot = max(vol_alloc.get('GLD', 0), gld_dd)

    total = gdx_v + gdxj_v + gld_tot
    if total <= 0 or not weights: return weights

    stock_frac = max(1.0 - total, 0.01)
    tot = sum(weights.values())
    if tot <= 0: return weights

    new = {t: w/tot*stock_frac for t, w in weights.items()}
    if gld_tot  > 0: new['GLD']  = new.get('GLD', 0) + gld_tot
    if gdx_v    > 0: new['GDX']  = new.get('GDX', 0) + gdx_v
    if gdxj_v   > 0: new['GDXJ'] = new.get('GDXJ', 0) + gdxj_v
    return new


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, params,
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

        w = select_stocks(sig, sectors, dt, prev_h, gld_p, gdx_p, gdxj_p, params)
        w = apply_overlays(w, spy_vol, dd, params)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t,0)-prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD','GDX','GDXJ')}

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


def metrics(eq):
    if len(eq) < 3: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    mo = eq.pct_change().dropna()
    sh = mo.mean()/mo.std()*np.sqrt(12) if mo.std() > 0 else 0
    dd = ((eq - eq.cummax())/eq.cummax()).min()
    cal = cagr/abs(dd) if dd != 0 else 0
    return dict(cagr=float(cagr), max_dd=float(dd), sharpe=float(sh), calmar=float(cal))


def evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, params):
    eq_f, to = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, params)
    eq_i, _  = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, params, '2015-01-01', '2020-12-31')
    eq_o, _  = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, params, '2021-01-01', '2025-12-31')
    m = metrics(eq_f); mi = metrics(eq_i); mo = metrics(eq_o)
    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
    return dict(full=m, is_m=mi, oos_m=mo, wf=float(wf), comp=float(comp), to=float(to))


def fmt(label, r, ref=1.6168):
    m = r['full']
    flag = '🚀' if r['comp']>1.65 and r['wf']>=0.7 else '✅' if r['comp']>1.60 and r['wf']>=0.7 else '⭐' if r['comp']>1.58 and r['wf']>=0.7 else ''
    return (f"  {label:35s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  "
            f"DD {m['max_dd']:.1%}  Cal {m['calmar']:.2f}  WF {r['wf']:.2f}  "
            f"Comp {r['comp']:.4f} ({r['comp']-ref:+.4f}) {flag}")


def main():
    print("=" * 82)
    print("🐻 动量轮动 v9f — GDXJ + GDX阈值精调 + 趋势一致性过滤")
    print("=" * 82)
    print(f"  Baseline: v9e (Composite 1.6168, Sharpe 1.638, WF 0.877)")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers + GLD, GDX, GDXJ, SHY")

    # v9e baseline params
    V9E = dict(
        gdx_thresh=0.30, gdx_frac=0.10,
        gdxj_thresh=0, gdxj_frac=0,
        vol_lo_thresh=0.30, vol_hi_thresh=0.45,
        lo_asset='GDX', hi_asset='GDX',
        vol_lo_frac=0.12, vol_hi_frac=0.25,
        min_consistency=0,
    )

    results = []
    print("\n🔄 v9e Baseline...")
    base = evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, V9E)
    print(fmt("v9e_baseline", base))
    results.append({'label': 'v9e_baseline', **base})

    # ── Sweep A: GDXJ as vol trigger ─────────────────────────────────────────
    print("\n📊 Sweep A: GDXJ as Vol-Trigger Asset")
    A_configs = [
        ("A_gdxj_lo_only",   dict(V9E, lo_asset='GDXJ', hi_asset='GDXJ', vol_lo_frac=0.10, vol_hi_frac=0.20)),
        ("A_gdxj_hi_only",   dict(V9E, hi_asset='GDXJ', vol_hi_frac=0.20)),   # GDX for lo, GDXJ for hi
        ("A_gdxj_lo_gdx_hi", dict(V9E, lo_asset='GDXJ', hi_asset='GDX', vol_lo_frac=0.10, vol_hi_frac=0.20)),
        ("A_gdxj_lo_12_hi_20", dict(V9E, lo_asset='GDXJ', hi_asset='GDXJ', vol_lo_frac=0.12, vol_hi_frac=0.22)),
        ("A_gdxj_lo_8_hi_18",  dict(V9E, lo_asset='GDXJ', hi_asset='GDXJ', vol_lo_frac=0.08, vol_hi_frac=0.18)),
    ]
    for label, p in A_configs:
        r = evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, p)
        print(fmt(label, r))
        results.append({'label': label, **r})

    # ── Sweep B: GDX Competition Threshold ──────────────────────────────────
    print("\n📊 Sweep B: GDX Competition Threshold (current: 30% → 10%)")
    B_configs = [
        ("B_gdx_thresh20_10", dict(V9E, gdx_thresh=0.20, gdx_frac=0.10)),
        ("B_gdx_thresh25_10", dict(V9E, gdx_thresh=0.25, gdx_frac=0.10)),
        ("B_gdx_thresh35_10", dict(V9E, gdx_thresh=0.35, gdx_frac=0.10)),
        ("B_gdx_thresh40_10", dict(V9E, gdx_thresh=0.40, gdx_frac=0.10)),
        ("B_gdx_thresh20_08", dict(V9E, gdx_thresh=0.20, gdx_frac=0.08)),
        ("B_gdx_thresh25_08", dict(V9E, gdx_thresh=0.25, gdx_frac=0.08)),
        ("B_gdx_thresh30_12", dict(V9E, gdx_thresh=0.30, gdx_frac=0.12)),
        ("B_gdx_thresh20_12", dict(V9E, gdx_thresh=0.20, gdx_frac=0.12)),
        ("B_gdx_off",         dict(V9E, gdx_thresh=9.99, gdx_frac=0)),  # disable GDX compete
    ]
    for label, p in B_configs:
        r = evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, p)
        print(fmt(label, r))
        results.append({'label': label, **r})

    # ── Sweep C: Trend Consistency Filter ────────────────────────────────────
    print("\n📊 Sweep C: Trend Consistency Filter (min positive months in last 6)")
    C_configs = [
        ("C_cons3",  dict(V9E, min_consistency=3)),
        ("C_cons4",  dict(V9E, min_consistency=4)),
        ("C_cons5",  dict(V9E, min_consistency=5)),
    ]
    for label, p in C_configs:
        r = evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, p)
        print(fmt(label, r))
        results.append({'label': label, **r})

    # ── Sweep D: GDXJ Natural Competition ────────────────────────────────────
    print("\n📊 Sweep D: GDXJ Natural Competition (alongside GDX)")
    D_configs = [
        ("D_gdxj_comp30_10", dict(V9E, gdxj_thresh=0.30, gdxj_frac=0.10)),
        ("D_gdxj_comp40_10", dict(V9E, gdxj_thresh=0.40, gdxj_frac=0.10)),
        ("D_gdxj_comp25_08", dict(V9E, gdxj_thresh=0.25, gdxj_frac=0.08)),
        ("D_gdxj_comp30_08", dict(V9E, gdxj_thresh=0.30, gdxj_frac=0.08)),
    ]
    for label, p in D_configs:
        r = evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, p)
        print(fmt(label, r))
        results.append({'label': label, **r})

    # ── Sweep E: Best Combos ──────────────────────────────────────────────────
    print("\n📊 Sweep E: Best Combos")
    best_A = max([r for r in results if r['label'].startswith('A_')], key=lambda x: x['comp'])
    best_B = max([r for r in results if r['label'].startswith('B_')], key=lambda x: x['comp'])
    best_C = max([r for r in results if r['label'].startswith('C_')], key=lambda x: x['comp'])
    best_D = max([r for r in results if r['label'].startswith('D_')], key=lambda x: x['comp'])
    print(f"  Best A: {best_A['label']} Comp={best_A['comp']:.4f}")
    print(f"  Best B: {best_B['label']} Comp={best_B['comp']:.4f}")
    print(f"  Best C: {best_C['label']} Comp={best_C['comp']:.4f}")
    print(f"  Best D: {best_D['label']} Comp={best_D['comp']:.4f}")

    # Combo B+C (best GDX thresh + consistency filter)
    b_label = best_B['label']
    B_map = {label: p for label, p in B_configs}
    best_B_p = B_map.get(b_label, V9E)
    best_C_cons = [p['min_consistency'] for label, p in C_configs if label == best_C['label']]
    best_C_cons_val = best_C_cons[0] if best_C_cons else 0

    combo_BC = dict(best_B_p); combo_BC['min_consistency'] = best_C_cons_val
    r = evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, combo_BC)
    print(fmt("E_bestB+bestC", r))
    results.append({'label': 'E_bestB+bestC', **r})

    # Combo B+D (best GDX thresh + GDXJ compete)
    D_map = {label: p for label, p in D_configs}
    best_D_p = D_map.get(best_D['label'], V9E)
    combo_BD = dict(best_B_p)
    combo_BD['gdxj_thresh'] = best_D_p.get('gdxj_thresh', 0)
    combo_BD['gdxj_frac']   = best_D_p.get('gdxj_frac', 0)
    r = evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, combo_BD)
    print(fmt("E_bestB+bestD", r))
    results.append({'label': 'E_bestB+bestD', **r})

    # All best combo
    combo_all = dict(best_B_p)
    combo_all['min_consistency'] = best_C_cons_val
    combo_all['gdxj_thresh'] = best_D_p.get('gdxj_thresh', 0)
    combo_all['gdxj_frac']   = best_D_p.get('gdxj_frac', 0)
    # Also try GDXJ for hi-vol
    combo_all['hi_asset'] = 'GDXJ'; combo_all['vol_hi_frac'] = 0.20
    r = evaluate(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, combo_all)
    print(fmt("E_AllBest_GDXJ_hi", r))
    results.append({'label': 'E_AllBest_GDXJ_hi', **r})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("🏆 TOP 8 Results (WF ≥ 0.70):")
    print("=" * 82)
    valid = sorted([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'], reverse=True)[:8]
    for r in valid:
        m = r['full']
        print(f"  {r['label']:35s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  DD {m['max_dd']:.1%}  WF {r['wf']:.2f}  Comp {r['comp']:.4f}")

    champion = max([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'])
    cm = champion['full']; comp = champion['comp']; wf = champion['wf']

    print(f"\n🏆 Champion: {champion['label']}")
    print(f"  CAGR: {cm['cagr']:.1%}  Sharpe: {cm['sharpe']:.2f}  MaxDD: {cm['max_dd']:.1%}")
    print(f"  IS Sharpe: {champion['is_m']['sharpe']:.2f}  OOS: {champion['oos_m']['sharpe']:.2f}")
    print(f"  WF: {wf:.2f}  Composite: {comp:.4f}  vs v9e ({comp-1.6168:+.4f})")

    if comp > 1.80 or cm['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】!")
    elif comp > 1.65:
        print(f"\n🚀 突破1.65! Composite {comp:.4f}")
    elif comp > 1.62:
        print(f"\n✅ 超越 v9e: Composite {comp:.4f}")

    out = {'champion': champion['label'],
           'champion_metrics': {k: float(v) for k, v in cm.items()} | {'wf': float(wf), 'composite': float(comp)},
           'baseline_v9e': 1.6168,
           'improvement': float(comp - 1.6168),
           'results': [{'label': r['label'], 'comp': float(r['comp']), 'wf': float(r['wf']),
                        'cagr': float(r['full']['cagr']), 'sharpe': float(r['full']['sharpe']),
                        'max_dd': float(r['full']['max_dd']), 'calmar': float(r['full']['calmar']),
                        'is_sharpe': float(r['is_m']['sharpe']), 'oos_sharpe': float(r['oos_m']['sharpe'])}
                       for r in results]}
    jf = Path(__file__).parent / "momentum_v9f_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")


if __name__ == '__main__':
    main()
