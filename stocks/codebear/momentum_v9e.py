#!/usr/bin/env python3
"""
动量轮动 v9e — GDX 参数精调 + GDX自然竞争 + 双资产波动率对冲
代码熊 🐻

v9d 发现: GDX (黄金矿工) 作为波动率触发资产优于 GLD!
  v9c (GLD vol): Composite 1.5669
  v9d (GDX vol): Composite 1.5892 (+0.022)

GDX 优势原因:
  - 比 GLD 更高的 beta (~2x)，在危机后反弹更猛
  - 黄金矿工有经营杠杆，黄金价格上涨时矿工利润翻倍
  - 波动率触发期间 (危机初始), GDX 的高 beta 有利

v9e 继续探索:

[A] GDX 波动率阈值调优
    当前: vol>30%→10%GDX, >45%→20%GDX
    尝试: 更低/更高阈值, 不同分配比例

[B] GDX 自然竞争机制
    类比 GLD 竞争: 当 GDX_6m > avg_stock_6m × threshold 时
    自然获得 10-20% 仓位
    → GDX 同时竞争"动量赚钱"和"防御保护"

[C] 双资产波动率对冲
    - vol>30%: GDX (黄金矿工，高beta危机对冲)
    - vol>50%: GDX + TLT (极端恐慌: 矿工+国债双保险)

[D] GDX + GLD 双竞争
    GLD 竞争(6m>股票70%) + GDX 竞争(6m>股票50%)
    两者同时竞争时分别获得 10% 仓位

目标: Composite > 1.60 🚀
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9c base params (same as always)
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
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200,
                sma50=sma50, close=close_df)


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


def gld_competition(sig, date, gld_prices, thresh=None, frac=None):
    thresh = thresh or GLD_AVG_THRESH
    frac   = frac   or GLD_COMPETE_FRAC
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    gld_h = gld_prices.loc[:d].dropna()
    if len(gld_h) < 130: return 0.0
    return frac if (gld_h.iloc[-1]/gld_h.iloc[-127]-1) >= avg_r6 * thresh else 0.0


def gdx_competition(sig, date, gdx_prices, thresh=0.50, frac=0.10):
    """GDX natural competition: similar to GLD but lower threshold"""
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    gdx_h = gdx_prices.loc[:d].dropna()
    if len(gdx_h) < 130: return 0.0
    gdx_r6 = gdx_h.iloc[-1] / gdx_h.iloc[-127] - 1
    return frac if gdx_r6 >= avg_r6 * thresh else 0.0


def get_vol_allocs(spy_vol, params):
    """Return {asset: allocation} dict from vol trigger params"""
    allocs = {}
    for (asset, low_thresh, high_thresh, low_frac, high_frac) in params.get('vol_rules', []):
        if spy_vol >= high_thresh:
            allocs[asset] = allocs.get(asset, 0) + high_frac
        elif spy_vol >= low_thresh:
            allocs[asset] = allocs.get(asset, 0) + low_frac
    return allocs


def select_stocks(sig, sectors, date, prev_hold, gld_prices, gdx_prices, params):
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
    }).dropna(subset=['mom', 'sma50'])

    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df = df[df['price'] >= df['hi52'] * HI52_FRAC]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += CONT_BONUS

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    # Competing safe assets (natural competition)
    gld_a = gld_competition(sig, date, gld_prices)
    gdx_a = params.get('gdx_compete_frac', 0.0)
    if gdx_a > 0:
        gdx_a = gdx_competition(sig, date, gdx_prices,
                                params.get('gdx_compete_thresh', 0.50),
                                params.get('gdx_compete_frac', 0.10))

    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)
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
        return w

    iv  = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}

    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = gld_a
    if gdx_a > 0: weights['GDX'] = gdx_a
    return weights


def apply_all_overlays(weights, spy_vol, dd, params):
    """Apply vol-trigger and DD-responsive overlays"""
    vol_allocs = get_vol_allocs(spy_vol, params)
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    # GLD: max of vol trigger and DD response
    gld_vol = vol_allocs.get('GLD', 0)
    gld_total = max(gld_vol, gld_dd)

    # GDX: only from vol trigger
    gdx_total = vol_allocs.get('GDX', 0)

    # TLT: only from vol trigger
    tlt_total = vol_allocs.get('TLT', 0)

    total_hedge = gld_total + gdx_total + tlt_total
    if total_hedge <= 0 or not weights: return weights

    stock_frac = max(1.0 - total_hedge, 0.01)
    tot = sum(weights.values())
    if tot <= 0: return weights

    new = {t: w/tot*stock_frac for t, w in weights.items()}
    if gld_total > 0: new['GLD'] = new.get('GLD', 0) + gld_total
    if gdx_total > 0: new['GDX'] = new.get('GDX', 0) + gdx_total
    if tlt_total > 0: new['TLT'] = new.get('TLT', 0) + tlt_total
    return new


def run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, params,
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

        w = select_stocks(sig, sectors, dt, prev_h, gld, gdx, params)
        w = apply_all_overlays(w, spy_vol, dd, params)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t,0)-prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD','TLT','GDX')}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD': s = gld.loc[dt:ndt].dropna()
            elif t == 'TLT': s = tlt.loc[dt:ndt].dropna()
            elif t == 'GDX': s = gdx.loc[dt:ndt].dropna()
            elif t in close_df.columns: s = close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt

        if USE_SHY and cash_frac > 0 and shy is not None:
            s = shy.loc[dt:ndt].dropna()
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


def evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, params):
    eq_f, to = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, params)
    eq_i, _  = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, params, '2015-01-01', '2020-12-31')
    eq_o, _  = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, params, '2021-01-01', '2025-12-31')
    m = metrics(eq_f); mi = metrics(eq_i); mo = metrics(eq_o)
    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
    return dict(full=m, is_m=mi, oos_m=mo, wf=float(wf), comp=float(comp), to=float(to))


def fmt(label, r, ref=1.5892):
    m = r['full']
    flag = '🚀' if r['comp']>1.60 and r['wf']>=0.7 else '✅' if r['comp']>1.57 and r['wf']>=0.7 else '⭐' if r['comp']>1.55 and r['wf']>=0.7 else ''
    return (f"  {label:33s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  "
            f"DD {m['max_dd']:.1%}  Cal {m['calmar']:.2f}  "
            f"WF {r['wf']:.2f}  Comp {r['comp']:.4f} ({r['comp']-ref:+.4f}) {flag}")


def make_params(vol_rules, gdx_compete_thresh=0.0, gdx_compete_frac=0.0):
    return {
        'vol_rules': vol_rules,
        'gdx_compete_thresh': gdx_compete_thresh,
        'gdx_compete_frac': gdx_compete_frac,
    }


def main():
    print("=" * 82)
    print("🐻 动量轮动 v9e — GDX 精调 + GDX自然竞争 + 双资产波动率对冲")
    print("=" * 82)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    tlt = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    gdx = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers + GLD, TLT, SHY, GDX")

    results = []
    REF = 1.5892   # v9d GDX champion

    # Baseline: v9d GDX (GDX vol trigger + GLD DD)
    base_params = make_params([('GDX', 0.30, 0.45, 0.10, 0.20)])
    print("\n🔄 Baseline (v9d GDX vol trigger)...")
    base = evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, base_params)
    print(fmt("v9d_gdx_baseline", base, REF))
    results.append({'label': 'v9d_gdx_baseline', **base})

    # ── Sweep A: GDX vol threshold variants ────────────────────────────────────
    print("\n📊 Sweep A: GDX Vol Threshold Variations")
    A_configs = [
        ("A_gdx_v25_v40",   [('GDX', 0.25, 0.40, 0.10, 0.20)]),
        ("A_gdx_v25_v45",   [('GDX', 0.25, 0.45, 0.10, 0.20)]),
        ("A_gdx_v30_v50",   [('GDX', 0.30, 0.50, 0.10, 0.20)]),
        ("A_gdx_v35_v50",   [('GDX', 0.35, 0.50, 0.10, 0.20)]),
        ("A_gdx_v25_8_18",  [('GDX', 0.25, 0.40, 0.08, 0.18)]),
        ("A_gdx_v30_12_25", [('GDX', 0.30, 0.45, 0.12, 0.25)]),
        ("A_gdx_v30_15_25", [('GDX', 0.30, 0.45, 0.15, 0.25)]),
    ]
    for label, vol_rules in A_configs:
        p = make_params(vol_rules)
        r = evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, p)
        print(fmt(label, r, REF))
        results.append({'label': label, **r})

    # ── Sweep B: GDX Natural Competition ──────────────────────────────────────
    print("\n📊 Sweep B: GDX Natural Competition (momentum-based)")
    B_configs = [
        ("B_gdx_compete_50_10", 0.50, 0.10),
        ("B_gdx_compete_40_10", 0.40, 0.10),
        ("B_gdx_compete_30_10", 0.30, 0.10),
        ("B_gdx_compete_50_15", 0.50, 0.15),
        ("B_gdx_compete_40_15", 0.40, 0.15),
    ]
    for label, thresh, frac in B_configs:
        # GDX competes + GDX vol trigger
        p = make_params([('GDX', 0.30, 0.45, 0.10, 0.20)],
                        gdx_compete_thresh=thresh, gdx_compete_frac=frac)
        r = evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, p)
        print(fmt(label, r, REF))
        results.append({'label': label, **r})

    # ── Sweep C: Dual-Asset Vol Hedge ──────────────────────────────────────────
    print("\n📊 Sweep C: Dual-Asset Vol Hedge (GDX + TLT or GDX + GLD)")
    C_configs = [
        ("C_gdx30+tlt50",  [('GDX', 0.30, 0.45, 0.10, 0.10), ('TLT', 0.50, 0.60, 0.10, 0.20)]),
        ("C_gdx30_tlt50_sm",[('GDX', 0.30, 0.45, 0.08, 0.08), ('TLT', 0.50, 0.60, 0.08, 0.15)]),
        ("C_gdx+gld_split", [('GDX', 0.30, 0.45, 0.08, 0.15), ('GLD', 0.30, 0.45, 0.05, 0.10)]),
        ("C_gdx_heavy",     [('GDX', 0.25, 0.35, 0.10, 0.15), ('GDX', 0.35, 0.50, 0.15, 0.25)]),  # wrong, use one rule
        ("C_v9c_+gdx10",    [('GLD', 0.30, 0.45, 0.10, 0.20), ('GDX', 0.30, 0.45, 0.05, 0.10)]),
    ]
    # Fix C_gdx_heavy to use single GDX rule
    C_configs[3] = ("C_gdx_bigger", [('GDX', 0.25, 0.40, 0.12, 0.22)])
    for label, vol_rules in C_configs:
        p = make_params(vol_rules)
        r = evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, p)
        print(fmt(label, r, REF))
        results.append({'label': label, **r})

    # ── Sweep D: Best A + GDX competition combo ────────────────────────────────
    print("\n📊 Sweep D: Best Combos")
    best_A = max([x for x in results if x['label'].startswith('A_')], key=lambda x: x['comp'])
    best_B = max([x for x in results if x['label'].startswith('B_')], key=lambda x: x['comp'])
    print(f"  Best A: {best_A['label']} Comp={best_A['comp']:.4f}")
    print(f"  Best B: {best_B['label']} Comp={best_B['comp']:.4f}")

    # Combo: Best A vol rules + best B competition
    p = make_params(
        vol_rules=[(k.split(',')[0], float(k.split(',')[1]), float(k.split(',')[2]),
                    float(k.split(',')[3]), float(k.split(',')[4]))
                   for k in []] or [('GDX', 0.30, 0.45, 0.10, 0.20)],
        gdx_compete_thresh=best_B['params'].get('gdx_compete_thresh', 0.50) if 'params' in best_B else 0.50,
        gdx_compete_frac=best_B['params'].get('gdx_compete_frac', 0.10) if 'params' in best_B else 0.10,
    )

    # Simpler: Best A + GDX competition (best of B)
    # Extract A's vol rules — use hardcoded best
    best_A_label = best_A['label']
    A_map = {label: vol_rules for label, vol_rules in A_configs}
    best_A_rules = A_map.get(best_A_label, [('GDX', 0.30, 0.45, 0.10, 0.20)])
    best_B_thresh = 0.50; best_B_frac = 0.10

    if 'B_gdx_compete_40' in best_B['label']:
        best_B_thresh = 0.40
    elif 'B_gdx_compete_30' in best_B['label']:
        best_B_thresh = 0.30
    if '15' in best_B['label']:
        best_B_frac = 0.15

    p_combo = make_params(best_A_rules, gdx_compete_thresh=best_B_thresh,
                          gdx_compete_frac=best_B_frac)
    r = evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, p_combo)
    print(fmt("D_bestA+bestB_compete", r, REF))
    results.append({'label': 'D_bestA+bestB_compete', **r})

    # Also try baseline + GDX competition (no vol change)
    for thresh, frac in [(0.50, 0.10), (0.40, 0.10), (0.30, 0.10)]:
        p = make_params([('GDX', 0.30, 0.45, 0.10, 0.20)],
                        gdx_compete_thresh=thresh, gdx_compete_frac=frac)
        r = evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, p)
        print(fmt(f"D_base+gdxC{thresh:.0%}_{frac:.0%}", r, REF))
        results.append({'label': f'D_base+gdxC{thresh:.0%}_{frac:.0%}', **r})

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("🏆 TOP 8 Results (WF ≥ 0.70):")
    print("=" * 82)
    valid = sorted([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'], reverse=True)[:8]
    for r in valid:
        m = r['full']
        print(f"  {r['label']:33s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  "
              f"DD {m['max_dd']:.1%}  WF {r['wf']:.2f}  Comp {r['comp']:.4f}")

    champion = max([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'])
    cm = champion['full']
    comp = champion['comp']
    wf = champion['wf']

    print(f"\n🏆 Champion: {champion['label']}")
    print(f"  CAGR: {cm['cagr']:.1%}  Sharpe: {cm['sharpe']:.2f}  MaxDD: {cm['max_dd']:.1%}")
    print(f"  IS Sharpe: {champion['is_m']['sharpe']:.2f}  OOS: {champion['oos_m']['sharpe']:.2f}")
    print(f"  WF: {wf:.2f}  Composite: {comp:.4f}  vs v9d (+{comp-REF:+.4f})")

    if comp > 1.80 or cm['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】!")
    elif comp > 1.60:
        print(f"\n🚀 突破1.60! Composite {comp:.4f}")
    elif comp > 1.57:
        print(f"\n✅ 超越 v9d: Composite {comp:.4f}")

    # Save
    out = {'champion': champion['label'],
           'champion_metrics': {k: float(v) for k, v in cm.items()} | {'wf': float(wf), 'composite': float(comp)},
           'baseline_v9d': REF,
           'improvement': float(comp - REF),
           'results': [{'label': r['label'], 'comp': float(r['comp']), 'wf': float(r['wf']),
                        'cagr': float(r['full']['cagr']), 'sharpe': float(r['full']['sharpe']),
                        'max_dd': float(r['full']['max_dd']), 'calmar': float(r['full']['calmar'])}
                       for r in results]}
    jf = Path(__file__).parent / "momentum_v9e_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")


if __name__ == '__main__':
    main()
