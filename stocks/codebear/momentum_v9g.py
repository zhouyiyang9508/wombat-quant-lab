#!/usr/bin/env python3
"""
动量轮动 v9g — 自适应动量权重 + IEF债券 + 动态行业集中度
代码熊 🐻

当前最佳: v9f — Composite 1.667, CAGR 34.6%, Sharpe 1.67, WF 0.88

v9g 三个新方向 (在v9f正确实现基础上):

[A] 自适应动量权重 (Adaptive Momentum Weights)
    基于SPY相对SMA50的位置，动态调整动量周期权重：
    - SPY/SMA50 > 1.03 & breadth>0.55: "强牛" → 偏短期 (0.28, 0.50, 0.14, 0.08)
    - SPY/SMA50 < 0.99: "弱牛/过渡" → 偏长期 (0.10, 0.38, 0.35, 0.17)
    - 默认: v9f权重 (0.20, 0.50, 0.20, 0.10)

[B] IEF债券整合
    极端波动期 (vol>45%) 加入5% IEF (7-10年期国债)
    原因: Fed极端恐慌时倾向降息 → 中期国债受益

[C] 动态行业集中度
    breadth > 0.65: 4行业×2=8支 (集中在强势行业)
    breadth 0.45-0.65: 5行业×2=10支 (当前)
    breadth < 0.45: 熊市, 减少持仓

测试: base / A / B / C / AB / AC / BC / ABC
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9f champion base params (fixed)
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


def get_spy_trend_strength(sig, date):
    """Return SPY/SMA50 ratio and breadth for adaptive weights"""
    if sig['spy'] is None or sig['s50'] is None:
        return 1.0, 0.5
    spy_now = sig['spy'].loc[:date].dropna()
    s50_now = sig['s50'].loc[:date].dropna()
    if len(spy_now) == 0 or len(s50_now) == 0:
        return 1.0, 0.5
    ratio   = float(spy_now.iloc[-1]) / float(s50_now.iloc[-1]) if float(s50_now.iloc[-1]) > 0 else 1.0
    breadth = compute_breadth(sig, date)
    return ratio, breadth


def get_adaptive_mom_weights(sig, date):
    """Adaptive momentum weights based on SPY/SMA50 strength"""
    ratio, breadth = get_spy_trend_strength(sig, date)
    if ratio > 1.03 and breadth > 0.55:
        # Strong bull: overweight 1m
        return (0.28, 0.50, 0.14, 0.08)
    elif ratio < 0.99:
        # Weakening: overweight long-term
        return (0.10, 0.38, 0.35, 0.17)
    else:
        return BASE_MOM_W


def get_dynamic_n_secs(sig, date, regime):
    """Dynamic sector count based on breadth"""
    if regime != 'bull': return 3
    breadth = compute_breadth(sig, date)
    return 4 if breadth > 0.65 else N_BULL_SECS


def asset_compete(sig, date, prices, thresh, frac, lb=127):
    r6  = sig['r6']
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

    # Momentum weights (adaptive or fixed)
    if cfg.get('adaptive_mom', False):
        w1, w3, w6, w12 = get_adaptive_mom_weights(sig, date)
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
        if t in prev_hold:
            df.loc[t, 'mom'] += CONT_BONUS

    if len(df) == 0: return {}
    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    # Competing safe assets
    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg = get_regime(sig, date)

    # Dynamic sector count (Direction C) or fixed
    if cfg.get('dynamic_secs', False):
        n_bull_target = get_dynamic_n_secs(sig, date, reg)
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


def apply_overlays(weights, spy_vol, dd, cfg):
    ief_hi = cfg.get('ief_hi', 0.0)

    # GDXJ vol trigger (v9f unchanged)
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    # IEF at extreme vol (Direction B)
    ief_v = ief_hi if (ief_hi > 0 and spy_vol >= GDXJ_VOL_HI_THRESH) else 0.0

    # GLD DD response (unchanged)
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdxj_v + ief_v + gld_dd
    if total <= 0 or not weights: return weights
    stock_frac = max(1.0 - total, 0.01)
    tot = sum(weights.values())
    if tot <= 0: return weights
    new = {t: w/tot*stock_frac for t, w in weights.items()}
    if gld_dd > 0: new['GLD'] = new.get('GLD', 0) + gld_dd
    if gdxj_v > 0: new['GDXJ'] = new.get('GDXJ', 0) + gdxj_v
    if ief_v  > 0: new['IEF']  = new.get('IEF',  0) + ief_v
    return new


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, ief_p, cfg,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd      = (val - peak) / peak if peak > 0 else 0
        spy_vol = get_spy_vol(sig, dt)

        w = select(sig, sectors, dt, prev_h, gld_p, gdx_p, cfg)
        w = apply_overlays(w, spy_vol, dd, cfg)

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t,0) - prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD','GDX','GDXJ','IEF')}

        invested   = sum(w.values()); cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t == 'IEF':  s = ief_p.loc[dt:ndt].dropna() if ief_p is not None else pd.Series()
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


CONFIGS = {
    'v9f_base':   {'adaptive_mom': False, 'dynamic_secs': False, 'ief_hi': 0.0},
    'A_adaptive': {'adaptive_mom': True,  'dynamic_secs': False, 'ief_hi': 0.0},
    'B_ief5':     {'adaptive_mom': False, 'dynamic_secs': False, 'ief_hi': 0.05},
    'B_ief8':     {'adaptive_mom': False, 'dynamic_secs': False, 'ief_hi': 0.08},
    'C_dynsecs':  {'adaptive_mom': False, 'dynamic_secs': True,  'ief_hi': 0.0},
    'AB_5':       {'adaptive_mom': True,  'dynamic_secs': False, 'ief_hi': 0.05},
    'AC':         {'adaptive_mom': True,  'dynamic_secs': True,  'ief_hi': 0.0},
    'BC_5':       {'adaptive_mom': False, 'dynamic_secs': True,  'ief_hi': 0.05},
    'ABC_5':      {'adaptive_mom': True,  'dynamic_secs': True,  'ief_hi': 0.05},
}


def main():
    print("=" * 70)
    print("🐻 动量轮动 v9g — 自适应权重 + IEF + 动态集中度")
    print("=" * 70)
    print(f"\nBase: v9f champion (Composite 1.667)")
    print(f"Testing {len(CONFIGS)} configurations...\n")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()

    ief_p = None
    ief_f = CACHE / "IEF.csv"
    if ief_f.exists():
        try:
            ief_p = load_csv(ief_f)['Close'].dropna()
            print(f"  ✅ IEF loaded: {len(ief_p)} rows")
        except:
            print("  ❌ IEF load failed")

    sig = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers")

    results = {}
    best_comp = 0.0
    best_name = ''

    for name, cfg in CONFIGS.items():
        print(f"\n--- {name} ---", flush=True)
        try:
            eq_f, to = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, ief_p, cfg)
            eq_is, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, ief_p, cfg,
                                     '2015-01-01', '2020-12-31')
            eq_oo, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, ief_p, cfg,
                                     '2021-01-01', '2025-12-31')
            m  = compute_metrics(eq_f)
            mi = compute_metrics(eq_is)
            mo = compute_metrics(eq_oo)
            wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            results[name] = {'full': m, 'is': mi, 'oos': mo, 'wf': wf, 'composite': comp, 'to': to}
            tag = '🚀🚀' if comp > 1.70 else ('🚀' if comp > 1.667 else ('✅' if comp > 1.60 else ''))
            print(f"  Comp={comp:.4f} {tag} | Sharpe={m['sharpe']:.2f} | CAGR={m['cagr']:.1%} | "
                  f"DD={m['max_dd']:.1%} | WF={wf:.2f}")
            if comp > best_comp:
                best_comp = comp
                best_name = name
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print("📊 FINAL RANKINGS")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda x: x[1]['composite'], reverse=True)
    for i, (name, r) in enumerate(ranked[:8]):
        m = r['full']
        tag = '🚀🚀' if r['composite'] > 1.70 else ('🚀' if r['composite'] > 1.667 else '')
        print(f"  #{i+1} {name}: Comp={r['composite']:.4f} {tag} | "
              f"Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%} WF={r['wf']:.2f}")

    print(f"\n🏆 Best: {best_name} → Composite {best_comp:.4f}")
    if best_comp > 1.80:
        print("🚨🚨🚨 【重大突破】Composite > 1.80!")
    elif best_comp > 1.70:
        print("🚀🚀 突破1.70!")
    elif best_comp > 1.667:
        print(f"🚀 超越v9f冠军! +{best_comp-1.667:.4f}")
    else:
        print("❌ 未超越v9f (1.667)")

    out = {
        'configs': CONFIGS,
        'results': {n: {'full': r['full'], 'is': r['is'], 'oos': r['oos'],
                        'wf': r['wf'], 'composite': r['composite']} for n, r in results.items()},
        'best': best_name, 'best_composite': best_comp
    }
    jf = Path(__file__).parent / "momentum_v9g_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results, best_name, best_comp


if __name__ == '__main__':
    main()
