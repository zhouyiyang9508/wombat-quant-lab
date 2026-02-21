#!/usr/bin/env python3
"""
动量轮动 v9d — TLT/GDX 多资产安全港 + 波动率触发优化
代码熊 🐻

核心思路: v9c 用 GLD 作为波动率触发的预警对冲资产
但 TLT (长期国债) 在 COVID 类型危机中表现远优于 GLD!

对比 2020年3月:
  GLD:  -3% (初期跌, 后恢复)
  TLT: +14% (美联储暴力降息, 长期国债飙涨)
  SPY: -13% (熊市)

现有 v9c:
  - 波动率触发 (vol>30%): 10% GLD (作为防御)
  - 回撤触发 (DD>8%): 40% GLD (作为对冲)
  - GLD 竞争: 20% GLD (基于6m动量)

v9d 改进:
  - 波动率触发 → TLT (利率切割受益)
  - 回撤触发   → GLD (保持现有, GLD 对缓慢熊市更好)
  - GLD 竞争   → 保持 (基于动量自然竞争)

理由: TLT 在快速崩盘+美联储干预场景效果最好
      GLD 在缓慢通胀/不确定性环境中效果最好
      波动率触发 ↔ 快速崩盘; DD触发 ↔ 缓慢熊市

变种对比:
  v9d_tlt_vol:   vol触发用TLT代替GLD
  v9d_tlt_gld:   vol触发用50%TLT+50%GLD
  v9d_gdx_vol:   vol触发用GDX (黄金矿工, 更高beta)
  v9d_tlt_comp:  TLT也参与动量竞争 (和GLD一起)
  v9d_max_tlt:   更激进TLT分配 (vol>30%→20%TLT)
  v9d_2asset:    GLD竞争 + TLT vol触发 + GLD DD (三层)

注意: 2022年 TLT 跌了30%，但若vol信号在2022不触发
      (2022是缓慢下跌，非高vol崩盘)，则可以避免
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9c champion params (baseline for all variants)
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
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1
    r1  = close_df / close_df.shift(22)  - 1
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol5  = log_r.rolling(5).std() * np.sqrt(252)   # 5-day vol (for vol-trigger)
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200,
                sma50=sma50, close=close_df)


def get_spy_vol(sig, date):
    """Get SPY 5-day annualized vol at date"""
    vol = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None
    if vol is None:
        return 0.0
    v = vol.loc[:date].dropna()
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
    spy_bear     = spy_now.iloc[-1] < s200_now.iloc[-1]
    breadth_bear = compute_breadth(sig, date) < BREADTH_NARROW
    return 'bear' if (spy_bear and breadth_bear) else 'bull'


def gld_competition(sig, date, gld_prices):
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
    gld_r6 = gld_h.iloc[-1] / gld_h.iloc[-127] - 1
    return GLD_COMPETE_FRAC if gld_r6 >= avg_r6 * GLD_AVG_THRESH else 0.0


def get_vol_allocation(spy_vol, variant):
    """
    Return (tlt_alloc, gld_alloc) based on vol signal and variant
    variant controls which asset(s) are used for vol-triggered hedge
    """
    v30, v45 = 0.30, 0.45
    if variant == 'gld_only':       # v9c baseline
        if spy_vol >= v45: return 0.0, 0.20
        elif spy_vol >= v30: return 0.0, 0.10
        else: return 0.0, 0.0
    elif variant == 'tlt_only':     # TLT replaces GLD for vol trigger
        if spy_vol >= v45: return 0.20, 0.0
        elif spy_vol >= v30: return 0.10, 0.0
        else: return 0.0, 0.0
    elif variant == 'tlt_gld':      # Split: 50% TLT + 50% GLD for vol
        if spy_vol >= v45: return 0.10, 0.10
        elif spy_vol >= v30: return 0.05, 0.05
        else: return 0.0, 0.0
    elif variant == 'tlt_max':      # More aggressive TLT
        if spy_vol >= v45: return 0.25, 0.0
        elif spy_vol >= v30: return 0.15, 0.0
        else: return 0.0, 0.0
    elif variant == 'tlt_v25':      # Lower threshold for TLT (25% vol)
        if spy_vol >= v45: return 0.20, 0.0
        elif spy_vol >= v30: return 0.10, 0.0
        elif spy_vol >= 0.25: return 0.05, 0.0
        else: return 0.0, 0.0
    elif variant == 'gdx_vol':      # GDX (gold miners) for vol trigger
        if spy_vol >= v45: return 0.0, 0.20    # reuse gld slot for GDX
        elif spy_vol >= v30: return 0.0, 0.10
        else: return 0.0, 0.0
    elif variant == 'tlt_v20':      # Very sensitive TLT trigger at 20% vol
        if spy_vol >= v45: return 0.20, 0.0
        elif spy_vol >= v30: return 0.10, 0.0
        elif spy_vol >= 0.20: return 0.05, 0.0
        else: return 0.0, 0.0
    return 0.0, 0.0


def select(sig, sectors, date, prev_hold, gld_prices, variant):
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
    gld_a   = gld_competition(sig, date, gld_prices)
    reg     = get_regime(sig, date)

    if reg == 'bull':
        n_secs = N_BULL_SECS - (1 if gld_a > 0 else 0)
        sps, cash = BULL_SPS, 0.0
    else:
        n_secs = 3 - (1 if gld_a > 0 else 0)
        sps, cash = BEAR_SPS, 0.20

    n_secs = max(n_secs, 1)
    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - cash - gld_a, 0.0)
    if not selected:
        return {'GLD': gld_a} if gld_a > 0 else {}

    iv  = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}

    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0:
        weights['GLD'] = gld_a
    return weights


def apply_overlays(weights, spy_vol, dd, variant, gdx_prices, sig, date):
    """Apply vol-trigger (TLT/GDX) and DD-responsive (GLD) overlays"""
    tlt_a, gld_vol_a = get_vol_allocation(spy_vol, variant)

    # DD-responsive GLD overlay
    gld_dd_a = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    # For GDX variant, use gld slot for GDX
    if variant == 'gdx_vol':
        gdx_vol_a = gld_vol_a
        gld_vol_a = 0.0
    else:
        gdx_vol_a = 0.0

    # Total safe haven = max(pre-emptive, DD response) + TLT + GDX
    # Strategy: pre-emptive and DD don't stack fully (take the max for GLD)
    gld_total = max(gld_vol_a, gld_dd_a)  # GLD: max of vol and DD trigger
    tlt_total = tlt_a
    gdx_total = gdx_vol_a

    total_hedge = gld_total + tlt_total + gdx_total
    if total_hedge <= 0 or not weights:
        return weights

    # Rescale existing weights
    stock_frac = max(1.0 - total_hedge, 0.01)
    tot = sum(weights.values())
    if tot <= 0:
        return weights

    new = {t: w/tot*stock_frac for t, w in weights.items()}
    if gld_total > 0:
        new['GLD'] = new.get('GLD', 0) + gld_total
    if tlt_total > 0:
        new['TLT'] = new.get('TLT', 0) + tlt_total
    if gdx_total > 0:
        new['GDX'] = new.get('GDX', 0) + gdx_total
    return new


def run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, variant,
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

        w = select(sig, sectors, dt, prev_h, gld, variant)
        w = apply_overlays(w, spy_vol, dd, variant, gdx, sig, dt)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t,0)-prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'TLT', 'GDX')}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)

        ret = 0.0
        for t, wt in w.items():
            if t == 'GLD':
                s = gld.loc[dt:ndt].dropna()
            elif t == 'TLT':
                s = tlt.loc[dt:ndt].dropna()
            elif t == 'GDX':
                s = gdx.loc[dt:ndt].dropna()
            elif t in close_df.columns:
                s = close_df[t].loc[dt:ndt].dropna()
            else:
                continue
            if len(s) >= 2:
                ret += (s.iloc[-1]/s.iloc[0]-1) * wt

        # SHY for cash
        if USE_SHY and cash_frac > 0 and shy is not None:
            s = shy.loc[dt:ndt].dropna()
            if len(s) >= 2:
                ret += (s.iloc[-1]/s.iloc[0]-1) * cash_frac

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


def evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, variant):
    eq_f, to = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, variant)
    eq_i, _  = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, variant,
                             '2015-01-01', '2020-12-31')
    eq_o, _  = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, variant,
                             '2021-01-01', '2025-12-31')
    m = metrics(eq_f); mi = metrics(eq_i); mo = metrics(eq_o)
    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
    return dict(full=m, is_m=mi, oos_m=mo, wf=float(wf), comp=float(comp), to=float(to))


def fmt(label, r, baseline_comp=1.5669):
    m = r['full']
    delta = r['comp'] - baseline_comp
    flag = '🚀' if r['comp']>1.6 and r['wf']>=0.7 else '✅' if r['comp']>1.55 and r['wf']>=0.7 else '⭐' if r['comp']>1.53 and r['wf']>=0.7 else ''
    return (f"  {label:25s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  "
            f"DD {m['max_dd']:.1%}  Cal {m['calmar']:.2f}  WF {r['wf']:.2f}  "
            f"Comp {r['comp']:.4f} ({delta:+.4f}) {flag}")


def main():
    print("=" * 78)
    print("🐻 动量轮动 v9d — TLT/GDX 多资产安全港 + 波动率触发优化")
    print("=" * 78)
    print("  Strategy: TLT for vol-trigger hedge (better in rate-cut crashes)")
    print("           GLD for DD-responsive hedge (better in slow bear markets)")
    print("  Baseline: v9c (Composite 1.5669, Sharpe 1.638, WF 0.889)")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    tlt = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    gdx = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers + GLD, TLT, SHY, GDX")

    results = []

    VARIANTS = [
        ('gld_only',  'v9c Baseline (GLD vol+DD)'),
        ('tlt_only',  'TLT vol-trigger (GLD DD)'),
        ('tlt_gld',   '50%TLT+50%GLD vol-trigger'),
        ('tlt_max',   'TLT vol-trigger 15-25%'),
        ('tlt_v25',   'TLT vol-trigger (25% thresh)'),
        ('tlt_v20',   'TLT vol-trigger (20% thresh)'),
        ('gdx_vol',   'GDX vol-trigger (miners)'),
    ]

    for var, label in VARIANTS:
        print(f"\n🔄 {label}...")
        r = evaluate(close_df, sig, sectors, gld, tlt, shy, gdx, var)
        print(fmt(label, r))
        results.append({'variant': var, 'label': label, **r})

    # ── Identify best variant ─────────────────────────────────────────────────
    valid = [r for r in results if r['wf'] >= 0.70]
    champion = max(valid, key=lambda x: x['comp']) if valid else results[0]

    print("\n" + "=" * 78)
    print("📊 Summary:")
    print("=" * 78)
    for r in sorted(results, key=lambda x: x['comp'], reverse=True):
        print(fmt(r['label'], r))

    cm = champion['full']
    wf = champion['wf']
    comp = champion['comp']

    print(f"\n🏆 Champion: {champion['label']}")
    print(f"  CAGR:       {cm['cagr']:.1%}")
    print(f"  Sharpe:     {cm['sharpe']:.2f}  {'✅' if cm['sharpe']>1.5 else ''}")
    print(f"  MaxDD:      {cm['max_dd']:.1%}")
    print(f"  Calmar:     {cm['calmar']:.2f}")
    print(f"  IS Sharpe:  {champion['is_m']['sharpe']:.2f}")
    print(f"  OOS Sharpe: {champion['oos_m']['sharpe']:.2f}")
    print(f"  WF ratio:   {wf:.2f}  {'✅' if wf>=0.70 else '❌'}")
    print(f"  Composite:  {comp:.4f}  {'✅ > 1.5!' if comp>1.5 else ''}")
    print(f"  vs v9c:     {comp - 1.5669:+.4f}")

    if comp > 1.8 or cm['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】!")
    elif comp > 1.60:
        print(f"\n🚀 重要突破: Composite {comp:.4f} > 1.60!")
    elif comp > 1.57:
        print(f"\n✅ 进一步改善: Composite {comp:.4f}")
    elif comp > 1.5:
        print(f"\n⭐ 维持在 Composite {comp:.4f} > 1.5")
    else:
        print(f"\n❌ 未超越 v9c (1.5669)")

    # Save
    out = {
        'champion': champion['label'],
        'champion_variant': champion['variant'],
        'champion_metrics': {k: float(v) for k, v in cm.items()} | {'wf': float(wf), 'composite': float(comp)},
        'baseline_v9c': 1.5669,
        'improvement': float(comp - 1.5669),
        'results': [{'label': r['label'], 'variant': r['variant'],
                     'comp': float(r['comp']), 'wf': float(r['wf']),
                     'cagr': float(r['full']['cagr']), 'sharpe': float(r['full']['sharpe']),
                     'max_dd': float(r['full']['max_dd']), 'calmar': float(r['full']['calmar']),
                     'is_sharpe': float(r['is_m']['sharpe']), 'oos_sharpe': float(r['oos_m']['sharpe'])}
                    for r in results]
    }
    jf = Path(__file__).parent / "momentum_v9d_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return champion


if __name__ == '__main__':
    main()
