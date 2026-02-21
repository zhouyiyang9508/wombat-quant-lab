#!/usr/bin/env python3
"""
动量轮动 v9c — Volatility Pre-emptive Hedge + Quality Combo
代码熊 🐻

当前最佳: v9b C_hi52_60% (Composite 1.526, Sharpe 1.59, WF 0.89)
目标: 进一步减少 MaxDD (当前-14.9%) → 提升 Calmar → 突破 Composite 1.55

核心问题: -14.9% MaxDD 的来源是急跌（如2020年3月COVID崩盘）
  - 月度再平衡无法及时响应月内急跌
  - DD响应GLD只能在下月才介入
  - 需要"提前预警"机制

新思路 — [A] 波动率预警 (Volatility Pre-emptive Hedge)
  - 不等 MaxDD 触发，在高波动性期间提前分配 GLD
  - VIX代理: SPY 5日已实现波动率 (annualized)
  - 如果 SPY_5d_vol > 25%: 预先分配 10-15% GLD
  - 如果 SPY_5d_vol > 40%: 预先分配 20-25% GLD
  - 这是 ADDITIVE 的叠加（加上现有的DD响应GLD）
  - 与 v5d 不同: v5d 替代DD响应，v9c 叠加使用

新思路 — [B] 更激进的提前DD触发
  - 当前: -8% → 30%GLD, -12% → 50%GLD, -18% → 60%GLD
  - 尝试: -5% → 15%GLD, -8% → 35%GLD, -12% → 55%GLD, -18% → 65%GLD
  - 更早介入 → 更小的 MaxDD

新思路 — [C] 组合 (最佳v9b参数 + 波动率预警 + 提前DD)
  - C_hi52_60% + SHY + vol_preemptive + tighter_dd

严格无前瞻:
  - SPY_5d_vol 用月末前5个交易日的收盘价计算
  - 使用 close[i-1] 原则
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


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
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    # SPY 5-day realized vol (annualized) — used as VIX proxy
    spy_log = np.log(close_df['SPY'] / close_df['SPY'].shift(1)) if 'SPY' in close_df.columns else None
    spy_vol5 = spy_log.rolling(5).std() * np.sqrt(252) if spy_log is not None else None
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi, vol30=vol30,
                spy_vol5=spy_vol5, spy=spy, s200=s200, sma50=sma50, close=close_df)


def compute_breadth(sig, date):
    close = sig['close'].loc[:date].dropna(how='all')
    sma50 = sig['sma50'].loc[:date].dropna(how='all')
    if len(close) < 50:
        return 1.0
    lc = close.iloc[-1]; ls = sma50.iloc[-1]
    mask = (lc > ls).dropna()
    return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0


def get_regime(sig, date, breadth_thresh):
    if sig['s200'] is None:
        return 'bull'
    spy_now = sig['spy'].loc[:date].dropna()
    s200_now = sig['s200'].loc[:date].dropna()
    if len(spy_now) == 0 or len(s200_now) == 0:
        return 'bull'
    spy_bear     = spy_now.iloc[-1] < s200_now.iloc[-1]
    breadth_bear = compute_breadth(sig, date) < breadth_thresh
    return 'bear' if (spy_bear and breadth_bear) else 'bull'


def get_spy_vol(sig, date):
    """Get SPY 5-day realized vol (annualized) at date — no lookahead"""
    if sig['spy_vol5'] is None:
        return 0.0
    vol_series = sig['spy_vol5'].loc[:date].dropna()
    return float(vol_series.iloc[-1]) if len(vol_series) > 0 else 0.0


def gld_competition(sig, date, gld_prices, gld_thresh, gld_frac):
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1:
        return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10:
        return 0.0
    avg_r6 = stock_r6.mean()
    gld_h = gld_prices.loc[:d].dropna()
    if len(gld_h) < 130:
        return 0.0
    gld_r6 = gld_h.iloc[-1] / gld_h.iloc[-127] - 1
    return gld_frac if gld_r6 >= avg_r6 * gld_thresh else 0.0


def select(sig, sectors, date, prev_hold, gld_prices, params):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    d = idx[-1]

    w1, w3, w6, w12 = params['mom_w']
    mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
           sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom':   mom,
        'r6':    sig['r6'].loc[d],
        'vol':   sig['vol30'].loc[d],
        'price': close.loc[d],
        'sma50': sig['sma50'].loc[d],
        'hi52':  sig['r52w_hi'].loc[d],
    }).dropna(subset=['mom', 'sma50'])

    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]

    # 52w-high filter
    hi52f = params.get('hi52_frac')
    if hi52f:
        df = df[df['price'] >= df['hi52'] * hi52f]

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    cont = params.get('cont_bonus', 0.03)
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += cont

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)
    gld_a = gld_competition(sig, date, gld_prices, params['gld_thresh'], params['gld_frac'])
    reg = get_regime(sig, date, params['breadth_thresh'])

    if reg == 'bull':
        n_secs = params['n_bull_secs'] - (1 if gld_a > 0 else 0)
        sps, cash = params['bull_sps'], 0.0
    else:
        n_secs = 3 - (1 if gld_a > 0 else 0)
        sps, cash = params['bear_sps'], 0.20

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


def apply_overlays(weights, dd, spy_vol, params):
    """Apply DD overlay + optional vol pre-emptive overlay"""
    dd_params = params.get('dd_params', {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60})
    dd_gld = max((dd_params[th] for th in sorted(dd_params) if dd < th), default=0.0)

    # Vol pre-emptive overlay (ADDITIVE to DD response)
    vol_gld = 0.0
    vol_params = params.get('vol_params')  # e.g. {0.25: 0.10, 0.40: 0.20}
    if vol_params and spy_vol > 0:
        for thresh in sorted(vol_params):
            if spy_vol > thresh:
                vol_gld = vol_params[thresh]

    # Total GLD to add (cap at reasonable level)
    total_add = max(dd_gld, vol_gld)  # use max (not sum) to avoid over-hedging

    if total_add <= 0 or not weights:
        return weights
    tot = sum(weights.values())
    if tot <= 0:
        return weights
    new = {t: w/tot*(1-total_add) for t, w in weights.items()}
    new['GLD'] = new.get('GLD', 0) + total_add
    return new


def run_backtest(close_df, sig, sectors, gld, shy, params,
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

        w = select(sig, sectors, dt, prev_h, gld, params)
        w = apply_overlays(w, dd, spy_vol, params)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t,0)-prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k != 'GLD'}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)

        ret = 0.0
        for t, wt in w.items():
            if t == 'GLD':
                s = gld.loc[dt:ndt].dropna()
            elif t in close_df.columns:
                s = close_df[t].loc[dt:ndt].dropna()
            else:
                continue
            if len(s) >= 2:
                ret += (s.iloc[-1]/s.iloc[0]-1) * wt

        if params.get('use_shy') and cash_frac > 0 and shy is not None:
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
    if len(eq) < 3:
        return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5:
        return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    mo = eq.pct_change().dropna()
    sh = mo.mean()/mo.std()*np.sqrt(12) if mo.std() > 0 else 0
    dd = ((eq - eq.cummax())/eq.cummax()).min()
    cal = cagr/abs(dd) if dd != 0 else 0
    return dict(cagr=float(cagr), max_dd=float(dd), sharpe=float(sh), calmar=float(cal))


def evaluate(close_df, sig, sectors, gld, shy, params):
    eq_f, to = run_backtest(close_df, sig, sectors, gld, shy, params)
    eq_i, _  = run_backtest(close_df, sig, sectors, gld, shy, params, '2015-01-01', '2020-12-31')
    eq_o, _  = run_backtest(close_df, sig, sectors, gld, shy, params, '2021-01-01', '2025-12-31')
    m  = metrics(eq_f); mi = metrics(eq_i); mo = metrics(eq_o)
    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
    return dict(full=m, is_m=mi, oos_m=mo, wf=float(wf), comp=float(comp), to=float(to))


def fmt(label, r):
    m = r['full']
    flag = '✅' if r['comp']>1.5 and r['wf']>=0.7 else '⭐' if r['comp']>1.48 and r['wf']>=0.7 else ''
    return (f"  {label:35s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  DD {m['max_dd']:.1%}  "
            f"Cal {m['calmar']:.2f}  WF {r['wf']:.2f}  Comp {r['comp']:.4f} {flag}")


def main():
    print("=" * 80)
    print("🐻 动量轮动 v9c — 波动率预警 + 更激进DD触发 + 质量组合")
    print("=" * 80)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers")

    # ── v9a baseline params ──────────────────────────────────────────────────
    BASE_P = dict(
        mom_w=(0.20, 0.50, 0.20, 0.10),
        n_bull_secs=5, bull_sps=2, bear_sps=2,
        breadth_thresh=0.45,
        gld_thresh=0.70, gld_frac=0.20,
        cont_bonus=0.03,
        hi52_frac=0.60,   # from v9b: 52w filter
        use_shy=True,     # from v9b: SHY cash
        dd_params={-0.08: 0.30, -0.12: 0.50, -0.18: 0.60},
        vol_params=None,  # no vol overlay (baseline)
    )

    results = []

    print("\n🔄 Baseline (v9b champion: hi52=60% + SHY)...")
    base = evaluate(close_df, sig, sectors, gld, shy, BASE_P)
    print(fmt("v9b_hi52_shy_baseline", base))
    results.append({'label': 'v9b_baseline', **base})

    # ── Sweep A: Vol Pre-emptive Overlay ──────────────────────────────────────
    print("\n📊 Sweep A: SPY Volatility Pre-emptive GLD Overlay")
    print("    (If SPY 5d-vol > threshold, add GLD BEFORE DD triggers)")
    vol_configs = [
        ({0.25: 0.10, 0.40: 0.20}, "A_vol25:10%,40:20%"),
        ({0.25: 0.15, 0.40: 0.25}, "A_vol25:15%,40:25%"),
        ({0.30: 0.10, 0.45: 0.20}, "A_vol30:10%,45:20%"),
        ({0.30: 0.15, 0.50: 0.25}, "A_vol30:15%,50:25%"),
        ({0.20: 0.10, 0.35: 0.20}, "A_vol20:10%,35:20%"),
        ({0.25: 0.10}, "A_vol25:10%_only"),
        ({0.20: 0.10, 0.30: 0.20, 0.45: 0.30}, "A_vol3tier"),
    ]
    for vp, label in vol_configs:
        p = dict(BASE_P); p['vol_params'] = vp
        r = evaluate(close_df, sig, sectors, gld, shy, p)
        print(fmt(label, r))
        results.append({'label': label, **r})

    # ── Sweep B: Tighter DD Thresholds ───────────────────────────────────────
    print("\n📊 Sweep B: Tighter DD Trigger Levels")
    dd_configs = [
        ({-0.05: 0.20, -0.08: 0.35, -0.12: 0.55, -0.18: 0.65}, "B_dd5:20,8:35,12:55,18:65"),
        ({-0.05: 0.15, -0.08: 0.30, -0.12: 0.50, -0.18: 0.60}, "B_dd5:15,8:30,12:50,18:60"),
        ({-0.06: 0.20, -0.10: 0.40, -0.15: 0.60}, "B_dd6:20,10:40,15:60"),
        ({-0.05: 0.10, -0.10: 0.30, -0.15: 0.50, -0.20: 0.65}, "B_dd5:10,10:30,15:50,20:65"),
        ({-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}, "B_more_aggressive"),
    ]
    for dp, label in dd_configs:
        p = dict(BASE_P); p['dd_params'] = dp
        r = evaluate(close_df, sig, sectors, gld, shy, p)
        print(fmt(label, r))
        results.append({'label': label, **r})

    # ── Sweep C: Vol + DD Combined ────────────────────────────────────────────
    print("\n📊 Sweep C: Best Vol + Best DD Combined")
    best_A = max([r for r in results if r['label'].startswith('A_')], key=lambda x: x['comp'])
    best_B = max([r for r in results if r['label'].startswith('B_')], key=lambda x: x['comp'])
    print(f"  Best A: {best_A['label']} Comp={best_A['comp']:.4f}")
    print(f"  Best B: {best_B['label']} Comp={best_B['comp']:.4f}")

    # A + B combined
    # Need to recover vol_params and dd_params from the best results
    # Since we can't easily recover, let's just test key combos manually
    for (vp, dp, label) in [
        ({0.25: 0.10, 0.40: 0.20}, {-0.05: 0.15, -0.08: 0.30, -0.12: 0.50, -0.18: 0.60}, "C_vol+dd5"),
        ({0.30: 0.10, 0.45: 0.20}, {-0.05: 0.15, -0.08: 0.30, -0.12: 0.50, -0.18: 0.60}, "C_vol30+dd5"),
        ({0.25: 0.10, 0.40: 0.20}, {-0.06: 0.20, -0.10: 0.40, -0.15: 0.60}, "C_vol+dd6"),
    ]:
        p = dict(BASE_P); p['vol_params'] = vp; p['dd_params'] = dp
        r = evaluate(close_df, sig, sectors, gld, shy, p)
        print(fmt(label, r))
        results.append({'label': label, **r})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🏆 TOP 5 Results (WF ≥ 0.70):")
    print("=" * 80)
    valid = sorted([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'], reverse=True)[:5]
    for r in valid:
        m = r['full']
        print(f"  {r['label']:38s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  "
              f"DD {m['max_dd']:.1%}  Cal {m['calmar']:.2f}  WF {r['wf']:.2f}  Comp {r['comp']:.4f}")

    champion = max([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'])
    cm = champion['full']
    wf = champion['wf']
    comp = champion['comp']

    print(f"\n🏆 Champion: {champion['label']}")
    print(f"  CAGR:       {cm['cagr']:.1%}  {'✅' if cm['cagr']>0.30 else ''}")
    print(f"  MaxDD:      {cm['max_dd']:.1%}")
    print(f"  Sharpe:     {cm['sharpe']:.2f}")
    print(f"  Calmar:     {cm['calmar']:.2f}")
    print(f"  IS Sharpe:  {champion['is_m']['sharpe']:.2f}")
    print(f"  OOS Sharpe: {champion['oos_m']['sharpe']:.2f}")
    print(f"  WF ratio:   {wf:.2f}")
    print(f"  Composite:  {comp:.4f}")
    print(f"\n  vs v9b baseline: Δ Composite = {comp - base['comp']:+.4f}")
    print(f"  vs v9a (1.512):  Δ Composite = {comp - 1.5116:+.4f}")

    if comp > 1.8 or cm['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】 Composite > 1.8!")
    elif comp > 1.55:
        print(f"\n✅ 突破 1.55! Composite {comp:.4f}")
    elif comp > 1.5:
        print(f"\n✅ 继续 > 1.5, Composite {comp:.4f}")
    else:
        print(f"\n⚠️ 未提升, Composite {comp:.4f}")

    out = {
        'champion': champion['label'],
        'champion_metrics': {'cagr': cm['cagr'], 'sharpe': cm['sharpe'],
                             'max_dd': cm['max_dd'], 'calmar': cm['calmar'],
                             'wf': wf, 'composite': comp},
        'baseline_v9b': float(base['comp']),
        'baseline_v9a': 1.5116,
        'improvement_vs_v9b': float(comp - base['comp']),
        'results': [{'label': r['label'], 'comp': r['comp'], 'wf': r['wf'],
                     'cagr': r['full']['cagr'], 'sharpe': r['full']['sharpe'],
                     'max_dd': r['full']['max_dd'], 'calmar': r['full']['calmar']}
                    for r in results]
    }
    jf = Path(__file__).parent / "momentum_v9c_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return champion


if __name__ == '__main__':
    main()
