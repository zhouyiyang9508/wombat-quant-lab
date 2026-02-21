#!/usr/bin/env python3
"""
动量轮动 v9b — Fine-Tuning v9a Champion (Composite 1.512 → 1.55+?)
代码熊 🐻

v9a 已突破 1.5 Composite! 现在尝试进一步优化:

[A] 持续奖励调优 (Continuation Bonus)
    当前: 0.03 持仓奖励 → 换仓成本 62.3%/月 → ~2.24%/年交易摩擦
    目标: 增大奖励 → 减少换仓 → 降低成本 → 提升收益
    尝试: 0.02, 0.03(baseline), 0.04, 0.05, 0.06, 0.08

[B] SHY替代熊市现金 (Bear Cash → SHY Returns)
    当前: 熊市20%现金 = 0%收益
    改进: 20%现金 = SHY实际收益 (~4-5%/年)
    预期: +0.3-0.5% CAGR

[C] 52周高点邻近度过滤 (52-Week High Proximity)
    只选 price > 52w_high × 0.70 的股票 (距高点<30%)
    过滤掉破位股，只买强势股
    这是 NOVEL 过滤器，此前未尝试

[D] 精细化参数扫描 (Fine-Grained around v9a Winner)
    n_bull_secs: 4, 5, 6  ×  bull_sps: 2, 3
    breadth_thresh: 0.42, 0.45, 0.47, 0.50
    更细粒度探索最优点

目标: Composite > 1.55, 进一步接近 1.6
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9a champion params (baseline)
V9A_PARAMS = dict(
    mom_w=(0.20, 0.50, 0.20, 0.10),  # 1m, 3m, 6m, 12m
    n_bull_secs=5, bull_sps=2, bear_sps=2,
    breadth_thresh=0.45,
    gld_thresh=0.70, gld_frac=0.20,
    cont_bonus=0.03,
    use_shy=False,
    hi52_frac=None,   # None = no filter
)
DD_PARAMS = {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}


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
    r52w_high = close_df.rolling(252).max()   # 52-week high (no lookahead: uses past 252 days)
    log_r = np.log(close_df / close_df.shift(1))
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, vol30=vol30,
                r52w_high=r52w_high, spy=spy, s200=s200, sma50=sma50, close=close_df)


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
    mom = (sig['r1'].loc[d] * w1 + sig['r3'].loc[d] * w3 +
           sig['r6'].loc[d] * w6 + sig['r12'].loc[d] * w12)

    df = pd.DataFrame({
        'mom':   mom,
        'r6':    sig['r6'].loc[d],
        'vol':   sig['vol30'].loc[d],
        'price': close.loc[d],
        'sma50': sig['sma50'].loc[d],
        'hi52':  sig['r52w_high'].loc[d],
    }).dropna(subset=['mom', 'sma50'])

    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]

    # 52-week high proximity filter (if enabled)
    hi52_frac = params.get('hi52_frac')
    if hi52_frac is not None:
        df = df[df['price'] >= df['hi52'] * hi52_frac]

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


def add_dd_gld(weights, dd):
    gld_a = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    if gld_a <= 0 or not weights:
        return weights
    tot = sum(weights.values())
    if tot <= 0:
        return weights
    new = {t: w/tot*(1-gld_a) for t, w in weights.items()}
    new['GLD'] = new.get('GLD', 0) + gld_a
    return new


def run_backtest(close_df, sig, sectors, gld, shy, params,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    use_shy = params.get('use_shy', False)

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        w = select(sig, sectors, dt, prev_h, gld, params)
        w = add_dd_gld(w, dd)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t,0) - prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k != 'GLD'}

        # Compute cash fraction
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
                ret += (s.iloc[-1]/s.iloc[0] - 1) * wt

        # SHY cash return (if enabled)
        if use_shy and cash_frac > 0 and shy is not None:
            s = shy.loc[dt:ndt].dropna()
            if len(s) >= 2:
                shy_ret = s.iloc[-1]/s.iloc[0] - 1
                ret += shy_ret * cash_frac

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
    return (f"  {label:32s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  DD {m['max_dd']:.1%}  "
            f"Cal {m['calmar']:.2f}  WF {r['wf']:.2f}  TO {r['to']:.0%}  Comp {r['comp']:.4f} {flag}")


def main():
    print("=" * 78)
    print("🐻 动量轮动 v9b — Fine-Tuning v9a Champion Toward Composite 1.55+")
    print("=" * 78)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers")

    results = []

    # Baseline v9a
    print("\n🔄 Baseline (v9a)...")
    base = evaluate(close_df, sig, sectors, gld, shy, V9A_PARAMS)
    print(fmt("v9a_baseline", base))
    results.append({'label': 'v9a_baseline', 'params': V9A_PARAMS, **base})

    # ── Sweep A: Continuation Bonus ───────────────────────────────────────────
    print("\n📊 Sweep A: Continuation Bonus (turnover control)")
    for cb in [0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        p = dict(V9A_PARAMS); p['cont_bonus'] = cb
        r = evaluate(close_df, sig, sectors, gld, shy, p)
        label = f"A_cb{cb:.2f}"
        print(fmt(label, r))
        results.append({'label': label, 'params': p, **r})

    # ── Sweep B: SHY Bear Cash ────────────────────────────────────────────────
    print("\n📊 Sweep B: SHY Bear Cash (cash earns SHY returns)")
    for use_shy in [True]:
        p = dict(V9A_PARAMS); p['use_shy'] = use_shy
        r = evaluate(close_df, sig, sectors, gld, shy, p)
        label = "B_shy_cash"
        print(fmt(label, r))
        results.append({'label': label, 'params': p, **r})

    # Best of A + SHY
    best_A = max([r for r in results if r['label'].startswith('A_')], key=lambda x: x['comp'])
    p = dict(best_A['params']); p['use_shy'] = True
    r = evaluate(close_df, sig, sectors, gld, shy, p)
    label = f"B_bestA+shy"
    print(fmt(label, r))
    results.append({'label': label, 'params': p, **r})

    # ── Sweep C: 52-Week High Proximity Filter ────────────────────────────────
    print("\n📊 Sweep C: 52-Week High Proximity Filter (novel quality filter)")
    for frac in [0.60, 0.65, 0.70, 0.75, 0.80]:
        p = dict(V9A_PARAMS); p['hi52_frac'] = frac
        r = evaluate(close_df, sig, sectors, gld, shy, p)
        label = f"C_hi52_{frac:.0%}"
        print(fmt(label, r))
        results.append({'label': label, 'params': p, **r})

    # ── Sweep D: Fine-grained Breadth + Stock Count ───────────────────────────
    print("\n📊 Sweep D: Fine-Grained Sector × Stock Count")
    for n_secs, sps in [(5,3), (6,2), (4,3), (5,2), (6,3)]:
        if (n_secs, sps) == (5, 2):  # skip baseline
            continue
        p = dict(V9A_PARAMS); p['n_bull_secs'] = n_secs; p['bull_sps'] = sps
        r = evaluate(close_df, sig, sectors, gld, shy, p)
        label = f"D_{n_secs}x{sps}"
        print(fmt(label, r))
        results.append({'label': label, 'params': p, **r})

    print("\n📊 Sweep D2: Fine-Grained Breadth Threshold")
    for bt in [0.40, 0.42, 0.43, 0.44, 0.46, 0.48, 0.50]:
        if bt == 0.45:
            continue  # skip baseline
        p = dict(V9A_PARAMS); p['breadth_thresh'] = bt
        r = evaluate(close_df, sig, sectors, gld, shy, p)
        label = f"D2_bread{bt:.2f}"
        print(fmt(label, r))
        results.append({'label': label, 'params': p, **r})

    # ── Sweep E: Combo of best improvements ──────────────────────────────────
    print("\n📊 Sweep E: Combos")
    best_A = max([r for r in results if r['label'].startswith('A_')], key=lambda x: x['comp'])
    best_C = max([r for r in results if r['label'].startswith('C_')], key=lambda x: x['comp'])
    best_D = max([r for r in results if r['label'].startswith('D_')], key=lambda x: x['comp'])
    best_D2 = max([r for r in results if r['label'].startswith('D2_')], key=lambda x: x['comp'])
    print(f"  Best A: {best_A['label']} Comp={best_A['comp']:.4f}  TO={best_A['to']:.0%}")
    print(f"  Best C: {best_C['label']} Comp={best_C['comp']:.4f}")
    print(f"  Best D: {best_D['label']} Comp={best_D['comp']:.4f}")
    print(f"  Best D2: {best_D2['label']} Comp={best_D2['comp']:.4f}")

    # A + SHY + D2 combo
    combo1 = dict(V9A_PARAMS)
    combo1['cont_bonus'] = best_A['params']['cont_bonus']
    combo1['use_shy'] = True
    combo1['breadth_thresh'] = best_D2['params']['breadth_thresh']
    r = evaluate(close_df, sig, sectors, gld, shy, combo1)
    print(fmt("E_A+SHY+D2", r))
    results.append({'label': 'E_A+SHY+D2', 'params': combo1, **r})

    # A + C (best two individual)
    combo2 = dict(V9A_PARAMS)
    combo2['cont_bonus'] = best_A['params']['cont_bonus']
    combo2['hi52_frac'] = best_C['params']['hi52_frac']
    r = evaluate(close_df, sig, sectors, gld, shy, combo2)
    print(fmt("E_A+C", r))
    results.append({'label': 'E_A+C', 'params': combo2, **r})

    # Best A + D (stocks) + D2 (breadth) + SHY
    combo3 = dict(V9A_PARAMS)
    combo3['cont_bonus'] = best_A['params']['cont_bonus']
    combo3['n_bull_secs'] = best_D['params']['n_bull_secs']
    combo3['bull_sps'] = best_D['params']['bull_sps']
    combo3['breadth_thresh'] = best_D2['params']['breadth_thresh']
    combo3['use_shy'] = True
    r = evaluate(close_df, sig, sectors, gld, shy, combo3)
    print(fmt("E_AllBest", r))
    results.append({'label': 'E_AllBest', 'params': combo3, **r})

    # Best A + D2 (keep v9a stock structure, tweak breadth + bonus)
    combo4 = dict(V9A_PARAMS)
    combo4['cont_bonus'] = best_A['params']['cont_bonus']
    combo4['breadth_thresh'] = best_D2['params']['breadth_thresh']
    r = evaluate(close_df, sig, sectors, gld, shy, combo4)
    print(fmt("E_A+D2", r))
    results.append({'label': 'E_A+D2', 'params': combo4, **r})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("🏆 TOP 5 Results (WF ≥ 0.70):")
    print("=" * 78)
    valid = sorted([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'], reverse=True)[:5]
    for r in valid:
        m = r['full']
        print(f"  {r['label']:32s}: CAGR {m['cagr']:.1%}  Sh {m['sharpe']:.2f}  "
              f"DD {m['max_dd']:.1%}  Cal {m['calmar']:.2f}  WF {r['wf']:.2f}  Comp {r['comp']:.4f}")

    champion = max([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'])
    cm = champion['full']
    wf = champion['wf']
    comp = champion['comp']

    print(f"\n🏆 Champion: {champion['label']}")
    print(f"  CAGR:       {cm['cagr']:.1%}  {'✅' if cm['cagr']>0.30 else ''}")
    print(f"  MaxDD:      {cm['max_dd']:.1%}")
    print(f"  Sharpe:     {cm['sharpe']:.2f}  {'✅' if cm['sharpe']>1.5 else ''}")
    print(f"  Calmar:     {cm['calmar']:.2f}")
    print(f"  IS Sharpe:  {champion['is_m']['sharpe']:.2f}")
    print(f"  OOS Sharpe: {champion['oos_m']['sharpe']:.2f}")
    print(f"  WF ratio:   {wf:.2f}  {'✅' if wf>=0.70 else '❌'}")
    print(f"  Turnover:   {champion['to']:.1%}")
    print(f"  Composite:  {comp:.4f}  {'✅ > 1.5!' if comp>1.5 else ''}")
    print(f"\n  vs v9a baseline: Δ Composite = {comp - base['comp']:+.4f}")

    if comp > 1.8 or cm['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】 Composite > 1.8 or Sharpe > 2.0!")
    elif comp > 1.55:
        print(f"\n✅ 超越 v9a: Composite {comp:.4f} > 1.55! 策略再升级!")
    elif comp > 1.51:
        print(f"\n⭐ 小幅提升: Composite {comp:.4f} > v9a {base['comp']:.4f}")

    # Save
    out = {
        'champion': champion['label'],
        'champion_metrics': {'cagr': float(cm['cagr']), 'sharpe': float(cm['sharpe']),
                             'max_dd': float(cm['max_dd']), 'calmar': float(cm['calmar']),
                             'wf': float(wf), 'composite': float(comp)},
        'baseline_v9a': float(base['comp']),
        'improvement': float(comp - base['comp']),
        'results': [{'label': r['label'], 'comp': float(r['comp']), 'wf': float(r['wf']),
                     'cagr': float(r['full']['cagr']), 'sharpe': float(r['full']['sharpe']),
                     'max_dd': float(r['full']['max_dd']), 'calmar': float(r['full']['calmar']),
                     'turnover': float(r['to'])} for r in results]
    }
    jf = Path(__file__).parent / "momentum_v9b_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return champion


if __name__ == '__main__':
    main()
