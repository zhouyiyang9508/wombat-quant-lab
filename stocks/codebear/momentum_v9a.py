#!/usr/bin/env python3
"""
动量轮动 v9a — Fine-Tuning v8d to Break the 1.5 Composite Barrier
代码熊 🐻

v8d 已达到 Composite 1.460, Sharpe 1.58, WF 0.90 — 非常接近1.5目标!
(v8d: Breadth+SPY双确认 + GLD自然竞争 + DD响应)

v9a 尝试3个针对性改进:

[A] 扩大持股数: 4×4=16支 vs 4×3=12支(当前)
    - 更多分散化降低单股风险 → 可能提升Sharpe
    - 但每股权重更小，可能稀释收益

[B] 动量权重调优: 探索不同的1m/3m/6m/12m权重组合
    - 当前: 0.20/0.40/0.30/0.10
    - 备选: 偏重3m/6m组合，或偏重6m长期

[C] GLD竞争阈值微调: 降低GLD入场门槛提高频率
    - GLD_AVG_THRESH: 0.60/0.70/0.80(当前)/0.90
    - GLD_COMPETE_FRAC: 0.15/0.20(当前)/0.25

[D] 行业宽度阈值: 0.35/0.40(当前)/0.45
    - 更宽松 = 更多时间在牛市模式(高持仓)
    - 更严格 = 更早进入防御模式

目标: Composite > 1.5, Sharpe > 1.5, WF > 0.70
"""

import json, warnings, itertools
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
    r2  = close_df / close_df.shift(44)  - 1   # 2-month (new)
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1
    log_r = np.log(close_df / close_df.shift(1))
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r2=r2, r3=r3, r6=r6, r12=r12, vol30=vol30,
                spy=spy, s200=s200, sma50=sma50, close=close_df)


def compute_breadth(sig, date):
    """Fraction of S&P500 stocks with price > SMA50"""
    close = sig['close'].loc[:date].dropna(how='all')
    sma50 = sig['sma50'].loc[:date].dropna(how='all')
    if len(close) < 50:
        return 1.0
    last_close = close.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    mask = last_close > last_sma50
    valid = mask.dropna()
    return float(valid.sum() / len(valid)) if len(valid) > 0 else 1.0


def get_regime(sig, date, breadth_thresh):
    """Dual confirmation: SPY<SMA200 AND breadth<thresh"""
    s200 = sig['s200']
    if s200 is None:
        return 'bull'
    spy_now = sig['spy'].loc[:date].dropna()
    s200_now = s200.loc[:date].dropna()
    if len(spy_now) == 0 or len(s200_now) == 0:
        return 'bull'
    spy_bear = spy_now.iloc[-1] < s200_now.iloc[-1]
    breadth_val = compute_breadth(sig, date)
    breadth_bear = breadth_val < breadth_thresh
    return 'bear' if (spy_bear and breadth_bear) else 'bull'


def gld_compete(sig, date, gld_prices, gld_thresh, gld_frac):
    """Check if GLD earned a natural slot via momentum competition"""
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1:
        return 0.0
    d = idx[-1]

    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10:
        return 0.0
    avg_stock_r6 = stock_r6.mean()

    gld_now = gld_prices.loc[:d].dropna()
    if len(gld_now) < 130:
        return 0.0
    gld_r6 = gld_now.iloc[-1] / gld_now.iloc[-127] - 1

    return gld_frac if gld_r6 >= avg_stock_r6 * gld_thresh else 0.0


def select(sig, sectors, date, prev_hold, gld_prices,
           mom_weights, n_bull_sectors, bull_sps, bear_sps,
           breadth_thresh, gld_thresh, gld_frac_param):
    """Stock selection with configurable parameters"""
    close = sig['close']
    r6 = sig['r6']
    idx = close.index[close.index <= date]
    if len(idx) == 0:
        return {}
    d = idx[-1]

    # Momentum blend
    w1, w2, w3, w6, w12 = mom_weights
    mom = (sig['r1'].loc[d] * w1 + sig['r2'].loc[d] * w2 +
           sig['r3'].loc[d] * w3 + sig['r6'].loc[d] * w6 +
           sig['r12'].loc[d] * w12)

    df = pd.DataFrame({
        'mom': mom,
        'r6': r6.loc[d],
        'vol': sig['vol30'].loc[d],
        'price': close.loc[d],
        'sma50': sig['sma50'].loc[d],
    }).dropna(subset=['mom', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += 0.03

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)
    gld_a = gld_compete(sig, date, gld_prices, gld_thresh, gld_frac_param)
    reg = get_regime(sig, date, breadth_thresh)

    if reg == 'bull':
        n_secs = n_bull_sectors - (1 if gld_a > 0 else 0)
        sps, cash = bull_sps, 0.0
    else:
        n_secs = 3 - (1 if gld_a > 0 else 0)
        sps, cash = bear_sps, 0.20

    n_secs = max(n_secs, 1)
    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps])

    stock_frac = max(1.0 - cash - gld_a, 0.0)
    if not selected:
        return {'GLD': gld_a} if gld_a > 0 else {}

    iv = {t: 1.0 / max(df.loc[t, 'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn = min(df.loc[t, 'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw = {t: df.loc[t, 'mom'] + sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}

    weights = {t: (0.70*iv_w[t] + 0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0:
        weights['GLD'] = gld_a
    return weights


def add_gld(weights, frac, dd_params):
    if not weights:
        return weights
    dd_frac = 0.0
    # frac is already the max DD level to add
    if frac <= 0:
        return weights
    tot = sum(weights.values())
    new = {t: w/tot*(1-frac) for t, w in weights.items()}
    new['GLD'] = new.get('GLD', 0) + frac
    return new


def run_backtest(close_df, sig, sectors, gld, params,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    mom_weights = params['mom_weights']
    n_bull_secs = params['n_bull_secs']
    bull_sps    = params['bull_sps']
    bear_sps    = params['bear_sps']
    breadth_t   = params['breadth_thresh']
    gld_thresh  = params['gld_thresh']
    gld_frac    = params['gld_frac']
    dd_params   = {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}

    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0

        w = select(sig, sectors, dt, prev_h, gld,
                   mom_weights, n_bull_secs, bull_sps, bear_sps,
                   breadth_t, gld_thresh, gld_frac)

        # DD overlay
        gld_a = max((dd_params[th] for th in sorted(dd_params) if dd < th), default=0)
        if gld_a > 0:
            tot = sum(w.values())
            if tot > 0:
                w2 = {t: wt/tot*(1-gld_a) for t, wt in w.items()}
                w2['GLD'] = w2.get('GLD', 0) + gld_a
                w = w2

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k != 'GLD'}

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
        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)

    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq, float(np.mean(tos)) if tos else 0.0


def metrics(eq, name=''):
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
    return dict(cagr=cagr, max_dd=dd, sharpe=sh, calmar=cal)


def evaluate(close_df, sig, sectors, gld, params):
    eq_f, to = run_backtest(close_df, sig, sectors, gld, params)
    eq_i, _  = run_backtest(close_df, sig, sectors, gld, params, '2015-01-01', '2020-12-31')
    eq_o, _  = run_backtest(close_df, sig, sectors, gld, params, '2021-01-01', '2025-12-31')
    m  = metrics(eq_f)
    mi = metrics(eq_i)
    mo = metrics(eq_o)
    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
    return dict(full=m, is_m=mi, oos_m=mo, wf=wf, comp=comp, to=to)


def main():
    print("=" * 75)
    print("🐻 动量轮动 v9a — Fine-Tuning v8d → Composite 1.5 突破")
    print("=" * 75)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld      = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    sig      = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers")

    # ── Baseline: v8d params ──────────────────────────────────────────────────
    v8d_params = dict(
        mom_weights=(0.20, 0.00, 0.40, 0.30, 0.10),  # 1m,2m,3m,6m,12m
        n_bull_secs=4, bull_sps=3, bear_sps=2,
        breadth_thresh=0.40, gld_thresh=0.80, gld_frac=0.20,
    )

    print("\n🔄 Baseline (v8d)...")
    base = evaluate(close_df, sig, sectors, gld, v8d_params)
    bm = base['full']
    print(f"  CAGR {bm['cagr']:.1%}  Sharpe {bm['sharpe']:.2f}  MaxDD {bm['max_dd']:.1%}  "
          f"Calmar {bm['calmar']:.2f}  WF {base['wf']:.2f}  Comp {base['comp']:.4f}")

    results = [{'label': 'v8d_baseline', 'params': v8d_params, **base}]

    # ── Sweep A: Stock Count ──────────────────────────────────────────────────
    print("\n📊 Sweep A: Stock Count (n_bull_secs × bull_sps)")
    for n_secs, sps in [(4,4), (5,3), (5,2), (3,4), (4,3)]:
        if (n_secs, sps) == (4, 3):  # skip baseline
            continue
        p = dict(v8d_params); p['n_bull_secs'] = n_secs; p['bull_sps'] = sps
        r = evaluate(close_df, sig, sectors, gld, p)
        m = r['full']
        label = f"A_{n_secs}x{sps}"
        print(f"  {label:10s}: CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
              f"MaxDD {m['max_dd']:.1%}  Calmar {m['calmar']:.2f}  "
              f"WF {r['wf']:.2f}  Comp {r['comp']:.4f} "
              f"{'✅' if r['comp'] > 1.5 else ('⭐' if r['comp'] > 1.46 else '')}")
        results.append({'label': label, 'params': p, **r})

    # ── Sweep B: Momentum Weights ─────────────────────────────────────────────
    print("\n📊 Sweep B: Momentum Weights (1m, 2m, 3m, 6m, 12m)")
    mom_combos = [
        (0.10, 0.00, 0.30, 0.45, 0.15, "6m-heavy"),
        (0.10, 0.10, 0.30, 0.40, 0.10, "with-2m"),
        (0.15, 0.05, 0.40, 0.30, 0.10, "3m-heavy"),
        (0.20, 0.00, 0.50, 0.20, 0.10, "3m-dominant"),
        (0.10, 0.00, 0.45, 0.35, 0.10, "3m+6m"),
        (0.25, 0.00, 0.35, 0.30, 0.10, "1m-boost"),
    ]
    for w1, w2, w3, w6, w12, label in mom_combos:
        p = dict(v8d_params); p['mom_weights'] = (w1, w2, w3, w6, w12)
        r = evaluate(close_df, sig, sectors, gld, p)
        m = r['full']
        print(f"  B_{label:15s}: CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
              f"MaxDD {m['max_dd']:.1%}  Calmar {m['calmar']:.2f}  "
              f"WF {r['wf']:.2f}  Comp {r['comp']:.4f} "
              f"{'✅' if r['comp'] > 1.5 else ('⭐' if r['comp'] > 1.46 else '')}")
        results.append({'label': f'B_{label}', 'params': p, **r})

    # ── Sweep C: GLD Threshold ────────────────────────────────────────────────
    print("\n📊 Sweep C: GLD Competition Parameters")
    for gld_thresh, gld_frac in [
        (0.60, 0.20), (0.70, 0.20), (0.80, 0.15), (0.80, 0.25),
        (0.90, 0.20), (0.70, 0.25), (0.60, 0.25), (1.00, 0.20),
    ]:
        p = dict(v8d_params); p['gld_thresh'] = gld_thresh; p['gld_frac'] = gld_frac
        r = evaluate(close_df, sig, sectors, gld, p)
        m = r['full']
        label = f"C_gld{gld_thresh:.0%}_frac{gld_frac:.0%}"
        print(f"  {label:25s}: CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
              f"MaxDD {m['max_dd']:.1%}  WF {r['wf']:.2f}  Comp {r['comp']:.4f} "
              f"{'✅' if r['comp'] > 1.5 else ('⭐' if r['comp'] > 1.46 else '')}")
        results.append({'label': label, 'params': p, **r})

    # ── Sweep D: Breadth Threshold ────────────────────────────────────────────
    print("\n📊 Sweep D: Breadth Threshold")
    for bt in [0.30, 0.35, 0.45, 0.50, 0.55]:
        p = dict(v8d_params); p['breadth_thresh'] = bt
        r = evaluate(close_df, sig, sectors, gld, p)
        m = r['full']
        label = f"D_bread{bt:.0%}"
        print(f"  {label:18s}: CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
              f"MaxDD {m['max_dd']:.1%}  Calmar {m['calmar']:.2f}  "
              f"WF {r['wf']:.2f}  Comp {r['comp']:.4f} "
              f"{'✅' if r['comp'] > 1.5 else ('⭐' if r['comp'] > 1.46 else '')}")
        results.append({'label': label, 'params': p, **r})

    # ── Combo: Best-of-each combined ─────────────────────────────────────────
    print("\n📊 Sweep E: Best-of-each Combinations")
    # Find best from each sweep
    best_A = max([r for r in results if r['label'].startswith('A_')], key=lambda x: x['comp'])
    best_B = max([r for r in results if r['label'].startswith('B_')], key=lambda x: x['comp'])
    best_C = max([r for r in results if r['label'].startswith('C_')], key=lambda x: x['comp'])
    best_D = max([r for r in results if r['label'].startswith('D_')], key=lambda x: x['comp'])
    print(f"  Best A: {best_A['label']} Comp={best_A['comp']:.4f}")
    print(f"  Best B: {best_B['label']} Comp={best_B['comp']:.4f}")
    print(f"  Best C: {best_C['label']} Comp={best_C['comp']:.4f}")
    print(f"  Best D: {best_D['label']} Comp={best_D['comp']:.4f}")

    # Combine B+C
    combo1 = dict(v8d_params)
    combo1['mom_weights'] = best_B['params']['mom_weights']
    combo1['gld_thresh']  = best_C['params']['gld_thresh']
    combo1['gld_frac']    = best_C['params']['gld_frac']
    r = evaluate(close_df, sig, sectors, gld, combo1)
    m = r['full']
    print(f"  E_BestB+BestC: CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
          f"MaxDD {m['max_dd']:.1%}  WF {r['wf']:.2f}  Comp {r['comp']:.4f} "
          f"{'✅' if r['comp'] > 1.5 else ('⭐' if r['comp'] > 1.46 else '')}")
    results.append({'label': 'E_BestB+BestC', 'params': combo1, **r})

    # Combine A+B+C+D best
    combo2 = {
        'mom_weights': best_B['params']['mom_weights'],
        'n_bull_secs': best_A['params']['n_bull_secs'],
        'bull_sps':    best_A['params']['bull_sps'],
        'bear_sps':    2,
        'breadth_thresh': best_D['params']['breadth_thresh'],
        'gld_thresh':  best_C['params']['gld_thresh'],
        'gld_frac':    best_C['params']['gld_frac'],
    }
    r = evaluate(close_df, sig, sectors, gld, combo2)
    m = r['full']
    print(f"  E_AllBest    : CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
          f"MaxDD {m['max_dd']:.1%}  WF {r['wf']:.2f}  Comp {r['comp']:.4f} "
          f"{'✅' if r['comp'] > 1.5 else ('⭐' if r['comp'] > 1.46 else '')}")
    results.append({'label': 'E_AllBest', 'params': combo2, **r})

    # Combine A+B+D (no GLD change)
    combo3 = {
        'mom_weights': best_B['params']['mom_weights'],
        'n_bull_secs': best_A['params']['n_bull_secs'],
        'bull_sps':    best_A['params']['bull_sps'],
        'bear_sps':    2,
        'breadth_thresh': best_D['params']['breadth_thresh'],
        'gld_thresh':  0.80,
        'gld_frac':    0.20,
    }
    r = evaluate(close_df, sig, sectors, gld, combo3)
    m = r['full']
    print(f"  E_A+B+D     : CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
          f"MaxDD {m['max_dd']:.1%}  WF {r['wf']:.2f}  Comp {r['comp']:.4f} "
          f"{'✅' if r['comp'] > 1.5 else ('⭐' if r['comp'] > 1.46 else '')}")
    results.append({'label': 'E_A+B+D', 'params': combo3, **r})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("🏆 TOP 5 Results:")
    print("=" * 75)
    top5 = sorted([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'], reverse=True)[:5]
    for r in top5:
        m = r['full']
        print(f"  {r['label']:28s}: CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
              f"MaxDD {m['max_dd']:.1%}  Calmar {m['calmar']:.2f}  "
              f"WF {r['wf']:.2f}  Comp {r['comp']:.4f}")

    champion = max([r for r in results if r['wf'] >= 0.70], key=lambda x: x['comp'])
    cm = champion['full']
    wf = champion['wf']
    comp = champion['comp']

    print(f"\n🏆 Champion: {champion['label']}")
    print(f"  CAGR:       {cm['cagr']:.1%}")
    print(f"  Sharpe:     {cm['sharpe']:.2f} {'✅' if cm['sharpe'] > 1.5 else ''}")
    print(f"  MaxDD:      {cm['max_dd']:.1%}")
    print(f"  Calmar:     {cm['calmar']:.2f}")
    print(f"  WF ratio:   {wf:.2f} {'✅' if wf >= 0.70 else '❌'}")
    print(f"  Composite:  {comp:.4f} {'✅ 突破1.5!' if comp > 1.5 else '⚠️ 接近'}")

    if comp > 1.8 or cm['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】 Composite > 1.8 or Sharpe > 2.0!")
    elif comp > 1.5:
        print(f"\n✅ 突破目标! Composite {comp:.4f} > 1.5 !! 策略升级成功!")
    elif comp > 1.46:
        print(f"\n⭐ 小幅改进: v9a Champion {comp:.4f} vs v8d {base['comp']:.4f}")

    # Save
    out = {
        'champion': champion['label'],
        'champion_params': {k: v for k, v in champion['params'].items() if k != 'mom_weights'} |
                           {'mom_weights': list(champion['params']['mom_weights'])},
        'champion_metrics': {
            'cagr': float(cm['cagr']), 'sharpe': float(cm['sharpe']),
            'max_dd': float(cm['max_dd']), 'calmar': float(cm['calmar']),
            'wf': float(wf), 'composite': float(comp),
        },
        'baseline_v8d': float(base['comp']),
        'all_results': [
            {
                'label': r['label'],
                'comp': float(r['comp']), 'wf': float(r['wf']),
                'cagr': float(r['full']['cagr']), 'sharpe': float(r['full']['sharpe']),
                'max_dd': float(r['full']['max_dd']), 'calmar': float(r['full']['calmar']),
                'valid': r['wf'] >= 0.70,
            }
            for r in results
        ]
    }
    jf = Path(__file__).parent / "momentum_v9a_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return champion


if __name__ == '__main__':
    main()
