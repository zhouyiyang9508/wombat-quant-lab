#!/usr/bin/env python3
"""
动量轮动 v8b — 行业内 Sharpe 排名（Quality Momentum）
代码熊 🐻

核心创新 vs v3b:
  v3b: 行业内按绝对动量(composite momentum)排名，选 top-3
  v8b: 行业内按 mini-Sharpe = 6m动量/30d波动率排名，选 top-3
  
学术背景:
  "Quality Momentum" (Novy-Marx 2013): 
  选高动量中低波动率的股票（risk-adjusted momentum）
  比纯动量有更高的信息比率和Sharpe
  
  "Combination of Value and Momentum" 研究表明：
  Momentum/Volatility 比率选股 > 纯Momentum选股
  
  实质：我们不只想要涨得多的，还想要"涨得稳"的

变种:
  base:     v3b + DD GLD (基线)
  sharpe6:  行业内按 6m_return/vol30 排名选股
  sharpe3:  行业内按 3m_return/vol30 排名选股
  blend:    行业内按 0.7*sharpe6 + 0.3*composite_mom 排名
  strict:   行业内按sharpe6，且要求sharpe6 > 0 (正向)
  
所有变种都保留:
  - v3b 的基础过滤 (SMA50, 6m>0, vol<65%)
  - 行业轮动逻辑 (sector rotation)
  - v4d DD-响应 GLD 对冲
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

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
    r1   = close_df / close_df.shift(22)  - 1
    r3   = close_df / close_df.shift(63)  - 1
    r6   = close_df / close_df.shift(126) - 1
    r12  = close_df / close_df.shift(252) - 1
    log_r = np.log(close_df / close_df.shift(1))
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, vol30=vol30,
                spy=spy, s200=s200, sma50=sma50, close=close_df)


def bull_bear(sig, date):
    if sig['s200'] is None: return 'bull'
    s = sig['s200'].loc[:date].dropna()
    p = sig['spy'].loc[:date].dropna()
    if len(s) == 0 or len(p) == 0: return 'bull'
    return 'bull' if p.iloc[-1] > s.iloc[-1] else 'bear'


def select_v8b(sig, sectors, date, prev_hold, variant='sharpe6'):
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0: return {}, 'bull'
    idx = idx_arr[-1]
    reg = bull_bear(sig, date)

    # Compute all signals
    mom_base = (sig['r1'].loc[idx] * 0.20 + sig['r3'].loc[idx] * 0.40 +
                sig['r6'].loc[idx] * 0.30 + sig['r12'].loc[idx] * 0.10)
    r3_val  = sig['r3'].loc[idx]
    r6_val  = sig['r6'].loc[idx]
    vol_val = sig['vol30'].loc[idx]

    df = pd.DataFrame({
        'mom': mom_base, 'r3': r3_val, 'r6': r6_val,
        'vol': vol_val, 'price': close.loc[idx], 'sma50': sig['sma50'].loc[idx],
    }).dropna(subset=['mom', 'sma50'])

    # Base filters (identical to v3b)
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]

    # Compute Sharpe scores (annualized momentum / volatility)
    eps = 0.001
    df['sharpe6'] = df['r6'] / (df['vol'] + eps)      # 6m mom / 30d vol
    df['sharpe3'] = df['r3'] / (df['vol'] + eps)      # 3m mom / 30d vol

    # Score for ranking
    if variant == 'sharpe6':
        df['score'] = df['sharpe6']
    elif variant == 'sharpe3':
        df['score'] = df['sharpe3']
    elif variant == 'blend':
        # Normalize both to [0,1] within universe before blending
        def zscore(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + eps) if mx > mn else s * 0
        mom_norm  = zscore(df['mom'])
        sh6_norm  = zscore(df['sharpe6'])
        df['score'] = 0.5 * mom_norm + 0.5 * sh6_norm
    elif variant == 'strict':
        df['score'] = df['sharpe6']
        df = df[df['sharpe6'] > 0]   # Must have positive risk-adjusted return
    elif variant == 'base':
        df['score'] = df['mom']      # Pure v3b
    else:
        df['score'] = df['mom']

    if len(df) == 0: return {}, reg

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'score'] += 0.03  # continuity bonus (in normalized space, small)

    # Sector ranking: use average score for sector (not just momentum)
    sec_score = df.groupby('sector')['score'].mean().sort_values(ascending=False)

    if reg == 'bull':
        top_secs = sec_score.head(4).index.tolist(); sps, cash = 3, 0.0
    else:
        top_secs = sec_score.head(3).index.tolist(); sps, cash = 2, 0.20

    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('score', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    if not selected: return {}, reg

    # Weighting: 70% inv-vol + 30% momentum (v3b standard)
    iv = {t: 1.0 / max(df.loc[t, 'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v / iv_t for t, v in iv.items()}
    mn = min(df.loc[t, 'score'] for t in selected); sh = max(-mn + 0.01, 0)
    mw = {t: df.loc[t, 'score'] + sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v / mw_t for t, v in mw.items()}

    invested = 1.0 - cash
    return {t: (0.70 * iv_w[t] + 0.30 * mw_w[t]) * invested for t in selected}, reg


def add_gld(weights, frac):
    if frac <= 0 or not weights: return weights
    tot = sum(weights.values())
    if tot <= 0: return weights
    new = {t: w / tot * (1 - frac) for t, w in weights.items()}
    new['GLD'] = frac
    return new


def backtest(close_df, sig, sectors, gld, variant,
             start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0

        w, _ = select_v8b(sig, sectors, dt, prev_h, variant)
        gld_a = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0)
        w = add_gld(w, gld_a)

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
                ret += (s.iloc[-1] / s.iloc[0] - 1) * wt
        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)

    return pd.Series(vals, index=pd.DatetimeIndex(dates)), float(np.mean(tos)) if tos else 0.0


def mets(eq, name=''):
    if len(eq) < 3: return dict(name=name, cagr=0, max_dd=0, sharpe=0, calmar=0)
    yrs  = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return dict(name=name, cagr=0, max_dd=0, sharpe=0, calmar=0)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mo   = eq.pct_change().dropna()
    sh   = mo.mean() / mo.std() * np.sqrt(12) if mo.std() > 0 else 0
    dd   = ((eq - eq.cummax()) / eq.cummax()).min()
    cal  = cagr / abs(dd) if dd != 0 else 0
    return dict(name=name, cagr=cagr, max_dd=dd, sharpe=sh, calmar=cal)


def main():
    print("=" * 70)
    print("🐻 动量轮动 v8b — 行业内 Sharpe 排名（Quality Momentum）")
    print("=" * 70)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld      = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    sig      = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers")

    VARIANTS = {
        'base':    'v3b+DD (baseline)',
        'sharpe6': 'Sharpe6 (6m/vol)',
        'sharpe3': 'Sharpe3 (3m/vol)',
        'blend':   'Blend (50% mom + 50% sh6)',
        'strict':  'Strict Sharpe6>0',
    }

    results = {}
    for var, label in VARIANTS.items():
        print(f"\n🔄 {label} ...")
        eq,   _ = backtest(close_df, sig, sectors, gld, var)
        eq_i, _ = backtest(close_df, sig, sectors, gld, var, '2015-01-01', '2020-12-31')
        eq_o, _ = backtest(close_df, sig, sectors, gld, var, '2021-01-01', '2025-12-31')

        m  = mets(eq,  var)
        mi = mets(eq_i)
        mo = mets(eq_o)
        wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] else 0
        comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2

        results[var] = dict(m=m, is_m=mi, oos_m=mo, wf=wf, comp=comp)
        print(f"  CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  MaxDD {m['max_dd']:.1%}  "
              f"Calmar {m['calmar']:.2f}  Comp {comp:.3f}  WF {wf:.2f} {'✅' if wf >= 0.7 else '❌'}")

    print("\n" + "=" * 105)
    print(f"{'Variant':<28} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 105)
    for var, label in VARIANTS.items():
        r = results[var]; m = r['m']
        flag = '✅' if r['wf'] >= 0.7 else '❌'
        print(f"{label:<28} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
              f"{r['is_m']['sharpe']:>7.2f} {r['oos_m']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f}{flag} {r['comp']:>8.3f}")

    base_comp = results['base']['comp']
    bests = [(v, r) for v, r in results.items() if v != 'base' and r['wf'] >= 0.7]
    if bests:
        bv, br = max(bests, key=lambda x: x[1]['comp'])
        print(f"\n🏆 Best v8b: {VARIANTS[bv]} → Composite {br['comp']:.3f} "
              f"(vs baseline {base_comp:.3f}, Δ{br['comp']-base_comp:+.3f})")
        if br['comp'] > 1.8 or br['m']['sharpe'] > 2.0:
            print("🚨🚨🚨 【重大突破】!")
        elif br['comp'] > 1.356:
            print("✅ Beats v4d champion (1.356)!")
        else:
            print(f"⚠️  Below v4d champion (1.356)")

    out = {v: {'cagr': float(r['m']['cagr']), 'max_dd': float(r['m']['max_dd']),
               'sharpe': float(r['m']['sharpe']), 'calmar': float(r['m']['calmar']),
               'wf': float(r['wf']), 'composite': float(r['comp'])}
           for v, r in results.items()}
    jf = Path(__file__).parent / "momentum_v8b_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results


if __name__ == '__main__':
    main()
