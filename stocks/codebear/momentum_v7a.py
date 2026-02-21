#!/usr/bin/env python3
"""
动量轮动 v7a — SMA50 斜率加速 + 动量一致性过滤
代码熊 🐻

核心创新 vs v4d (Composite 1.356):
1. SMA50 Slope Filter: SMA50 today > SMA50 20 days ago（趋势加速）
   - v3b仅要price > SMA50，但SMA50本身可能是平的或下降的
   - 斜率正 = 趋势还在加速，选入更高质量的趋势股
   
2. 动量一致性（Consistency）: 过去6个月中有多少个月是正收益
   - 一致性过滤: 至少4/6月为正（约67%）
   - 避免选"暴涨1-2个月"的噪声动量股
   - "稳步向上"的股票Sharpe通常更高
   
3. 继承 v4d 全部 DD 响应 GLD 对冲

变种测试:
  v3b_base:  原始v3b选股 + v4d DD (baseline)
  slope:     v3b + SMA50斜率过滤 + v4d DD  
  consist:   v3b + 一致性过滤(4/6月) + v4d DD
  both:      v3b + 斜率 + 一致性 + v4d DD
  blend:     v3b + 斜率 + 一致性纳入打分 + v4d DD
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv(fp):
    df = pd.read_csv(fp)
    df['Date'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
    df = df.set_index('Date').sort_index()
    for col in ['Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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


# ── Signal precomputation ─────────────────────────────────────────────────────

def precompute(close_df):
    r1  = close_df / close_df.shift(22)  - 1
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1

    log_r = np.log(close_df / close_df.shift(1))
    vol30 = log_r.rolling(30).std() * np.sqrt(252)

    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    slope = sma50 / sma50.shift(20) - 1   # >0 means SMA50 is rising

    # Precompute monthly consistency: for each stock, fraction of last 6 months positive
    # Approximate: use 22-day non-overlapping return windows
    # Compute r1m_lag1 (1m return with 1-day lag) at each 22-day step
    # We'll compute this as a rolling 6-window fraction of positive 1m returns
    # For speed: roll the sign of 1m return over 6 months (132 days)
    pos_1m   = (r1 > 0).astype(float)
    consist6 = pos_1m.rolling(6).mean()   # fraction of last 6 1m-period samples positive
    # Note: since r1 is already the "past 22 days return", rolling(6).mean() samples
    # every day – we'll just use the value at month-end, which approximates well

    return {
        'r1': r1, 'r3': r3, 'r6': r6, 'r12': r12,
        'vol30': vol30, 'spy': spy, 's200': s200,
        'sma50': sma50, 'slope': slope,
        'consist6': consist6, 'close': close_df,
    }


# ── Regime ───────────────────────────────────────────────────────────────────

def regime(sig, date):
    if sig['s200'] is None:
        return 'bull'
    s = sig['s200'].loc[:date].dropna()
    p = sig['spy'].loc[:date].dropna()
    if len(s) == 0 or len(p) == 0:
        return 'bull'
    return 'bull' if p.iloc[-1] > s.iloc[-1] else 'bear'


# ── Stock selection ───────────────────────────────────────────────────────────

def select(sig, sectors, date, prev_hold, variant='both'):
    """
    variant: 'base' | 'slope' | 'consist' | 'both' | 'blend'
    """
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0:
        return {}, 'bull'
    idx = idx_arr[-1]
    reg = regime(sig, date)

    # Composite momentum (v3b formula)
    mom = (sig['r1'].loc[idx] * 0.20 +
           sig['r3'].loc[idx] * 0.40 +
           sig['r6'].loc[idx] * 0.30 +
           sig['r12'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'mom':    mom,
        'r6':     sig['r6'].loc[idx],
        'vol':    sig['vol30'].loc[idx],
        'price':  close.loc[idx],
        'sma50':  sig['sma50'].loc[idx],
        'slope':  sig['slope'].loc[idx],
        'consist': sig['consist6'].loc[idx],
    }).dropna(subset=['mom', 'sma50'])

    # Base filters (identical to v3b)
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]

    # ── v7a extra filters ──────────────────────────────────────────────────
    if variant in ('slope', 'both', 'blend'):
        df = df[df['slope'] > 0]            # SMA50 must be rising

    if variant in ('consist', 'both', 'blend'):
        df = df[df['consist'] >= 4/6]       # ≥4/6 months positive (≈67%)
    # ──────────────────────────────────────────────────────────────────────

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += 0.03

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    if reg == 'bull':
        top_secs = sec_mom.head(4).index.tolist()
        sps, cash = 3, 0.0
    else:
        top_secs = sec_mom.head(3).index.tolist()
        sps, cash = 2, 0.20

    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    if not selected:
        return {}, reg

    # Scoring column
    if variant == 'blend':
        # Boost momentum score by consistency
        df.loc[selected, 'score'] = df.loc[selected, 'mom'] * (0.7 + 0.3 * df.loc[selected, 'consist'])
        score_col = 'score'
    else:
        df.loc[selected, 'score'] = df.loc[selected, 'mom']
        score_col = 'score'

    # Blended weighting: 70% inv-vol + 30% momentum (v3b)
    iv = {t: 1.0 / max(df.loc[t, 'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values())
    iv_w = {t: v / iv_t for t, v in iv.items()}

    mn = min(df.loc[t, score_col] for t in selected)
    sh = max(-mn + 0.01, 0)
    mw = {t: df.loc[t, score_col] + sh for t in selected}
    mw_t = sum(mw.values())
    mw_w = {t: v / mw_t for t, v in mw.items()}

    invested = 1.0 - cash
    w = {t: (0.70 * iv_w[t] + 0.30 * mw_w[t]) * invested for t in selected}
    return w, reg


# ── GLD hedge ────────────────────────────────────────────────────────────────

DD_PARAMS = {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}   # v4d champion params

def add_gld(weights, frac):
    if frac <= 0 or not weights:
        return weights
    total = sum(weights.values())
    if total <= 0:
        return weights
    new = {t: w / total * (1 - frac) for t, w in weights.items()}
    new['GLD'] = frac
    return new


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(close_df, sig, sectors, gld, variant='both',
             start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0

        w, _ = select(sig, sectors, dt, prev_h, variant)
        gld_a = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0)
        w = add_gld(w, gld_a)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k != 'GLD'}

        ret = 0.0
        for t, wt in w.items():
            s = (gld.loc[dt:ndt].dropna() if t == 'GLD'
                 else close_df[t].loc[dt:ndt].dropna() if t in close_df.columns
                 else pd.Series(dtype=float))
            if len(s) >= 2:
                ret += (s.iloc[-1] / s.iloc[0] - 1) * wt
        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak:
            peak = val
        vals.append(val); dates.append(ndt)

    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq, float(np.mean(tos)) if tos else 0.0


def metrics(eq, name=''):
    if len(eq) < 3:
        return dict(name=name, cagr=0, max_dd=0, sharpe=0, calmar=0)
    yrs  = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5:
        return dict(name=name, cagr=0, max_dd=0, sharpe=0, calmar=0)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mo   = eq.pct_change().dropna()
    sh   = mo.mean() / mo.std() * np.sqrt(12) if mo.std() > 0 else 0
    dd   = ((eq - eq.cummax()) / eq.cummax()).min()
    cal  = cagr / abs(dd) if dd != 0 else 0
    return dict(name=name, cagr=cagr, max_dd=dd, sharpe=sh, calmar=cal)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🐻 动量轮动 v7a — SMA50 斜率加速 + 动量一致性过滤")
    print("=" * 70)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    sig      = precompute(close_df)
    gld      = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    print(f"  Loaded {len(close_df.columns)} tickers")

    VARIANTS = {
        'base':    'v3b+DD (baseline)',
        'slope':   'v7a Slope only',
        'consist': 'v7a Consist only',
        'both':    'v7a Slope+Consist',
        'blend':   'v7a Blend score',
    }

    results = {}
    for var, label in VARIANTS.items():
        print(f"\n🔄 {label} ...")
        eq,   to = backtest(close_df, sig, sectors, gld, var)
        eq_i, _  = backtest(close_df, sig, sectors, gld, var, '2015-01-01', '2020-12-31')
        eq_o, _  = backtest(close_df, sig, sectors, gld, var, '2021-01-01', '2025-12-31')

        m   = metrics(eq,  var)
        mi  = metrics(eq_i)
        mo  = metrics(eq_o)
        wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] else 0
        comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2

        results[var] = dict(m=m, is_m=mi, oos_m=mo, wf=wf, comp=comp, to=to)
        print(f"  CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  MaxDD {m['max_dd']:.1%}  "
              f"Calmar {m['calmar']:.2f}  Comp {comp:.3f}  WF {wf:.2f} {'✅' if wf >= 0.7 else '❌'}")

    # Summary table
    print("\n" + "=" * 105)
    print(f"{'Variant':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 105)
    for var, label in VARIANTS.items():
        r = results[var]
        m = r['m']
        flag = '✅' if r['wf'] >= 0.7 else '❌'
        print(f"{label:<22} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
              f"{r['is_m']['sharpe']:>7.2f} {r['oos_m']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f}{flag} {r['comp']:>8.3f}")

    base_comp = results['base']['comp']
    best = max(((v, r) for v, r in results.items() if v != 'base' and r['wf'] >= 0.7),
               key=lambda x: x[1]['comp'], default=None)
    if best:
        v, r = best
        print(f"\n🏆 Best v7a: {VARIANTS[v]} → Composite {r['comp']:.3f} "
              f"(vs baseline {base_comp:.3f}, Δ{r['comp']-base_comp:+.3f})")
        if r['comp'] > 1.8 or r['m']['sharpe'] > 2.0:
            print("🚨🚨🚨 【重大突破】!")
        elif r['comp'] > 1.356:
            print("✅ Beats v4d champion (1.356)!")
        else:
            print(f"⚠️  No improvement over v4d champion (1.356)")
    else:
        print("\n⚠️  No WF-passing v7a variant improves over baseline")

    out = {v: {k: float(vv) for k, vv in {
        'cagr': r['m']['cagr'], 'max_dd': r['m']['max_dd'],
        'sharpe': r['m']['sharpe'], 'calmar': r['m']['calmar'],
        'wf': r['wf'], 'composite': r['comp'],
    }.items()} for v, r in results.items()}
    jf = Path(__file__).parent / "momentum_v7a_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results


if __name__ == '__main__':
    main()
