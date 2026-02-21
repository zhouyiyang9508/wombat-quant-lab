#!/usr/bin/env python3
"""
动量轮动 v7b — 纯 Sector ETF 轮动 + GLD/TLT 对冲
代码熊 🐻

完全不同于 v3b/v4d 的框架！
v3b/v4d: S&P500 个股选股 + 行业轮转
v7b:     直接用 Sector ETF 做行业轮转（更简单、流动性更好）

宇宙:
  行业ETF: XLK, XLE, XLV, XLF, XLI, XLY, XLP, XLU, XLB, XLRE
  (XLC 仅2018年有, 暂不用)
  对冲: GLD, TLT, SHY

策略逻辑:
  1. 相对动量: 选 Top-3/4 行业 ETF by 6m momentum
  2. 绝对动量: 入选ETF须有正的6m momentum（vs SHY）
  3. 若无ETF通过绝对动量门槛 → 100% GLD
  4. 熊市(SPY<SMA200): top-3 行业，加20% GLD/TLT hedge
  5. DD响应 GLD: 与v4d相同
  6. 再平衡: 月度

优势:
  - 比个股更低风险（行业ETF天然分散）  
  - 更高信息比率（行业间动量 > 个股内行业动量）
  - 流动性完美，交易成本低

变种:
  v7b_3sec: Top-3 行业 ETF
  v7b_4sec: Top-4 行业 ETF  
  v7b_eq:   等权（不用inverse-vol）
  v7b_abs:  严格绝对动量过滤（vs SHY 3m）
  v7b_gld:  + v4d DD-responsive GLD hedge
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"

# Sector ETF universe (exclude XLC due to short history starting 2018)
SECTOR_ETFS = ['XLK', 'XLE', 'XLV', 'XLF', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE']
SAFE_ASSETS = {'GLD': 'GLD.csv', 'TLT': 'TLT.csv', 'SHY': 'SHY.csv', 'IEF': 'IEF.csv'}


def load_csv(fp):
    df = pd.read_csv(fp)
    col_date = 'Date' if 'Date' in df.columns else df.columns[0]
    df[col_date] = pd.to_datetime(df[col_date])
    df = df.set_index(col_date).sort_index()
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df


def load_etfs(tickers, cache_dir):
    d = {}
    for t in tickers:
        f = cache_dir / f"{t}.csv"
        if not f.exists():
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 100:
                d[t] = df['Close'].dropna()
        except:
            pass
    return pd.DataFrame(d)


def precompute(close_df):
    r1  = close_df / close_df.shift(22)  - 1
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1

    log_r = np.log(close_df / close_df.shift(1))
    vol63 = log_r.rolling(63).std() * np.sqrt(252)

    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    s50   = spy.rolling(50).mean()  if spy is not None else None
    sma50 = close_df.rolling(50).mean()

    return {
        'r1': r1, 'r3': r3, 'r6': r6, 'r12': r12,
        'vol63': vol63, 'spy': spy, 's200': s200, 's50': s50,
        'sma50': sma50, 'close': close_df,
    }


def regime(sig, date):
    if sig['s200'] is None:
        return 'bull'
    s = sig['s200'].loc[:date].dropna()
    p = sig['spy'].loc[:date].dropna()
    if len(s) == 0 or len(p) == 0:
        return 'bull'
    return 'bull' if p.iloc[-1] > s.iloc[-1] else 'bear'


def get_shy_3m(shy, date):
    avail = shy.loc[:date].dropna()
    if len(avail) < 64:
        return 0.0
    return float(avail.iloc[-1] / avail.iloc[-64] - 1)


def select_etf(sig, sector_etfs, safe, date, prev_hold, variant='v7b_4sec'):
    """Select sector ETFs by momentum."""
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0:
        return {}
    idx = idx_arr[-1]

    reg = regime(sig, date)
    shy_3m = get_shy_3m(safe['SHY'], date) if 'SHY' in safe else 0.0

    # Available sector ETFs at this date
    avail = []
    for t in sector_etfs:
        if t not in close.columns or t not in sig['r6'].columns:
            continue
        try:
            if pd.isna(close.loc[idx, t]) or pd.isna(sig['r6'].loc[idx, t]):
                continue
        except (KeyError, TypeError):
            continue
        avail.append(t)

    # Compute composite momentum for each ETF
    scores = {}
    for t in avail:
        def safe_get(df, col):
            try:
                v = df.loc[idx, col]
                return float(v) if not pd.isna(v) else 0.0
            except:
                return 0.0
        r1v  = safe_get(sig['r1'],  t)
        r3v  = safe_get(sig['r3'],  t)
        r6v  = safe_get(sig['r6'],  t)
        r12v = safe_get(sig['r12'], t)
        scores[t] = 0.10 * r1v + 0.20 * r3v + 0.40 * r6v + 0.30 * r12v

    # Continuity bonus
    for t in avail:
        if t in prev_hold:
            scores[t] += 0.02

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Select top N
    if variant in ('v7b_3sec',):
        top_n = 3
    else:
        top_n = 4  # v7b_4sec, v7b_eq, v7b_abs, v7b_gld

    # Regime adjustment
    if reg == 'bear':
        top_n = 3

    # Absolute momentum filter (v7b_abs): must beat SHY
    if variant == 'v7b_abs':
        qualified = [(t, s) for t, s in ranked if s > shy_3m]
    else:
        qualified = [(t, s) for t, s in ranked if s > 0]  # At least positive

    selected = qualified[:top_n]

    if not selected:
        # All negative → GLD/TLT defensive
        return {'GLD': 1.0}

    # Weighting
    if variant == 'v7b_eq':
        n = len(selected)
        weights = {t: 1.0 / n for t, _ in selected}
    else:
        # Inverse-vol weighting
        vols = {}
        for t, _ in selected:
            try:
                v = float(sig['vol63'].loc[idx, t]) if t in sig['vol63'].columns else 0.20
                vols[t] = max(v if not np.isnan(v) else 0.20, 0.05)
            except:
                vols[t] = 0.20
        iv = {t: 1.0 / vols[t] for t in vols}
        iv_t = sum(iv.values())
        weights = {t: iv[t] / iv_t for t in iv}

    # Bear regime: add GLD/TLT hedge
    if reg == 'bear':
        bear_hedge_frac = 0.20
        total = sum(weights.values())
        weights = {t: w / total * (1 - bear_hedge_frac) for t, w in weights.items()}
        # Pick GLD or TLT by 3m momentum
        if 'TLT' in safe and 'GLD' in safe:
            tlt_3m = get_shy_3m(safe['TLT'], date)
            gld_3m = get_shy_3m(safe['GLD'], date)
            hedge_t = 'GLD' if gld_3m >= tlt_3m else 'TLT'
        else:
            hedge_t = 'GLD'
        weights[hedge_t] = weights.get(hedge_t, 0) + bear_hedge_frac

    return weights


# ── DD-responsive GLD overlay ────────────────────────────────────────────────

DD_PARAMS = {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}

def add_gld(weights, frac):
    if frac <= 0 or not weights:
        return weights
    orig_total = sum(weights.values())
    if orig_total <= 0:
        return weights
    new = {t: w / orig_total * (1.0 - frac) for t, w in weights.items()}
    new['GLD'] = new.get('GLD', 0) + frac
    return new


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(sig, sector_etfs, safe_assets, variant='v7b_gld',
             start='2015-01-01', end='2025-12-31', cost=0.001):
    close = sig['close']
    rng   = close.loc[start:end].dropna(how='all')
    ends  = rng.resample('ME').last().index

    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0

        w = select_etf(sig, sector_etfs, safe_assets, dt, prev_h, variant)

        # v7b_gld: add DD-responsive GLD on top
        if variant == 'v7b_gld':
            gld_a = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0)
            if gld_a > 0:
                w = add_gld(w, gld_a)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in safe_assets}

        ret = 0.0
        for t, wt in w.items():
            if t in safe_assets:
                s = safe_assets[t].loc[dt:ndt].dropna()
            elif t in close.columns:
                s = close[t].loc[dt:ndt].dropna()
            else:
                s = pd.Series(dtype=float)
            if len(s) >= 2:
                ret += (s.iloc[-1] / s.iloc[0] - 1) * wt

        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)

    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq, float(np.mean(tos)) if tos else 0.0


def mets(eq, name=''):
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


def main():
    print("=" * 70)
    print("🐻 动量轮动 v7b — 纯 Sector ETF 轮动 + GLD/TLT 对冲")
    print("=" * 70)

    # Load sector ETFs + SPY
    all_etfs = SECTOR_ETFS + ['SPY']
    close_df = load_etfs(all_etfs, CACHE)
    print(f"  Loaded ETFs: {list(close_df.columns)}")

    # Load safe assets
    safe = {}
    for name, fname in SAFE_ASSETS.items():
        f = CACHE / fname
        if f.exists():
            df = load_csv(f)
            if 'Close' in df.columns:
                safe[name] = df['Close'].dropna()

    sig = precompute(close_df)

    # Date range: XLRE launched 2015, so start 2016 for proper warmup
    variants = {
        'v7b_3sec': 'Top-3 Sector ETF',
        'v7b_4sec': 'Top-4 Sector ETF',
        'v7b_eq':   'Top-4 Equal Weight',
        'v7b_abs':  'Top-4 AbsMom filter',
        'v7b_gld':  'Top-4 + DD GLD',
    }

    results = {}
    for var, label in variants.items():
        print(f"\n🔄 {label} ...")
        eq,   _  = backtest(sig, SECTOR_ETFS, safe, var, '2016-01-01', '2025-12-31')
        eq_i, _  = backtest(sig, SECTOR_ETFS, safe, var, '2016-01-01', '2021-12-31')
        eq_o, _  = backtest(sig, SECTOR_ETFS, safe, var, '2022-01-01', '2025-12-31')

        m  = mets(eq,  var)
        mi = mets(eq_i)
        mo = mets(eq_o)
        wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] else 0
        comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2
        results[var] = dict(m=m, is_m=mi, oos_m=mo, wf=wf, comp=comp)
        print(f"  CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  MaxDD {m['max_dd']:.1%}  "
              f"Calmar {m['calmar']:.2f}  Comp {comp:.3f}  WF {wf:.2f} {'✅' if wf >= 0.6 else '❌'}")

    # Add SPY benchmark
    print(f"\n🔄 SPY Buy&Hold (benchmark) ...")
    spy_eq = sig['close']['SPY'].loc['2016-01-01':'2025-12-31']
    spy_eq = spy_eq / spy_eq.iloc[0]
    spy_m  = mets(spy_eq.resample('ME').last())
    spy_comp = spy_m['sharpe'] * 0.4 + spy_m['calmar'] * 0.4 + spy_m['cagr'] * 0.2
    print(f"  CAGR {spy_m['cagr']:.1%}  Sharpe {spy_m['sharpe']:.2f}  MaxDD {spy_m['max_dd']:.1%}  "
          f"Calmar {spy_m['calmar']:.2f}  Comp {spy_comp:.3f}")

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Variant':<22} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 100)
    for var, label in variants.items():
        r = results[var]; m = r['m']
        flag = '✅' if r['wf'] >= 0.6 else '❌'
        print(f"{label:<22} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
              f"{r['is_m']['sharpe']:>7.2f} {r['oos_m']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f}{flag} {r['comp']:>8.3f}")
    print(f"{'SPY Buy&Hold':<22} {spy_m['cagr']:>6.1%} {spy_m['max_dd']:>7.1%} "
          f"{spy_m['sharpe']:>8.2f} {spy_m['calmar']:>8.2f} {'---':>7} {'---':>7} "
          f"{'---':>6} {spy_comp:>8.3f}")

    best = max(((v, r) for v, r in results.items() if r['wf'] >= 0.5),
               key=lambda x: x[1]['comp'], default=None)
    if best:
        v, r = best
        print(f"\n🏆 Best v7b: {variants[v]} → Composite {r['comp']:.3f}")
        # Compare to v4d champion
        if r['comp'] > 1.8 or r['m']['sharpe'] > 2.0:
            print("🚨🚨🚨 【重大突破】!")
        elif r['comp'] > 1.356:
            print(f"✅ Beats v4d champion (1.356)!")
        else:
            print(f"⚠️  Below v4d champion (1.356) — but sector ETF is simpler & more liquid")

    out = {v: {k: float(vv) for k, vv in {
        'cagr': r['m']['cagr'], 'max_dd': r['m']['max_dd'],
        'sharpe': r['m']['sharpe'], 'calmar': r['m']['calmar'],
        'wf': r['wf'], 'composite': r['comp'],
    }.items()} for v, r in results.items()}
    jf = Path(__file__).parent / "momentum_v7b_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results


if __name__ == '__main__':
    main()
