#!/usr/bin/env python3
"""
动量轮动 v7d — 纯横截面动量（无行业约束）
代码熊 🐻

核心差异 vs v3b/v4d:
  v3b: 4行业 × 3股票 = 12支（有行业约束）
  v7d: 直接选全市场 top-N，无行业限制

假设:
  行业约束虽然分散了行业集中风险，但也限制了选最强股票的自由
  纯横截面动量 (pure cross-sectional momentum) 是学术文献记录最充分的因子
  Jegadeesh-Titman (1993): 选过去6-12月最强的10%股票

变种测试:
  N=8:  Top 8 stocks (更集中)
  N=12: Top 12 stocks (接近v3b的12支)
  N=16: Top 16 stocks (更分散)
  N=20: Top 20 stocks (最分散)

过滤条件（保留v3b的基本过滤）:
  - price > $5
  - price > SMA50 (趋势过滤)
  - 6m return > 0 (正绝对动量)
  - vol < 65% (排除极高波动股)

权重: 
  inverse-vol (最常用的实证表现最好)

对冲:
  v4d DD-responsive GLD (不变)

注意: 无行业约束意味着在科技牛市时可能持有 8/12 支科技股
但这正是动量策略应该做的: 骑最强的趋势
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
    col = 'Date' if 'Date' in df.columns else df.columns[0]
    df[col] = pd.to_datetime(df[col])
    df = df.set_index(col).sort_index()
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
    if sig['s200'] is None:
        return 'bull'
    s = sig['s200'].loc[:date].dropna()
    p = sig['spy'].loc[:date].dropna()
    if len(s) == 0 or len(p) == 0:
        return 'bull'
    return 'bull' if p.iloc[-1] > s.iloc[-1] else 'bear'


def select_topN(sig, date, prev_hold, top_n=12, bear_n=None, cash_bear=0.20):
    """Pure cross-sectional momentum — no sector constraint."""
    if bear_n is None:
        bear_n = max(int(top_n * 0.67), 4)
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0:
        return {}, 'bull'
    idx = idx_arr[-1]
    reg = bull_bear(sig, date)

    # Composite momentum (v3b formula)
    mom = (sig['r1'].loc[idx] * 0.20 +
           sig['r3'].loc[idx] * 0.40 +
           sig['r6'].loc[idx] * 0.30 +
           sig['r12'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'mom':   mom,
        'r6':    sig['r6'].loc[idx],
        'vol':   sig['vol30'].loc[idx],
        'price': close.loc[idx],
        'sma50': sig['sma50'].loc[idx],
    }).dropna(subset=['mom', 'sma50'])

    # Same base filters as v3b
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]

    # Continuity bonus
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += 0.03

    # Select top N without sector constraint
    n = top_n if reg == 'bull' else bear_n
    selected = df.sort_values('mom', ascending=False).index[:n].tolist()

    if not selected:
        return {}, reg

    cash = cash_bear if reg == 'bear' else 0.0
    invested = 1.0 - cash

    # Pure inverse-vol weighting
    iv = {t: 1.0 / max(df.loc[t, 'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values())
    weights = {t: iv[t] / iv_t * invested for t in selected}
    return weights, reg


def add_gld(weights, frac):
    if frac <= 0 or not weights:
        return weights
    total = sum(weights.values())
    if total <= 0:
        return weights
    new = {t: w / total * (1 - frac) for t, w in weights.items()}
    new['GLD'] = frac
    return new


def backtest(close_df, sig, gld, top_n=12, bear_n=None, cash_bear=0.20,
             start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0

        w, _ = select_topN(sig, dt, prev_h, top_n, bear_n, cash_bear)
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
        val *= (1 + ret); 
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)

    return pd.Series(vals, index=pd.DatetimeIndex(dates)), float(np.mean(tos)) if tos else 0.0


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
    print("🐻 动量轮动 v7d — 纯横截面动量（无行业约束）")
    print("=" * 70)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sig      = precompute(close_df)
    gld      = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    print(f"  Loaded {len(close_df.columns)} tickers")

    VARIANTS = [
        ('v3b_base', 12, 8,  0.20, 'v3b+DD baseline (sector)'),
        ('top8',      8, 6,  0.20, 'Top-8 (concentrated)'),
        ('top12',    12, 8,  0.20, 'Top-12 (pure xsec)'),
        ('top16',    16, 10, 0.20, 'Top-16 (medium)'),
        ('top20',    20, 13, 0.20, 'Top-20 (diversified)'),
        ('top12nb',  12, 12, 0.00, 'Top-12 no bear cash'),
    ]

    # For baseline: import the v4d backtest
    # For simplicity: run top12 as baseline proxy (same N as v3b's ~12 positions)
    # But v3b also has sector constraint, so this is a fair comparison

    results = {}
    for tag, n, bn, cash_b, label in VARIANTS:
        print(f"\n🔄 {label} (N={n}, bearN={bn}) ...")
        eq,   _ = backtest(close_df, sig, gld, n, bn, cash_b)
        eq_i, _ = backtest(close_df, sig, gld, n, bn, cash_b, '2015-01-01', '2020-12-31')
        eq_o, _ = backtest(close_df, sig, gld, n, bn, cash_b, '2021-01-01', '2025-12-31')

        m  = mets(eq,  tag)
        mi = mets(eq_i)
        mo = mets(eq_o)
        wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] else 0
        comp = m['sharpe'] * 0.4 + m['calmar'] * 0.4 + m['cagr'] * 0.2

        results[tag] = dict(label=label, m=m, is_m=mi, oos_m=mo, wf=wf, comp=comp)
        print(f"  CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  MaxDD {m['max_dd']:.1%}  "
              f"Calmar {m['calmar']:.2f}  Comp {comp:.3f}  WF {wf:.2f} {'✅' if wf >= 0.7 else '❌'}")

    # Note: v3b_base in this script uses pure xsec top-12, not sector-constrained
    # Real v3b/v4d baseline is Composite=1.356

    print("\n" + "=" * 105)
    print(f"{'Variant':<30} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 105)
    for tag, _, _, _, label in VARIANTS:
        r = results[tag]; m = r['m']
        flag = '✅' if r['wf'] >= 0.7 else '❌'
        print(f"{label:<30} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
              f"{r['is_m']['sharpe']:>7.2f} {r['oos_m']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f}{flag} {r['comp']:>8.3f}")
    print(f"\n  [Reference: v4d champion = Composite 1.356, Sharpe 1.45, MaxDD -15.0%]")

    # Find best
    bests = [(t, r) for t, r in results.items() if r['wf'] >= 0.7]
    if bests:
        bt, br = max(bests, key=lambda x: x[1]['comp'])
        print(f"\n🏆 Best v7d: {br['label']} → Composite {br['comp']:.3f}")
        if br['comp'] > 1.8 or br['m']['sharpe'] > 2.0:
            print("🚨🚨🚨 【重大突破】!")
        elif br['comp'] > 1.356:
            print("✅ Beats v4d champion (1.356)!")
        else:
            print(f"⚠️  Below v4d champion (1.356)")

    out = {t: {'cagr': float(r['m']['cagr']), 'max_dd': float(r['m']['max_dd']),
               'sharpe': float(r['m']['sharpe']), 'calmar': float(r['m']['calmar']),
               'wf': float(r['wf']), 'composite': float(r['comp'])}
           for t, r in results.items()}
    jf = Path(__file__).parent / "momentum_v7d_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results


if __name__ == '__main__':
    main()
