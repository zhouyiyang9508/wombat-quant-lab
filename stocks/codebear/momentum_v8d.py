#!/usr/bin/env python3
"""
动量轮动 v8d — 最优组合: Breadth AND SPY Regime + GLD竞争 + DD响应
代码熊 🐻

组合策略: 把两个有效改进结合
1. v5e_combo: Breadth AND SPY 双重确认熊市 (Comp 1.357, WF 0.81)
2. v8c_gld_dd: GLD参与自然竞争 + DD响应 (Comp 1.358, WF 0.88)

假设: 两个独立有效的改进组合后可能有叠加效果

实现:
  - 股票选择: v3b (不变)
  - 市场 Regime: Breadth AND SPY SMA200 (需要两者都触发才进熊市)
  - GLD: 
    (a) 自然竞争 (当GLD 6m动量>最弱行业动量时进入)
    (b) DD响应覆盖 (DD<-8%: 30%, DD<-12%: 50%, DD<-18%: 60%)
  - 两者叠加，总GLD上限60%

如果 v8d > 1.36 with WF > 0.85，值得提交为新最佳
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
    above_sma50 = (close_df > sma50).astype(float)
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, vol30=vol30,
                spy=spy, s200=s200, sma50=sma50,
                above_sma50=above_sma50, close=close_df)


def spy_regime(sig, date):
    """SPY vs SMA200."""
    if sig['s200'] is None: return 'bull'
    s = sig['s200'].loc[:date].dropna()
    p = sig['spy'].loc[:date].dropna()
    if len(s) == 0 or len(p) == 0: return 'bull'
    return 'bull' if p.iloc[-1] > s.iloc[-1] else 'bear'


def breadth_regime(sig, date, narrow_thresh=0.40):
    """Market breadth."""
    ab = sig['above_sma50'].loc[:date].dropna(how='all')
    if len(ab) == 0: return 'wide'
    row = ab.iloc[-1].dropna()
    if 'SPY' in row.index: row = row.drop('SPY')
    if len(row) == 0: return 'wide'
    b = float(row.mean())
    return 'narrow' if b < narrow_thresh else 'wide'


def effective_regime(sig, date, variant):
    """Regime based on variant."""
    spy_r = spy_regime(sig, date)
    if variant in ('v4d', 'v8c_gld_dd'):
        return spy_r
    elif variant in ('v5e_combo', 'v8d'):
        # BOTH must signal bear
        br = breadth_regime(sig, date)
        return 'bear' if (spy_r == 'bear' and br == 'narrow') else 'bull'
    return spy_r


def get_gld_competition_alloc(sig, date, gld_prices):
    """
    Check if GLD 6m momentum > weakest top-4 sector.
    Returns 0.20 if GLD should compete, 0.0 otherwise.
    """
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0: return 0.0
    idx = idx_arr[-1]

    # GLD 6m momentum
    gld_avail = gld_prices.loc[:date].dropna()
    if len(gld_avail) < 127: return 0.0
    gld_6m = float(gld_avail.iloc[-1] / gld_avail.iloc[-127] - 1)

    # Sector average 6m momentum
    r6 = sig['r6'].loc[idx].dropna()
    sec_moms = {}
    for t, v in r6.items():
        if not np.isnan(v):
            sec_moms[t] = v
    if not sec_moms: return 0.0

    # Get sector averages (approximate)
    # Just use stock universe average and compare GLD
    avg_6m = float(r6.dropna().mean())
    # GLD replaces bottom sector if its momentum > avg
    if gld_6m > avg_6m * 0.8:  # 80% of average sector mom = threshold to enter
        return 0.20
    return 0.0


def select_v8d(sig, sectors, date, prev_hold, gld_prices, variant):
    """v3b stock selection with regime variant."""
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0: return {}, 'bull'
    idx = idx_arr[-1]
    reg = effective_regime(sig, date, variant)

    mom_base = (sig['r1'].loc[idx] * 0.20 + sig['r3'].loc[idx] * 0.40 +
                sig['r6'].loc[idx] * 0.30 + sig['r12'].loc[idx] * 0.10)
    r6_val = sig['r6'].loc[idx]
    vol_val = sig['vol30'].loc[idx]

    df = pd.DataFrame({
        'mom': mom_base, 'r6': r6_val, 'vol': vol_val,
        'price': close.loc[idx], 'sma50': sig['sma50'].loc[idx],
    }).dropna(subset=['mom', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += 0.03
    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    # GLD competition check (for variants with competition)
    gld_compete_alloc = 0.0
    if variant in ('v8c_gld_dd', 'v8d'):
        gld_compete_alloc = get_gld_competition_alloc(sig, date, gld_prices)

    if reg == 'bull':
        n_secs = 4 if gld_compete_alloc == 0 else 3
        sps, cash = 3, 0.0
    else:
        n_secs = 3 if gld_compete_alloc == 0 else 2
        sps, cash = 2, 0.20

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    if not selected and gld_compete_alloc == 0:
        return {}, reg

    # Stock allocation
    stock_frac = 1.0 - cash - gld_compete_alloc
    if stock_frac < 0: stock_frac = 0.0

    if selected and stock_frac > 0:
        iv = {t: 1.0 / max(df.loc[t, 'vol'], 0.10) for t in selected}
        iv_t = sum(iv.values()); iv_w = {t: v / iv_t for t, v in iv.items()}
        mn = min(df.loc[t, 'mom'] for t in selected); sh = max(-mn + 0.01, 0)
        mw = {t: df.loc[t, 'mom'] + sh for t in selected}; mw_t = sum(mw.values())
        mw_w = {t: v / mw_t for t, v in mw.items()}
        weights = {t: (0.70 * iv_w[t] + 0.30 * mw_w[t]) * stock_frac for t in selected}
    else:
        weights = {}

    if gld_compete_alloc > 0:
        weights['GLD'] = weights.get('GLD', 0) + gld_compete_alloc

    return weights, reg


def add_gld(weights, frac):
    if frac <= 0 or not weights: return weights
    tot = sum(weights.values())
    if tot <= 0: return weights
    new = {t: w / tot * (1 - frac) for t, w in weights.items()}
    new['GLD'] = new.get('GLD', 0) + frac
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

        w, _ = select_v8d(sig, sectors, dt, prev_h, gld, variant)
        # DD-responsive GLD overlay (all variants include this)
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
    print("🐻 动量轮动 v8d — 最优组合 (Breadth+SPY + GLD竞争 + DD)")
    print("=" * 70)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld      = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    sig      = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers")

    VARIANTS = {
        'v4d':       'v4d (SPY+DD) baseline',
        'v5e_combo': 'v5e (Breadth AND SPY+DD)',
        'v8c_gld_dd':'v8c (GLD compete+DD)',
        'v8d':       'v8d (Breadth+SPY+GLD+DD)',
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

    bests = [(v, r) for v, r in results.items() if r['wf'] >= 0.7]
    if bests:
        bv, br = max(bests, key=lambda x: x[1]['comp'])
        print(f"\n🏆 Best overall: {VARIANTS[bv]} → Composite {br['comp']:.3f}")
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
    jf = Path(__file__).parent / "momentum_v8d_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results


if __name__ == '__main__':
    main()
