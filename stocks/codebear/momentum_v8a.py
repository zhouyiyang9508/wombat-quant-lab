#!/usr/bin/env python3
"""
动量轮动 v8a — 信用利差预警 + 提前 GLD 对冲
代码熊 🐻

核心创新: 用债券市场信号（HYG/IEF信用利差）提前预警股票风险

学术背景:
  信用利差 (Credit Spread) 往往领先股票市场回撤 1-3 个月
  当高收益债券 (HYG) 相对国债 (IEF) 开始下跌时，
  这是"smart money"正在撤离风险资产的信号
  
  历史案例：
  - 2018 Q4: HYG 10月初开始跌，比SPY提前2-3周
  - 2020 COVID: HYG 2月20日开始跌，SPY 2月24日跟随
  - 2022: HYG持续下跌贯穿全年，TLT也跌（通胀周期特殊性）

信号设计:
  Credit_Stress = HYG_3m_return < IEF_3m_return - threshold
  (当高收益债券跑输中期国债时 = 信用压力)

  信用压力档:
  Level 0 (正常): HYG-IEF > -2% → 无预警，仅DD对冲
  Level 1 (轻度): HYG-IEF in [-5%, -2%) → 提前加 10% GLD
  Level 2 (中度): HYG-IEF in [-10%, -5%) → 提前加 20% GLD  
  Level 3 (重度): HYG-IEF < -10% → 提前加 30% GLD
  
  + 原v4d DD响应（在上述基础上叠加）
  总对冲上限: 60%

变种:
  v8a_cs:    信用利差预警 only（无DD对冲）→ 测试信号质量
  v8a_dd:    v4d DD only（基线对照）
  v8a_combo: 信用利差 + DD（主策略）
  v8a_soft:  信用利差 soft (threshold -3%)
  v8a_tight: 信用利差 tight (threshold -7%)
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# Credit spread thresholds (HYG - IEF 3m return)
CS_LEVELS = {
    'combo': [(-0.02, 0.0), (-0.05, 0.10), (-0.10, 0.20), (-999, 0.30)],
    'soft':  [(-0.03, 0.0), (-0.06, 0.10), (-0.12, 0.20), (-999, 0.30)],
    'tight': [(-0.07, 0.0), (-0.10, 0.10), (-0.15, 0.20), (-999, 0.30)],
}

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


def precompute(close_df, hyg, ief):
    r1   = close_df / close_df.shift(22)  - 1
    r3   = close_df / close_df.shift(63)  - 1
    r6   = close_df / close_df.shift(126) - 1
    r12  = close_df / close_df.shift(252) - 1
    log_r = np.log(close_df / close_df.shift(1))
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()

    # Credit spread signal: HYG - IEF 3-month return
    hyg_r3 = hyg / hyg.shift(63) - 1
    ief_r3 = ief / ief.shift(63) - 1
    cs_diff = hyg_r3 - ief_r3   # negative = stress

    return dict(r1=r1, r3=r3, r6=r6, r12=r12, vol30=vol30,
                spy=spy, s200=s200, sma50=sma50, close=close_df,
                hyg_r3=hyg_r3, ief_r3=ief_r3, cs_diff=cs_diff)


def regime(sig, date):
    if sig['s200'] is None:
        return 'bull'
    s = sig['s200'].loc[:date].dropna()
    p = sig['spy'].loc[:date].dropna()
    if len(s) == 0 or len(p) == 0:
        return 'bull'
    return 'bull' if p.iloc[-1] > s.iloc[-1] else 'bear'


def get_cs_alloc(sig, date, variant_levels):
    """
    Get credit-spread-based GLD allocation.
    Returns fraction based on HYG-IEF 3m spread.
    """
    cs = sig['cs_diff'].loc[:date].dropna()
    if len(cs) == 0:
        return 0.0
    cs_val = float(cs.iloc[-1])
    for threshold, alloc in variant_levels:
        if cs_val <= threshold:
            return alloc
    return 0.0


def select_v3b(sig, sectors, date, prev_hold):
    """Identical v3b stock selection — proven baseline."""
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0:
        return {}, 'bull'
    idx = idx_arr[-1]
    reg = regime(sig, date)

    mom = (sig['r1'].loc[idx] * 0.20 +
           sig['r3'].loc[idx] * 0.40 +
           sig['r6'].loc[idx] * 0.30 +
           sig['r12'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'mom': mom, 'r6': sig['r6'].loc[idx],
        'vol': sig['vol30'].loc[idx], 'price': close.loc[idx],
        'sma50': sig['sma50'].loc[idx],
    }).dropna(subset=['mom', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += 0.03
    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)
    if reg == 'bull':
        top_secs = sec_mom.head(4).index.tolist(); sps, cash = 3, 0.0
    else:
        top_secs = sec_mom.head(3).index.tolist(); sps, cash = 2, 0.20
    selected = []
    for sec in top_secs:
        selected.extend(df[df['sector'] == sec].sort_values('mom', ascending=False).index[:sps])
    if not selected:
        return {}, reg
    iv = {t: 1.0 / max(df.loc[t, 'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v / iv_t for t, v in iv.items()}
    mn = min(df.loc[t, 'mom'] for t in selected); sh = max(-mn + 0.01, 0)
    mw = {t: df.loc[t, 'mom'] + sh for t in selected}; mw_t = sum(mw.values())
    mw_w = {t: v / mw_t for t, v in mw.items()}
    invested = 1.0 - cash
    return {t: (0.70 * iv_w[t] + 0.30 * mw_w[t]) * invested for t in selected}, reg


def add_hedge(weights, frac, ticker='GLD'):
    if frac <= 0 or not weights:
        return weights
    total = sum(weights.values())
    if total <= 0:
        return weights
    new = {t: w / total * (1 - frac) for t, w in weights.items()}
    new[ticker] = new.get(ticker, 0) + frac
    return new


def backtest(close_df, sig, sectors, gld, variant,
             start='2015-01-01', end='2025-12-31', cost=0.0015):
    cs_levels_map = CS_LEVELS.get(variant.replace('v8a_', ''), CS_LEVELS['combo'])

    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0

        w, _ = select_v3b(sig, sectors, dt, prev_h)

        # CS-based allocation
        cs_alloc = 0.0
        if variant == 'v8a_cs':
            cs_alloc = get_cs_alloc(sig, dt, cs_levels_map)
        elif variant in ('v8a_combo', 'v8a_soft', 'v8a_tight'):
            cs_alloc = get_cs_alloc(sig, dt, cs_levels_map)

        # DD-based allocation
        dd_alloc = 0.0
        if variant in ('v8a_dd', 'v8a_combo', 'v8a_soft', 'v8a_tight'):
            dd_alloc = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0)

        # Combined (cap at 55%)
        total_hedge = min(cs_alloc + dd_alloc, 0.55)
        w = add_hedge(w, total_hedge)

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
    print("🐻 动量轮动 v8a — 信用利差预警 + 提前 GLD 对冲")
    print("=" * 70)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld      = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    hyg      = load_csv(CACHE / "HYG.csv")['Close'].dropna()
    ief      = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    sig      = precompute(close_df, hyg, ief)
    print(f"  Loaded {len(close_df.columns)} stocks, HYG {len(hyg)} days, IEF {len(ief)} days")

    # Show some CS stats
    print("\n📊 Credit Spread (HYG-IEF 3m) at key dates:")
    for ds in ['2018-10-31','2019-12-31','2020-03-31','2021-12-31','2022-06-30','2022-12-31','2023-12-31']:
        try:
            d = pd.Timestamp(ds)
            cs = sig['cs_diff'].loc[:d].dropna()
            if len(cs) > 0:
                cv = float(cs.iloc[-1])
                level = 'STRESS' if cv < -0.05 else ('mild' if cv < -0.02 else 'normal')
                print(f"  {ds}: CS={cv:.2%} ({level})")
        except:
            pass

    VARIANTS = {
        'v8a_dd':    'v4d DD only (base)',
        'v8a_cs':    'CS only (no DD)',
        'v8a_combo': 'CS + DD combo',
        'v8a_soft':  'CS soft + DD',
        'v8a_tight': 'CS tight + DD',
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
    print(f"{'Variant':<24} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 105)
    for var, label in VARIANTS.items():
        r = results[var]; m = r['m']
        flag = '✅' if r['wf'] >= 0.7 else '❌'
        print(f"{label:<24} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
              f"{r['is_m']['sharpe']:>7.2f} {r['oos_m']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f}{flag} {r['comp']:>8.3f}")
    print(f"\n  [v4d champion: Composite 1.356, Sharpe 1.45, MaxDD -15.0%]")

    bests = [(v, r) for v, r in results.items() if v != 'v8a_dd' and r['wf'] >= 0.7]
    if bests:
        bv, br = max(bests, key=lambda x: x[1]['comp'])
        base_c = results['v8a_dd']['comp']
        print(f"\n🏆 Best v8a: {VARIANTS[bv]} → Composite {br['comp']:.3f} "
              f"(vs DD-only {base_c:.3f}, Δ{br['comp']-base_c:+.3f})")
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
    jf = Path(__file__).parent / "momentum_v8a_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results


if __name__ == '__main__':
    main()
