#!/usr/bin/env python3
"""
动量轮动 v8c — GLD/TLT 作为竞争性"行业" 参与轮转
代码熊 🐻

核心思路: GLD 和 TLT 不再作为对冲资产，而是作为正常资产参与动量竞争

传统 v4d:
  - 股票: 主要资产（~80-100%）
  - GLD: 对冲资产（只在DD时出现）

v8c 思路:
  - 把 GLD 的6个月动量和行业动量放在同一个评分体系中
  - 如果 GLD 的6月动量高于排名最低的股票行业, GLD 自然替代一个行业
  - 比较的是: GLD_6m_return vs 各股票行业平均6m_return
  - 当 GLD 动量排进 top-5 时, 分配 20-25% 给 GLD

学术依据: Gary Antonacci 双动量框架
  把 GLD/TLT 作为替代资产, 当它们超过股票动量时自动轮换进来

变种:
  v8c_gld:  GLD竞争 top-5行业 (GLD取代最弱行业)
  v8c_tlt:  TLT竞争
  v8c_both: GLD+TLT共同竞争 (同时评估)
  v8c_gld_dd: GLD竞争 + 额外DD响应 GLD (叠加)
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


def get_alt_mom(prices, date, lb=126):
    """Get momentum of an alternative asset at date."""
    avail = prices.loc[:date].dropna()
    if len(avail) < lb + 1:
        return None
    return float(avail.iloc[-1] / avail.iloc[-lb - 1] - 1)


def select_v8c(sig, sectors, date, prev_hold, variant,
               alt_prices=None):
    """
    v3b base + GLD/TLT competition.
    alt_prices: dict of {'GLD': pd.Series, 'TLT': pd.Series}
    """
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0: return {}, 'bull'
    idx = idx_arr[-1]
    reg = bull_bear(sig, date)

    # v3b composite momentum
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

    # Compute alternative asset 6m momentum
    alt_moms = {}
    if alt_prices and variant != 'base':
        for alt_name, alt_p in alt_prices.items():
            m = get_alt_mom(alt_p, date, 126)
            if m is not None:
                alt_moms[alt_name] = m

    # Determine top sectors and whether alt replaces one
    if reg == 'bull':
        n_sectors = 4; sps = 3; cash = 0.0
    else:
        n_sectors = 3; sps = 2; cash = 0.20

    # Get sector rankings
    ranked_secs = sec_mom.index.tolist()

    # Compare alt assets to nth sector (threshold for entry)
    alt_allocations = {}
    remaining_sectors = n_sectors

    if alt_moms and variant in ('v8c_gld', 'v8c_tlt', 'v8c_both', 'v8c_gld_dd'):
        for alt_name, alt_m in sorted(alt_moms.items(), key=lambda x: x[1], reverse=True):
            if remaining_sectors == 0:
                break
            # Compare to the nth sector (weakest top sector)
            nth_sec_idx = min(remaining_sectors - 1, len(sec_mom) - 1)
            if nth_sec_idx < 0 or len(sec_mom) == 0:
                nth_sec_mom = 0
            else:
                nth_sec_mom = sec_mom.iloc[nth_sec_idx]

            if alt_m > nth_sec_mom:
                # Alt replaces one sector (20% allocation)
                alt_allocations[alt_name] = 0.20  # Fixed 20% per alt asset
                remaining_sectors -= 1  # One fewer sector slot

    top_secs = ranked_secs[:remaining_sectors]

    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    if not selected and not alt_allocations:
        return {}, reg

    # Total alt allocation
    total_alt = sum(alt_allocations.values())
    stock_frac = 1.0 - cash - total_alt
    if stock_frac < 0:
        stock_frac = 0.0

    # Weight stocks
    if selected and stock_frac > 0:
        iv = {t: 1.0 / max(df.loc[t, 'vol'], 0.10) for t in selected}
        iv_t = sum(iv.values()); iv_w = {t: v / iv_t for t, v in iv.items()}
        mn = min(df.loc[t, 'mom'] for t in selected); sh = max(-mn + 0.01, 0)
        mw = {t: df.loc[t, 'mom'] + sh for t in selected}
        mw_t = sum(mw.values()); mw_w = {t: v / mw_t for t, v in mw.items()}
        weights = {t: (0.70 * iv_w[t] + 0.30 * mw_w[t]) * stock_frac for t in selected}
    else:
        weights = {}

    # Add alt allocations
    for alt_name, alloc in alt_allocations.items():
        weights[alt_name] = weights.get(alt_name, 0) + alloc

    return weights, reg


def add_gld(weights, frac):
    if frac <= 0 or not weights: return weights
    tot = sum(weights.values())
    if tot <= 0: return weights
    new = {t: w / tot * (1 - frac) for t, w in weights.items()}
    new['GLD'] = new.get('GLD', 0) + frac
    return new


def backtest(close_df, sig, sectors, alt_prices, variant,
             start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0

        # Select which alts compete
        if variant == 'v8c_gld':
            competing = {'GLD': alt_prices['GLD']}
        elif variant == 'v8c_tlt':
            competing = {'TLT': alt_prices['TLT']}
        elif variant in ('v8c_both', 'v8c_gld_dd'):
            competing = alt_prices
        else:
            competing = {}

        w, _ = select_v8c(sig, sectors, dt, prev_h, variant, competing)

        # All variants with 'dd' suffix or base: also add DD-responsive GLD on top
        if variant in ('v8c_gld_dd', 'base'):
            gld_a = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0)
            w = add_gld(w, gld_a)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in alt_prices}

        ret = 0.0
        for t, wt in w.items():
            if t in alt_prices:
                s = alt_prices[t].loc[dt:ndt].dropna()
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
    print("🐻 动量轮动 v8c — GLD/TLT 作为竞争性行业参与轮转")
    print("=" * 70)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    sig      = precompute(close_df)
    alt_prices = {
        'GLD': load_csv(CACHE / "GLD.csv")['Close'].dropna(),
        'TLT': load_csv(CACHE / "TLT.csv")['Close'].dropna(),
    }
    print(f"  Loaded {len(close_df.columns)} stocks")

    # Print GLD momentum vs sector ranking at key dates
    print("\n📊 GLD 6m momentum vs sector ranking:")
    for ds in ['2018-10-31', '2019-06-30', '2020-06-30', '2022-09-30', '2023-12-31']:
        try:
            d = pd.Timestamp(ds)
            gld_m = alt_prices['GLD'] / alt_prices['GLD'].shift(126) - 1
            gld_m = gld_m.loc[:d].dropna().iloc[-1]
            # Get sector moms
            idx_arr = close_df.index[close_df.index <= d]
            if len(idx_arr) == 0: continue
            idx = idx_arr[-1]
            r6 = sig['r6'].loc[idx]
            sec_s = {}
            for t in close_df.columns:
                sec = sectors.get(t, 'Unknown')
                if sec not in sec_s: sec_s[sec] = []
                v = r6.get(t, float('nan'))
                if not np.isnan(v): sec_s[sec].append(v)
            sec_avg = {s: np.mean(v) for s, v in sec_s.items() if v}
            ranked = sorted(sec_avg.items(), key=lambda x: x[1], reverse=True)[:5]
            rank = sum(1 for s, m in ranked if m > gld_m) + 1
            print(f"  {ds}: GLD_6m={gld_m:.1%}, rank among top-5 sectors: #{rank}")
        except Exception as e:
            pass

    VARIANTS = {
        'base':       'v3b+DD (baseline)',
        'v8c_gld':    'GLD competes',
        'v8c_tlt':    'TLT competes',
        'v8c_both':   'GLD+TLT compete',
        'v8c_gld_dd': 'GLD competes + DD',
    }

    results = {}
    for var, label in VARIANTS.items():
        print(f"\n🔄 {label} ...")
        eq,   _ = backtest(close_df, sig, sectors, alt_prices, var)
        eq_i, _ = backtest(close_df, sig, sectors, alt_prices, var, '2015-01-01', '2020-12-31')
        eq_o, _ = backtest(close_df, sig, sectors, alt_prices, var, '2021-01-01', '2025-12-31')

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

    bests = [(v, r) for v, r in results.items() if v != 'base' and r['wf'] >= 0.7]
    if bests:
        bv, br = max(bests, key=lambda x: x[1]['comp'])
        base_c = results['base']['comp']
        print(f"\n🏆 Best v8c: {VARIANTS[bv]} → Composite {br['comp']:.3f} "
              f"(vs baseline {base_c:.3f}, Δ{br['comp']-base_c:+.3f})")
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
    jf = Path(__file__).parent / "momentum_v8c_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results


if __name__ == '__main__':
    main()
