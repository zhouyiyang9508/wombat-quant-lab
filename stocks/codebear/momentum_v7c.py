#!/usr/bin/env python3
"""
动量轮动 v7c — 股票级别 Alpha 动量 (Sector-Relative Momentum)
代码熊 🐻

核心创新 vs v3b/v4d:
  传统动量: 选行业内绝对动量最高的股票
  v7c Alpha动量: 选相对行业ETF超额收益最高的股票

原理:
  一支股票6m涨了30%，但它所在的XLK涨了35% → 这是落后者
  另一支涨了25%，但XLK只涨了10% → 这是真正的强者
  
  Alpha Momentum = Stock_6m_return - Sector_ETF_6m_return
  
  选股时优先选 alpha > 0 且 alpha 最大的股票

为什么可能有效:
  学术研究表明 Industry-Adjusted Momentum (IAM) 比纯动量有更高夏普
  参考: Novy-Marx (2012), "Is Momentum Really Momentum?"
  IAM 过滤掉纯行业轮动，只保留个股超额Alpha
  预期: 更低的个股同向回落风险（当行业回调时，有alpha的股票跌得少）

信号设计:
  1. 计算每支股票相对行业ETF的3m/6m超额收益
  2. 行业ETF映射: 用S&P500行业分类匹配XLK/XLE/XLV等
  3. 选择条件: alpha > 0 (必须跑赢行业)
  4. 行业轮动: 行业ranking用ETF动量（而非个股平均）
  5. 权重: inverse-vol (不变)
  6. 对冲: v4d DD-responsive GLD (不变)

变种:
  v7c_alpha: 纯alpha动量选股（行业仍用v3b方式轮转）
  v7c_both:  行业ETF轮转 + alpha选股
  v7c_strict: 要求3m AND 6m alpha均为正
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# Sector ETF mapping for S&P 500 GICS sectors
SECTOR_ETF_MAP = {
    'Information Technology': 'XLK',
    'Technology':             'XLK',
    'Energy':                 'XLE',
    'Health Care':            'XLV',
    'Healthcare':             'XLV',
    'Financials':             'XLF',
    'Industrials':            'XLI',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples':       'XLP',
    'Utilities':              'XLU',
    'Materials':              'XLB',
    'Real Estate':            'XLRE',
    'Communication Services': 'XLK',  # Proxy (XLC too short)
    'Unknown':                'SPY',  # Fallback
}


def load_csv(fp):
    df = pd.read_csv(fp)
    col_date = 'Date' if 'Date' in df.columns else df.columns[0]
    df[col_date] = pd.to_datetime(df[col_date])
    df = df.set_index(col_date).sort_index()
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


def load_etf(ticker):
    f = CACHE / f"{ticker}.csv"
    if not f.exists():
        return None
    try:
        df = load_csv(f)
        return df['Close'].dropna() if 'Close' in df.columns else None
    except:
        return None


def precompute(close_df, etf_prices):
    r1  = close_df / close_df.shift(22)  - 1
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1

    log_r = np.log(close_df / close_df.shift(1))
    vol30 = log_r.rolling(30).std() * np.sqrt(252)

    spy  = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200 = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()

    # Precompute ETF returns for alpha calculation
    etf_r3  = {t: p / p.shift(63)  - 1 for t, p in etf_prices.items() if p is not None}
    etf_r6  = {t: p / p.shift(126) - 1 for t, p in etf_prices.items() if p is not None}
    etf_r12 = {t: p / p.shift(252) - 1 for t, p in etf_prices.items() if p is not None}

    return {
        'r1': r1, 'r3': r3, 'r6': r6, 'r12': r12,
        'vol30': vol30, 'spy': spy, 's200': s200,
        'sma50': sma50, 'close': close_df,
        'etf_r3': etf_r3, 'etf_r6': etf_r6, 'etf_r12': etf_r12,
    }


def regime(sig, date):
    if sig['s200'] is None:
        return 'bull'
    s = sig['s200'].loc[:date].dropna()
    p = sig['spy'].loc[:date].dropna()
    if len(s) == 0 or len(p) == 0:
        return 'bull'
    return 'bull' if p.iloc[-1] > s.iloc[-1] else 'bear'


def get_etf_val(etf_series, date):
    """Get ETF return value at date."""
    avail = etf_series.loc[:date].dropna()
    if len(avail) == 0:
        return None
    return float(avail.iloc[-1])


def select_v7c(sig, sectors, date, prev_hold, variant='v7c_alpha'):
    close = sig['close']
    idx_arr = close.index[close.index <= date]
    if len(idx_arr) == 0:
        return {}, 'bull'
    idx = idx_arr[-1]
    reg = regime(sig, date)

    # Base momentum (v3b formula)
    mom_base = (sig['r1'].loc[idx] * 0.20 +
                sig['r3'].loc[idx] * 0.40 +
                sig['r6'].loc[idx] * 0.30 +
                sig['r12'].loc[idx] * 0.10)

    df = pd.DataFrame({
        'mom':   mom_base,
        'r3':    sig['r3'].loc[idx],
        'r6':    sig['r6'].loc[idx],
        'r12':   sig['r12'].loc[idx],
        'vol':   sig['vol30'].loc[idx],
        'price': close.loc[idx],
        'sma50': sig['sma50'].loc[idx],
    }).dropna(subset=['mom', 'sma50'])

    # Base filters (same as v3b)
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))

    # ── Alpha momentum calculation ──────────────────────────────────────────
    # For each stock, compute excess return vs its sector ETF
    df['etf']    = df['sector'].map(lambda s: SECTOR_ETF_MAP.get(s, 'SPY'))
    df['alpha3'] = 0.0
    df['alpha6'] = 0.0

    for etf_code in df['etf'].unique():
        mask = df['etf'] == etf_code
        if etf_code in sig['etf_r3'] and etf_code in sig['etf_r6']:
            try:
                etf3 = get_etf_val(sig['etf_r3'][etf_code], date)
                etf6 = get_etf_val(sig['etf_r6'][etf_code], date)
                if etf3 is not None:
                    df.loc[mask, 'alpha3'] = df.loc[mask, 'r3'] - etf3
                if etf6 is not None:
                    df.loc[mask, 'alpha6'] = df.loc[mask, 'r6'] - etf6
            except:
                pass

    # ── Alpha filters ───────────────────────────────────────────────────────
    if variant == 'v7c_alpha':
        # Require positive 6m alpha (stock outperforms its sector ETF)
        df = df[df['alpha6'] > 0]

    elif variant == 'v7c_strict':
        # Require both 3m AND 6m alpha positive
        df = df[(df['alpha3'] > 0) & (df['alpha6'] > 0)]

    elif variant == 'v7c_both':
        # No alpha filter on selection, but alpha affects sector ranking
        pass  # fall through

    # Use alpha-adjusted momentum score
    if variant in ('v7c_alpha', 'v7c_strict', 'v7c_both'):
        # Scoring: blend base mom with alpha6
        df['score'] = df['mom'] * 0.6 + df['alpha6'] * 2.0 * 0.4
    else:
        df['score'] = df['mom']

    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'score'] += 0.03

    # Sector ranking (use ETF momentum if available for v7c_both)
    if variant == 'v7c_both':
        # Rank sectors by sector ETF momentum
        def get_sec_etf_mom(sec):
            etf_code = SECTOR_ETF_MAP.get(sec, 'SPY')
            if etf_code in sig['etf_r6']:
                v = get_etf_val(sig['etf_r6'][etf_code], date)
                return v if v is not None else 0
            return 0
        sector_scores = {sec: get_sec_etf_mom(sec) for sec in df['sector'].unique()}
        sec_rank = pd.Series(sector_scores).sort_values(ascending=False)
    else:
        sec_rank = df.groupby('sector')['score'].mean().sort_values(ascending=False)

    if reg == 'bull':
        top_secs = sec_rank.head(4).index.tolist()
        sps, cash = 3, 0.0
    else:
        top_secs = sec_rank.head(3).index.tolist()
        sps, cash = 2, 0.20

    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('score', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    if not selected:
        return {}, reg

    # Blended weighting (same as v3b: 70% inv-vol + 30% momentum)
    iv = {t: 1.0 / max(df.loc[t, 'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values())
    iv_w = {t: v / iv_t for t, v in iv.items()}

    mn = min(df.loc[t, 'score'] for t in selected)
    sh = max(-mn + 0.01, 0)
    mw = {t: df.loc[t, 'score'] + sh for t in selected}
    mw_t = sum(mw.values())
    mw_w = {t: v / mw_t for t, v in mw.items()}

    invested = 1.0 - cash
    weights = {t: (0.70 * iv_w[t] + 0.30 * mw_w[t]) * invested for t in selected}
    return weights, reg


# ── GLD hedge (v4d params) ───────────────────────────────────────────────────

DD_PARAMS = {-0.08: 0.30, -0.12: 0.50, -0.18: 0.60}

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

def backtest(close_df, sig, sectors, gld, variant='v7c_alpha',
             start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i + 1]
        dd = (val - peak) / peak if peak > 0 else 0

        w, _ = select_v7c(sig, sectors, dt, prev_h, variant)
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
    print("🐻 动量轮动 v7c — Stock Alpha Momentum (相对行业ETF超额收益)")
    print("=" * 70)

    tickers = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld      = load_etf('GLD')
    print(f"  Loaded {len(close_df.columns)} stocks")

    # Load sector ETFs for alpha calculation
    etf_codes = set(SECTOR_ETF_MAP.values())
    etf_prices = {}
    for etf in etf_codes:
        p = load_etf(etf)
        if p is not None:
            etf_prices[etf] = p
            print(f"  ETF {etf}: {len(p)} days")

    sig = precompute(close_df, etf_prices)

    VARIANTS = {
        'base':      'v3b+DD (baseline)',
        'v7c_alpha': 'v7c Alpha(6m>0)',
        'v7c_strict':'v7c Alpha(3m&6m>0)',
        'v7c_both':  'v7c ETF-rank+Alpha',
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
    print(f"{'Variant':<26} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'IS Sh':>7} {'OOS Sh':>7} {'WF':>6} {'Comp':>8}")
    print("-" * 105)
    for var, label in VARIANTS.items():
        r = results[var]; m = r['m']
        flag = '✅' if r['wf'] >= 0.7 else '❌'
        print(f"{label:<26} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['calmar']:>8.2f} "
              f"{r['is_m']['sharpe']:>7.2f} {r['oos_m']['sharpe']:>7.2f} "
              f"{r['wf']:>5.2f}{flag} {r['comp']:>8.3f}")

    base_comp = results['base']['comp']
    bests = [(v, r) for v, r in results.items() if v != 'base' and r['wf'] >= 0.7]
    if bests:
        bv, br = max(bests, key=lambda x: x[1]['comp'])
        print(f"\n🏆 Best v7c: {VARIANTS[bv]} → Comp {br['comp']:.3f} "
              f"(vs baseline {base_comp:.3f}, Δ{br['comp']-base_comp:+.3f})")
        if br['comp'] > 1.8 or br['m']['sharpe'] > 2.0:
            print("🚨🚨🚨 【重大突破】!")
        elif br['comp'] > 1.356:
            print("✅ Beats v4d champion (1.356)!")
        else:
            print(f"⚠️  No improvement over v4d champion (1.356)")

    out = {v: {'cagr': float(r['m']['cagr']), 'max_dd': float(r['m']['max_dd']),
               'sharpe': float(r['m']['sharpe']), 'calmar': float(r['m']['calmar']),
               'wf': float(r['wf']), 'composite': float(r['comp'])}
           for v, r in results.items()}
    jf = Path(__file__).parent / "momentum_v7c_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results


if __name__ == '__main__':
    main()
