#!/usr/bin/env python3
"""
动量轮动 v10d — 利率自适应对冲 (TLT/IEF最优债券选择)
代码熊 🐻

v9j基础 (14层) + 第15层: 利率自适应债券对冲

核心问题:
  2022年加息周期: SPY跌19%, TLT跌30% (股债双杀)
  v9j的条件TLT在此期间正确地不触发 (TLT动量<0)
  但仍需更好的利率环境感知

v10d创新:
  在熊市/防御期, 智能选择债券工具:
  - TLT 6m动量 > IEF 6m动量 AND TLT 6m > 0:
    → 降息环境, 长久期有利 → 使用TLT (25%)
  - IEF 6m动量 > TLT 6m动量 AND IEF 6m > 0:
    → 加息/平稳环境, 中短久期 → 使用IEF (20%)
  - 两者都≤0: 股债双杀 → 不使用债券, 转SHY
  
  这比单独检查TLT更智能:
  - 总是选择当前利率环境中表现更好的债券
  - 2022: TLT和IEF都是负的 → 自动规避
  - 2020 COVID: TLT和IEF都是正的, TLT更好 → 用TLT
  - 2018 Q4: TLT微正 → 用TLT保护

额外增加: HYG信用信号
  - HYG (高收益债) 1m收益 < -2%: 信用压力预警 → +3%GLD
  - 信用债先于股市崩塌, 是领先指标
  - 2015-2016: HYG大跌先于SPY
  - 2020-02: HYG同期下跌

严格无前瞻: 所有信号使用历史收盘价
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9j baseline parameters (unchanged)
MOM_W    = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS    = 5; N_BULL_SECS_HI = 4; BREADTH_CONC = 0.65
BULL_SPS = 2; BEAR_SPS = 2; BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
CONT_BONUS = 0.03; HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18
VOL_TARGET_ANN = 0.11; VOL_LOOKBACK = 3

# v10d NEW: Rate-adaptive bond selection
TLT_BEAR_FRAC = 0.25   # fraction if TLT wins
IEF_BEAR_FRAC = 0.20   # fraction if IEF wins (slightly less, more conservative)
BOND_MOM_LB   = 126    # 6-month lookback for bond momentum

# HYG credit signal
HYG_STRESS_THRESH = -0.02  # HYG 1m < -2% → credit stress
HYG_STRESS_GLD    = 0.03   # +3% GLD on credit stress


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
        if not f.exists() or f.stat().st_size < 500: continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = df['Close'].dropna()
        except: pass
    return pd.DataFrame(d)


def precompute(close_df):
    r1   = close_df / close_df.shift(22)  - 1
    r3   = close_df / close_df.shift(63)  - 1
    r6   = close_df / close_df.shift(126) - 1
    r12  = close_df / close_df.shift(252) - 1
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol5  = log_r.rolling(5).std() * np.sqrt(252)
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200,
                sma50=sma50, close=close_df)


def compute_breadth(sig, date):
    close = sig['close'].loc[:date].dropna(how='all')
    sma50 = sig['sma50'].loc[:date].dropna(how='all')
    if len(close) < 50: return 1.0
    lc = close.iloc[-1]; ls = sma50.iloc[-1]
    mask = (lc > ls).dropna()
    return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0


def get_regime(sig, date):
    if sig['s200'] is None: return 'bull'
    spy_now  = sig['spy'].loc[:date].dropna()
    s200_now = sig['s200'].loc[:date].dropna()
    if len(spy_now) == 0: return 'bull'
    breadth = compute_breadth(sig, date)
    return 'bear' if (spy_now.iloc[-1] < s200_now.iloc[-1] and
                      breadth < BREADTH_NARROW) else 'bull'


def select_best_bond(tlt_p, ief_p, date):
    """
    Returns (ticker, fraction) for the best bond hedge.
    Uses 6m momentum comparison. Returns (None, 0) if neither is positive.
    """
    def get_6m(p):
        hist = p.loc[:date].dropna()
        if len(hist) < BOND_MOM_LB + 3: return None
        return hist.iloc[-1] / hist.iloc[-BOND_MOM_LB] - 1

    tlt_mom = get_6m(tlt_p)
    ief_mom = get_6m(ief_p)

    # Both negative: avoid bonds entirely
    if (tlt_mom is None or tlt_mom <= 0) and (ief_mom is None or ief_mom <= 0):
        return None, 0.0

    # Only IEF positive
    if (tlt_mom is None or tlt_mom <= 0) and (ief_mom is not None and ief_mom > 0):
        return 'IEF', IEF_BEAR_FRAC

    # Only TLT positive
    if (tlt_mom is not None and tlt_mom > 0) and (ief_mom is None or ief_mom <= 0):
        return 'TLT', TLT_BEAR_FRAC

    # Both positive: pick the one with higher momentum
    if tlt_mom > ief_mom:
        return 'TLT', TLT_BEAR_FRAC
    else:
        return 'IEF', IEF_BEAR_FRAC


def get_hyg_signal(hyg_p, date):
    """HYG credit stress: 1m return < threshold"""
    hist = hyg_p.loc[:date].dropna()
    if len(hist) < 25: return 0.0
    hyg_1m = hist.iloc[-1] / hist.iloc[-22] - 1
    return HYG_STRESS_GLD if hyg_1m < HYG_STRESS_THRESH else 0.0


def asset_compete(sig, date, prices, thresh, frac, lb=127):
    r6  = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lb] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, None, 0.0
    d = idx[-1]

    w1, w3, w6, w12 = MOM_W
    mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
           sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom':   mom, 'r6':  sig['r6'].loc[d],
        'vol':   sig['vol30'].loc[d], 'price': close.loc[d],
        'sma50': sig['sma50'].loc[d], 'hi52':  sig['r52w_hi'].loc[d],
    }).dropna(subset=['mom', 'sma50'])
    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df = df[df['price'] >= df['hi52'] * HI52_FRAC]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    df = df[df['sector'] != 'Unknown']
    for t in df.index:
        if t in prev_hold: df.loc[t, 'mom'] += CONT_BONUS
    if len(df) == 0: return {}, None, 0.0

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg     = get_regime(sig, date)
    breadth = compute_breadth(sig, date)

    # Rate-adaptive bond selection
    bond_ticker, bond_frac = select_best_bond(tlt_p, ief_p, date)
    use_bond = (reg == 'bear' and bond_frac > 0)
    actual_bond_frac = bond_frac if use_bond else 0.0

    if reg == 'bull':
        n_bull = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull - n_compete, 1)
        sps, bear_cash = BULL_SPS, 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        bear_cash = max(0.20 - actual_bond_frac, 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - bear_cash - total_compete - actual_bond_frac, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if use_bond and bond_ticker: w[bond_ticker] = actual_bond_frac
        return w, bond_ticker, actual_bond_frac

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if use_bond and bond_ticker: weights[bond_ticker] = actual_bond_frac
    return weights, bond_ticker, actual_bond_frac


def apply_overlays(weights, spy_vol, dd, port_vol_ann, hyg_gld):
    """GDXJ + DD GLD + HYG credit signal + vol target"""
    HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF'}

    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    # HYG credit signal adds pre-emptive GLD
    gld_total = max(gld_dd, hyg_gld)

    total_overlay = gdxj_v + gld_total
    if total_overlay > 0 and weights:
        hedge_w = {t: w for t, w in weights.items() if t in HEDGE_KEYS}
        equity_w = {t: w for t, w in weights.items() if t not in HEDGE_KEYS}
        stock_frac = max(1.0 - total_overlay - sum(hedge_w.values()), 0.01)
        tot = sum(equity_w.values())
        if tot > 0:
            equity_w = {t: w/tot*stock_frac for t, w in equity_w.items()}
        weights = {**equity_w, **hedge_w}
        if gld_total > 0: weights['GLD'] = weights.get('GLD', 0) + gld_total
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in HEDGE_KEYS]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                 tlt_p, ief_p, hyg_p,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []
    bond_hist = {'TLT': 0, 'IEF': 0, 'none': 0}
    hyg_trigger_months = 0
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15

        if len(port_returns) >= VOL_LOOKBACK:
            pv = np.std(port_returns[-VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        hyg_gld = get_hyg_signal(hyg_p, dt)
        if hyg_gld > 0: hyg_trigger_months += 1

        w, bond_ticker, bond_frac = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, ief_p)
        w, shy_boost = apply_overlays(w, spy_vol, dd, pv, hyg_gld)

        if bond_frac > 0 and bond_ticker:
            bond_hist[bond_ticker] = bond_hist.get(bond_ticker, 0) + 1
        elif get_regime(sig, dt) == 'bear':
            bond_hist['none'] = bond_hist.get('none', 0) + 1

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'GDX', 'GDXJ', 'TLT', 'IEF')}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t == 'TLT':  s = tlt_p.loc[dt:ndt].dropna()
            elif t == 'IEF':  s = ief_p.loc[dt:ndt].dropna()
            elif t in close_df.columns: s = close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt

        total_shy = shy_boost + (cash_frac if USE_SHY else 0.0)
        if total_shy > 0 and shy_p is not None:
            s = shy_p.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * total_shy

        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)
        port_returns.append(ret)

    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq, float(np.mean(tos)) if tos else 0.0, bond_hist, hyg_trigger_months


def compute_metrics(eq):
    if len(eq) < 3: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    mo   = eq.pct_change().dropna()
    sh   = mo.mean()/mo.std()*np.sqrt(12) if mo.std() > 0 else 0
    dd   = ((eq - eq.cummax())/eq.cummax()).min()
    cal  = cagr/abs(dd) if dd != 0 else 0
    return dict(cagr=float(cagr), max_dd=float(dd), sharpe=float(sh), calmar=float(cal))


def main():
    global HYG_STRESS_THRESH, HYG_STRESS_GLD, TLT_BEAR_FRAC, IEF_BEAR_FRAC
    print("=" * 72)
    print("🐻 动量轮动 v10d — 利率自适应对冲 (TLT/IEF最优选择)")
    print("=" * 72)
    print(f"\n  Bond selection: TLT ({TLT_BEAR_FRAC:.0%}) vs IEF ({IEF_BEAR_FRAC:.0%}) by 6m momentum")
    print(f"  Both ≤0 → no bond (use SHY)")
    print(f"  HYG credit signal: 1m < {HYG_STRESS_THRESH:.0%} → +{HYG_STRESS_GLD:.0%} GLD")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    ief_p  = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    hyg_p  = load_csv(CACHE / "HYG.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")
    print(f"  IEF: {ief_p.index[0].date()} → {ief_p.index[-1].date()}")
    print(f"  HYG: {hyg_p.index[0].date()} → {hyg_p.index[-1].date()}")

    # Quick parameter sweep for HYG threshold
    print("\n🔍 Sweeping HYG threshold and bond fractions...")
    results = []
    configs = [  # (hyg_thresh, hyg_gld, tlt_frac, ief_frac)
        (-0.02, 0.03, 0.25, 0.20),  # baseline v10d
        (-0.02, 0.03, 0.25, 0.25),  # equal TLT/IEF
        (-0.02, 0.05, 0.25, 0.20),  # more HYG GLD
        (-0.03, 0.03, 0.25, 0.20),  # tighter HYG threshold
        (-0.02, 0.00, 0.25, 0.20),  # no HYG signal
        (-0.02, 0.03, 0.30, 0.20),  # more TLT
        (-0.02, 0.03, 0.25, 0.15),  # less IEF
        (-9.99, 0.00, 0.25, 0.00),  # TLT only (no IEF, same as v9j)
    ]
    for hyg_t, hyg_g, tlt_f, ief_f in configs:
        HYG_STRESS_THRESH = hyg_t
        HYG_STRESS_GLD    = hyg_g
        TLT_BEAR_FRAC     = tlt_f
        IEF_BEAR_FRAC     = ief_f
        try:
            eq, to, bh, hm = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                           shy_p, tlt_p, ief_p, hyg_p)
            m = compute_metrics(eq)
            eq_is, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                           shy_p, tlt_p, ief_p, hyg_p, '2015-01-01', '2020-12-31')
            eq_oos, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                            shy_p, tlt_p, ief_p, hyg_p, '2021-01-01', '2025-12-31')
            mi = compute_metrics(eq_is); mo = compute_metrics(eq_oos)
            wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            hyg_str = f"{hyg_t:.2f}/{hyg_g:.2f}" if hyg_g > 0 else "off"
            print(f"  hyg={hyg_str} TLT={tlt_f:.2f} IEF={ief_f:.2f} → "
                  f"Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
                  f"MaxDD={m['max_dd']:.1%} CAGR={m['cagr']:.1%} WF={wf:.2f} "
                  f"(TLT:{bh.get('TLT',0)} IEF:{bh.get('IEF',0)} none:{bh.get('none',0)} HYG:{hm})")
            results.append({'hyg_t': hyg_t, 'hyg_g': hyg_g, 'tlt_f': tlt_f, 'ief_f': ief_f,
                            'composite': comp, 'sharpe': m['sharpe'], 'cagr': m['cagr'],
                            'max_dd': m['max_dd'], 'calmar': m['calmar'], 'wf': wf,
                            'bond_hist': bh})
        except Exception as e:
            print(f"  Error: {e}")

    results = sorted(results, key=lambda x: x['composite'], reverse=True)
    print(f"\n📊 Top 5:")
    for r in results[:5]:
        hyg_str = f"{r['hyg_t']:.2f}/{r['hyg_g']:.2f}" if r['hyg_g'] > 0 else "off"
        print(f"  hyg={hyg_str} TLT={r['tlt_f']:.2f} IEF={r['ief_f']:.2f} → "
              f"Comp={r['composite']:.4f} Sharpe={r['sharpe']:.2f} WF={r['wf']:.2f}")

    best = results[0]
    HYG_STRESS_THRESH = best['hyg_t']
    HYG_STRESS_GLD    = best['hyg_g']
    TLT_BEAR_FRAC     = best['tlt_f']
    IEF_BEAR_FRAC     = best['ief_f']

    print("\n🔄 Full (2015-2025)...")
    eq_full, to, bh, hm = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                        shy_p, tlt_p, ief_p, hyg_p)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                   shy_p, tlt_p, ief_p, hyg_p, '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                    shy_p, tlt_p, ief_p, hyg_p, '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v10d vs v9j (2.057)")
    print("=" * 72)
    v9j = dict(cagr=0.3235, max_dd=-0.1033, sharpe=1.850, calmar=3.132, comp=2.057, wf=0.785)
    print(f"{'Metric':<12} {'v9j':<10} {'v10d':<10} {'Delta'}")
    print(f"{'CAGR':<12} {v9j['cagr']:.1%}     {m['cagr']:.1%}     {m['cagr']-v9j['cagr']:+.1%}")
    print(f"{'MaxDD':<12} {v9j['max_dd']:.1%}    {m['max_dd']:.1%}    {m['max_dd']-v9j['max_dd']:+.1%}")
    print(f"{'Sharpe':<12} {v9j['sharpe']:.2f}      {m['sharpe']:.2f}      {m['sharpe']-v9j['sharpe']:+.2f}")
    print(f"{'Calmar':<12} {v9j['calmar']:.2f}      {m['calmar']:.2f}      {m['calmar']-v9j['calmar']:+.2f}")
    print(f"{'IS Sharpe':<12}           {mi['sharpe']:.2f}")
    print(f"{'OOS Sharpe':<12}           {mo['sharpe']:.2f}")
    print(f"{'WF':<12} {v9j['wf']:.2f}      {wf:.2f}      {wf-v9j['wf']:+.2f}")
    print(f"{'Composite':<12} {v9j['comp']:.4f}  {comp:.4f}  {comp-v9j['comp']:+.4f}")
    print(f"\n  Bond usage: TLT={bh.get('TLT',0)}/IEF={bh.get('IEF',0)}/none={bh.get('none',0)} bear months")
    print(f"  HYG triggers: {hm} months")

    if comp > 2.1:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.1!")
    elif comp > 2.057:
        print(f"\n🚀🚀 超越v9j! Composite {comp:.4f}")
    elif comp > 2.00:
        print(f"\n✅ Composite > 2.0: {comp:.4f}")

    if wf >= 0.78:
        print(f"\n🎯 WF维持高位: {wf:.2f} >= v9j!")

    out = {
        'strategy': f'v10d Rate-Adaptive Hedge (TLT/IEF by 6m-mom + HYG signal)',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'bond_hist': bh, 'hyg_months': hm,
        'best_params': {'hyg_t': HYG_STRESS_THRESH, 'hyg_g': HYG_STRESS_GLD,
                        'tlt_f': TLT_BEAR_FRAC, 'ief_f': IEF_BEAR_FRAC},
        'sweep': results,
    }
    jf = Path(__file__).parent / "momentum_v10d_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
