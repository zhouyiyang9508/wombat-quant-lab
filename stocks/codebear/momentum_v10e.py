#!/usr/bin/env python3
"""
动量轮动 v10e — Sharpe调整动量排名 (风险调整选股)
代码熊 🐻

v9j基础 (14层) + 第15层: Sharpe调整动量

核心思路:
  当前选股逻辑: 按 composite_momentum 排名 (1m×20%+3m×50%+6m×20%+12m×10%)
  问题: 两只股票都有30% 6m动量, 但一只vol=20%,另一只vol=50%
        当前策略对它们同等对待, 但高vol股票风险更大
  
  v10e改进: 引入"动量质量分"
  momentum_quality = composite_momentum / max(vol, 0.15)
  
  这是一个Sharpe比率代理 (近似的"每单位风险的动量回报")
  高质量动量 = 稳定上涨而非剧烈波动
  
  实现方式:
  1. 保留所有现有过滤器 (r6>0, SMA50, 52w-hi过滤等)
  2. 用于行业排名: sector_quality = mean(momentum_quality) for stocks in sector
  3. 用于行业内选股: 按 momentum_quality 降序选股

预期效果:
  - 过滤掉高动量但高波动的股票 (例如biotech在pump阶段)
  - 降低月度收益方差 → Sharpe提升
  - MaxDD可能改善 (高vol股票在下跌时跌幅更大)

参数探索:
  - pure_sharpe: 完全用 mom/vol 排名
  - mixed: 0.7 × pure_mom + 0.3 × sharpe_adj  (混合)
  - mixed2: 0.5 × pure_mom + 0.5 × sharpe_adj  (对半)

严格无前瞻: vol使用过去30日实现波动率 (无前瞻)
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
TLT_BEAR_FRAC = 0.25; TLT_MOM_LOOKBACK = 126
VOL_TARGET_ANN = 0.11; VOL_LOOKBACK = 3

# v10e: Sharpe-adjusted ranking weight
# rank_score = RANK_MOM × mom + RANK_SHARPE × (mom/vol)
RANK_MOM    = 0.70   # weight on pure momentum
RANK_SHARPE = 0.30   # weight on Sharpe-adjusted momentum
VOL_FLOOR   = 0.15   # minimum vol for Sharpe calculation


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


def get_tlt_momentum(tlt_p, date):
    hist = tlt_p.loc[:date].dropna()
    if len(hist) < TLT_MOM_LOOKBACK + 3: return False
    return bool(hist.iloc[-1] / hist.iloc[-TLT_MOM_LOOKBACK] - 1 > 0)


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


def compute_rank_score(mom, vol):
    """Combined rank score: RANK_MOM × mom + RANK_SHARPE × (mom/vol)"""
    sharpe_adj = mom / max(vol, VOL_FLOOR)
    return RANK_MOM * mom + RANK_SHARPE * sharpe_adj


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, False
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

    # v10e KEY: Compute Sharpe-adjusted rank score
    df['rank_score'] = df.apply(lambda row: compute_rank_score(row['mom'], row['vol']), axis=1)
    if len(df) == 0: return {}, False

    # Use rank_score for sector scoring and within-sector selection
    sec_rank = df.groupby('sector')['rank_score'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg     = get_regime(sig, date)
    breadth = compute_breadth(sig, date)
    tlt_pos = get_tlt_momentum(tlt_p, date)
    use_tlt = (reg == 'bear' and tlt_pos)

    if reg == 'bull':
        n_bull = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull - n_compete, 1)
        sps, bear_cash = BULL_SPS, 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        bear_cash = max(0.20 - (TLT_BEAR_FRAC if use_tlt else 0), 0.0)

    top_secs = sec_rank.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('rank_score', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    tlt_alloc = TLT_BEAR_FRAC if use_tlt else 0.0
    stock_frac = max(1.0 - bear_cash - total_compete - tlt_alloc, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if tlt_alloc > 0: w['TLT'] = tlt_alloc
        return w, use_tlt

    # Weighting: use rank_score instead of momentum for mw (momentum-proportional component)
    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'rank_score'] for t in selected); sh = max(-mn+0.01, 0)
    rsw  = {t: df.loc[t,'rank_score']+sh for t in selected}
    rs_t = sum(rsw.values()); rsw_n = {t: v/rs_t for t, v in rsw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*rsw_n[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if tlt_alloc > 0: weights['TLT'] = tlt_alloc
    return weights, use_tlt


def apply_overlays(weights, spy_vol, dd, port_vol_ann):
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdxj_v + gld_dd
    if total > 0 and weights:
        hedge_keys = {'GLD', 'GDX', 'GDXJ', 'TLT'}
        hedge_w = {t: w for t, w in weights.items() if t in hedge_keys}
        equity_w = {t: w for t, w in weights.items() if t not in hedge_keys}
        stock_frac = max(1.0 - total - sum(hedge_w.values()), 0.01)
        tot = sum(equity_w.values())
        if tot > 0:
            equity_w = {t: w/tot*stock_frac for t, w in equity_w.items()}
        weights = {**equity_w, **hedge_w}
        if gld_dd > 0: weights['GLD'] = weights.get('GLD', 0) + gld_dd
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in ('GLD', 'GDX', 'GDXJ', 'TLT')]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []
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

        w, use_tlt = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p)
        w, shy_boost = apply_overlays(w, spy_vol, dd, pv)

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'GDX', 'GDXJ', 'TLT')}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t == 'TLT':  s = tlt_p.loc[dt:ndt].dropna()
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
    return eq, float(np.mean(tos)) if tos else 0.0


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
    global RANK_MOM, RANK_SHARPE
    print("=" * 72)
    print("🐻 动量轮动 v10e — Sharpe调整动量排名")
    print("=" * 72)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")

    print("\n🔍 Sweeping Sharpe-adjustment weight...")
    results = []
    # (rank_mom_weight, rank_sharpe_weight)
    configs = [
        (1.00, 0.00),  # pure momentum (control = v9j)
        (0.80, 0.20),
        (0.70, 0.30),
        (0.60, 0.40),
        (0.50, 0.50),
        (0.40, 0.60),
        (0.20, 0.80),
        (0.00, 1.00),  # pure sharpe-adjusted
    ]
    for rm, rs in configs:
        RANK_MOM = rm; RANK_SHARPE = rs
        try:
            eq, to = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p)
            m = compute_metrics(eq)
            eq_is, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                                     '2015-01-01', '2020-12-31')
            eq_oos, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                                      '2021-01-01', '2025-12-31')
            mi = compute_metrics(eq_is); mo = compute_metrics(eq_oos)
            wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            results.append({'rm': rm, 'rs': rs, 'composite': comp, 'sharpe': m['sharpe'],
                            'cagr': m['cagr'], 'max_dd': m['max_dd'], 'calmar': m['calmar'],
                            'wf': wf})
            label = f"mom={rm:.0%} sharpe={rs:.0%}"
            print(f"  {label} → Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
                  f"MaxDD={m['max_dd']:.1%} CAGR={m['cagr']:.1%} WF={wf:.2f}")
        except Exception as e:
            print(f"  Error ({rm}/{rs}): {e}")

    results = sorted(results, key=lambda x: x['composite'], reverse=True)
    print(f"\n📊 All results ranked:")
    for r in results:
        print(f"  mom={r['rm']:.0%} sharpe={r['rs']:.0%} → "
              f"Comp={r['composite']:.4f} Sharpe={r['sharpe']:.2f} WF={r['wf']:.2f}")

    best = results[0]
    RANK_MOM = best['rm']; RANK_SHARPE = best['rs']
    print(f"\n🏆 Best: mom={RANK_MOM:.0%} sharpe={RANK_SHARPE:.0%}")

    print("\n🔄 Full (2015-2025)...")
    eq_full, to = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p)
    print("🔄 IS (2015-2020)...")
    eq_is, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                             '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                              '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v10e vs v9j (2.057)")
    print("=" * 72)
    v9j = dict(cagr=0.3235, max_dd=-0.1033, sharpe=1.850, calmar=3.132, comp=2.057, wf=0.785)
    print(f"{'Metric':<12} {'v9j(100%mom)':<16} {'v10e(best)':<16} {'Delta'}")
    print(f"{'CAGR':<12} {v9j['cagr']:.1%}           {m['cagr']:.1%}           {m['cagr']-v9j['cagr']:+.1%}")
    print(f"{'MaxDD':<12} {v9j['max_dd']:.1%}          {m['max_dd']:.1%}          {m['max_dd']-v9j['max_dd']:+.1%}")
    print(f"{'Sharpe':<12} {v9j['sharpe']:.2f}             {m['sharpe']:.2f}             {m['sharpe']-v9j['sharpe']:+.2f}")
    print(f"{'Calmar':<12} {v9j['calmar']:.2f}             {m['calmar']:.2f}             {m['calmar']-v9j['calmar']:+.2f}")
    print(f"{'IS Sharpe':<12}                  {mi['sharpe']:.2f}")
    print(f"{'OOS Sharpe':<12}                  {mo['sharpe']:.2f}")
    print(f"{'WF':<12} {v9j['wf']:.2f}             {wf:.2f}             {wf-v9j['wf']:+.2f}")
    print(f"{'Composite':<12} {v9j['comp']:.4f}         {comp:.4f}         {comp-v9j['comp']:+.4f}")
    print(f"\n  Best config: mom={RANK_MOM:.0%} sharpe={RANK_SHARPE:.0%}")

    if comp > 2.1:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.1!")
    elif comp > 2.057:
        print(f"\n🚀🚀 超越v9j! Composite {comp:.4f}")
    elif comp > 2.00:
        print(f"\n✅ Composite > 2.0: {comp:.4f}")

    if wf >= 0.80:
        print(f"\n🎯 WF提升: {wf:.2f} > 0.80!")
    elif wf >= 0.78:
        print(f"\n🎯 WF维持: {wf:.2f} (保住v9j水平)")

    out = {
        'strategy': f'v10e Sharpe-sorted (mom={RANK_MOM:.0%}, sharpe={RANK_SHARPE:.0%})',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'best_config': {'rank_mom': RANK_MOM, 'rank_sharpe': RANK_SHARPE},
        'sweep': results,
    }
    jf = Path(__file__).parent / "momentum_v10e_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
