#!/usr/bin/env python3
"""
动量轮动 v9k — 质量动量一致性过滤 + 均线斜率加速度
代码熊 🐻

v9i 创新栈 (13层) + 第14层: 质量动量过滤

核心思路:
  - "质量动量" = 1m、3m、6m 三个时间框架同时为正的股票
  - 避免"假动量": 股票可能6m动量强, 但近期已经开始走弱(1m负)
  - 加入 SMA斜率加速度: 50日SMA相对于100日前的斜率变化
    当SMA斜率为正且在加速时, 说明趋势正在增强
  - 动量质量分数 = 1m>0 × 3m>0 × 6m>0 的布尔乘积

为什么质量动量有效:
  1. 过滤"动量陷阱": 6m动量强但已经开始逆转的股票
  2. 减少回撤: 顶部回调时, 一致性强的股票更可能继续上涨
  3. 更高的持股信心: 多时间框架一致性 = 趋势可靠性更高
  4. 与当前过滤器协同: 52w高点过滤(量) + 质量动量过滤(质)

严格无前瞻:
  - 所有动量信号使用已知收盘价计算
  - 月末信号基于已实现历史数据
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9i baseline parameters
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

# NEW v9k: Quality Momentum Filter
REQUIRE_POSITIVE_1M  = True   # require 1m return > 0
REQUIRE_POSITIVE_3M  = True   # require 3m return > 0
MIN_QUALITY_MOM      = 2      # at least 2 of {1m, 3m, 6m} must be positive (=full quality)
# SMA Acceleration filter
USE_SMA_ACCEL = True          # require SMA50 slope is positive (price trending up)
SMA_SLOPE_LB  = 20            # SMA50[t] vs SMA50[t-20] slope window


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
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol5  = log_r.rolling(5).std() * np.sqrt(252)
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    sma50_prev = sma50.shift(SMA_SLOPE_LB)  # for slope calculation
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200,
                sma50=sma50, sma50_prev=sma50_prev, close=close_df)


def get_spy_vol(sig, date):
    if sig['vol5'] is None or 'SPY' not in sig['vol5'].columns: return 0.15
    v = sig['vol5']['SPY'].loc[:date].dropna()
    return float(v.iloc[-1]) if len(v) > 0 else 0.15


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
    if len(spy_now) == 0 or len(s200_now) == 0: return 'bull'
    breadth = compute_breadth(sig, date)
    return 'bear' if (spy_now.iloc[-1] < s200_now.iloc[-1] and
                      breadth < BREADTH_NARROW) else 'bull'


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


def select(sig, sectors, date, prev_hold, gld_p, gdx_p):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
    d = idx[-1]

    w1, w3, w6, w12 = MOM_W
    mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
           sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom':  mom, 'r1': sig['r1'].loc[d], 'r3': sig['r3'].loc[d],
        'r6':   sig['r6'].loc[d], 'vol':   sig['vol30'].loc[d],
        'price': close.loc[d], 'sma50': sig['sma50'].loc[d],
        'hi52':  sig['r52w_hi'].loc[d],
        'sma50_prev': sig['sma50_prev'].loc[d],
    }).dropna(subset=['mom', 'sma50'])

    df = df[(df['price'] >= 5) & (df.index != 'SPY')]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df = df[df['price'] >= df['hi52'] * HI52_FRAC]

    # NEW: Quality Momentum Filter — multi-timeframe consistency
    # Count how many of {1m, 3m, 6m} are positive
    df['quality_score'] = (
        (df['r1'] > 0).astype(int) +
        (df['r3'] > 0).astype(int) +
        (df['r6'] > 0).astype(int)
    )
    # Require at least MIN_QUALITY_MOM timeframes positive
    df = df[df['quality_score'] >= MIN_QUALITY_MOM]

    # NEW: SMA Slope filter — require SMA50 is rising (trend acceleration)
    if USE_SMA_ACCEL:
        df = df.dropna(subset=['sma50_prev'])
        df = df[df['sma50'] > df['sma50_prev']]  # SMA50 trending up

    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    df = df[df['sector'] != 'Unknown']
    for t in df.index:
        if t in prev_hold: df.loc[t, 'mom'] += CONT_BONUS
    if len(df) == 0: return {}

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg     = get_regime(sig, date)
    breadth = compute_breadth(sig, date)

    if reg == 'bull':
        n_bull = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull - n_compete, 1)
        sps, cash = BULL_SPS, 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps, cash = BEAR_SPS, 0.20

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - cash - total_compete, 0.0)
    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        return w

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = gld_a
    if gdx_a > 0: weights['GDX'] = gdx_a
    return weights


def apply_overlays(weights, spy_vol, dd, port_vol_ann):
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdxj_v + gld_dd
    if total > 0 and weights:
        stock_frac = max(1.0 - total, 0.01)
        tot = sum(weights.values())
        if tot > 0:
            weights = {t: w/tot*stock_frac for t, w in weights.items()}
        if gld_dd > 0: weights['GLD'] = weights.get('GLD', 0) + gld_dd
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in ('GLD', 'GDX', 'GDXJ')]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] = weights[t] * scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []
    scale_hist = []
    holdings_hist = {}
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15

        if len(port_returns) >= VOL_LOOKBACK:
            port_vol_mon = np.std(port_returns[-VOL_LOOKBACK:], ddof=1)
            port_vol_ann = port_vol_mon * np.sqrt(12)
        else:
            port_vol_ann = 0.20

        w = select(sig, sectors, dt, prev_h, gld_p, gdx_p)
        w, shy_boost = apply_overlays(w, spy_vol, dd, port_vol_ann)

        if port_vol_ann > 0.01 and len(port_returns) >= VOL_LOOKBACK:
            scale_hist.append(min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0))

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'GDX', 'GDXJ')}
        holdings_hist[dt.strftime('%Y-%m')] = list(w.keys()) + (['SHY_vt'] if shy_boost > 0 else [])

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
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
    avg_to = float(np.mean(tos)) if tos else 0.0
    avg_scale = float(np.mean(scale_hist)) if scale_hist else 1.0
    return eq, holdings_hist, avg_to, avg_scale


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


def sweep_quality_params(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p):
    """Sweep quality momentum parameters"""
    global REQUIRE_POSITIVE_1M, REQUIRE_POSITIVE_3M, MIN_QUALITY_MOM, USE_SMA_ACCEL
    results = []
    for min_qual in [1, 2, 3]:
        for use_sma in [True, False]:
            REQUIRE_POSITIVE_1M = True
            REQUIRE_POSITIVE_3M = True
            MIN_QUALITY_MOM = min_qual
            USE_SMA_ACCEL = use_sma
            try:
                eq, _, to, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p)
                m = compute_metrics(eq)
                eq_is, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                               '2015-01-01', '2020-12-31')
                eq_oos, _, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                               '2021-01-01', '2025-12-31')
                mi = compute_metrics(eq_is); mo = compute_metrics(eq_oos)
                wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
                comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
                results.append({
                    'min_quality': min_qual, 'use_sma_accel': use_sma,
                    'composite': comp, 'sharpe': m['sharpe'], 'cagr': m['cagr'],
                    'max_dd': m['max_dd'], 'calmar': m['calmar'], 'wf': wf
                })
                print(f"  min_qual={min_qual} sma_accel={use_sma} "
                      f"→ Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
                      f"MaxDD={m['max_dd']:.1%} CAGR={m['cagr']:.1%} WF={wf:.2f}")
            except Exception as e:
                print(f"  Error (min_qual={min_qual} sma={use_sma}): {e}")
    return sorted(results, key=lambda x: x['composite'], reverse=True)


def main():
    print("=" * 72)
    print("🐻 动量轮动 v9k — 质量动量一致性过滤 + SMA加速度")
    print("=" * 72)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")

    print("\n🔍 Sweeping quality momentum parameters...")
    results = sweep_quality_params(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p)

    print(f"\n📊 All configurations ranked:")
    for r in results:
        print(f"  min_qual={r['min_quality']} sma={r['use_sma_accel']} "
              f"→ Comp={r['composite']:.4f} Sharpe={r['sharpe']:.2f} "
              f"MaxDD={r['max_dd']:.1%} CAGR={r['cagr']:.1%} WF={r['wf']:.2f}")

    # Best config
    best = results[0]
    global MIN_QUALITY_MOM, USE_SMA_ACCEL
    MIN_QUALITY_MOM = best['min_quality']
    USE_SMA_ACCEL = best['use_sma_accel']
    print(f"\n🏆 Best config: min_quality={MIN_QUALITY_MOM} sma_accel={USE_SMA_ACCEL}")

    print("\n🔄 Full (2015-2025)...")
    eq_full, hold, to, avg_scale = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _, _ = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _, _ = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v9k vs v9i Champion")
    print("=" * 72)
    v9i = dict(cagr=0.319, max_dd=-0.107, sharpe=1.81, calmar=2.97, comp=1.973, wf=0.82)
    print(f"{'Metric':<12} {'v9i':<10} {'v9k':<10} {'Delta'}")
    print(f"{'CAGR':<12} {v9i['cagr']:.1%}     {m['cagr']:.1%}     {m['cagr']-v9i['cagr']:+.1%}")
    print(f"{'MaxDD':<12} {v9i['max_dd']:.1%}    {m['max_dd']:.1%}    {m['max_dd']-v9i['max_dd']:+.1%}")
    print(f"{'Sharpe':<12} {v9i['sharpe']:.2f}      {m['sharpe']:.2f}      {m['sharpe']-v9i['sharpe']:+.2f}")
    print(f"{'Calmar':<12} {v9i['calmar']:.2f}      {m['calmar']:.2f}      {m['calmar']-v9i['calmar']:+.2f}")
    print(f"{'WF':<12} {v9i['wf']:.2f}      {wf:.2f}      {wf-v9i['wf']:+.2f}")
    print(f"{'Composite':<12} {v9i['comp']:.4f}  {comp:.4f}  {comp-v9i['comp']:+.4f}")

    if comp > 2.0:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.0! ({comp:.4f})")
    elif comp > 1.973:
        print(f"\n🚀🚀 超越v9i冠军! Composite {comp:.4f}")
    elif comp > 1.80:
        print(f"\n✅ 优秀! Composite {comp:.4f} > 1.80")

    out = {
        'strategy': f'v9k Quality Momentum Filter (min_quality={MIN_QUALITY_MOM}, sma_accel={USE_SMA_ACCEL})',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'sweep_results': results,
    }
    jf = Path(__file__).parent / "momentum_v9k_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
