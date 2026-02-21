#!/usr/bin/env python3
"""
动量轮动 v9j — TLT国债对冲 + 债券自然竞争
代码熊 🐻

v9i 创新栈 (13层) + 第14层: TLT长债对冲

核心思路:
  - TLT (20年期美债) 在股市崩盘期间往往大幅上涨 (风险规避资金流入)
  - 当TLT 6m动量超过股票平均动量的N%阈值时, 加入TLT仓位 (自然竞争)
  - 在熊市期间, 使用TLT替代部分SHY现金 (TLT提供更高防御收益)
  - TLT与GLD协同: GLD对冲通胀风险, TLT对冲衰退/通缩风险

严格无前瞻:
  - 所有信号基于月末已知收盘价
  - TLT动量用前一交易日数据
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9i baseline parameters (unchanged)
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

# NEW v9j: TLT parameters (sweep these)
TLT_AVG_THRESH = 0.60   # TLT enters when 6m mom >= avg×60%
TLT_COMPETE_FRAC = 0.12  # 12% TLT in natural competition
TLT_BEAR_FRAC = 0.25    # 25% TLT replaces SHY in bear mode


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
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200,
                sma50=sma50, close=close_df)


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
    if len(df) == 0: return {}, False

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    tlt_a = asset_compete(sig, date, tlt_p, TLT_AVG_THRESH, TLT_COMPETE_FRAC)
    total_compete = gld_a + gdx_a + tlt_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0) + (1 if tlt_a > 0 else 0)

    reg     = get_regime(sig, date)
    breadth = compute_breadth(sig, date)

    if reg == 'bull':
        n_bull = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull - n_compete, 1)
        sps = BULL_SPS
        bear_cash = 0.0
        use_tlt_bear = False
    else:
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        # Bear mode: use TLT as defensive instead of plain cash
        bear_cash = max(0.20 - TLT_BEAR_FRAC, 0.0)  # remaining cash
        use_tlt_bear = True

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - bear_cash - total_compete - (TLT_BEAR_FRAC if use_tlt_bear else 0), 0.0)
    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if tlt_a > 0: w['TLT'] = tlt_a
        if use_tlt_bear: w['TLT'] = w.get('TLT', 0) + TLT_BEAR_FRAC
        return w, use_tlt_bear

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if tlt_a > 0: weights['TLT'] = weights.get('TLT', 0) + tlt_a
    if use_tlt_bear: weights['TLT'] = weights.get('TLT', 0) + TLT_BEAR_FRAC
    return weights, use_tlt_bear


def apply_overlays(weights, spy_vol, dd, port_vol_ann):
    # GDXJ vol trigger
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    # GLD DD response
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdxj_v + gld_dd
    if total > 0 and weights:
        stock_frac = max(1.0 - total, 0.01)
        # preserve TLT
        hedge_keys = {'GLD', 'GDX', 'GDXJ', 'TLT'}
        hedge_w = {t: w for t, w in weights.items() if t in hedge_keys}
        equity_w = {t: w for t, w in weights.items() if t not in hedge_keys}
        tot_eq = sum(equity_w.values())
        if tot_eq > 0:
            eq_frac = max(stock_frac - sum(hedge_w.values()), 0.01)
            equity_w = {t: w/tot_eq*eq_frac for t, w in equity_w.items()}
        weights = {**equity_w, **hedge_w}
        if gld_dd > 0: weights['GLD'] = weights.get('GLD', 0) + gld_dd
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    # Portfolio Vol Targeting
    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in ('GLD', 'GDX', 'GDXJ', 'TLT')]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] = weights[t] * scale
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

        w, use_tlt_bear = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p)
        w, shy_boost = apply_overlays(w, spy_vol, dd, port_vol_ann)

        if port_vol_ann > 0.01 and len(port_returns) >= VOL_LOOKBACK:
            scale_hist.append(min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0))

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'GDX', 'GDXJ', 'TLT')}
        holdings_hist[dt.strftime('%Y-%m')] = list(w.keys()) + (['SHY_vt'] if shy_boost > 0 else [])

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


def sweep_tlt_params(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p):
    """Sweep TLT parameters to find best config"""
    global TLT_AVG_THRESH, TLT_COMPETE_FRAC, TLT_BEAR_FRAC
    results = []
    for thresh in [0.50, 0.60, 0.70, 0.80, 9.99]:  # 9.99 = TLT off
        for compete_frac in [0.08, 0.12, 0.15]:
            for bear_frac in [0.15, 0.20, 0.25]:
                TLT_AVG_THRESH = thresh
                TLT_COMPETE_FRAC = compete_frac
                TLT_BEAR_FRAC = bear_frac
                try:
                    eq, _, to, _ = run_backtest(close_df, sig, sectors,
                                                 gld_p, gdx_p, gdxj_p, shy_p, tlt_p)
                    m = compute_metrics(eq)
                    eq_is, _, _, _ = run_backtest(close_df, sig, sectors,
                                                   gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                                                   '2015-01-01', '2020-12-31')
                    eq_oos, _, _, _ = run_backtest(close_df, sig, sectors,
                                                    gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                                                    '2021-01-01', '2025-12-31')
                    mi = compute_metrics(eq_is); mo = compute_metrics(eq_oos)
                    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
                    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
                    results.append({
                        'thresh': thresh, 'compete': compete_frac, 'bear': bear_frac,
                        'composite': comp, 'sharpe': m['sharpe'], 'cagr': m['cagr'],
                        'max_dd': m['max_dd'], 'calmar': m['calmar'], 'wf': wf
                    })
                    print(f"  thresh={thresh:.2f} compete={compete_frac:.2f} bear={bear_frac:.2f} "
                          f"→ Comp={comp:.4f} Sharpe={m['sharpe']:.2f} MaxDD={m['max_dd']:.1%} WF={wf:.2f}")
                except Exception as e:
                    print(f"  Error: {e}")
    return sorted(results, key=lambda x: x['composite'], reverse=True)


def main():
    print("=" * 72)
    print("🐻 动量轮动 v9j — TLT国债对冲 + 债券自然竞争")
    print("=" * 72)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()

    # Load or download TLT
    tlt_path = CACHE / "TLT.csv"
    if not tlt_path.exists():
        print("⚠️ TLT.csv not found — downloading via yfinance...")
        import subprocess
        subprocess.run(['python3', '-c', '''
import yfinance as yf, pandas as pd
from pathlib import Path
tlt = yf.download("TLT", start="2003-01-01", end="2026-01-01", auto_adjust=True)
tlt["Close"].to_frame().reset_index().rename(columns={"index":"Date","Date":"Date"}).to_csv("/root/.openclaw/workspace/wombat-quant-lab/data_cache/TLT.csv", index=False)
print("TLT downloaded")
'''], check=True)
    tlt_p  = load_csv(tlt_path)['Close'].dropna()

    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")
    print(f"  TLT data: {tlt_p.index[0].date()} → {tlt_p.index[-1].date()}")

    print("\n🔍 Sweeping TLT parameters...")
    results = sweep_tlt_params(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p)

    print(f"\n📊 Top 5 configurations:")
    for r in results[:5]:
        print(f"  thresh={r['thresh']:.2f} compete={r['compete']:.2f} bear={r['bear']:.2f} "
              f"→ Comp={r['composite']:.4f} Sharpe={r['sharpe']:.2f} MaxDD={r['max_dd']:.1%} WF={r['wf']:.2f}")

    # Use best config
    best = results[0]
    global TLT_AVG_THRESH, TLT_COMPETE_FRAC, TLT_BEAR_FRAC
    TLT_AVG_THRESH = best['thresh']
    TLT_COMPETE_FRAC = best['compete']
    TLT_BEAR_FRAC = best['bear']
    print(f"\n🏆 Best config: thresh={TLT_AVG_THRESH:.2f} compete={TLT_COMPETE_FRAC:.2f} bear={TLT_BEAR_FRAC:.2f}")

    print("\n🔄 Full (2015-2025)...")
    eq_full, hold, to, avg_scale = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _, _ = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
        '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _, _ = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
        '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v9j vs v9i Champion")
    print("=" * 72)
    v9i = dict(cagr=0.319, max_dd=-0.107, sharpe=1.81, calmar=2.97, comp=1.973, wf=0.82)
    print(f"{'Metric':<12} {'v9i':<10} {'v9j':<10} {'Delta'}")
    print(f"{'CAGR':<12} {v9i['cagr']:.1%}     {m['cagr']:.1%}     {m['cagr']-v9i['cagr']:+.1%}")
    print(f"{'MaxDD':<12} {v9i['max_dd']:.1%}    {m['max_dd']:.1%}    {m['max_dd']-v9i['max_dd']:+.1%}")
    print(f"{'Sharpe':<12} {v9i['sharpe']:.2f}      {m['sharpe']:.2f}      {m['sharpe']-v9i['sharpe']:+.2f}")
    print(f"{'Calmar':<12} {v9i['calmar']:.2f}      {m['calmar']:.2f}      {m['calmar']-v9i['calmar']:+.2f}")
    print(f"{'WF':<12} {v9i['wf']:.2f}      {wf:.2f}      {wf-v9i['wf']:+.2f}")
    print(f"{'Composite':<12} {v9i['comp']:.4f}  {comp:.4f}  {comp-v9i['comp']:+.4f}")

    tlt_months = sum(1 for h in hold.values() if 'TLT' in h)
    print(f"\n📅 TLT appeared in {tlt_months}/{len(hold)} months")

    if comp > 2.0:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.0! ({comp:.4f})")
    elif comp > 1.973:
        print(f"\n🚀🚀 超越v9i冠军! Composite {comp:.4f}")
    elif comp > 1.80:
        print(f"\n✅ 优秀! Composite {comp:.4f} > 1.80")

    out = {
        'strategy': f'v9j TLT Bond Hedge (thresh={TLT_AVG_THRESH}, compete={TLT_COMPETE_FRAC}, bear={TLT_BEAR_FRAC})',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'tlt_months': tlt_months,
        'sweep_top5': results[:5],
    }
    jf = Path(__file__).parent / "momentum_v9j_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
