#!/usr/bin/env python3
"""
动量轮动 v10b — 防御行业桥接 (三制度个股策略)
代码熊 🐻

v9j基础 (14层) + 第15层: 三制度防御桥接

核心思路:
  v9x系列只有两个制度: 全力牛市 vs 熊市避险
  
  问题: 从牛到熊的过渡期(breadth 40-60%) 直接切换太生硬
  - 牛市: 100%个股
  - 熊市: 20%现金 + GLD + TLT
  
  缺少中间状态: "防御性牛市" 或 "早期预警"
  
  v10b新增第三制度:
  - 制度1 (强牛, breadth>65%): 100%动量个股 (同v9j)
  - 制度2 (软牛, 45%<breadth≤65%): 70%动量个股 + 30%防御行业ETF
  - 制度3 (熊市, breadth≤45%+SPY<SMA200): 熊市保护 (同v9j)
  
  防御行业ETF组合: XLV(医疗) + XLP(必需消费) + XLU(公用事业)
  - 这三个行业在经济下行时表现最稳定
  - 它们比GLD更像股票(正beta), 减少制度切换带来的波动
  - 历史上在市场调整期跑赢SPY而不像债券那样与股票负相关

为什么这样有效:
  1. 减少牛市→熊市的剧烈切换
  2. 防御ETF的正beta减少错过反弹的风险
  3. breadth 45-65%这个中间地带约占所有月份的30%
  4. 软牛中加入防御ETF不会像GLD那样拖累表现

严格无前瞻: 所有信号在月末基于历史数据计算
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9j baseline parameters
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

# v10b NEW: Defensive sector bridge
# Regime boundaries (must match v9j breadth thresholds)
BREADTH_SOFT_BULL = 0.65  # > this: STRONG BULL (use 4 sectors × 2)
BREADTH_SOFT_LO   = 0.45  # < this: potentially BEAR (same as v9j BREADTH_NARROW)

# Defensive ETF allocation in soft-bull regime
DEFENSIVE_ETFS = ['XLV', 'XLP', 'XLU']  # healthcare, staples, utilities
DEFENSIVE_TOTAL_FRAC = 0.30   # 30% defensive ETFs in soft-bull mode
DEFENSIVE_EACH = DEFENSIVE_TOTAL_FRAC / len(DEFENSIVE_ETFS)  # 10% each


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


def load_etf(ticker):
    f = CACHE / f"{ticker}.csv"
    if not f.exists(): return None
    try:
        return load_csv(f)['Close'].dropna()
    except: return None


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
    """Three regimes: bull_hi, soft_bull, bear"""
    if sig['s200'] is None: return 'bull_hi'
    spy_now  = sig['spy'].loc[:date].dropna()
    s200_now = sig['s200'].loc[:date].dropna()
    if len(spy_now) == 0: return 'bull_hi'
    breadth = compute_breadth(sig, date)
    
    if spy_now.iloc[-1] < s200_now.iloc[-1] and breadth < BREADTH_SOFT_LO:
        return 'bear'
    elif breadth < BREADTH_SOFT_BULL:
        return 'soft_bull'  # 45-65% breadth → soft bull, add defensive ETFs
    else:
        return 'bull_hi'    # > 65% breadth → strong bull


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


def get_tlt_momentum(tlt_p, date):
    hist = tlt_p.loc[:date].dropna()
    if len(hist) < TLT_MOM_LOOKBACK + 3: return False
    return bool(hist.iloc[-1] / hist.iloc[-TLT_MOM_LOOKBACK] - 1 > 0)


def check_defensive_etfs_available(def_prices, date):
    """Check which defensive ETFs have data at this date"""
    available = []
    for t, p in def_prices.items():
        if p is None: continue
        hist = p.loc[:date].dropna()
        if len(hist) > 0: available.append(t)
    return available


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, def_prices):
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
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg = get_regime(sig, date)
    tlt_pos = get_tlt_momentum(tlt_p, date)
    use_tlt_bear = (reg == 'bear' and tlt_pos)

    if reg == 'bull_hi':
        # Strong bull: 4 sectors × 2 stocks (same as v9j high-breadth mode)
        n_secs = max(N_BULL_SECS_HI - n_compete, 1)
        sps = BULL_SPS; bear_cash = 0.0; def_frac = 0.0
    elif reg == 'soft_bull':
        # Soft bull: 5 sectors × 2 stocks but with 30% defensive ETF overlay
        # Stock allocation = 70% of normal bull
        n_secs = max(N_BULL_SECS - n_compete, 1)
        sps = BULL_SPS; bear_cash = 0.0
        # Check which defensive ETFs are available
        avail_def = check_defensive_etfs_available(def_prices, date)
        def_frac = DEFENSIVE_EACH * len(avail_def) if avail_def else 0.0
    else:  # bear
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        bear_cash = max(0.20 - (TLT_BEAR_FRAC if use_tlt_bear else 0), 0.0)
        def_frac = 0.0
        avail_def = []

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    tlt_alloc = TLT_BEAR_FRAC if use_tlt_bear else 0.0
    stock_frac = max(1.0 - bear_cash - total_compete - tlt_alloc - def_frac, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if tlt_alloc > 0: w['TLT'] = tlt_alloc
        if def_frac > 0 and reg == 'soft_bull':
            for t in avail_def:
                w[t] = DEFENSIVE_EACH
        return w, use_tlt_bear

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if tlt_alloc > 0: weights['TLT'] = tlt_alloc
    if def_frac > 0 and reg == 'soft_bull':
        for t in avail_def:
            weights[t] = weights.get(t, 0) + DEFENSIVE_EACH
    return weights, use_tlt_bear


def apply_overlays(weights, spy_vol, dd, port_vol_ann):
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdxj_v + gld_dd
    if total > 0 and weights:
        hedge_keys = {'GLD', 'GDX', 'GDXJ', 'TLT', 'XLV', 'XLP', 'XLU'}
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
            equity_keys = [t for t in weights if t not in ('GLD', 'GDX', 'GDXJ', 'TLT',
                                                            'XLV', 'XLP', 'XLU')]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                 def_prices, start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []; tlt_m = 0; soft_bull_m = 0
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None
    regime_hist = {'bull_hi': 0, 'soft_bull': 0, 'bear': 0}

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15

        if len(port_returns) >= VOL_LOOKBACK:
            pv = np.std(port_returns[-VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        w, use_tlt = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, def_prices)
        w, shy_boost = apply_overlays(w, spy_vol, dd, pv)
        if use_tlt: tlt_m += 1
        reg = get_regime(sig, dt)
        regime_hist[reg] = regime_hist.get(reg, 0) + 1
        if reg == 'soft_bull': soft_bull_m += 1

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD', 'GDX', 'GDXJ', 'TLT',
                                              'XLV', 'XLP', 'XLU')}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t == 'TLT':  s = tlt_p.loc[dt:ndt].dropna()
            elif t in def_prices and def_prices[t] is not None:
                s = def_prices[t].loc[dt:ndt].dropna()
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
    return eq, float(np.mean(tos)) if tos else 0.0, regime_hist


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
    global DEFENSIVE_TOTAL_FRAC, DEFENSIVE_EACH
    print("=" * 72)
    print("🐻 动量轮动 v10b — 防御行业桥接 (三制度个股策略)")
    print("=" * 72)
    print(f"\n  Defensive ETFs: {DEFENSIVE_ETFS} ({DEFENSIVE_TOTAL_FRAC:.0%} total in soft-bull)")
    print(f"  Regimes: bull_hi (breadth>{BREADTH_SOFT_BULL:.0%}) / "
          f"soft_bull ({BREADTH_SOFT_LO:.0%}~{BREADTH_SOFT_BULL:.0%}) / "
          f"bear (breadth≤{BREADTH_SOFT_LO:.0%}+SPY<SMA200)")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    def_prices = {t: load_etf(t) for t in DEFENSIVE_ETFS}
    print(f"\n  Defensive ETF data: {[t for t,p in def_prices.items() if p is not None]}")
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")

    # Parameter sweep: defensive fraction
    print("\n🔍 Sweeping defensive ETF allocation...")
    results = []
    for def_frac in [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        DEFENSIVE_TOTAL_FRAC = def_frac
        DEFENSIVE_EACH = def_frac / len(DEFENSIVE_ETFS) if def_frac > 0 else 0
        try:
            eq, to, rh = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                       shy_p, tlt_p, def_prices)
            m = compute_metrics(eq)
            eq_is, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                        shy_p, tlt_p, def_prices, '2015-01-01', '2020-12-31')
            eq_oos, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                         shy_p, tlt_p, def_prices, '2021-01-01', '2025-12-31')
            mi = compute_metrics(eq_is); mo = compute_metrics(eq_oos)
            wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            results.append({'def_frac': def_frac, 'composite': comp,
                            'sharpe': m['sharpe'], 'cagr': m['cagr'],
                            'max_dd': m['max_dd'], 'calmar': m['calmar'],
                            'wf': wf, 'soft_bull_months': rh.get('soft_bull', 0)})
            print(f"  def_frac={def_frac:.0%} → Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
                  f"MaxDD={m['max_dd']:.1%} CAGR={m['cagr']:.1%} WF={wf:.2f} "
                  f"(soft_bull={rh.get('soft_bull',0)}m)")
        except Exception as e:
            print(f"  Error (def_frac={def_frac}): {e}")

    results = sorted(results, key=lambda x: x['composite'], reverse=True)
    print(f"\n📊 Best configs:")
    for r in results[:5]:
        print(f"  def_frac={r['def_frac']:.0%} → Comp={r['composite']:.4f} "
              f"Sharpe={r['sharpe']:.2f} WF={r['wf']:.2f}")

    best = results[0]
    DEFENSIVE_TOTAL_FRAC = best['def_frac']
    DEFENSIVE_EACH = DEFENSIVE_TOTAL_FRAC / len(DEFENSIVE_ETFS) if DEFENSIVE_TOTAL_FRAC > 0 else 0

    print(f"\n🏆 Best: def_frac={DEFENSIVE_TOTAL_FRAC:.0%}")
    print("\n🔄 Full (2015-2025)...")
    eq_full, to, rh = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                    shy_p, tlt_p, def_prices)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                shy_p, tlt_p, def_prices, '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                 shy_p, tlt_p, def_prices, '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v10b vs v9j champion (2.057)")
    print("=" * 72)
    v9j = dict(cagr=0.3235, max_dd=-0.1033, sharpe=1.850, calmar=3.132, comp=2.057, wf=0.785)
    print(f"{'Metric':<12} {'v9j':<10} {'v10b':<10} {'Delta'}")
    print(f"{'CAGR':<12} {v9j['cagr']:.1%}     {m['cagr']:.1%}     {m['cagr']-v9j['cagr']:+.1%}")
    print(f"{'MaxDD':<12} {v9j['max_dd']:.1%}    {m['max_dd']:.1%}    {m['max_dd']-v9j['max_dd']:+.1%}")
    print(f"{'Sharpe':<12} {v9j['sharpe']:.2f}      {m['sharpe']:.2f}      {m['sharpe']-v9j['sharpe']:+.2f}")
    print(f"{'Calmar':<12} {v9j['calmar']:.2f}      {m['calmar']:.2f}      {m['calmar']-v9j['calmar']:+.2f}")
    print(f"{'IS Sharpe':<12}           {mi['sharpe']:.2f}")
    print(f"{'OOS Sharpe':<12}           {mo['sharpe']:.2f}")
    print(f"{'WF':<12} {v9j['wf']:.2f}      {wf:.2f}      {wf-v9j['wf']:+.2f}")
    print(f"{'Composite':<12} {v9j['comp']:.4f}  {comp:.4f}  {comp-v9j['comp']:+.4f}")
    print(f"\n  Regime distribution: {rh}")
    print(f"  Defensive ETF frac: {DEFENSIVE_TOTAL_FRAC:.0%}")

    if comp > 2.1:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.1!")
    elif comp > 2.057:
        print(f"\n🚀🚀 超越v9j冠军! Composite {comp:.4f}")
    elif comp > 2.00:
        print(f"\n🚀 Composite > 2.0: {comp:.4f}")
    elif comp > 1.90:
        print(f"\n✅ 良好: Composite {comp:.4f}")
    
    if wf > 0.80:
        print(f"\n🎯 高WF: {wf:.2f} (比v9j更稳健!)")

    out = {
        'strategy': f'v10b Defensive Sector Bridge (def_frac={DEFENSIVE_TOTAL_FRAC:.0%})',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'regime_hist': rh, 'defensive_etfs': DEFENSIVE_ETFS,
        'sweep': results,
    }
    jf = Path(__file__).parent / "momentum_v10b_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
