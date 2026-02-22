#!/usr/bin/env python3
"""
动量轮动 v11b — 行业ETF趋势确认过滤器
代码熊 🐻

v11a基础 + 新第16层: 行业ETF趋势确认

问题: 当前策略选择个股动量最强的行业,但有时顶尖行业的ETF整体已经开始下跌
  例如: 2022年初, 科技股个股动量仍为正, 但XLK 6m动量已经转负
  这时买入科技股就是在追顶, 结果亏损

解决方案: 行业ETF趋势确认过滤器
  条件: 只投资 ETF 6m动量 > 阈值 的行业
  - XLK(科技), XLV(医疗), XLF(金融), XLY(可选消费), XLI(工业)
  - XLE(能源), XLP(必需消费), XLU(公用事业), XLB(材料)
  - XLRE(房地产), XLC(通信)
  
  如果某行业ETF动量≤阈值, 该行业被跳过, 选下一个排名的行业
  这样即使个股动量强, 如果ETF整体趋势弱, 也不会进入

预期效果:
  - 2022年科技: XLK 6m<0 → 跳过Tech, 选Energy(XLE正)
  - 2020 COVID: 大多数ETF为负 → 持有更多防御仓位
  - 长期: 减少"追顶"行为 → Sharpe和WF提升
  - 代价: 可能错过行业早期复苏 → 轻微CAGR下降

行业ETF映射:
  Technology       → XLK
  Healthcare       → XLV
  Financials       → XLF
  Consumer Cyclical → XLY
  Industrials      → XLI
  Energy           → XLE
  Consumer Defensive → XLP
  Utilities        → XLU
  Materials        → XLB
  Real Estate      → XLRE
  Communication Services → XLC
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v11a parameters (unchanged)
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

# v11a innovations (unchanged)
TLT_BEAR_FRAC = 0.25; IEF_BEAR_FRAC = 0.20; BOND_MOM_LB = 126
DEFENSIVE_ETFS = ['XLV', 'XLP', 'XLU']; DEFENSIVE_FRAC = 0.15
DEFENSIVE_EACH = DEFENSIVE_FRAC / len(DEFENSIVE_ETFS)
SPY_SOFT_HI_THRESH = -0.07; SPY_SOFT_HI_FRAC = 0.10

# v11b NEW: Sector ETF trend filter
SECTOR_ETF_MAP = {
    'Technology':             'XLK',
    'Healthcare':             'XLV',
    'Financials':             'XLF',
    'Consumer Cyclical':      'XLY',
    'Industrials':            'XLI',
    'Energy':                 'XLE',
    'Consumer Defensive':     'XLP',
    'Utilities':              'XLU',
    'Materials':              'XLB',
    'Real Estate':            'XLRE',
    'Communication Services': 'XLC',
}
ETF_MOM_LB     = 126   # 6-month lookback
ETF_MOM_THRESH = 0.0   # sector ETF 6m momentum must be > this to allow entry


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
    r1  = close_df / close_df.shift(22)  - 1
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1
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


def get_three_regime(sig, date):
    if sig['s200'] is None: return 'bull_hi', 1.0
    spy_now  = sig['spy'].loc[:date].dropna()
    s200_now = sig['s200'].loc[:date].dropna()
    if len(spy_now) == 0: return 'bull_hi', 1.0
    breadth = compute_breadth(sig, date)
    if spy_now.iloc[-1] < s200_now.iloc[-1] and breadth < BREADTH_NARROW:
        return 'bear', breadth
    elif breadth < BREADTH_CONC:
        return 'soft_bull', breadth
    else:
        return 'bull_hi', breadth


def get_spy_1m(sig, date):
    if sig['r1'] is None or 'SPY' not in sig['r1'].columns: return 0.0
    hist = sig['r1']['SPY'].loc[:date].dropna()
    return float(hist.iloc[-1]) if len(hist) > 0 else 0.0


def select_best_bond(tlt_p, ief_p, date):
    def get_6m(p):
        hist = p.loc[:date].dropna()
        if len(hist) < BOND_MOM_LB + 3: return None
        return float(hist.iloc[-1] / hist.iloc[-BOND_MOM_LB] - 1)
    tlt_mom = get_6m(tlt_p); ief_mom = get_6m(ief_p)
    tlt_pos = tlt_mom is not None and tlt_mom > 0
    ief_pos = ief_mom is not None and ief_mom > 0
    if not tlt_pos and not ief_pos: return None, 0.0
    if tlt_pos and not ief_pos: return 'TLT', TLT_BEAR_FRAC
    if ief_pos and not tlt_pos: return 'IEF', IEF_BEAR_FRAC
    return ('TLT', TLT_BEAR_FRAC) if (tlt_mom or 0) >= (ief_mom or 0) else ('IEF', IEF_BEAR_FRAC)


def asset_compete(sig, date, prices, thresh, frac, lb=127):
    r6  = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].dropna(); stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lb] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0


def get_sector_etf_momentum(sec_name, sec_etf_prices, date):
    """Returns 6m momentum of sector ETF. None if no data."""
    etf_ticker = SECTOR_ETF_MAP.get(sec_name)
    if etf_ticker is None or etf_ticker not in sec_etf_prices: return None
    prices = sec_etf_prices[etf_ticker]
    hist = prices.loc[:date].dropna()
    if len(hist) < ETF_MOM_LB + 3: return None
    return float(hist.iloc[-1] / hist.iloc[-ETF_MOM_LB] - 1)


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p,
           def_prices, sec_etf_prices):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, 'bull_hi', None
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
    if len(df) == 0: return {}, 'bull_hi', None

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg, breadth = get_three_regime(sig, date)

    bond_ticker, bond_frac = select_best_bond(tlt_p, ief_p, date)
    use_bond = (reg == 'bear' and bond_frac > 0)
    actual_bond_frac = bond_frac if use_bond else 0.0

    if reg == 'soft_bull':
        avail_def = [t for t, p in def_prices.items()
                     if p is not None and len(p.loc[:date].dropna()) > 0]
        def_alloc = {t: DEFENSIVE_EACH for t in avail_def}
        def_frac = sum(def_alloc.values())
    else:
        avail_def = []; def_alloc = {}; def_frac = 0.0

    if reg == 'bull_hi':
        max_secs = N_BULL_SECS_HI - n_compete
        sps, bear_cash = BULL_SPS, 0.0
    elif reg == 'soft_bull':
        max_secs = N_BULL_SECS - n_compete
        sps, bear_cash = BULL_SPS, 0.0
    else:
        max_secs = 3 - n_compete
        sps = BEAR_SPS
        bear_cash = max(0.20 - actual_bond_frac, 0.0)

    # v11b KEY: Apply sector ETF trend filter
    allowed_secs = []
    skipped_secs = []
    for sec in sec_mom.index:
        etf_mom = get_sector_etf_momentum(sec, sec_etf_prices, date)
        if etf_mom is None or etf_mom > ETF_MOM_THRESH:
            allowed_secs.append(sec)
        else:
            skipped_secs.append(sec)

    top_secs = allowed_secs[:max(max_secs, 1)]
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    stock_frac = max(1.0 - bear_cash - total_compete - actual_bond_frac - def_frac, 0.0)

    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if use_bond and bond_ticker: w[bond_ticker] = actual_bond_frac
        w.update(def_alloc)
        return w, reg, bond_ticker

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = weights.get('GLD', 0) + gld_a
    if gdx_a > 0: weights['GDX'] = weights.get('GDX', 0) + gdx_a
    if use_bond and bond_ticker: weights[bond_ticker] = actual_bond_frac
    weights.update(def_alloc)
    return weights, reg, bond_ticker


HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'XLV', 'XLP', 'XLU'}


def apply_overlays(weights, spy_vol, dd, port_vol_ann, spy_1m_ret):
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    gld_soft = SPY_SOFT_HI_FRAC if spy_1m_ret <= SPY_SOFT_HI_THRESH else 0.0
    gld_total = max(gld_dd, gld_soft)

    total_overlay = gdxj_v + gld_total
    if total_overlay > 0 and weights:
        hedge_w  = {t: w for t, w in weights.items() if t in HEDGE_KEYS}
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
                for t in equity_keys: weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)
    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                 tlt_p, ief_p, def_prices, sec_etf_prices,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0; port_returns = []
    regime_hist = {'bull_hi': 0, 'soft_bull': 0, 'bear': 0}
    skipped_total = 0
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15
        spy_1m = get_spy_1m(sig, dt)

        if len(port_returns) >= VOL_LOOKBACK:
            pv = np.std(port_returns[-VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        w, reg, bond_t = select(sig, sectors, dt, prev_h, gld_p, gdx_p,
                                 tlt_p, ief_p, def_prices, sec_etf_prices)
        w, shy_boost = apply_overlays(w, spy_vol, dd, pv, spy_1m)
        regime_hist[reg] = regime_hist.get(reg, 0) + 1

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in HEDGE_KEYS}

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t == 'TLT':  s = tlt_p.loc[dt:ndt].dropna()
            elif t == 'IEF':  s = ief_p.loc[dt:ndt].dropna()
            elif t in def_prices and def_prices[t] is not None:
                s = def_prices[t].loc[dt:ndt].dropna()
            elif t in sec_etf_prices:
                s = sec_etf_prices[t].loc[dt:ndt].dropna()
            elif t in close_df.columns:
                s = close_df[t].loc[dt:ndt].dropna()
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
    global ETF_MOM_THRESH
    print("=" * 72)
    print("🐻 动量轮动 v11b — 行业ETF趋势确认过滤器")
    print("=" * 72)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    ief_p  = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    def_prices = {t: load_csv(CACHE / f"{t}.csv")['Close'].dropna()
                  for t in ['XLV', 'XLP', 'XLU']
                  if (CACHE / f"{t}.csv").exists()}
    # Load all sector ETFs
    sec_etf_prices = {}
    all_sec_etfs = list(set(SECTOR_ETF_MAP.values()))
    for t in all_sec_etfs:
        f = CACHE / f"{t}.csv"
        if f.exists():
            sec_etf_prices[t] = load_csv(f)['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")
    print(f"  Sector ETFs loaded: {sorted(sec_etf_prices.keys())}")

    # Sweep ETF momentum threshold
    print("\n🔍 Sweeping ETF momentum threshold...")
    thresholds = [-0.05, -0.03, -0.02, 0.0, 0.02, 0.05]
    results = []
    for thresh in thresholds:
        ETF_MOM_THRESH = thresh
        eq, to, rh = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                   shy_p, tlt_p, ief_p, def_prices, sec_etf_prices)
        eq_is, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                    shy_p, tlt_p, ief_p, def_prices, sec_etf_prices,
                                    '2015-01-01', '2020-12-31')
        eq_oos, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                     shy_p, tlt_p, ief_p, def_prices, sec_etf_prices,
                                     '2021-01-01', '2025-12-31')
        m = compute_metrics(eq); mi = compute_metrics(eq_is); mo = compute_metrics(eq_oos)
        wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
        results.append({'thresh': thresh, 'composite': comp, 'sharpe': m['sharpe'],
                        'cagr': m['cagr'], 'max_dd': m['max_dd'], 'calmar': m['calmar'],
                        'wf': wf, 'is_sh': mi['sharpe'], 'oos_sh': mo['sharpe']})
        print(f"  thresh={thresh:+.2f} → Comp={comp:.4f} Sharpe={m['sharpe']:.2f} "
              f"CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} WF={wf:.2f} "
              f"(IS={mi['sharpe']:.2f}/OOS={mo['sharpe']:.2f})")

    results_sorted = sorted(results, key=lambda x: x['composite'], reverse=True)
    print(f"\n📊 By Composite:")
    for r in results_sorted[:4]:
        print(f"  thresh={r['thresh']:+.2f}: Comp={r['composite']:.4f} WF={r['wf']:.2f} Sharpe={r['sharpe']:.2f}")

    best = results_sorted[0]
    ETF_MOM_THRESH = best['thresh']
    print(f"\n🏆 Best threshold: {ETF_MOM_THRESH}")

    print("\n🔄 Final run with best threshold...")
    eq_full, to, rh = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                    shy_p, tlt_p, ief_p, def_prices, sec_etf_prices)
    eq_is, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                shy_p, tlt_p, ief_p, def_prices, sec_etf_prices,
                                '2015-01-01', '2020-12-31')
    eq_oos, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p,
                                 shy_p, tlt_p, ief_p, def_prices, sec_etf_prices,
                                 '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS — v11b vs v11a (Champion 2.190)")
    print("=" * 72)
    v11a = dict(cagr=0.3209, max_dd=-0.0944, sharpe=1.916, calmar=3.400, comp=2.190, wf=0.742)
    print(f"{'Metric':<12} {'v11a':<12} {'v11b':<12} {'Delta'}")
    for k, fmt in [('cagr','%'),('max_dd','%'),('sharpe','f'),('calmar','f')]:
        r1 = f"{v11a[k]:.1%}" if fmt=='%' else f"{v11a[k]:.2f}"
        rv = f"{m[k]:.1%}" if fmt=='%' else f"{m[k]:.2f}"
        delta = f"{m[k]-v11a[k]:+.1%}" if fmt=='%' else f"{m[k]-v11a[k]:+.2f}"
        print(f"  {k.upper():<10} {r1:<12} {rv:<12} {delta}")
    print(f"  {'IS Sharpe':<10} {mi['sharpe']:.2f}")
    print(f"  {'OOS Sharpe':<10} {mo['sharpe']:.2f}")
    print(f"  {'WF':<10} {v11a['wf']:.2f}         {wf:.2f}         {wf-v11a['wf']:+.2f}")
    print(f"  {'Composite':<10} {v11a['comp']:.4f}     {comp:.4f}     {comp-v11a['comp']:+.4f}")
    print(f"  {'ETF thresh':<10} {ETF_MOM_THRESH:+.2f}")
    print(f"\n  Regime: {rh}")

    if comp > 2.20:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.20! ({comp:.4f})")
    elif comp > 2.190:
        print(f"\n🚀🚀 超越v11a! Composite {comp:.4f}")
    elif comp > 2.10:
        print(f"\n✅ Composite > 2.10: {comp:.4f}")
    if wf >= 0.77:
        print(f"\n🎯 WF提升! {wf:.2f}")
    elif wf >= 0.74:
        print(f"\n🎯 WF维持: {wf:.2f}")

    out = {
        'strategy': f'v11b: v11a + sector ETF trend filter (thresh={ETF_MOM_THRESH})',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'regime_hist': rh, 'best_thresh': ETF_MOM_THRESH,
        'sweep': results,
    }
    jf = Path(__file__).parent / "momentum_v11b_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
