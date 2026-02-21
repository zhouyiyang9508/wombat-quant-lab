#!/usr/bin/env python3
"""
v9h3 — 动量加速 + 行业ETF加权排名 + Alpha动量 (v9g基础上的精准改进)
代码熊 🐻

当前最佳: v9g — Composite 1.759

v9h2 教训:
  - 横截面选股 (Cross-Sectional): Composite仅1.37, 远不如sector-first (行业优先)
  - 跳过1月: 月度频率下反转效应弱, 灾难性 (Composite 0.92)
  - 确认: v9g的sector-first top-down框架在月度策略中是最优的结构

v9h3 三个干净的新尝试 (在v9g sector-first框架内):

[A] 动量加速过滤器 (Acceleration Filter)
    新增: 只选3m动量 > 6m动量 * thresh 的股票 (动量正在加速)
    理由: 衰减的动量即将逆转, 加速的动量更可能持续
    参数: thresh ∈ {0.5, 0.7, 0.9, 1.0}
    
    纠正v9h2中的bug: select_topdown()签名错误导致B系列全部出错

[B] 行业ETF增强排名 (Sector ETF Enhanced Ranking)
    当前: 行业分数 = 行业内股票平均动量 (有噪声)
    新: 行业分数 = α × 行业ETF动量 + (1-α) × 股票平均动量
    
    原理: 行业ETF是行业趋势的干净代理, 混合后减少噪声
    参数: α ∈ {0.2, 0.4}
    ETF数据: XLK, XLV, XLF, XLY, XLI, XLE, XLU, XLB, XLRE, XLC, XLP

[C] Alpha动量 (Excess Return vs SPY)
    当前: 绝对动量 (原始return)
    新: Alpha动量 = 股票return - SPY return (超额收益)
    
    原理: 
      - 只有真正跑赢大盘的股票才值得持有
      - 在牛市中过滤掉"只是随大盘涨"的股票
      - 在熊市中找到真正的抵抗性股票
    实现: mom_alpha = r3 - spy_r3 (for 3m) etc.
    
    注意: Alpha动量在月度策略中的效果有文献支持 (Fama-French alpha momentum)

[D] 组合: A+B, A+C, B+C
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

N_BULL_SECS    = 5
N_BULL_SECS_HI = 4
BREADTH_CONC   = 0.65
BULL_SPS       = 2
BEAR_SPS       = 2
BREADTH_NARROW   = 0.45
GLD_AVG_THRESH   = 0.70
GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH   = 0.20
GDX_COMPETE_FRAC = 0.04
CONT_BONUS       = 0.03
HI52_FRAC        = 0.60
USE_SHY          = True
DD_PARAMS        = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30
GDXJ_VOL_LO_FRAC   = 0.08
GDXJ_VOL_HI_THRESH = 0.45
GDXJ_VOL_HI_FRAC   = 0.18
MOM_W = (0.20, 0.50, 0.20, 0.10)

SECTOR_ETF = {
    'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
    'Consumer Discretionary': 'XLY', 'Industrials': 'XLI', 'Energy': 'XLE',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP',
}


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


def precompute(close_df, spy_ser, etf_prices):
    r1   = close_df / close_df.shift(22)  - 1
    r3   = close_df / close_df.shift(63)  - 1
    r6   = close_df / close_df.shift(126) - 1
    r12  = close_df / close_df.shift(252) - 1
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol5  = log_r.rolling(5).std() * np.sqrt(252)
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = spy_ser
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    # SPY returns for alpha computation
    spy_r1 = spy / spy.shift(22)  - 1  if spy is not None else None
    spy_r3 = spy / spy.shift(63)  - 1  if spy is not None else None
    spy_r6 = spy / spy.shift(126) - 1  if spy is not None else None
    spy_r12= spy / spy.shift(252) - 1  if spy is not None else None
    # ETF 3m returns for sector ETF ranking
    etf_r3 = {}
    for sec, etf in SECTOR_ETF.items():
        if etf in etf_prices:
            ep = etf_prices[etf]
            etf_r3[sec] = ep / ep.shift(63) - 1
    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200, sma50=sma50,
                close=close_df, spy_r1=spy_r1, spy_r3=spy_r3,
                spy_r6=spy_r6, spy_r12=spy_r12, etf_r3=etf_r3)


def get_spy_vol(sig, date):
    if sig['vol5'] is None: return 0.15
    # SPY vol stored as a series
    v = sig['vol5_spy'].loc[:date].dropna() if 'vol5_spy' in sig else pd.Series()
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
    r6 = sig['r6']
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


def get_spy_scalar(spy_series, date):
    """Get scalar SPY return at date"""
    v = spy_series.loc[:date].dropna()
    return float(v.iloc[-1]) if len(v) > 0 else 0.0


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, cfg):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
    d = idx[-1]

    use_alpha   = cfg.get('alpha_mom', False)
    accel_min   = cfg.get('accel_filter', None)   # None = disabled
    etf_frac    = cfg.get('etf_frac', 0.0)

    w1, w3, w6, w12 = MOM_W

    if use_alpha and sig['spy_r3'] is not None:
        spy_r1v = get_spy_scalar(sig['spy_r1'], date)
        spy_r3v = get_spy_scalar(sig['spy_r3'], date)
        spy_r6v = get_spy_scalar(sig['spy_r6'], date)
        spy_r12v= get_spy_scalar(sig['spy_r12'], date)
        mom = ((sig['r1'].loc[d] - spy_r1v)*w1 +
               (sig['r3'].loc[d] - spy_r3v)*w3 +
               (sig['r6'].loc[d] - spy_r6v)*w6 +
               (sig['r12'].loc[d]- spy_r12v)*w12)
    else:
        mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
               sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom':   mom, 'r3': sig['r3'].loc[d], 'r6': sig['r6'].loc[d],
        'vol':   sig['vol30'].loc[d], 'price': close.loc[d],
        'sma50': sig['sma50'].loc[d], 'hi52': sig['r52w_hi'].loc[d],
    }).dropna(subset=['mom', 'sma50', 'r3', 'r6'])
    df = df[(df['price'] >= 5)]
    df = df[(df['r6'] > 0) & (df['vol'] < 0.65)]
    df = df[df['price'] > df['sma50']]
    df = df[df['price'] >= df['hi52'] * HI52_FRAC]
    df['sector'] = df.index.map(lambda t: sectors.get(t, 'Unknown'))
    df = df[df['sector'] != 'Unknown']

    for t in df.index:
        if t in prev_hold: df.loc[t, 'mom'] += CONT_BONUS

    # Acceleration filter (Direction A)
    if accel_min is not None:
        df = df[df['r3'] >= df['r6'] * accel_min]

    if len(df) == 0: return {}

    # Sector ranking (with optional ETF blend)
    if etf_frac > 0 and sig['etf_r3']:
        etf_mom_at_date = {}
        for sec, ser in sig['etf_r3'].items():
            v = ser.loc[:date].dropna()
            etf_mom_at_date[sec] = float(v.iloc[-1]) if len(v) > 0 else 0.0
        # Blend: sec_score = etf_frac * ETF_3m + (1-etf_frac) * avg_stock_3m
        stock_sec_mom = df.groupby('sector')['mom'].mean()
        sec_scores = {}
        for sec in stock_sec_mom.index:
            etf_s = etf_mom_at_date.get(sec, 0.0)
            sec_scores[sec] = etf_frac * etf_s + (1 - etf_frac) * stock_sec_mom[sec]
        sec_mom = pd.Series(sec_scores).sort_values(ascending=False)
    else:
        sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg     = get_regime(sig, date)
    breadth = compute_breadth(sig, date)

    if reg == 'bull':
        n_bull_secs = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull_secs - n_compete, 1)
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


def apply_overlays(weights, spy_vol, dd):
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)
    total = gdxj_v + gld_dd
    if total <= 0 or not weights: return weights
    stock_frac = max(1.0 - total, 0.01)
    tot = sum(weights.values())
    if tot <= 0: return weights
    new = {t: w/tot*stock_frac for t, w in weights.items()}
    if gld_dd > 0: new['GLD'] = new.get('GLD', 0) + gld_dd
    if gdxj_v > 0: new['GDXJ'] = new.get('GDXJ', 0) + gdxj_v
    return new


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                 spy_vol5, cfg, start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        v = spy_vol5.loc[:dt].dropna()
        spy_vol = float(v.iloc[-1]) if len(v) > 0 else 0.15

        w = select(sig, sectors, dt, prev_h, gld_p, gdx_p, cfg)
        w = apply_overlays(w, spy_vol, dd)
        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t,0) - prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD','GDX','GDXJ')}
        invested = sum(w.values()); cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t in close_df.columns: s = close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt
        if USE_SHY and cash_frac > 0 and shy_p is not None:
            s = shy_p.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * cash_frac
        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)
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


CONFIGS = {
    'v9g_base':   {'alpha_mom': False, 'accel_filter': None, 'etf_frac': 0.0},
    # [A] Accel filter
    'A_acc09':    {'alpha_mom': False, 'accel_filter': 0.9,  'etf_frac': 0.0},
    'A_acc07':    {'alpha_mom': False, 'accel_filter': 0.7,  'etf_frac': 0.0},
    'A_acc05':    {'alpha_mom': False, 'accel_filter': 0.5,  'etf_frac': 0.0},
    'A_acc03':    {'alpha_mom': False, 'accel_filter': 0.3,  'etf_frac': 0.0},
    # [B] Sector ETF enhanced
    'B_etf20':    {'alpha_mom': False, 'accel_filter': None, 'etf_frac': 0.20},
    'B_etf40':    {'alpha_mom': False, 'accel_filter': None, 'etf_frac': 0.40},
    # [C] Alpha momentum
    'C_alpha':    {'alpha_mom': True,  'accel_filter': None, 'etf_frac': 0.0},
    # Combos
    'AB_acc07_etf20': {'alpha_mom': False, 'accel_filter': 0.7, 'etf_frac': 0.20},
    'AC_alpha_acc07': {'alpha_mom': True,  'accel_filter': 0.7, 'etf_frac': 0.0},
    'BC_alpha_etf20': {'alpha_mom': True,  'accel_filter': None, 'etf_frac': 0.20},
}


def main():
    print("=" * 72)
    print("🐻 v9h3 — 动量加速 + ETF行业排名 + Alpha动量")
    print("=" * 72)

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers)
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    spy_p  = load_csv(CACHE / "SPY.csv")['Close'].dropna()

    # Load sector ETFs
    etf_prices = {}
    for sec, etf in SECTOR_ETF.items():
        fp = CACHE / f"{etf}.csv"
        if fp.exists():
            try:
                etf_prices[sec] = load_csv(fp)['Close'].dropna()
            except: pass
    print(f"  Sector ETFs: {len(etf_prices)}/{len(SECTOR_ETF)}")

    sig = precompute(close_df, spy_p, etf_prices)
    spy_vol5 = np.log(spy_p / spy_p.shift(1)).rolling(5).std() * np.sqrt(252)
    print(f"  Loaded {len(close_df.columns)} stock tickers")
    print(f"  Base: v9g champion (Composite 1.759)\n")

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"--- {name} ---", flush=True)
        try:
            eq_f, to = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, spy_vol5, cfg)
            eq_is, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, spy_vol5, cfg,
                                     '2015-01-01', '2020-12-31')
            eq_oo, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, spy_vol5, cfg,
                                     '2021-01-01', '2025-12-31')
            m  = compute_metrics(eq_f)
            mi = compute_metrics(eq_is)
            mo = compute_metrics(eq_oo)
            wf   = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            results[name] = {'full': m, 'is': mi, 'oos': mo, 'wf': wf, 'composite': comp}
            tag = '🚀🚀' if comp > 1.80 else ('🚀' if comp > 1.759 else ('✅' if comp > 1.70 else ''))
            wt  = '✅' if wf >= 0.70 else ('⚠️' if wf >= 0.60 else '❌')
            print(f"  Comp={comp:.4f} {tag} | Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} "
                  f"DD={m['max_dd']:.1%} WF={wf:.2f} {wt}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 72)
    print("📊 RANKINGS")
    print("=" * 72)
    ranked = sorted(results.items(), key=lambda x: x[1]['composite'], reverse=True)
    for i, (n, r) in enumerate(ranked[:8]):
        m = r['full']
        tag = '🚀🚀' if r['composite'] > 1.80 else ('🚀' if r['composite'] > 1.759 else '')
        wft = '✅' if r['wf'] >= 0.70 else '⚠️'
        print(f"  #{i+1} {n:22s}: Comp={r['composite']:.4f} {tag} | "
              f"Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%} WF={r['wf']:.2f} {wft}")

    best = max(results.items(), key=lambda x: x[1]['composite'])
    print(f"\n🏆 Best: {best[0]} → Composite {best[1]['composite']:.4f}")
    if best[1]['composite'] > 1.80:
        print("🚨🚨🚨 【重大突破】Composite > 1.80!")
    elif best[1]['composite'] > 1.759:
        print(f"🚀 超越v9g!")
    else:
        print("❌ 未超越v9g (1.759)")

    jf = Path(__file__).parent / "momentum_v9h3_results.json"
    jf.write_text(json.dumps({'results': {n: {'full': r['full'], 'is': r['is'],
        'oos': r['oos'], 'wf': r['wf'], 'composite': r['composite']} for n, r in results.items()},
        'best': best[0], 'best_composite': best[1]['composite']}, indent=2))
    print(f"💾 Results → {jf}")

if __name__ == '__main__':
    main()
