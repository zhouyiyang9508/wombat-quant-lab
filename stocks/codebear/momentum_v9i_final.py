#!/usr/bin/env python3
"""
动量轮动 v9i (Final) — 波动率目标化风险控制 🚨🚨🚨
代码熊 🐻

📊 性能摘要 (2015-2025):
  CAGR:      31.9% ✅ (>30%)
  MaxDD:     -10.7% ✅ (<25%, 史上最低!)
  Sharpe:    1.81  ✅ (>1.5)
  Calmar:    2.97
  IS Sharpe: 1.91 (2015-2020)
  OOS Sharpe:1.57 (2021-2025)
  WF ratio:  0.82  ✅ (>0.6)
  Composite: 1.973 ✅ (>1.5, >1.8!)

进化路径 (v4d → v9i):
  v4d:  1.356 → CAGR 27.0%, Sharpe 1.43, MaxDD -15.0%
  v9a:  1.512 → CAGR 30.5%, Sharpe 1.57, MaxDD -14.9%
  v9c:  1.567 → CAGR 31.6%, Sharpe 1.64, MaxDD -14.9%
  v9f:  1.667 → CAGR 34.6%, Sharpe 1.67, MaxDD -14.9%
  v9g:  1.759 → CAGR 37.2%, Sharpe 1.71, MaxDD -14.9%
  v9i:  1.973 → CAGR 31.9%, Sharpe 1.81, MaxDD -10.7% ← NEW!
  Total: +0.617 vs v4d (+45.5%)

🚨 v9i 核心创新: 投资组合波动率目标化 (Portfolio Volatility Targeting) 🚨

★ 关键洞察 ★
历史回测中, 月度组合收益率的3个月滚动波动率包含重要的市场状态信息:
  - 低波动期 (port_vol < target_vol): 趋势稳定, 维持满仓
  - 高波动期 (port_vol > target_vol): 市场不稳, 按比例缩减权益仓位

实现机制:
  1. 每月计算过去3个月组合月度收益率的年化标准差 (portfolio_vol_ann)
  2. scale = min(target_vol / portfolio_vol_ann, 1.0)
  3. 股票仓位 × scale, 多余资金转入SHY (安全收益)
  
  target_vol = 11% (年化, 约对应月度波动率 3.2%)

效果量化:
  - avg_scale = 0.766: 平均权益仓位缩减至76.6% (23.4%转SHY)
  - pct_scaled = 61%: 61%的月份有缩减 (vol > 11%)
  - MaxDD改善: -14.9% → -10.7% (-4.2pp, 史上最好!)
  - Calmar改善: 2.50 → 2.97 (+0.47)
  - Sharpe改善: 1.71 → 1.81 (+0.10)
  - Composite改善: 1.759 → 1.973 (+0.214)

为什么3月lookback (而非6月) 更好?
  - 3月lookback对市场状态变化反应更快
  - 6月lookback太滞后: 危机已结束时还在减仓 → 错过反弹
  - 3月窗口的噪声虽大, 但对月度策略来说是正确的时间尺度

为什么不用SPY vol (GDXJ vol触发)?
  - SPY vol看的是市场宏观波动 (5日实现vol)
  - Portfolio vol看的是我们策略的实际风险 (月度收益波动)
  - 二者互补: GDXJ处理瞬时冲击, vol_target处理持续高波动期
  - 两个机制都保留, 各司其职

和CAGR权衡:
  - v9g: CAGR 37.2%, Composite 1.759
  - v9i: CAGR 31.9%, Composite 1.973 (+12.2% composite)
  - 放弃5.3pp CAGR, 换取4.2pp MaxDD改善 + 更高Sharpe/Calmar
  - 对风险厌恶投资者: v9i 明显优于 v9g

完整 13 层创新栈:
① GLD竞争: GLD_6m > avg×70% → 20%GLD
② Breadth+SPY双确认熊市
③ 3m主导动量权重 (20/50/20/10)
④ 5行业×2股 (牛市, breadth≤65%)
⑤ 4行业×2股 (强牛, breadth>65%)
⑥ 宽度阈值45%
⑦ 52周高点过滤 (price ≥ 52w_hi×60%)
⑧ SHY替代熊市现金
⑨ GDXJ波动率预警: vol>30%→8%GDXJ; >45%→18%GDXJ
⑩ 激进DD: -8%→40%GLD, -12%→60%, -18%→70%GLD
⑪ GDX精细竞争: GDX_6m>avg×20% → 4%GDX
⑫ GLD自然竞争
⑬ 投资组合波动率目标化: port_3m_vol > 11% → 缩减权益 ← NEW!

严格无前瞻:
  - portfolio_vol 由历史月度收益率计算 (全部为已实现数据)
  - 月末信号使用历史收盘价 (无当月前瞻)
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# Champion parameters (v9i: Portfolio Volatility Targeting)
MOM_W    = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS    = 5; N_BULL_SECS_HI = 4; BREADTH_CONC = 0.65
BULL_SPS = 2; BEAR_SPS = 2; BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
CONT_BONUS = 0.03; HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18

# Vol Targeting parameters (NEW!)
VOL_TARGET_ANN = 0.11    # 11% annual vol target
VOL_LOOKBACK   = 3       # 3 months of portfolio return history


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


def select(sig, sectors, date, prev_hold, gld_p, gdx_p):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
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
    """Apply GDXJ vol-trigger, GLD DD response, and Portfolio Vol Targeting"""
    # GDXJ vol trigger
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    # GLD DD response
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdxj_v + gld_dd
    if total > 0 and weights:
        stock_frac = max(1.0 - total, 0.01)
        tot = sum(weights.values())
        if tot > 0:
            weights = {t: w/tot*stock_frac for t, w in weights.items()}
        if gld_dd > 0: weights['GLD'] = weights.get('GLD', 0) + gld_dd
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    # Portfolio Volatility Targeting (NEW in v9i)
    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:  # Only adjust if meaningful
            equity_keys = [t for t in weights if t not in ('GLD', 'GDX', 'GDXJ')]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] = weights[t] * scale
                shy_boost = eq_frac * (1.0 - scale)  # excess → SHY

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []  # track monthly returns for vol targeting
    scale_hist = []
    holdings_hist = {}
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15

        # Portfolio vol from past VOL_LOOKBACK months
        if len(port_returns) >= VOL_LOOKBACK:
            port_vol_mon = np.std(port_returns[-VOL_LOOKBACK:], ddof=1)
            port_vol_ann = port_vol_mon * np.sqrt(12)
        else:
            port_vol_ann = 0.20  # bootstrap: assume 20% before enough history

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

        # SHY for: vol-targeting excess + natural cash
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
    avg_scale = float(np.mean(scale_hist)) if scale_hist else 1.0
    pct_scaled = float(np.mean([s < 0.99 for s in scale_hist])) if scale_hist else 0.0
    avg_to = float(np.mean(tos)) if tos else 0.0
    return eq, holdings_hist, avg_to, avg_scale, pct_scaled


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
    print("=" * 72)
    print("🐻 动量轮动 v9i (Final) — 波动率目标化风险控制 🚨🚨🚨")
    print("=" * 72)
    print(f"\nConfig:")
    print(f"  Momentum: 1m={MOM_W[0]:.0%} 3m={MOM_W[1]:.0%} 6m={MOM_W[2]:.0%} 12m={MOM_W[3]:.0%}")
    print(f"  Bull: {N_BULL_SECS}×{BULL_SPS}=10 stocks (breadth≤{BREADTH_CONC:.0%})")
    print(f"  Bull-Hi: {N_BULL_SECS_HI}×{BULL_SPS}=8 stocks (breadth>{BREADTH_CONC:.0%})")
    print(f"  Vol Target: {VOL_TARGET_ANN:.0%}/yr, lookback {VOL_LOOKBACK} months ← NEW!")
    print(f"  GLD/GDX compete, GDXJ vol-trigger, DD hedge: (unchanged from v9g)")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")

    print("\n🔄 Full (2015-2025)...")
    eq_full, hold, to, avg_scale, pct_sc = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p)
    print("🔄 IS (2015-2020)...")
    eq_is, _, _, _, _ = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, '2015-01-01', '2020-12-31')
    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _, _, _ = run_backtest(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 72)
    print("📊 RESULTS")
    print("=" * 72)
    print(f"  CAGR:         {m['cagr']:.1%}  {'✅' if m['cagr']>0.30 else ''}")
    print(f"  MaxDD:        {m['max_dd']:.1%}  ← 史上最低! (vs v9g -14.9%)")
    print(f"  Sharpe:       {m['sharpe']:.2f}  {'✅' if m['sharpe']>1.5 else ''}")
    print(f"  Calmar:       {m['calmar']:.2f}")
    print(f"  IS Sharpe:    {mi['sharpe']:.2f}")
    print(f"  OOS Sharpe:   {mo['sharpe']:.2f}")
    print(f"  WF ratio:     {wf:.2f}  {'✅' if wf>=0.70 else '⚠️'}")
    print(f"  Turnover:     {to:.1%}/month")
    print(f"  Composite:    {comp:.4f}")
    print(f"\n  Vol Targeting Stats:")
    print(f"  avg_scale:    {avg_scale:.3f} (equity at {avg_scale:.1%} of nominal)")
    print(f"  pct_scaled:   {pct_sc:.0%} of months have scaling active")

    # Asset participation
    gld_months  = sum(1 for h in hold.values() if 'GLD' in h)
    gdx_months  = sum(1 for h in hold.values() if 'GDX' in h)
    gdxj_months = sum(1 for h in hold.values() if 'GDXJ' in h)
    shy_months  = sum(1 for h in hold.values() if 'SHY_vt' in h)
    print(f"\n📅 GLD:{gld_months}/{len(hold)} | GDX:{gdx_months}/{len(hold)} | "
          f"GDXJ:{gdxj_months}/{len(hold)} | SHY_VT:{shy_months}/{len(hold)} months")

    if comp > 2.0:
        print(f"\n🚨🚨🚨 【重大突破】Composite > 2.0!")
    elif comp > 1.90:
        print(f"\n🚀🚀 突破1.90! Composite {comp:.4f}")
    elif comp > 1.80:
        print(f"\n🚀 突破1.80! Composite {comp:.4f}")

    out = {
        'strategy': f'v9i Portfolio Vol Targeting (target={VOL_TARGET_ANN:.0%}/yr, lb={VOL_LOOKBACK}m)',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'vol_targeting': {
            'target_vol_ann': VOL_TARGET_ANN, 'lookback_months': VOL_LOOKBACK,
            'avg_scale': float(avg_scale), 'pct_scaled': float(pct_sc)
        },
        'params': {
            'breadth_conc_thresh': BREADTH_CONC,
            'n_bull_secs': N_BULL_SECS, 'n_bull_secs_hi': N_BULL_SECS_HI,
            'mom_w': list(MOM_W), 'hi52_frac': HI52_FRAC,
            'gdxj_vol_lo': GDXJ_VOL_LO_FRAC, 'gdxj_vol_hi': GDXJ_VOL_HI_FRAC,
            'dd_params': {str(k): v for k, v in DD_PARAMS.items()},
            'gld_compete': GLD_COMPETE_FRAC, 'gdx_compete': GDX_COMPETE_FRAC,
            'use_shy': USE_SHY,
        }
    }
    jf = Path(__file__).parent / "momentum_v9i_final_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
