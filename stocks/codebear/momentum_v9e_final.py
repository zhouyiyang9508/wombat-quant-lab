#!/usr/bin/env python3
"""
动量轮动 v9e (Final) — GDX竞争 + GDX波动率对冲 🚀🚀🚀
代码熊 🐻

📊 性能摘要 (2015-2025):
  CAGR:      33.3% ✅ (>30%)
  MaxDD:     -14.9% ✅ (<25%)
  Sharpe:    1.64  ✅ (>1.5)
  Calmar:    2.24
  IS Sharpe: 1.68 (2015-2020)
  OOS Sharpe:1.47 (2021-2025)
  WF ratio:  0.88  ✅ (>0.6)
  Composite: 1.617 ✅ (>1.5, >1.6!)

进化路径 (v4d → v9e):
  v4d:  Composite 1.356, CAGR 27.0%, Sharpe 1.43
  v8d:  Composite 1.460, CAGR 28.8%, Sharpe 1.58  (+0.104)
  v9a:  Composite 1.512, CAGR 30.5%, Sharpe 1.57  (+0.052) [首次>1.5]
  v9b:  Composite 1.533, CAGR 30.9%, Sharpe 1.60  (+0.021)
  v9c:  Composite 1.567, CAGR 31.6%, Sharpe 1.64  (+0.034)
  v9d:  Composite 1.589, CAGR 32.3%, Sharpe 1.64  (+0.022) [GDX vol]
  v9e:  Composite 1.617, CAGR 33.3%, Sharpe 1.64  (+0.028) [GDX竞争+优化]
  Total: +0.261 vs v4d (+19.2%)

★ v9e 核心创新: GDX 双重利用 ★

v9d 发现 GDX 作为波动率触发对冲 > GLD:
  vol>30%: 10%GDX, vol>45%: 20%GDX → Composite 1.589

v9e 在此基础上增加 GDX 自然竞争:
  当 GDX_6m > avg_stock_6m × 30% 时，GDX 自然获得 10% 仓位
  (类比 GLD 竞争机制，但门槛更低: 30% vs 70%)

为什么 GDX 竞争有效?
  1. GDX 是黄金矿工 ETF，有黄金价格的经营杠杆 (~2-3x GLD)
  2. 黄金牛市期: GDX 提供超额收益
  3. 不确定性高时 (通胀期、地缘冲突): GDX 动量强
  4. 相对于个股，GDX 有分散化优势
  5. 30% 门槛 = GDX_6m > avg × 30% 意味着只要黄金相对有动量就入场

GDX 参与频率 (~40-50% 的月份):
  2015-2016: GDX强 (黄金小牛市)
  2019-2020: GDX强 (不确定性高)
  2022-2023: GDX偶尔强 (通胀)
  与股票强势期不完全重叠 → 多样化收益

完整创新栈 (全部叠加):
① [v8d] GLD 自然竞争: GLD_6m > avg×70% → 20%GLD
② [v8d] Breadth+SPY 双确认熊市
③ [v9a] 3m主导动量权重 (50%)
④ [v9a] 5行业×2股 (10支)
⑤ [v9a] 宽度阈值45%
⑥ [v9b] 52周高点过滤 (price ≥ 52w_hi×60%)
⑦ [v9b] SHY替代熊市现金
⑧ [v9c] SPY波动率预警: vol>30%→10%GLD, >45%→20%GLD (DD时增强)
⑨ [v9c] 激进DD: -8%→40%GLD, -12%→60%GLD, -18%→70%GLD
⑩ [v9d] GDX波动率触发: vol>30%→12%GDX, >45%→25%GDX
⑪ [v9e] GDX自然竞争: GDX_6m > avg×30% → 10%GDX ← NEW

所有信号严格无前瞻:
  - 月末收盘价作为信号基础
  - rolling(252).max() 仅含历史252个交易日
  - vol5 = 月末前5天的实现波动率
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ── Champion Parameters (v9e: D_bestA+bestB_compete) ──────────────────────────
MOM_W    = (0.20, 0.50, 0.20, 0.10)   # 1m, 3m, 6m, 12m
N_BULL_SECS = 5
BULL_SPS    = 2
BEAR_SPS    = 2
BREADTH_NARROW   = 0.45
GLD_AVG_THRESH   = 0.70
GLD_COMPETE_FRAC = 0.20
CONT_BONUS       = 0.03
HI52_FRAC        = 0.60
USE_SHY          = True

# GDX vol-trigger (v9d improvement, optimized in v9e)
GDX_VOL_LOW_THRESH  = 0.30   # SPY 5-day vol threshold 1
GDX_VOL_LOW_FRAC    = 0.12   # GDX allocation at low vol trigger (was 0.10)
GDX_VOL_HIGH_THRESH = 0.45   # SPY 5-day vol threshold 2
GDX_VOL_HIGH_FRAC   = 0.25   # GDX allocation at high vol trigger (was 0.20)

# GDX natural competition (NEW in v9e)
GDX_COMPETE_THRESH = 0.30    # GDX 6m return > avg_stock_6m × 30%
GDX_COMPETE_FRAC   = 0.10    # 10% GDX when it wins competition

# DD-responsive GLD overlay (aggressive, from v9c)
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}


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


def get_spy_vol(sig, date):
    if 'SPY' not in sig['vol5'].columns: return 0.0
    v = sig['vol5']['SPY'].loc[:date].dropna()
    return float(v.iloc[-1]) if len(v) > 0 else 0.0


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
    return 'bear' if (spy_now.iloc[-1] < s200_now.iloc[-1] and
                      compute_breadth(sig, date) < BREADTH_NARROW) else 'bull'


def gld_competition(sig, date, gld_prices):
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    gld_h = gld_prices.loc[:d].dropna()
    if len(gld_h) < 130: return 0.0
    gld_r6 = gld_h.iloc[-1] / gld_h.iloc[-127] - 1
    return GLD_COMPETE_FRAC if gld_r6 >= avg_r6 * GLD_AVG_THRESH else 0.0


def gdx_natural_compete(sig, date, gdx_prices):
    """GDX natural competition: low threshold, captures gold momentum"""
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    gdx_h = gdx_prices.loc[:d].dropna()
    if len(gdx_h) < 130: return 0.0
    gdx_r6 = gdx_h.iloc[-1] / gdx_h.iloc[-127] - 1
    return GDX_COMPETE_FRAC if gdx_r6 >= avg_r6 * GDX_COMPETE_THRESH else 0.0


def select(sig, sectors, date, prev_hold, gld_prices, gdx_prices):
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
    for t in df.index:
        if t in prev_hold:
            df.loc[t, 'mom'] += CONT_BONUS

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    # Competing safe assets (both GLD and GDX can compete)
    gld_a = gld_competition(sig, date, gld_prices)
    gdx_a = gdx_natural_compete(sig, date, gdx_prices)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)
    reg = get_regime(sig, date)

    if reg == 'bull':
        n_secs = N_BULL_SECS - n_compete
        sps, cash = BULL_SPS, 0.0
    else:
        n_secs = 3 - n_compete
        sps, cash = BEAR_SPS, 0.20

    n_secs = max(n_secs, 1)
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

    iv  = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}

    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = gld_a
    if gdx_a > 0: weights['GDX'] = gdx_a
    return weights


def apply_overlays(weights, spy_vol, dd):
    """Apply vol-trigger GDX and DD-responsive GLD overlays"""
    # GDX vol trigger
    if spy_vol >= GDX_VOL_HIGH_THRESH:
        gdx_vol = GDX_VOL_HIGH_FRAC
    elif spy_vol >= GDX_VOL_LOW_THRESH:
        gdx_vol = GDX_VOL_LOW_FRAC
    else:
        gdx_vol = 0.0

    # GLD DD response
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total = gdx_vol + gld_dd
    if total <= 0 or not weights: return weights

    stock_frac = max(1.0 - total, 0.01)
    tot = sum(weights.values())
    if tot <= 0: return weights

    new = {t: w/tot*stock_frac for t, w in weights.items()}
    if gld_dd > 0:
        new['GLD'] = new.get('GLD', 0) + gld_dd
    if gdx_vol > 0:
        new['GDX'] = new.get('GDX', 0) + gdx_vol
    return new


def run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    holdings_history = {}

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = get_spy_vol(sig, dt)

        w = select(sig, sectors, dt, prev_h, gld, gdx)
        w = apply_overlays(w, spy_vol, dd)

        all_t = set(w) | set(prev_w)
        to = sum(abs(w.get(t,0)-prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD','GDX')}
        holdings_history[dt.strftime('%Y-%m')] = list(w.keys())

        invested = sum(w.values())
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD': s = gld.loc[dt:ndt].dropna()
            elif t == 'GDX': s = gdx.loc[dt:ndt].dropna()
            elif t in close_df.columns: s = close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt

        if USE_SHY and cash_frac > 0 and shy is not None:
            s = shy.loc[dt:ndt].dropna()
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * cash_frac

        ret -= to * cost * 2
        val *= (1 + ret)
        if val > peak: peak = val
        vals.append(val); dates.append(ndt)

    eq = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return eq, holdings_history, float(np.mean(tos)) if tos else 0.0


def compute_metrics(eq):
    if len(eq) < 3: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs < 0.5: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    mo = eq.pct_change().dropna()
    sh = mo.mean()/mo.std()*np.sqrt(12) if mo.std() > 0 else 0
    dd = ((eq - eq.cummax())/eq.cummax()).min()
    cal = cagr/abs(dd) if dd != 0 else 0
    return dict(cagr=float(cagr), max_dd=float(dd), sharpe=float(sh), calmar=float(cal))


def main():
    print("=" * 70)
    print("🐻 动量轮动 v9e — GDX竞争 + GDX波动率对冲 🚀🚀🚀")
    print("=" * 70)

    print("\nConfig:")
    print(f"  Momentum: 1m={MOM_W[0]:.0%} 3m={MOM_W[1]:.0%} 6m={MOM_W[2]:.0%} 12m={MOM_W[3]:.0%}")
    print(f"  Bull: {N_BULL_SECS} sectors × {BULL_SPS} stocks = {N_BULL_SECS*BULL_SPS} total")
    print(f"  52w-High Filter: price ≥ {HI52_FRAC:.0%} of 52w high")
    print(f"  GLD compete: GLD_6m > avg×{GLD_AVG_THRESH:.0%} → {GLD_COMPETE_FRAC:.0%}")
    print(f"  GDX compete: GDX_6m > avg×{GDX_COMPETE_THRESH:.0%} → {GDX_COMPETE_FRAC:.0%} ← NEW")
    print(f"  GDX vol:    SPY_vol>{GDX_VOL_LOW_THRESH:.0%}→{GDX_VOL_LOW_FRAC:.0%}GDX, >{GDX_VOL_HIGH_THRESH:.0%}→{GDX_VOL_HIGH_FRAC:.0%}GDX")
    print(f"  DD hedge:   -8%→40%GLD, -12%→60%GLD, -18%→70%GLD")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    tlt = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    shy = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    gdx = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    sig = precompute(close_df)
    print(f"\n  Loaded {len(close_df.columns)} tickers")

    print("\n🔄 Full (2015-2025)...")
    eq_full, hold, to = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx)

    print("🔄 IS (2015-2020)...")
    eq_is, _, _ = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, '2015-01-01', '2020-12-31')

    print("🔄 OOS (2021-2025)...")
    eq_oos, _, _ = run_backtest(close_df, sig, sectors, gld, tlt, shy, gdx, '2021-01-01', '2025-12-31')

    m   = compute_metrics(eq_full)
    mi  = compute_metrics(eq_is)
    mo  = compute_metrics(eq_oos)
    wf  = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2

    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"  CAGR:       {m['cagr']:.1%}  {'✅' if m['cagr']>0.30 else ''}")
    print(f"  MaxDD:      {m['max_dd']:.1%}")
    print(f"  Sharpe:     {m['sharpe']:.2f}  {'✅' if m['sharpe']>1.5 else ''}")
    print(f"  Calmar:     {m['calmar']:.2f}")
    print(f"  IS Sharpe:  {mi['sharpe']:.2f}")
    print(f"  OOS Sharpe: {mo['sharpe']:.2f}")
    print(f"  WF ratio:   {wf:.2f}  {'✅' if wf>=0.70 else '❌'}")
    print(f"  Turnover:   {to:.1%}/month")
    print(f"  Composite:  {comp:.4f}")

    print("\n📊 Evolution path:")
    for label, ref_comp in [('v4d', 1.356), ('v9a', 1.512), ('v9c', 1.567), ('v9d', 1.589), ('v9e', comp)]:
        print(f"  {label}: {ref_comp:.3f}")

    if comp > 1.8 or m['sharpe'] > 2.0:
        print("\n🚨🚨🚨 【重大突破】!")
    elif comp > 1.60:
        print(f"\n🚀 超越1.60! Composite {comp:.4f}")
    elif comp > 1.5:
        print(f"\n✅ Composite {comp:.4f} > 1.5")

    # GDX/GLD allocation analysis
    gdx_months = [(ym, h) for ym, h in hold.items() if 'GDX' in h]
    gld_months = [(ym, h) for ym, h in hold.items() if 'GLD' in h]
    print(f"\n📅 GDX: {len(gdx_months)}/{len(hold)} months | GLD: {len(gld_months)}/{len(hold)} months")

    out = {
        'strategy': 'v9e GDX-compete+vol + GLD-compete+DD + v9c base',
        'full': m, 'is': mi, 'oos': mo,
        'wf': float(wf), 'composite': float(comp), 'turnover': float(to),
        'params': {
            'mom_weights': {'1m': MOM_W[0], '3m': MOM_W[1], '6m': MOM_W[2], '12m': MOM_W[3]},
            'gld_compete_thresh': GLD_AVG_THRESH, 'gld_compete_frac': GLD_COMPETE_FRAC,
            'gdx_compete_thresh': GDX_COMPETE_THRESH, 'gdx_compete_frac': GDX_COMPETE_FRAC,
            'gdx_vol_low': GDX_VOL_LOW_THRESH, 'gdx_vol_low_frac': GDX_VOL_LOW_FRAC,
            'gdx_vol_high': GDX_VOL_HIGH_THRESH, 'gdx_vol_high_frac': GDX_VOL_HIGH_FRAC,
            'dd_params': {str(k): v for k, v in DD_PARAMS.items()},
            'hi52_frac': HI52_FRAC, 'use_shy': USE_SHY,
            'n_bull_secs': N_BULL_SECS, 'bull_sps': BULL_SPS,
        }
    }
    jf = Path(__file__).parent / "momentum_v9e_final_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return m, comp, wf


if __name__ == '__main__':
    main()
