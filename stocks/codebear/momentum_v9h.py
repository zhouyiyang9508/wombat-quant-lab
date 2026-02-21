#!/usr/bin/env python3
"""
动量轮动 v9h — 3级集中度 + 加权行业配额 + 跳过1月动量
代码熊 🐻

当前最佳: v9g — Composite 1.759, CAGR 37.2%, Sharpe 1.71, WF 0.78

v9h 三个新方向 (在v9g基础上):

[A] 3级动态行业集中度 (3-Tier Sector Concentration)
    v9g: 2级 (breadth>0.65→4secs, else→5secs)
    v9h: 3级
      breadth > 0.75: 3行业×2=6支 (极度集中,强牛)
      breadth > 0.65: 4行业×2=8支 (集中,宽幅牛)
      breadth ≤ 0.65: 5行业×2=10支 (分散,正常)
    
    理由: 极端breadth (>0.75)时,动量非常集中,
          第4/5个行业更是边际弱势,
          进一步集中到3行业能获得更高α

[B] 加权行业配额 (Weighted Sector Slots)
    当前: 每个行业固定2支
    新: 顶部行业获得额外1支
    
    实现方案1 (weighted_top1):
      顶部1个行业: 3支; 其他: 2支
      5行业: 3+2+2+2+2=11支; 4行业: 3+2+2+2=9支
    
    实现方案2 (weighted_top2):
      顶部2个行业: 3支; 其他: 2支
      5行业: 3+3+2+2+2=12支; 4行业: 3+3+2+2=10支
    
    理由: 排名第1的行业往往拥有显著领先的动量,
          给它额外1个槽位 → 更多α

[C] 跳过最近1月动量 (Skip-1m Momentum)
    学术研究: 过去1个月动量存在"短期反转"效应
    (Jegadeesh 1990, many replication studies)
    当前权重: 1m:20%, 3m:50%, 6m:20%, 12m:10%
    
    测试方案:
      C1 (skip_1m): 1m:0%, 3m:60%, 6m:25%, 12m:15%
        → 完全跳过1月动量
      C2 (reduce_1m): 1m:8%, 3m:55%, 6m:25%, 12m:12%
        → 大幅降低1月权重
    
    注意: 在月度再平衡中, 短期反转效应较弱,
          但值得验证。同时提高3m权重。

[D] 组合: A+B / A+C / B+C / A+B+C
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v9g champion base params
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

BASE_MOM_W = (0.20, 0.50, 0.20, 0.10)


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
    # Skip-month momentum: 2m return = close[t-22] / close[t-44] - 1
    # (return from 2 months ago to 1 month ago, skip most recent month)
    # Actually for "skip 1m, start from 2m": use close/close.shift(44)-1 for "2m ago"
    # More useful: r_2to12 = close.shift(22) / close.shift(252) - 1  (12m momentum skipping 1m)
    # But we already have r12 and r1 separately. For clean implementation:
    # r_skip1m = r3 * (1/(1+r1+1e-9))  — imperfect, just use r3 as "skip1m" proxy
    # Better: use rolling window starting 1m back:
    # r3_skip = close.shift(22) / close.shift(63+22) - 1 = close.shift(22)/close.shift(85)-1
    r3_s1 = close_df.shift(22) / close_df.shift(85)  - 1   # 3m return, start from -1m
    r6_s1 = close_df.shift(22) / close_df.shift(148) - 1   # 6m return, start from -1m
    r12_s1= close_df.shift(22) / close_df.shift(274) - 1   # 12m return, start from -1m
    r52w_hi = close_df.rolling(252).max()
    log_r = np.log(close_df / close_df.shift(1))
    vol5  = log_r.rolling(5).std() * np.sqrt(252)
    vol30 = log_r.rolling(30).std() * np.sqrt(252)
    spy   = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200  = spy.rolling(200).mean() if spy is not None else None
    sma50 = close_df.rolling(50).mean()
    return dict(r1=r1, r3=r3, r6=r6, r12=r12,
                r3_s1=r3_s1, r6_s1=r6_s1, r12_s1=r12_s1,
                r52w_hi=r52w_hi, vol5=vol5, vol30=vol30,
                spy=spy, s200=s200, sma50=sma50, close=close_df)


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
    r6 = sig['r6']
    idx = r6.index[r6.index <= date]
    if len(idx) < 1: return 0.0
    d = idx[-1]
    stock_r6 = r6.loc[d].drop('SPY', errors='ignore').dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lb] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, cfg):
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}
    d = idx[-1]

    # Choose momentum weights
    skip_1m = cfg.get('skip_1m', False)
    mom_w   = cfg.get('mom_w', BASE_MOM_W)
    if skip_1m:
        # Use skip-1m returns: r3_s1, r6_s1, r12_s1 (start from 1m ago)
        w3, w6, w12 = mom_w[1], mom_w[2], mom_w[3]
        # Redistribute 1m weight to 3m
        total = w3 + w6 + w12; w3 /= total; w6 /= total; w12 /= total
        mom = (sig['r3_s1'].loc[d]*w3 + sig['r6_s1'].loc[d]*w6 + sig['r12_s1'].loc[d]*w12)
    else:
        w1, w3, w6, w12 = mom_w
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
    if len(df) == 0: return {}

    sec_mom = df.groupby('sector')['mom'].mean().sort_values(ascending=False)

    gld_a = asset_compete(sig, date, gld_p, GLD_AVG_THRESH, GLD_COMPETE_FRAC)
    gdx_a = asset_compete(sig, date, gdx_p, GDX_AVG_THRESH, GDX_COMPETE_FRAC)
    total_compete = gld_a + gdx_a
    n_compete = (1 if gld_a > 0 else 0) + (1 if gdx_a > 0 else 0)

    reg     = get_regime(sig, date)
    breadth = compute_breadth(sig, date)

    # Sector count decision
    three_tier   = cfg.get('three_tier', False)
    thresh_hi    = cfg.get('thresh_hi', 0.75)
    thresh_lo    = cfg.get('thresh_lo', 0.65)
    weighted_top = cfg.get('weighted_top', 0)   # 0=equal, 1=top1+1slot, 2=top2+1slot

    if reg == 'bull':
        if three_tier and breadth > thresh_hi:
            n_bull_secs = 3   # extreme concentration
        elif breadth > thresh_lo:
            n_bull_secs = N_BULL_SECS_HI  # = 4
        else:
            n_bull_secs = N_BULL_SECS     # = 5
        n_secs = n_bull_secs - n_compete
        base_sps, cash = BULL_SPS, 0.0
    else:
        n_secs   = 3 - n_compete
        n_bull_secs = 3
        base_sps, cash = BEAR_SPS, 0.20
    n_secs = max(n_secs, 1)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []

    if weighted_top > 0 and reg == 'bull':
        # Give extra slot to top N sectors
        for i, sec in enumerate(top_secs):
            extra = 1 if i < weighted_top else 0
            sps_i = base_sps + extra
            sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
            selected.extend(sdf.index[:sps_i].tolist())
    else:
        for sec in top_secs:
            sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
            selected.extend(sdf.index[:base_sps].tolist())

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


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, cfg,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = get_spy_vol(sig, dt)
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
    # Baseline (v9g)
    'v9g_base':    {'three_tier': False, 'weighted_top': 0, 'skip_1m': False},

    # [A] 3-tier concentration
    'A_3tier_75':  {'three_tier': True, 'thresh_hi': 0.75, 'thresh_lo': 0.65, 'weighted_top': 0, 'skip_1m': False},
    'A_3tier_70':  {'three_tier': True, 'thresh_hi': 0.70, 'thresh_lo': 0.65, 'weighted_top': 0, 'skip_1m': False},
    'A_3tier_78':  {'three_tier': True, 'thresh_hi': 0.78, 'thresh_lo': 0.65, 'weighted_top': 0, 'skip_1m': False},

    # [B] Weighted sector slots
    'B_top1':      {'three_tier': False, 'weighted_top': 1, 'skip_1m': False},
    'B_top2':      {'three_tier': False, 'weighted_top': 2, 'skip_1m': False},

    # [C] Skip-1m momentum
    'C_skip1m':    {'three_tier': False, 'weighted_top': 0, 'skip_1m': True,
                    'mom_w': (0.0, 0.60, 0.28, 0.12)},
    'C_reduce1m':  {'three_tier': False, 'weighted_top': 0, 'skip_1m': False,
                    'mom_w': (0.08, 0.55, 0.25, 0.12)},

    # [A+B] Combinations
    'AB_3t75_top1':{'three_tier': True, 'thresh_hi': 0.75, 'thresh_lo': 0.65,
                    'weighted_top': 1, 'skip_1m': False},
    'AB_3t70_top1':{'three_tier': True, 'thresh_hi': 0.70, 'thresh_lo': 0.65,
                    'weighted_top': 1, 'skip_1m': False},

    # [A+C]
    'AC_3t75_skip':{'three_tier': True, 'thresh_hi': 0.75, 'thresh_lo': 0.65,
                    'weighted_top': 0, 'skip_1m': True, 'mom_w': (0.0, 0.60, 0.28, 0.12)},
}


def main():
    print("=" * 72)
    print("🐻 动量轮动 v9h — 3级集中度 + 加权配额 + 跳过1月")
    print("=" * 72)
    print(f"\nBase: v9g champion (Composite 1.759)")
    print(f"Testing {len(CONFIGS)} configurations...\n")

    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    sig    = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers\n")

    results = {}
    best_comp = 0.0
    best_name = ''

    for name, cfg in CONFIGS.items():
        print(f"--- {name} ---", flush=True)
        try:
            eq_f, to = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, cfg)
            eq_is, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, cfg,
                                     '2015-01-01', '2020-12-31')
            eq_oo, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, cfg,
                                     '2021-01-01', '2025-12-31')
            m  = compute_metrics(eq_f)
            mi = compute_metrics(eq_is)
            mo = compute_metrics(eq_oo)
            wf   = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            results[name] = {'full': m, 'is': mi, 'oos': mo, 'wf': wf, 'composite': comp, 'to': to}
            tag = '🚀🚀' if comp > 1.80 else ('🚀' if comp > 1.759 else ('✅' if comp > 1.70 else ''))
            wf_tag = '✅' if wf >= 0.70 else ('⚠️' if wf >= 0.60 else '❌')
            print(f"  Comp={comp:.4f} {tag} | Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} "
                  f"DD={m['max_dd']:.1%} WF={wf:.2f} {wf_tag}")
            if comp > best_comp:
                best_comp = comp
                best_name = name
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 72)
    print("📊 FINAL RANKINGS")
    print("=" * 72)
    ranked = sorted(results.items(), key=lambda x: x[1]['composite'], reverse=True)
    for i, (n, r) in enumerate(ranked[:8]):
        m = r['full']
        tag = '🚀🚀' if r['composite'] > 1.80 else ('🚀' if r['composite'] > 1.759 else '')
        wft = '✅' if r['wf'] >= 0.70 else ('⚠️' if r['wf'] >= 0.60 else '❌')
        print(f"  #{i+1} {n:20s}: Comp={r['composite']:.4f} {tag} | "
              f"Sh={m['sharpe']:.2f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%} WF={r['wf']:.2f} {wft}")

    print(f"\n🏆 Best: {best_name} → Composite {best_comp:.4f}")
    if best_comp > 1.80:
        print("🚨🚨🚨 【重大突破】Composite > 1.80!")
    elif best_comp > 1.759:
        print(f"🚀 超越v9g! +{best_comp-1.759:.4f}")
    else:
        print(f"❌ 未超越v9g (1.759)")

    out = {'configs': CONFIGS, 'results': {n: {'full': r['full'], 'is': r['is'], 'oos': r['oos'],
           'wf': r['wf'], 'composite': r['composite']} for n, r in results.items()},
           'best': best_name, 'best_composite': best_comp}
    jf = Path(__file__).parent / "momentum_v9h_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results → {jf}")
    return results, best_name, best_comp


if __name__ == '__main__':
    main()
