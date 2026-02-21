#!/usr/bin/env python3
"""
动量轮动 v9i — 投资组合波动率目标化 + TLT利率环境过滤
代码熊 🐻

当前最佳: v9g — Composite 1.759, CAGR 37.2%, Sharpe 1.71, WF 0.78

v9h/v9h2/v9h3 教训总结:
  ❌ 动量加速过滤 (accel>0.5): Comp 0.90-1.48, 严重破坏性
  ❌ 横截面选股 (cross-sectional): Comp 1.34-1.37, 远低于sector-first
  ❌ 行业ETF加权排名: 中性 (Comp不变), ETF动量已被股票价格捕捉  
  ❌ Alpha动量 vs 绝对动量: 基本中性
  ❌ 跳过1月动量: 月度频率下无意义 (Comp 0.92)
  ✅ Sector-first, breadth>0.65→4secs 是最优框架 (v9g)

v9i 两个新方向 (完全不同的维度):

[A] 投资组合波动率目标化 (Portfolio Volatility Targeting)
    当前: 固定持仓结构, 不管组合整体风险
    新: 计算过去3个月组合月度收益率的标准差
       目标波动率 = 20%/年 (约5.77%/月×√12)
       当portfolio_vol > target_vol 时, 按比例缩减股票仓位至目标
       缩减部分放入SHY (安全利息)
    
    数学: scale = min(target_vol / port_vol, 1.0)
          equity_total_new = equity_total_old × scale
          shy_new = shy + (equity_total_old - equity_total_new)
    
    优势:
      - 降低2022年式熊市的损失 (market_vol↑ → scale↓ → 减少股票)
      - 在低波动时保持正常杠杆 (scale=1.0)
      - 不依赖任何预测信号, 纯机械式风险控制
      - 与现有GDXJ vol触发层是互补的 (不是替代)
    
    参数: target_vol_annual ∈ {0.15, 0.18, 0.20, 0.22}
    初始portfolio_vol = 先用全期均值bootstrap前3个月

[B] TLT利率环境过滤 (TLT Rate Environment Filter)
    当前: 策略不感知利率环境
    新: 当TLT 3m动量 < -X% (利率快速上升), 减少权益仓位
    
    2022年验证:
      - TLT在2022年跌约27% (6m return约-18%)
      - 这种级别的债券跌幅预示急剧紧缩
      - 此时权益应更加防御
    
    实现:
      - 月末计算TLT 3m return (无前瞻, 使用close[i-1])
      - 如果 TLT_3m < thresh (e.g. -5%, -8%, -10%):
          添加5-10% SHY并缩减同等权益仓位
    
    参数: thresh ∈ {-0.05, -0.08, -0.10}; shy_boost ∈ {0.05, 0.10}

[C] 组合: A+B
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


def get_tlt_r3(tlt_p, date):
    """TLT 3m return ending at date"""
    hist = tlt_p.loc[:date].dropna()
    if len(hist) < 64: return 0.0
    return float(hist.iloc[-1] / hist.iloc[-64] - 1)


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


def apply_overlays(weights, spy_vol, dd, port_vol_ann, cfg):
    """Apply GDXJ vol-trigger, GLD DD response, vol targeting, TLT filter"""
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    # Apply GDXJ and GLD first (standard overlays)
    total = gdxj_v + gld_dd
    if total > 0 and weights:
        stock_frac = max(1.0 - total, 0.01)
        tot = sum(weights.values())
        if tot > 0:
            weights = {t: w/tot*stock_frac for t, w in weights.items()}
        if gld_dd > 0: weights['GLD'] = weights.get('GLD', 0) + gld_dd
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    # [A] Portfolio volatility targeting
    target_vol = cfg.get('target_vol', None)
    if target_vol is not None and port_vol_ann > 0.01:
        # Scale equity (non-hedge) positions down
        equity_keys = [t for t in weights if t not in ('GLD','GDX','GDXJ','SHY')]
        equity_frac = sum(weights[t] for t in equity_keys)
        if equity_frac > 0:
            # Estimate equity portion vol (approximate: port_vol proportional to equity_frac)
            est_eq_vol = port_vol_ann / max(equity_frac, 0.01)
            scale = min(target_vol / max(port_vol_ann, 0.01), 1.0)
            if scale < 0.98:  # Only adjust if meaningful
                new_equity = equity_frac * scale
                shy_boost = equity_frac - new_equity
                for t in equity_keys:
                    weights[t] = weights[t] * scale
                weights['_SHY_VTAR'] = shy_boost  # tagged separately

    # [B] TLT rate environment filter
    tlt_r3  = cfg.get('_tlt_r3_now', 0.0)
    tlt_thresh = cfg.get('tlt_thresh', None)
    if tlt_thresh is not None and tlt_r3 < tlt_thresh:
        shy_boost2 = cfg.get('tlt_shy_boost', 0.05)
        equity_keys = [t for t in weights if t not in ('GLD','GDX','GDXJ','SHY','_SHY_VTAR')]
        equity_frac = sum(weights[t] for t in equity_keys)
        if equity_frac > shy_boost2:
            scale2 = (equity_frac - shy_boost2) / equity_frac
            for t in equity_keys:
                weights[t] = weights[t] * scale2
            weights['_SHY_TLT'] = shy_boost2

    return weights


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, cfg,
                 start='2015-01-01', end='2025-12-31', cost=0.0015):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates, tos = [], [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []  # track monthly returns for vol targeting
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0 else 0.15

        # Compute portfolio vol from past 3 months of returns
        if len(port_returns) >= 3:
            port_vol_mon = np.std(port_returns[-3:], ddof=1)
            port_vol_ann = port_vol_mon * np.sqrt(12)
        else:
            port_vol_ann = 0.20  # bootstrap

        # TLT rate environment
        cfg_run = dict(cfg)
        if cfg.get('tlt_thresh') is not None:
            cfg_run['_tlt_r3_now'] = get_tlt_r3(tlt_p, dt)

        w = select(sig, sectors, dt, prev_h, gld_p, gdx_p)
        w = apply_overlays(w, spy_vol, dd, port_vol_ann, cfg_run)

        # Consolidate SHY entries
        shy_total = w.pop('_SHY_VTAR', 0) + w.pop('_SHY_TLT', 0)

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t,0) - prev_w.get(t,0)) for t in all_t) / 2
        tos.append(to); prev_w = w.copy()
        prev_h = {k for k in w if k not in ('GLD','GDX','GDXJ')}

        invested = sum(w.values()) + shy_total
        cash_frac = max(1.0 - invested, 0.0)
        ret = 0.0
        for t, wt in w.items():
            if   t == 'GLD':  s = gld_p.loc[dt:ndt].dropna()
            elif t == 'GDX':  s = gdx_p.loc[dt:ndt].dropna()
            elif t == 'GDXJ': s = gdxj_p.loc[dt:ndt].dropna()
            elif t in close_df.columns: s = close_df[t].loc[dt:ndt].dropna()
            else: continue
            if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt
        # SHY for: vol-targeting excess + TLT filter excess + natural cash
        total_shy = shy_total + (cash_frac if USE_SHY else 0.0)
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


CONFIGS = {
    'v9g_base':       {'target_vol': None, 'tlt_thresh': None},
    # [A] Vol targeting
    'A_vt20':         {'target_vol': 0.20, 'tlt_thresh': None},
    'A_vt18':         {'target_vol': 0.18, 'tlt_thresh': None},
    'A_vt15':         {'target_vol': 0.15, 'tlt_thresh': None},
    'A_vt22':         {'target_vol': 0.22, 'tlt_thresh': None},
    # [B] TLT filter
    'B_tlt05_5':      {'target_vol': None, 'tlt_thresh': -0.05, 'tlt_shy_boost': 0.05},
    'B_tlt08_10':     {'target_vol': None, 'tlt_thresh': -0.08, 'tlt_shy_boost': 0.10},
    'B_tlt05_10':     {'target_vol': None, 'tlt_thresh': -0.05, 'tlt_shy_boost': 0.10},
    'B_tlt10_10':     {'target_vol': None, 'tlt_thresh': -0.10, 'tlt_shy_boost': 0.10},
    # [C] Combos
    'AB_vt18_tlt08':  {'target_vol': 0.18, 'tlt_thresh': -0.08, 'tlt_shy_boost': 0.10},
    'AB_vt20_tlt05':  {'target_vol': 0.20, 'tlt_thresh': -0.05, 'tlt_shy_boost': 0.05},
}


def main():
    print("=" * 72)
    print("🐻 v9i — 波动率目标化 + TLT利率过滤")
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
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    sig    = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers, TLT loaded: {len(tlt_p)} rows")

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"--- {name} ---", flush=True)
        try:
            eq_f, to = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, cfg)
            eq_is, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, cfg,
                                     '2015-01-01', '2020-12-31')
            eq_oo, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, cfg,
                                     '2021-01-01', '2025-12-31')
            m  = compute_metrics(eq_f)
            mi = compute_metrics(eq_is)
            mo = compute_metrics(eq_oo)
            wf   = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
            comp = m['sharpe']*0.4 + m['calmar']*0.4 + m['cagr']*0.2
            results[name] = {'full': m, 'is': mi, 'oos': mo, 'wf': wf, 'composite': comp, 'to': to}
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
        print(f"🚀 超越v9g! +{best[1]['composite']-1.759:.4f}")
    else:
        print("❌ 未超越v9g (1.759)")

    jf = Path(__file__).parent / "momentum_v9i_results.json"
    jf.write_text(json.dumps({'results': {n: {'full': r['full'], 'is': r['is'],
        'oos': r['oos'], 'wf': r['wf'], 'composite': r['composite']} for n, r in results.items()},
        'best': best[0], 'best_composite': best[1]['composite']}, indent=2))
    print(f"💾 Results → {jf}")

if __name__ == '__main__':
    main()
