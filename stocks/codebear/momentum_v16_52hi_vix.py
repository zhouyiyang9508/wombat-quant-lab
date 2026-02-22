#!/usr/bin/env python3
"""
动量轮动 v16 — 52周高位信号增强 + VIX宏观风险门控
代码熊 🐻 | 2026-02-22

基于 v11b Final，叠加两个正交改进：

  ★ Direction A: 52周高位信号增强
    - 研究依据：George & Hwang 2004 发现接近52周高的股票动量更强
    - 实现：price/hi52（接近度比率）→ 增强动量得分
    - 参数：hi52_alpha（0.0=不用，0.5=中等增强，1.0=强增强）
    - 公式：enhanced_mom = raw_mom × (1 + alpha × max(0, price/hi52 - proximity_thresh))
    - 注意：hi52 已在 precompute 中计算，只做排名增强不改过滤

  ★ Direction B: VIX/波动率宏观门控
    - 数据：用 SPY 30日已实现波动率替代 VIX（已有，高度相关）
    - 逻辑：
      Level 1: vol30_SPY < 0.15 (VIX≈15) → 正常
      Level 2: 0.15 ≤ vol30 < 0.25 (VIX≈15-25) → vol_target↑ to 0.13
      Level 3: 0.25 ≤ vol30 < 0.35 (VIX≈25-35) → vol_target↓ to 0.08
      Level 4: vol30 ≥ 0.35 (VIX≈35+) → 全仓 SHY
    - 与现有 vol_target (0.11) 的区别：现在 vol_target 基于组合月频收益标准差
      新层基于 SPY 实时日频波动率，更前瞻

评分公式：Composite = Sharpe×0.4 + Calmar×0.4 + min(CAGR,1.0)×0.2
WF：IS=2015-2020，OOS=2021-2025（固定日期边界）
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ══════════════════════════════════════════════════════════════════════════════
# v11b 原始参数（不改动）
# ══════════════════════════════════════════════════════════════════════════════
MOM_W = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS = 5; N_BULL_SECS_HI = 4; BREADTH_CONC = 0.65
BULL_SPS = 2; BEAR_SPS = 2; BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
CONT_BONUS = 0.03; HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18
VOL_TARGET_ANN = 0.11; VOL_LOOKBACK = 3
TLT_BEAR_FRAC = 0.25; IEF_BEAR_FRAC = 0.20; BOND_MOM_LB = 126
DEFENSIVE_ETFS = ['XLV', 'XLP', 'XLU']
DEFENSIVE_FRAC = 0.12
DEFENSIVE_EACH = DEFENSIVE_FRAC / len(DEFENSIVE_ETFS)
SPY_SOFT_HI_THRESH = -0.07; SPY_SOFT_HI_FRAC = 0.08

# ══════════════════════════════════════════════════════════════════════════════
# 新参数 — Direction A: 52周高位信号增强
# ══════════════════════════════════════════════════════════════════════════════
HI52_ALPHA = 0.50          # 增强强度（0=关闭，1=强增强）
HI52_PROXIMITY_THRESH = 0.85  # 只有 price/hi52 > 0.85 时才增强

# ══════════════════════════════════════════════════════════════════════════════
# 新参数 — Direction B: VIX/波动率宏观门控
# ══════════════════════════════════════════════════════════════════════════════
VIX_L2_THRESH = 0.15   # SPY vol30 阈值（≈VIX 15）
VIX_L3_THRESH = 0.25   # ≈VIX 25
VIX_L4_THRESH = 0.35   # ≈VIX 35（极端恐慌 → 全SHY）
VIX_L2_VOL_TARGET = 0.13  # Level 2: 略放松 vol target
VIX_L3_VOL_TARGET = 0.08  # Level 3: 收紧 vol target
VIX_ENABLED = True

HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'XLV', 'XLP', 'XLU'}


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# 信号预计算（与 v11b 完全相同）
# ══════════════════════════════════════════════════════════════════════════════
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


def get_spy_vol30(sig, date):
    """获取SPY 30日已实现波动率（VIX proxy）"""
    if 'SPY' not in sig['vol30'].columns: return 0.15
    hist = sig['vol30']['SPY'].loc[:date].dropna()
    return float(hist.iloc[-1]) if len(hist) > 0 else 0.15


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
    stock_r6 = r6.loc[d].dropna()
    stock_r6 = stock_r6[stock_r6 > 0]
    if len(stock_r6) < 10: return 0.0
    avg_r6 = stock_r6.mean()
    hist = prices.loc[:d].dropna()
    if len(hist) < lb + 3: return 0.0
    asset_r6 = hist.iloc[-1] / hist.iloc[-lb] - 1
    return frac if asset_r6 >= avg_r6 * thresh else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 核心选股逻辑（含 Direction A: 52周高位增强）
# ══════════════════════════════════════════════════════════════════════════════
def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices,
           hi52_alpha=HI52_ALPHA, hi52_prox_thresh=HI52_PROXIMITY_THRESH):
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

    # ★ Direction A: 52周高位信号增强 ★
    if hi52_alpha > 0 and len(df) > 0:
        proximity = df['price'] / df['hi52']  # 0.6 ~ 1.0
        # 只对接近52周高位的股票增强：proximity > thresh 时才加分
        boost = (proximity - hi52_prox_thresh).clip(lower=0)  # 0 ~ 0.15
        df['mom'] = df['mom'] * (1 + hi52_alpha * boost)

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
        n_secs = max(N_BULL_SECS_HI - n_compete, 1)
        sps, bear_cash = BULL_SPS, 0.0
    elif reg == 'soft_bull':
        n_secs = max(N_BULL_SECS - n_compete, 1)
        sps, bear_cash = BULL_SPS, 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        bear_cash = max(0.20 - actual_bond_frac, 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
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


# ══════════════════════════════════════════════════════════════════════════════
# Overlays（含 Direction B: VIX门控）
# ══════════════════════════════════════════════════════════════════════════════
def apply_overlays(weights, spy_vol5, dd, port_vol_ann, spy_1m_ret,
                   spy_vol30=0.15, vix_enabled=VIX_ENABLED):
    """
    v11b overlays + Direction B: VIX门控

    Direction B 逻辑:
      spy_vol30 是 SPY 30日已实现波动率（VIX proxy）
      - < 0.15: 正常
      - 0.15-0.25: vol_target 调到 0.13（轻微放松，因为此时市场波动正常偏高）
      - 0.25-0.35: vol_target 调到 0.08（大幅收紧）
      - > 0.35: 全仓 SHY（极端恐慌）
    """
    # ★ Direction B: VIX门控 ★
    if vix_enabled and spy_vol30 >= VIX_L4_THRESH:
        # Level 4: 极端恐慌 → 全仓 SHY
        return {'SHY': 1.0}, 1.0

    # 确定有效 vol_target
    if vix_enabled:
        if spy_vol30 >= VIX_L3_THRESH:
            effective_vol_target = VIX_L3_VOL_TARGET  # 0.08
        elif spy_vol30 >= VIX_L2_THRESH:
            effective_vol_target = VIX_L2_VOL_TARGET  # 0.13
        else:
            effective_vol_target = VOL_TARGET_ANN     # 0.11 (default)
    else:
        effective_vol_target = VOL_TARGET_ANN

    # GDXJ overlay
    if spy_vol5 >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol5 >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0

    # GLD DD + SPY soft hedge
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

    # Vol targeting（使用 VIX 调整后的 effective_vol_target）
    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(effective_vol_target / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            equity_keys = [t for t in weights if t not in HEDGE_KEYS]
            eq_frac = sum(weights[t] for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys: weights[t] *= scale
                shy_boost = eq_frac * (1.0 - scale)

    return weights, shy_boost


# ══════════════════════════════════════════════════════════════════════════════
# 月频回测
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                 tlt_p, ief_p, def_prices,
                 start='2015-01-01', end='2025-12-31', cost=0.0015,
                 hi52_alpha=HI52_ALPHA, hi52_prox_thresh=HI52_PROXIMITY_THRESH,
                 vix_enabled=VIX_ENABLED):
    rng  = close_df.loc[start:end].dropna(how='all')
    ends = rng.resample('ME').last().index
    vals, dates = [], []
    prev_w, prev_h = {}, set()
    val = 1.0; peak = 1.0
    port_returns = []
    regime_hist = {'bull_hi': 0, 'soft_bull': 0, 'bear': 0}
    vix_actions = {'normal': 0, 'L2_relax': 0, 'L3_tight': 0, 'L4_shy': 0}
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    for i in range(len(ends) - 1):
        dt, ndt = ends[i], ends[i+1]
        dd = (val - peak) / peak if peak > 0 else 0
        spy_vol5 = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
            SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15
        spy_1m = get_spy_1m(sig, dt)
        spy_vol30 = get_spy_vol30(sig, dt)

        # Track VIX actions
        if vix_enabled:
            if spy_vol30 >= VIX_L4_THRESH: vix_actions['L4_shy'] += 1
            elif spy_vol30 >= VIX_L3_THRESH: vix_actions['L3_tight'] += 1
            elif spy_vol30 >= VIX_L2_THRESH: vix_actions['L2_relax'] += 1
            else: vix_actions['normal'] += 1

        if len(port_returns) >= VOL_LOOKBACK:
            pv = np.std(port_returns[-VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        else:
            pv = 0.20

        w, reg, bond_t = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, ief_p,
                                def_prices, hi52_alpha, hi52_prox_thresh)
        w, shy_boost = apply_overlays(w, spy_vol5, dd, pv, spy_1m,
                                       spy_vol30, vix_enabled)

        regime_hist[reg] = regime_hist.get(reg, 0) + 1

        all_t = set(w) | set(prev_w)
        to    = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
        prev_w = w.copy()
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
            elif t == 'SHY':  s = shy_p.loc[dt:ndt].dropna()
            elif t in def_prices and def_prices[t] is not None:
                s = def_prices[t].loc[dt:ndt].dropna()
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
    return eq, regime_hist, vix_actions


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


def compute_wf_fixed(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                     hi52_alpha=HI52_ALPHA, hi52_prox_thresh=HI52_PROXIMITY_THRESH,
                     vix_enabled=VIX_ENABLED):
    """Walk-Forward IS=2015-2020, OOS=2021-2025 (fixed boundary)"""
    eq_is, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                tlt_p, ief_p, def_prices, '2015-01-01', '2020-12-31',
                                hi52_alpha=hi52_alpha, hi52_prox_thresh=hi52_prox_thresh,
                                vix_enabled=vix_enabled)
    eq_oos, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                 tlt_p, ief_p, def_prices, '2021-01-01', '2025-12-31',
                                 hi52_alpha=hi52_alpha, hi52_prox_thresh=hi52_prox_thresh,
                                 vix_enabled=vix_enabled)
    mi = compute_metrics(eq_is)
    mo = compute_metrics(eq_oos)
    wf = mo['sharpe'] / mi['sharpe'] if mi['sharpe'] > 0 else 0
    return wf, mi, mo


# ══════════════════════════════════════════════════════════════════════════════
# 主流程：参数扫描
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("🐻 动量轮动 v16 — 52周高位信号增强 + VIX宏观风险门控")
    print("基于 v11b Final，叠加 Direction A + Direction B")
    print("=" * 80)

    # 加载数据
    tickers  = (CACHE / "sp500_tickers.txt").read_text().strip().split('\n')
    close_df = load_stocks(tickers + ['SPY'])
    sectors  = json.load(open(CACHE / "sp500_sectors.json"))
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna()
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna()
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna()
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna()
    ief_p  = load_csv(CACHE / "IEF.csv")['Close'].dropna()
    def_prices = {}
    for t in DEFENSIVE_ETFS:
        f = CACHE / f"{t}.csv"
        def_prices[t] = load_csv(f)['Close'].dropna() if f.exists() else None
    sig = precompute(close_df)
    print(f"  Loaded {len(close_df.columns)} tickers\n")

    # ══════════════════════════════════════════════════════════════════════════
    # 0. v11b Baseline（hi52_alpha=0, vix_enabled=False）
    # ══════════════════════════════════════════════════════════════════════════
    print("━" * 80)
    print("▶ [0] v11b Baseline (原始，无改动)")
    eq0, rh0, vx0 = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                   tlt_p, ief_p, def_prices,
                                   hi52_alpha=0.0, vix_enabled=False)
    m0 = compute_metrics(eq0)
    wf0, mi0, mo0 = compute_wf_fixed(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                       tlt_p, ief_p, def_prices,
                                       hi52_alpha=0.0, vix_enabled=False)
    comp0 = m0['sharpe']*0.4 + m0['calmar']*0.4 + min(m0['cagr'],1.0)*0.2
    print(f"  CAGR={m0['cagr']:.1%}  MaxDD={m0['max_dd']:.1%}  Sharpe={m0['sharpe']:.2f}  "
          f"Calmar={m0['calmar']:.2f}  Composite={comp0:.3f}")
    print(f"  WF={wf0:.3f}  IS_Sh={mi0['sharpe']:.2f}→OOS_Sh={mo0['sharpe']:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Direction A only: 52周高位信号（扫描 alpha）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n━" * 80)
    print("▶ [1] Direction A: 52周高位信号增强 (alpha sweep)")
    print(f"  公式: enhanced_mom = mom × (1 + alpha × max(0, price/hi52 - {HI52_PROXIMITY_THRESH}))")
    print(f"{'alpha':>8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Composite':>10} "
          f"{'WF':>6} {'IS_Sh':>7} {'OOS_Sh':>7}  {'ΔCAGR':>7} {'ΔComp':>7}")
    print("-" * 100)

    best_a_comp = comp0; best_a_alpha = 0.0
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]:
        eq, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                 tlt_p, ief_p, def_prices,
                                 hi52_alpha=alpha, vix_enabled=False)
        m = compute_metrics(eq)
        wf, mi, mo = compute_wf_fixed(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                        tlt_p, ief_p, def_prices,
                                        hi52_alpha=alpha, vix_enabled=False)
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + min(m['cagr'],1.0)*0.2
        dc = m['cagr'] - m0['cagr']
        dcomp = comp - comp0
        flag = ""
        if comp > best_a_comp:
            best_a_comp = comp; best_a_alpha = alpha
            flag = " ★"
        wf_flag = "✅" if wf >= 0.60 else ("⚠️" if wf >= 0.55 else "❌")
        print(f"  {alpha:>6.2f}  {m['cagr']:>7.1%}  {m['max_dd']:>7.1%}  {m['sharpe']:>7.2f}  "
              f"{m['calmar']:>7.2f}  {comp:>9.3f}  {wf_flag}{wf:.2f}  "
              f"{mi['sharpe']:>6.2f}  {mo['sharpe']:>6.2f}    {dc:>+6.1%}  {dcomp:>+6.3f}{flag}")

    print(f"\n  → Direction A 最优 alpha = {best_a_alpha}")

    # ══════════════════════════════════════════════════════════════════════════
    # 1b. Direction A: 不同 proximity threshold
    # ══════════════════════════════════════════════════════════════════════════
    print("\n▶ [1b] Direction A: proximity threshold sweep (alpha=best)")
    alpha_use = best_a_alpha if best_a_alpha > 0 else 0.50  # 如果 0 最优，用 0.5 做 threshold 测试
    print(f"  使用 alpha={alpha_use}")
    best_a2_comp = comp0; best_a2_thresh = HI52_PROXIMITY_THRESH
    for thresh in [0.80, 0.85, 0.90, 0.93, 0.95]:
        eq, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                 tlt_p, ief_p, def_prices,
                                 hi52_alpha=alpha_use, hi52_prox_thresh=thresh,
                                 vix_enabled=False)
        m = compute_metrics(eq)
        wf, mi, mo = compute_wf_fixed(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                        tlt_p, ief_p, def_prices,
                                        hi52_alpha=alpha_use, hi52_prox_thresh=thresh,
                                        vix_enabled=False)
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + min(m['cagr'],1.0)*0.2
        dc = m['cagr'] - m0['cagr']; dcomp = comp - comp0
        flag = ""
        if comp > best_a2_comp:
            best_a2_comp = comp; best_a2_thresh = thresh; flag = " ★"
        wf_flag = "✅" if wf >= 0.60 else ("⚠️" if wf >= 0.55 else "❌")
        print(f"  thresh={thresh:.2f}  CAGR={m['cagr']:>7.1%}  MaxDD={m['max_dd']:>7.1%}  "
              f"Sharpe={m['sharpe']:>7.2f}  Comp={comp:>9.3f}  {wf_flag}WF={wf:.3f}  "
              f"ΔCAGR={dc:>+6.1%}  ΔComp={dcomp:>+6.3f}{flag}")
    print(f"  → 最优 thresh = {best_a2_thresh}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Direction B only: VIX 门控
    # ══════════════════════════════════════════════════════════════════════════
    print("\n━" * 80)
    print("▶ [2] Direction B: VIX/波动率宏观门控 (SPY vol30 proxy)")
    eq_b, rh_b, vx_b = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                      tlt_p, ief_p, def_prices,
                                      hi52_alpha=0.0, vix_enabled=True)
    m_b = compute_metrics(eq_b)
    wf_b, mi_b, mo_b = compute_wf_fixed(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                          tlt_p, ief_p, def_prices,
                                          hi52_alpha=0.0, vix_enabled=True)
    comp_b = m_b['sharpe']*0.4 + m_b['calmar']*0.4 + min(m_b['cagr'],1.0)*0.2
    dc_b = m_b['cagr'] - m0['cagr']; dcomp_b = comp_b - comp0
    wf_flag = "✅" if wf_b >= 0.60 else ("⚠️" if wf_b >= 0.55 else "❌")
    print(f"  CAGR={m_b['cagr']:.1%}  MaxDD={m_b['max_dd']:.1%}  Sharpe={m_b['sharpe']:.2f}  "
          f"Calmar={m_b['calmar']:.2f}  Composite={comp_b:.3f}")
    print(f"  {wf_flag}WF={wf_b:.3f}  IS_Sh={mi_b['sharpe']:.2f}→OOS_Sh={mo_b['sharpe']:.2f}")
    print(f"  vs Baseline: ΔCAGR={dc_b:+.1%}  ΔComp={dcomp_b:+.3f}")
    print(f"  VIX actions: {vx_b}")

    # 2b. VIX threshold sweep
    print("\n▶ [2b] VIX L3/L4 threshold sweep")
    best_b_comp = comp0
    configs_b = [
        (0.20, 0.30, 'L3=0.20,L4=0.30'),
        (0.25, 0.35, 'L3=0.25,L4=0.35 (default)'),
        (0.25, 0.40, 'L3=0.25,L4=0.40'),
        (0.30, 0.40, 'L3=0.30,L4=0.40'),
        (0.30, 0.45, 'L3=0.30,L4=0.45'),
        (0.35, 0.50, 'L3=0.35,L4=0.50'),
    ]
    best_b_cfg = None
    for l3, l4, label in configs_b:
        global VIX_L3_THRESH, VIX_L4_THRESH
        old_l3, old_l4 = VIX_L3_THRESH, VIX_L4_THRESH
        VIX_L3_THRESH, VIX_L4_THRESH = l3, l4
        eq, _, vx = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                   tlt_p, ief_p, def_prices,
                                   hi52_alpha=0.0, vix_enabled=True)
        m = compute_metrics(eq)
        wf, mi, mo = compute_wf_fixed(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                        tlt_p, ief_p, def_prices,
                                        hi52_alpha=0.0, vix_enabled=True)
        comp = m['sharpe']*0.4 + m['calmar']*0.4 + min(m['cagr'],1.0)*0.2
        flag = ""
        if comp > best_b_comp:
            best_b_comp = comp; best_b_cfg = (l3, l4, label); flag = " ★"
        wf_flag = "✅" if wf >= 0.60 else ("⚠️" if wf >= 0.55 else "❌")
        dc = m['cagr'] - m0['cagr']; dcomp = comp - comp0
        print(f"  {label:<30} CAGR={m['cagr']:>7.1%}  MaxDD={m['max_dd']:>7.1%}  "
              f"Comp={comp:>8.3f}  {wf_flag}WF={wf:.3f}  "
              f"ΔCAGR={dc:>+6.1%}  ΔComp={dcomp:>+6.3f}  L4={vx.get('L4_shy',0)} L3={vx.get('L3_tight',0)}{flag}")
        VIX_L3_THRESH, VIX_L4_THRESH = old_l3, old_l4

    if best_b_cfg:
        VIX_L3_THRESH, VIX_L4_THRESH = best_b_cfg[0], best_b_cfg[1]
        print(f"  → Direction B 最优: {best_b_cfg[2]}")
    else:
        print(f"  → Direction B 无改进")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Combined: Direction A + B
    # ══════════════════════════════════════════════════════════════════════════
    print("\n━" * 80)
    print(f"▶ [3] Combined: Direction A (alpha={best_a_alpha}, thresh={best_a2_thresh}) "
          f"+ Direction B (VIX)")

    eq_c, rh_c, vx_c = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                      tlt_p, ief_p, def_prices,
                                      hi52_alpha=best_a_alpha,
                                      hi52_prox_thresh=best_a2_thresh,
                                      vix_enabled=True)
    m_c = compute_metrics(eq_c)
    wf_c, mi_c, mo_c = compute_wf_fixed(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                          tlt_p, ief_p, def_prices,
                                          hi52_alpha=best_a_alpha,
                                          hi52_prox_thresh=best_a2_thresh,
                                          vix_enabled=True)
    comp_c = m_c['sharpe']*0.4 + m_c['calmar']*0.4 + min(m_c['cagr'],1.0)*0.2
    dc_c = m_c['cagr'] - m0['cagr']; dcomp_c = comp_c - comp0

    wf_flag = "✅" if wf_c >= 0.60 else ("⚠️" if wf_c >= 0.55 else "❌")
    print(f"  CAGR={m_c['cagr']:.1%}  MaxDD={m_c['max_dd']:.1%}  Sharpe={m_c['sharpe']:.2f}  "
          f"Calmar={m_c['calmar']:.2f}  Composite={comp_c:.3f}")
    print(f"  {wf_flag}WF={wf_c:.3f}  IS_Sh={mi_c['sharpe']:.2f}→OOS_Sh={mo_c['sharpe']:.2f}")
    print(f"  vs Baseline: ΔCAGR={dc_c:+.1%}  ΔComp={dcomp_c:+.3f}")
    print(f"  VIX actions: {vx_c}")
    print(f"  Regime: {rh_c}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Direction C: 信用利差门控 (HYG < 200dMA)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n━" * 80)
    print("▶ [4] Direction C: 信用利差门控 (HYG < 200dMA)")
    hyg_p = load_csv(CACHE / "HYG.csv")['Close'].dropna() if (CACHE / "HYG.csv").exists() else None
    if hyg_p is not None:
        hyg_200ma = hyg_p.rolling(200, min_periods=160).mean()
        print(f"  HYG data: {len(hyg_p)} days ({hyg_p.index[0].date()} → {hyg_p.index[-1].date()})")

        # Add credit stress check to overlays
        def run_credit_backtest(credit_cut=0.12, cost=0.0015):
            """v11b + HYG<200dMA → equity 减仓 credit_cut%，补 SHY"""
            rng  = close_df.loc['2015-01-01':'2025-12-31'].dropna(how='all')
            ends = rng.resample('ME').last().index
            vals, dates = [], []
            prev_w, prev_h = {}, set()
            val2 = 1.0; peak2 = 1.0
            port_returns2 = []
            credit_count = 0

            for i in range(len(ends) - 1):
                dt, ndt = ends[i], ends[i+1]
                dd = (val2 - peak2) / peak2 if peak2 > 0 else 0
                spy_vol5 = float(SPY_VOL.loc[:dt].dropna().iloc[-1]) if (
                    SPY_VOL is not None and len(SPY_VOL.loc[:dt].dropna()) > 0) else 0.15
                spy_1m = get_spy_1m(sig, dt)

                if len(port_returns2) >= VOL_LOOKBACK:
                    pv = np.std(port_returns2[-VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
                else:
                    pv = 0.20

                w, reg, bond_t = select(sig, sectors, dt, prev_h, gld_p, gdx_p, tlt_p, ief_p,
                                        def_prices, hi52_alpha=0.0)
                w, shy_boost = apply_overlays(w, spy_vol5, dd, pv, spy_1m,
                                               spy_vol30=0.15, vix_enabled=False)

                # ★ 信用利差门控 ★
                hyg_hist = hyg_p.loc[:dt].dropna()
                hyg_ma_hist = hyg_200ma.loc[:dt].dropna()
                credit_stressed = False
                if len(hyg_hist) > 0 and len(hyg_ma_hist) > 0:
                    if hyg_hist.iloc[-1] < hyg_ma_hist.iloc[-1]:
                        credit_stressed = True
                        credit_count += 1

                if credit_stressed:
                    # 减仓 credit_cut% equity → SHY
                    equity_keys = [t for t in w if t not in HEDGE_KEYS and t != 'SHY']
                    eq_total = sum(w.get(t, 0) for t in equity_keys)
                    if eq_total > 0:
                        cut = min(credit_cut, eq_total)
                        scale = (eq_total - cut) / eq_total
                        for t in equity_keys: w[t] *= scale
                        shy_boost += cut

                all_t = set(w) | set(prev_w)
                to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
                prev_w = w.copy()
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
                    elif t == 'SHY':  s = shy_p.loc[dt:ndt].dropna()
                    elif t in def_prices and def_prices[t] is not None:
                        s = def_prices[t].loc[dt:ndt].dropna()
                    elif t in close_df.columns:
                        s = close_df[t].loc[dt:ndt].dropna()
                    else: continue
                    if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * wt

                total_shy = shy_boost + (cash_frac if USE_SHY else 0.0)
                if total_shy > 0 and shy_p is not None:
                    s = shy_p.loc[dt:ndt].dropna()
                    if len(s) >= 2: ret += (s.iloc[-1]/s.iloc[0]-1) * total_shy

                ret -= to * cost * 2
                val2 *= (1 + ret)
                if val2 > peak2: peak2 = val2
                vals.append(val2); dates.append(ndt)
                port_returns2.append(ret)

            eq_cr = pd.Series(vals, index=pd.DatetimeIndex(dates))
            return eq_cr, credit_count

        SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

        for cut in [0.08, 0.10, 0.12, 0.15, 0.20]:
            eq_cr, n_cr = run_credit_backtest(cut)
            m_cr = compute_metrics(eq_cr)
            # WF
            eq_cr_is = eq_cr[eq_cr.index <= pd.Timestamp('2020-12-31')]
            eq_cr_oos = eq_cr[eq_cr.index >= pd.Timestamp('2021-01-01')]
            mi_cr = compute_metrics(eq_cr_is)
            mo_cr = compute_metrics(eq_cr_oos)
            wf_cr = mo_cr['sharpe'] / mi_cr['sharpe'] if mi_cr['sharpe'] > 0 else 0
            comp_cr = m_cr['sharpe']*0.4 + m_cr['calmar']*0.4 + min(m_cr['cagr'],1.0)*0.2
            dc = m_cr['cagr'] - m0['cagr']; dcomp = comp_cr - comp0
            wf_flag = "✅" if wf_cr >= 0.60 else ("⚠️" if wf_cr >= 0.55 else "❌")
            print(f"  cut={cut:.0%}  CAGR={m_cr['cagr']:>7.1%}  MaxDD={m_cr['max_dd']:>7.1%}  "
                  f"Comp={comp_cr:>8.3f}  {wf_flag}WF={wf_cr:.3f}  "
                  f"ΔCAGR={dc:>+6.1%}  ΔComp={dcomp:>+6.3f}  (credit months={n_cr})")
    else:
        print("  ⚠️  HYG 数据缺失，跳过")

    # ══════════════════════════════════════════════════════════════════════════
    # 总结
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📊 总结")
    print("=" * 80)
    configs_summary = [
        ('v11b Baseline', m0, comp0, wf0, mi0, mo0),
        (f'Dir A only (α={best_a_alpha})', None, None, None, None, None),  # 用best alpha重新计算
        ('Dir B only (VIX)', m_b, comp_b, wf_b, mi_b, mo_b),
        ('Dir A+B Combined', m_c, comp_c, wf_c, mi_c, mo_c),
    ]

    # 重新计算 Dir A best
    eq_a_best, _, _ = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                    tlt_p, ief_p, def_prices,
                                    hi52_alpha=best_a_alpha, hi52_prox_thresh=best_a2_thresh,
                                    vix_enabled=False)
    m_a_best = compute_metrics(eq_a_best)
    wf_a_best, mi_a_best, mo_a_best = compute_wf_fixed(
        close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
        tlt_p, ief_p, def_prices,
        hi52_alpha=best_a_alpha, hi52_prox_thresh=best_a2_thresh,
        vix_enabled=False)
    comp_a_best = m_a_best['sharpe']*0.4 + m_a_best['calmar']*0.4 + min(m_a_best['cagr'],1.0)*0.2

    print(f"\n{'策略':<28} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'Composite':>10} {'WF':>7} {'IS_Sh':>7} {'OOS_Sh':>7}")
    print("-" * 100)
    for label, m, comp_v, wf_v, mi, mo in [
        ('v11b Baseline', m0, comp0, wf0, mi0, mo0),
        (f'Dir A (α={best_a_alpha},t={best_a2_thresh})', m_a_best, comp_a_best, wf_a_best, mi_a_best, mo_a_best),
        ('Dir B (VIX gate)', m_b, comp_b, wf_b, mi_b, mo_b),
        ('Dir A+B Combined', m_c, comp_c, wf_c, mi_c, mo_c),
    ]:
        wf_flag = "✅" if wf_v >= 0.60 else ("⚠️" if wf_v >= 0.55 else "❌")
        print(f"  {label:<26} {m['cagr']:>7.1%}  {m['max_dd']:>7.1%}  {m['sharpe']:>7.2f}  "
              f"{m['calmar']:>7.2f}  {comp_v:>9.3f}  {wf_flag}{wf_v:.2f}  "
              f"{mi['sharpe']:>6.2f}  {mo['sharpe']:>6.2f}")

    # 判断突破
    best_comp = max(comp0, comp_a_best, comp_b, comp_c)
    if best_comp > 2.20:
        print("\n🚀🚀🚀 【重大突破】Composite > 2.20!")
    elif best_comp > comp0:
        print(f"\n✅ 有改进: 最佳 Composite={best_comp:.3f} (vs baseline {comp0:.3f})")
    else:
        print(f"\n⚠️  无改进（baseline={comp0:.3f} 仍为最优）")

    # 保存
    out = {
        'baseline': {'metrics': m0, 'composite': comp0, 'wf': wf0, 'is': mi0, 'oos': mo0},
        'dir_a': {'alpha': best_a_alpha, 'thresh': best_a2_thresh,
                  'metrics': m_a_best, 'composite': comp_a_best, 'wf': wf_a_best,
                  'is': mi_a_best, 'oos': mo_a_best},
        'dir_b': {'metrics': m_b, 'composite': comp_b, 'wf': wf_b,
                  'is': mi_b, 'oos': mo_b, 'vix_actions': vx_b},
        'combined': {'metrics': m_c, 'composite': comp_c, 'wf': wf_c,
                     'is': mi_c, 'oos': mo_c, 'vix_actions': vx_c},
    }
    jf = Path(__file__).parent / "momentum_v16_results.json"
    jf.write_text(json.dumps(out, indent=2))
    print(f"\n💾 → {jf}")

    return out


if __name__ == '__main__':
    main()
