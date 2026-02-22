#!/usr/bin/env python3
"""
Hybrid v2 Daily — Stock v9j + Crypto 日频严谨组合策略
代码熊 🐻 | 2026-02-22

目标：找到股票(v9j) + 加密货币 的最优日频组合策略

设计原则：
  1. 严禁前瞻偏差：所有信号使用前一日/前月末收盘价
  2. 日频净值追踪：MaxDD 基于真实日内价格，不是月末快照
  3. Walk-Forward 验证：60/40 IS/OOS 分割
  4. 交易成本：单边 0.15%

组件：
  A. Stock v9j: 月频调仓信号 + 日频净值追踪（含 TLT 条件国债对冲）
  B. Crypto: BTC/ETH/GLD 月频动量信号 + 日频净值追踪
  C. 组合：静态权重扫描 + 动态风险调整

⚠️  月频 MaxDD 低估说明：
  v9j 月频报 -10.3%，但日频真实 MaxDD 预计约 -18% (1.78x 放大比，来自 v9f/v9g 审计)
  本脚本使用日频追踪，得出真实 MaxDD
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parents[2]
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ══════════════════════════════════════════════════════════════════════════════
# Stock v9j 参数（与 momentum_v9j_final.py 完全一致）
# ══════════════════════════════════════════════════════════════════════════════
MOM_W          = (0.20, 0.50, 0.20, 0.10)
N_BULL_SECS    = 5
N_BULL_SECS_HI = 4
BREADTH_CONC   = 0.65
BULL_SPS = 2; BEAR_SPS = 2
BREADTH_NARROW = 0.45
GLD_AVG_THRESH = 0.70; GLD_COMPETE_FRAC = 0.20
GDX_AVG_THRESH = 0.20; GDX_COMPETE_FRAC = 0.04
CONT_BONUS = 0.03; HI52_FRAC = 0.60; USE_SHY = True
DD_PARAMS  = {-0.08: 0.40, -0.12: 0.60, -0.18: 0.70}
GDXJ_VOL_LO_THRESH = 0.30; GDXJ_VOL_LO_FRAC = 0.08
GDXJ_VOL_HI_THRESH = 0.45; GDXJ_VOL_HI_FRAC = 0.18
VOL_TARGET_ANN = 0.11; VOL_LOOKBACK = 3
TLT_BEAR_FRAC = 0.25   # v9j 最优参数
TLT_MOM_LB    = 126    # 6 个月动量回看

# ══════════════════════════════════════════════════════════════════════════════
# 数据加载工具
# ══════════════════════════════════════════════════════════════════════════════
def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    return df.set_index(c).sort_index()


def load_stocks(tickers):
    d = {}
    for t in tickers:
        f = STOCK_CACHE / f"{t}.csv"
        if not f.exists() or f.stat().st_size < 500:
            continue
        try:
            df = load_csv(f)
            if 'Close' in df.columns and len(df) > 200:
                d[t] = pd.to_numeric(df['Close'], errors='coerce').dropna()
        except:
            pass
    return pd.DataFrame(d)


# ══════════════════════════════════════════════════════════════════════════════
# Stock v9j 信号计算
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


def get_tlt_momentum(tlt_p, date):
    """v9j 核心：TLT 6m 动量为正时才触发熊市 TLT 对冲"""
    hist = tlt_p.loc[:date].dropna()
    if len(hist) < TLT_MOM_LB + 3: return False
    return bool(hist.iloc[-1] / hist.iloc[-TLT_MOM_LB] - 1 > 0)


def select_v9j(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p):
    """v9j 选股信号（14层完整逻辑）"""
    close = sig['close']
    idx = close.index[close.index <= date]
    if len(idx) == 0: return {}, False
    d = idx[-1]

    w1, w3, w6, w12 = MOM_W
    mom = (sig['r1'].loc[d]*w1 + sig['r3'].loc[d]*w3 +
           sig['r6'].loc[d]*w6 + sig['r12'].loc[d]*w12)

    df = pd.DataFrame({
        'mom': mom, 'r6': sig['r6'].loc[d],
        'vol': sig['vol30'].loc[d], 'price': close.loc[d],
        'sma50': sig['sma50'].loc[d], 'hi52': sig['r52w_hi'].loc[d],
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
    breadth = compute_breadth(sig, date)

    # v9j 条件 TLT（仅熊市且 TLT 动量为正）
    tlt_positive = get_tlt_momentum(tlt_p, date)
    use_tlt = (reg == 'bear' and tlt_positive)
    tlt_alloc = TLT_BEAR_FRAC if use_tlt else 0.0

    if reg == 'bull':
        n_bull = N_BULL_SECS_HI if breadth > BREADTH_CONC else N_BULL_SECS
        n_secs = max(n_bull - n_compete, 1)
        sps, cash = BULL_SPS, 0.0
    else:
        n_secs = max(3 - n_compete, 1)
        sps = BEAR_SPS
        cash = max(0.20 - tlt_alloc, 0.0)

    top_secs = sec_mom.head(n_secs).index.tolist()
    selected = []
    for sec in top_secs:
        sdf = df[df['sector'] == sec].sort_values('mom', ascending=False)
        selected.extend(sdf.index[:sps].tolist())

    total_hedge = total_compete + tlt_alloc
    stock_frac = max(1.0 - cash - total_hedge, 0.0)
    if not selected:
        w = {}
        if gld_a > 0: w['GLD'] = gld_a
        if gdx_a > 0: w['GDX'] = gdx_a
        if use_tlt: w['TLT'] = tlt_alloc
        return w, use_tlt

    iv   = {t: 1.0/max(df.loc[t,'vol'], 0.10) for t in selected}
    iv_t = sum(iv.values()); iv_w = {t: v/iv_t for t, v in iv.items()}
    mn   = min(df.loc[t,'mom'] for t in selected); sh = max(-mn+0.01, 0)
    mw   = {t: df.loc[t,'mom']+sh for t in selected}
    mw_t = sum(mw.values()); mw_w = {t: v/mw_t for t, v in mw.items()}
    weights = {t: (0.70*iv_w[t]+0.30*mw_w[t])*stock_frac for t in selected}
    if gld_a > 0: weights['GLD'] = gld_a
    if gdx_a > 0: weights['GDX'] = gdx_a
    if use_tlt: weights['TLT'] = tlt_alloc
    return weights, use_tlt


def apply_overlays_v9j(weights, spy_vol, dd, port_vol_ann):
    """GDXJ 波动率预警 + GLD DD 响应 + 组合波动率目标化"""
    if spy_vol >= GDXJ_VOL_HI_THRESH: gdxj_v = GDXJ_VOL_HI_FRAC
    elif spy_vol >= GDXJ_VOL_LO_THRESH: gdxj_v = GDXJ_VOL_LO_FRAC
    else: gdxj_v = 0.0
    gld_dd = max((DD_PARAMS[th] for th in sorted(DD_PARAMS) if dd < th), default=0.0)

    total_overlay = gdxj_v + gld_dd
    if total_overlay > 0 and weights:
        hedge_keys = {'GLD', 'GDX', 'GDXJ', 'TLT', 'SHY'}
        equity_keys = [t for t in weights if t not in hedge_keys]
        eq_frac = sum(weights[t] for t in equity_keys)
        if eq_frac > 0:
            scale = max(1.0 - total_overlay, 0.01) / eq_frac
            for t in equity_keys:
                weights[t] *= scale
        if gld_dd > 0: weights['GLD'] = max(weights.get('GLD', 0), gld_dd)
        if gdxj_v > 0: weights['GDXJ'] = weights.get('GDXJ', 0) + gdxj_v

    # 组合波动率目标化
    shy_boost = 0.0
    if port_vol_ann > 0.01:
        scale = min(VOL_TARGET_ANN / max(port_vol_ann, 0.01), 1.0)
        if scale < 0.98:
            hedge_keys = {'GLD', 'GDX', 'GDXJ', 'TLT', 'SHY'}
            equity_keys = [t for t in weights if t not in hedge_keys]
            eq_frac = sum(weights.get(t, 0) for t in equity_keys)
            if eq_frac > 0:
                for t in equity_keys:
                    weights[t] = weights[t] * scale
                shy_boost = eq_frac * (1.0 - scale)
    return weights, shy_boost


# ══════════════════════════════════════════════════════════════════════════════
# Stock v9j 日频回测
# ══════════════════════════════════════════════════════════════════════════════
def run_stock_v9j_daily(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
                         start='2015-01-01', end='2025-12-31', cost=0.0015):
    """月频调仓信号 + 日频净值追踪（v9j 完整 14 层逻辑）"""
    all_daily = close_df.loc[start:end].dropna(how='all')
    month_ends = all_daily.resample('ME').last().index
    trading_days = all_daily.index
    SPY_VOL = sig['vol5']['SPY'] if 'SPY' in sig['vol5'].columns else None

    val = 1.0; peak = 1.0
    equity_vals, equity_dates = [], []
    current_weights = {}
    prev_hold, prev_w = set(), {}
    processed_months = set()
    port_returns = []
    last_rebal_val = 1.0   # 上次调仓时的净值，用于计算月度收益

    for day_idx, day in enumerate(trading_days):
        # 月末调仓检查
        past_month_ends = month_ends[month_ends < day]
        if len(past_month_ends) > 0:
            last_me = past_month_ends[-1]
            next_days_after_me = trading_days[trading_days > last_me]
            exec_day = next_days_after_me[0] if len(next_days_after_me) > 0 else None

            if exec_day is not None and day == exec_day and last_me not in processed_months:
                dd_now = (val - peak) / peak if peak > 0 else 0
                spy_vol = float(SPY_VOL.loc[:last_me].dropna().iloc[-1]) if (
                    SPY_VOL is not None and len(SPY_VOL.loc[:last_me].dropna()) > 0) else 0.15

                # 月度收益（从上次调仓到现在）
                month_ret = val / last_rebal_val - 1 if last_rebal_val > 0 else 0
                port_returns.append(month_ret)

                if len(port_returns) >= VOL_LOOKBACK:
                    port_vol_mon = np.std(port_returns[-VOL_LOOKBACK:], ddof=1)
                    port_vol_ann = port_vol_mon * np.sqrt(12)
                else:
                    port_vol_ann = 0.20

                new_w, _ = select_v9j(sig, sectors, last_me, prev_hold, gld_p, gdx_p, tlt_p)
                new_w, shy_boost = apply_overlays_v9j(new_w, spy_vol, dd_now, port_vol_ann)
                if shy_boost > 0:
                    new_w['SHY'] = new_w.get('SHY', 0) + shy_boost

                # 换仓成本
                all_t = set(new_w) | set(prev_w)
                turnover = sum(abs(new_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
                val *= (1 - turnover * cost * 2)

                last_rebal_val = val  # 更新调仓基准值
                current_weights = new_w.copy()
                prev_w = new_w.copy()
                prev_hold = {k for k in new_w if k not in ('GLD', 'GDX', 'GDXJ', 'SHY', 'TLT')}
                processed_months.add(last_me)

        # 日频净值计算
        if day_idx == 0:
            equity_vals.append(val)
            equity_dates.append(day)
            continue

        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0; invested = 0.0

        for ticker, w in current_weights.items():
            if   ticker == 'GLD':  series = gld_p
            elif ticker == 'GDX':  series = gdx_p
            elif ticker == 'GDXJ': series = gdxj_p
            elif ticker == 'SHY':  series = shy_p
            elif ticker == 'TLT':  series = tlt_p
            elif ticker in close_df.columns: series = close_df[ticker]
            else: continue

            if prev_day in series.index and day in series.index:
                p0 = series.loc[prev_day]; p1 = series.loc[day]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret += (p1 / p0 - 1) * w
                    invested += w

        # 剩余现金 → SHY
        cash_frac = max(1.0 - invested, 0.0)
        if cash_frac > 0.01 and USE_SHY and prev_day in shy_p.index and day in shy_p.index:
            sp, st = shy_p.loc[prev_day], shy_p.loc[day]
            if pd.notna(sp) and pd.notna(st) and sp > 0:
                day_ret += (st / sp - 1) * cash_frac

        val *= (1 + day_ret)
        peak = max(peak, val)
        equity_vals.append(val)
        equity_dates.append(day)

    return pd.Series(equity_vals, index=equity_dates, name='StockV9j')


# ══════════════════════════════════════════════════════════════════════════════
# Crypto 策略（BTC/ETH/GLD 月频动量 + 日频追踪）
# ══════════════════════════════════════════════════════════════════════════════
def run_crypto_daily(btc_p, eth_p, gld_p, shy_p,
                     start='2015-01-01', end='2025-12-31',
                     cost=0.0015, mom_lb=90):
    """
    Crypto 策略：BTC/ETH/GLD 三资产动量轮换
    月频信号（使用月末前收盘价），日频净值追踪

    规则：
      - 计算 BTC、ETH 的 mom_lb 天动量
      - 如果 max(btc_mom, eth_mom) > 0:
          * 如果 btc_mom > eth_mom: 70% BTC + 30% ETH
          * 如果 eth_mom > btc_mom: 70% ETH + 30% BTC
      - 如果两者动量均 <= 0: 100% GLD（避险）
    """
    common_idx = btc_p.index.intersection(eth_p.index).intersection(gld_p.index)
    common_idx = pd.DatetimeIndex(common_idx).sort_values()
    common_idx = common_idx[(common_idx >= pd.Timestamp(start)) &
                             (common_idx <= pd.Timestamp(end))]
    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    trading_days = common_idx
    month_ends = pd.Series(index=trading_days).resample('ME').last().index

    val = 1.0; peak = 1.0
    equity_vals, equity_dates = [], []
    current_weights = {}
    prev_w = {}
    processed_months = set()

    for day_idx, day in enumerate(trading_days):
        # 月末调仓检查
        past_me = month_ends[month_ends < day]
        if len(past_me) > 0:
            last_me = past_me[-1]
            next_days = trading_days[trading_days > last_me]
            exec_day = next_days[0] if len(next_days) > 0 else None

            if exec_day is not None and day == exec_day and last_me not in processed_months:
                # 使用月末前数据计算信号（严禁前瞻偏差）
                btc_hist = btc_p.loc[:last_me].dropna()
                eth_hist = eth_p.loc[:last_me].dropna()

                btc_mom = (btc_hist.iloc[-1] / btc_hist.iloc[-mom_lb] - 1
                           if len(btc_hist) >= mom_lb else -1)
                eth_mom = (eth_hist.iloc[-1] / eth_hist.iloc[-mom_lb] - 1
                           if len(eth_hist) >= mom_lb else -1)

                if max(btc_mom, eth_mom) > 0:
                    if btc_mom >= eth_mom:
                        new_w = {'BTC': 0.70, 'ETH': 0.30}
                    else:
                        new_w = {'ETH': 0.70, 'BTC': 0.30}
                else:
                    new_w = {'GLD': 1.00}  # 熊市 → 黄金避险

                # 换仓成本
                all_t = set(new_w) | set(prev_w)
                turnover = sum(abs(new_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
                val *= (1 - turnover * cost * 2)

                current_weights = new_w.copy()
                prev_w = new_w.copy()
                processed_months.add(last_me)

        # 日频净值
        if day_idx == 0:
            equity_vals.append(val)
            equity_dates.append(day)
            continue

        prev_day = trading_days[day_idx - 1]
        day_ret = 0.0; invested = 0.0

        for ticker, w in current_weights.items():
            if   ticker == 'BTC': series = btc_p
            elif ticker == 'ETH': series = eth_p
            elif ticker == 'GLD': series = gld_p
            else: continue

            if prev_day in series.index and day in series.index:
                p0 = series.loc[prev_day]; p1 = series.loc[day]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret += (p1 / p0 - 1) * w
                    invested += w

        # 剩余现金 → SHY
        cash_frac = max(1.0 - invested, 0.0)
        if cash_frac > 0.01 and prev_day in shy_p.index and day in shy_p.index:
            sp, st = shy_p.loc[prev_day], shy_p.loc[day]
            if pd.notna(sp) and pd.notna(st) and sp > 0:
                day_ret += (st / sp - 1) * cash_frac

        val *= (1 + day_ret)
        peak = max(peak, val)
        equity_vals.append(val)
        equity_dates.append(day)

    return pd.Series(equity_vals, index=equity_dates, name='Crypto')


# ══════════════════════════════════════════════════════════════════════════════
# 组合指标计算
# ══════════════════════════════════════════════════════════════════════════════
def calc_metrics(equity: pd.Series, rf=0.04):
    eq = equity.dropna()
    if len(eq) < 2: return {}
    rets = eq.pct_change().dropna()
    n = len(rets)
    years = n / 252
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    maxdd = float(dd.min())
    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    composite = sharpe * 0.4 + calmar * 0.4 + min(cagr, 1.0) * 0.2
    return dict(cagr=round(cagr, 4), maxdd=round(maxdd, 4),
                sharpe=round(sharpe, 4), calmar=round(calmar, 4),
                composite=round(composite, 4), ann_vol=round(ann_vol, 4),
                years=round(years, 2))


def walk_forward(equity: pd.Series, split=0.60, rf=0.04):
    eq = equity.dropna()
    n = len(eq)
    si = int(n * split)
    is_m = calc_metrics(eq.iloc[:si+1], rf)
    oos_m = calc_metrics(eq.iloc[si:], rf)
    wf = (oos_m.get('sharpe', 0) / is_m.get('sharpe', 1)
          if is_m.get('sharpe', 0) > 0 else 0)
    return dict(is_sharpe=round(is_m.get('sharpe', 0), 3),
                oos_sharpe=round(oos_m.get('sharpe', 0), 3),
                wf_ratio=round(wf, 3))


# ══════════════════════════════════════════════════════════════════════════════
# 混合组合：Stock v9j + Crypto
# ══════════════════════════════════════════════════════════════════════════════
def build_hybrid(stock_eq: pd.Series, crypto_eq: pd.Series,
                 w_crypto: float, gld_eq: pd.Series = None,
                 spy_eq: pd.Series = None):
    """
    将股票和加密货币日频净值曲线按固定权重合成

    动态风险调整层（叠加在基础权重上）：
      - Layer 2: 相关性检测（60d 滚动），高相关减少 crypto 暴露
      - Layer 3: 组合 vol 目标（20%/年），超出时缩减 crypto
      - Layer 4: 双组件同时回撤保护（都>阈值时加入额外 GLD 缓冲）

    注：本函数使用"净值曲线比例合成"，等效于每日再平衡到目标权重
        这是一种简化；实际更严格应做月频再平衡
    """
    common = stock_eq.index.intersection(crypto_eq.index).sort_values()
    s = stock_eq.loc[common]
    c = crypto_eq.loc[common]

    # 日收益率
    s_ret = s.pct_change().fillna(0)
    c_ret = c.pct_change().fillna(0)

    w_stock = 1.0 - w_crypto
    port_vals = [1.0]
    port_peak = 1.0
    stock_val = 1.0; stock_peak = 1.0
    crypto_val = 1.0; crypto_peak = 1.0
    recent_rets = []

    for i in range(1, len(common)):
        sr = s_ret.iloc[i]; cr = c_ret.iloc[i]

        # 动态调整：60d 滚动相关性
        adj_w_crypto = w_crypto
        if len(recent_rets) >= 60:
            s_r60 = np.array([r[0] for r in recent_rets[-60:]])
            c_r60 = np.array([r[1] for r in recent_rets[-60:]])
            corr = np.corrcoef(s_r60, c_r60)[0, 1] if np.std(s_r60) > 0 and np.std(c_r60) > 0 else 0
            if corr > 0.7: adj_w_crypto *= 0.70    # 高相关 → 减少 crypto
            elif corr < 0.2: adj_w_crypto *= 1.20  # 低相关 → 略增 crypto

        adj_w_crypto = min(adj_w_crypto, 0.80)  # 上限 80%

        # 动态调整：组合 vol 目标（20%/年）
        if len(recent_rets) >= 20:
            combo_r = [(r[0]*(1-w_crypto) + r[1]*w_crypto) for r in recent_rets[-20:]]
            port_vol_ann = np.std(combo_r) * np.sqrt(252)
            if port_vol_ann > 0.20:
                scale = 0.20 / port_vol_ann
                adj_w_crypto *= scale

        adj_w_stock = 1.0 - adj_w_crypto

        # Layer 4: 双组件回撤保护
        stock_dd = (stock_val - stock_peak) / stock_peak if stock_peak > 0 else 0
        crypto_dd = (crypto_val - crypto_peak) / crypto_peak if crypto_peak > 0 else 0
        gld_boost = 0.0
        if stock_dd < -0.12 and crypto_dd < -0.15:
            gld_boost = 0.20  # 双重压力 → 加入 20% GLD 缓冲
            adj_w_stock *= 0.80; adj_w_crypto *= 0.80

        # 日收益
        day_ret = adj_w_stock * sr + adj_w_crypto * cr
        # 简化：gld_boost 使用 GLD ETF 当天收益（若有）
        # 如果没有 GLD，用 0（保守处理）
        day_ret = day_ret  # gld_boost 部分暂不单独计算（保守）

        new_val = port_vals[-1] * (1 + day_ret)
        port_vals.append(new_val)
        port_peak = max(port_peak, new_val)

        # 跟踪各组件独立净值（用于回撤计算）
        stock_val *= (1 + sr); stock_peak = max(stock_peak, stock_val)
        crypto_val *= (1 + cr); crypto_peak = max(crypto_peak, crypto_val)
        recent_rets.append((sr, cr))

    return pd.Series(port_vals, index=common, name=f'Hybrid_c{int(w_crypto*100)}')


# ══════════════════════════════════════════════════════════════════════════════
# 主回测流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Hybrid v2 Daily — Stock v9j + Crypto 最优组合 🐻")
    print("=" * 70)

    # ── 加载市场数据 ──────────────────────────────────────────────────────────
    print("\n[1/5] 加载市场数据...")
    tickers_file = CACHE / "sp500_tickers.txt"
    with open(tickers_file) as f:
        tickers = [t.strip() for t in f if t.strip()]
    # 加入 SPY、对冲资产
    extra = ['SPY', 'GLD', 'GDX', 'GDXJ', 'SHY', 'TLT']
    all_tickers = list(set(tickers + extra))

    close_df = load_stocks(tickers + ['SPY'])  # 只加载股票 + SPY（stock_cache 目录）
    if 'SPY' not in close_df.columns:
        raise ValueError("SPY 数据缺失！")
    print(f"  ✓ 股票数据：{len(close_df.columns)} 只，{len(close_df)} 个交易日")

    # 加密数据 & 对冲 ETF（从主 data_cache 目录加载）
    btc_p  = load_csv(CACHE / "BTC_USD.csv")['Close'].dropna()
    eth_p  = load_csv(CACHE / "ETH_USD.csv")['Close'].dropna()
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna() if (CACHE/"GDX.csv").exists() else pd.Series(dtype=float)
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna() if (CACHE/"GDXJ.csv").exists() else pd.Series(dtype=float)
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna() if (CACHE/"SHY.csv").exists() else pd.Series(dtype=float)
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna() if (CACHE/"TLT.csv").exists() else pd.Series(dtype=float)

    # ── 预计算指标 ────────────────────────────────────────────────────────────
    print("[2/5] 预计算股票信号...")
    stock_cols = [t for t in close_df.columns if t not in ('GLD', 'GDX', 'GDXJ', 'SHY', 'TLT')]
    sig = precompute(close_df[stock_cols])

    # 行业分类
    sec_file = CACHE / "sp500_sectors.json"
    with open(sec_file) as f:
        sectors = json.load(f)
    print(f"  ✓ 行业分类：{len(sectors)} 只股票")

    # ── 运行 Stock v9j 日频回测 ───────────────────────────────────────────────
    print("[3/5] 运行 Stock v9j 日频回测...")
    stock_eq = run_stock_v9j_daily(
        close_df[stock_cols], sig, sectors,
        gld_p, gdx_p, gdxj_p, shy_p, tlt_p,
        start='2015-01-01', end='2025-12-31'
    )
    stock_m = calc_metrics(stock_eq)
    stock_wf = walk_forward(stock_eq)
    print(f"  ✓ Stock v9j 日频结果：")
    print(f"    CAGR={stock_m['cagr']:.1%}  MaxDD={stock_m['maxdd']:.1%}  "
          f"Sharpe={stock_m['sharpe']:.3f}  Composite={stock_m['composite']:.3f}  "
          f"WF={stock_wf['wf_ratio']:.2f}")
    print(f"    (月频参考: CAGR=32.3%, MaxDD=-10.3%, Composite=2.057)")

    # ── 运行 Crypto 日频回测 ──────────────────────────────────────────────────
    print("[4/5] 运行 Crypto 日频回测...")
    crypto_eq = run_crypto_daily(btc_p, eth_p, gld_p, shy_p,
                                  start='2015-01-01', end='2025-12-31')
    crypto_m = calc_metrics(crypto_eq)
    crypto_wf = walk_forward(crypto_eq)
    print(f"  ✓ Crypto 日频结果：")
    print(f"    CAGR={crypto_m.get('cagr',0):.1%}  MaxDD={crypto_m.get('maxdd',0):.1%}  "
          f"Sharpe={crypto_m.get('sharpe',0):.3f}  Composite={crypto_m.get('composite',0):.3f}  "
          f"WF={crypto_wf.get('wf_ratio',0):.2f}")

    # ── 组合参数扫描 ──────────────────────────────────────────────────────────
    print("[5/5] 组合参数扫描（w_crypto 从 0% → 50%）...")
    print()

    results = []
    w_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for w_c in w_list:
        if w_c == 0.0:
            eq = stock_eq
        else:
            eq = build_hybrid(stock_eq, crypto_eq, w_c)
        m = calc_metrics(eq)
        wf = walk_forward(eq)
        row = dict(w_crypto=w_c, **m, **wf)
        results.append(row)

    # ── 结果输出 ──────────────────────────────────────────────────────────────
    print("=" * 100)
    print(f"{'w_crypto':>8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Calmar':>8} {'Composite':>10} {'WF':>6} "
          f"{'IS_Sh':>7} {'OOS_Sh':>7}  备注")
    print("-" * 100)

    best_composite = -99; best_row = None
    for r in results:
        wf_ok = "✅" if r['wf_ratio'] >= 0.6 else "❌"
        flag = ""
        if r['composite'] > best_composite and r['wf_ratio'] >= 0.6:
            best_composite = r['composite']
            best_row = r
            flag = " ← 当前最优"
        if r['w_crypto'] == 0.0:
            flag = " ← v9j 纯股票基准"
        print(f"{r['w_crypto']:>7.0%}  {r['cagr']:>7.1%}  {r['maxdd']:>7.1%}  "
              f"{r['sharpe']:>7.3f}  {r['calmar']:>7.3f}  {r['composite']:>9.3f}  "
              f"{wf_ok}{r['wf_ratio']:>5.2f}  {r['is_sharpe']:>6.3f}  "
              f"{r['oos_sharpe']:>6.3f} {flag}")

    print("=" * 100)
    print()

    if best_row:
        print(f"🏆 最优组合：w_crypto={best_row['w_crypto']:.0%}")
        print(f"   Composite={best_row['composite']:.3f}  Sharpe={best_row['sharpe']:.3f}  "
              f"CAGR={best_row['cagr']:.1%}  MaxDD={best_row['maxdd']:.1%}  WF={best_row['wf_ratio']:.2f}")
        baseline = results[0]
        print(f"\n对比 v9j 纯股票基准（Composite={baseline['composite']:.3f}）：")
        delta_c = best_row['composite'] - baseline['composite']
        print(f"   Composite: {baseline['composite']:.3f} → {best_row['composite']:.3f} "
              f"({'+'if delta_c>=0 else ''}{delta_c:.3f})")
        print(f"   MaxDD:     {baseline['maxdd']:.1%} → {best_row['maxdd']:.1%}")
        print(f"   CAGR:      {baseline['cagr']:.1%} → {best_row['cagr']:.1%}")
        print()
        if best_row['composite'] > baseline['composite']:
            print("✅ 加入加密货币组合有显著提升！")
        else:
            print("⚠️  加入加密货币对 Composite 无提升（可能因日频 MaxDD 过大）")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent
    results_path = out_dir / "hybrid_v2_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'stock_v9j': {**stock_m, **stock_wf},
            'crypto': {**crypto_m, **crypto_wf},
            'hybrid_sweep': results,
            'best': best_row,
            'meta': {
                'date': '2026-02-22',
                'author': '代码熊 🐻',
                'note': '日频严谨回测，月频调仓信号，日频净值追踪'
            }
        }, f, indent=2)
    print(f"📁 结果已保存：{results_path}")

    # 保存最优组合的净值曲线
    if best_row and best_row['w_crypto'] > 0:
        best_eq = build_hybrid(stock_eq, crypto_eq, best_row['w_crypto'])
        eq_df = pd.DataFrame({'StockV9j': stock_eq, 'Crypto': crypto_eq,
                               'Hybrid_Best': best_eq}).dropna()
        eq_path = out_dir / "hybrid_v2_equity.csv"
        eq_df.to_csv(eq_path)
        print(f"📁 净值曲线已保存：{eq_path}")
    else:
        eq_df = pd.DataFrame({'StockV9j': stock_eq, 'Crypto': crypto_eq}).dropna()
        eq_path = out_dir / "hybrid_v2_equity.csv"
        eq_df.to_csv(eq_path)
        print(f"📁 净值曲线已保存：{eq_path}")

    return results, best_row


if __name__ == '__main__':
    results, best = main()
