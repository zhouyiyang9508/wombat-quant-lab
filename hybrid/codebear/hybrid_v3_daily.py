#!/usr/bin/env python3
"""
Hybrid v3 Daily — Stock v9j + Crypto(趋势过滤+波动率目标化)
代码熊 🐻 | 2026-02-22

核心改进（相比 v2）：
  A. BTC 200日均线趋势过滤（Direction A）
     - 只在 BTC > 200dMA 时持有 Crypto
     - BTC < 200dMA → Crypto 仓位全切 GLD（或 SHY）
     - 2022年初 BTC 跌破200dMA → 自动退出加密 → 改善 OOS WF
  D. 动态 Crypto 权重（vol targeting）（Direction D）
     - BTC 60日波动率目标化：目标 crypto sleeve vol = 20%/年
     - 高 vol 时自动降低 crypto 仓位
  额外：BTC 回撤保护（Direction B）
     - BTC 从近期高点回撤 >25% → Crypto 切 GLD
     - 回撤恢复到 <-15% → 允许重新持有 Crypto

评估标准（固定 Walk-Forward 日期边界）：
  IS = 2015-01-01 to 2020-12-31
  OOS = 2021-01-01 to 2025-12-31
  Composite = Sharpe×0.25 + Calmar×0.25 + min(CAGR,2.0)×0.50（CAGR 权重提升）
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parents[2]
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# ══════════════════════════════════════════════════════════════════════════════
# Stock v9j 参数（原封不动）
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
TLT_BEAR_FRAC = 0.25
TLT_MOM_LB    = 126

# ══════════════════════════════════════════════════════════════════════════════
# Crypto v3 参数
# ══════════════════════════════════════════════════════════════════════════════
BTC_MA_WINDOW    = 200     # 200日均线趋势过滤
BTC_DD_EXIT      = -0.25   # BTC 回撤超过 25% 时退出 Crypto
BTC_DD_REENTRY   = -0.15   # 回撤恢复到 -15% 才重入
BTC_VOL_WINDOW   = 60      # 波动率计算窗口（日）
BTC_VOL_TARGET   = 0.80    # crypto sleeve 目标年化波动率（80%，因加密天然高波动）
BTC_VOL_MIN      = 0.40    # crypto 仓位下限（相对 base weight 的比例）
BTC_VOL_MAX      = 1.50    # crypto 仓位上限（相对 base weight 的比例）
CRYPTO_MOM_LB    = 90      # 加密动量回看窗口（天）

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
# Stock v9j 信号计算（完整复制，保持原版精确性）
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
    hist = tlt_p.loc[:date].dropna()
    if len(hist) < TLT_MOM_LB + 3: return False
    return bool(hist.iloc[-1] / hist.iloc[-TLT_MOM_LB] - 1 > 0)


def select_v9j(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p):
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
    last_rebal_val = 1.0

    for day_idx, day in enumerate(trading_days):
        past_month_ends = month_ends[month_ends < day]
        if len(past_month_ends) > 0:
            last_me = past_month_ends[-1]
            next_days_after_me = trading_days[trading_days > last_me]
            exec_day = next_days_after_me[0] if len(next_days_after_me) > 0 else None

            if exec_day is not None and day == exec_day and last_me not in processed_months:
                dd_now = (val - peak) / peak if peak > 0 else 0
                spy_vol = float(SPY_VOL.loc[:last_me].dropna().iloc[-1]) if (
                    SPY_VOL is not None and len(SPY_VOL.loc[:last_me].dropna()) > 0) else 0.15

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

                all_t = set(new_w) | set(prev_w)
                turnover = sum(abs(new_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
                val *= (1 - turnover * cost * 2)

                last_rebal_val = val
                current_weights = new_w.copy()
                prev_w = new_w.copy()
                prev_hold = {k for k in new_w if k not in ('GLD', 'GDX', 'GDXJ', 'SHY', 'TLT')}
                processed_months.add(last_me)

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
# Crypto v3 策略 — 趋势过滤 + 波动率目标化 + 回撤保护
# ══════════════════════════════════════════════════════════════════════════════
def run_crypto_v3_daily(btc_p, eth_p, gld_p, shy_p,
                        start='2015-01-01', end='2025-12-31',
                        cost=0.0015, mom_lb=CRYPTO_MOM_LB,
                        ma_window=BTC_MA_WINDOW,
                        dd_exit=BTC_DD_EXIT,
                        dd_reentry=BTC_DD_REENTRY,
                        vol_window=BTC_VOL_WINDOW,
                        vol_target=BTC_VOL_TARGET):
    """
    Crypto v3 策略（改进版）：
    月频调仓信号 + 三重保护层 + 日频净值追踪

    三重保护层（按信号日使用前一月末数据，日频层面实时监控）：
      Layer 1 - 月频动量信号：BTC vs ETH 动量轮换，否则 GLD
      Layer 2 - BTC 200dMA 趋势过滤：BTC 日内价格 < 200dMA → 切换到 GLD
      Layer 3 - BTC 回撤保护：BTC 日内 DD > 25% → 切换到 GLD，DD < 15% 才重入
      Layer 4 - 波动率目标化：根据 BTC 60日 vol 动态调整月频信号权重
                             （在组合层面调整 crypto 仓位大小）
    """
    common_idx = btc_p.index.intersection(eth_p.index).intersection(gld_p.index)
    common_idx = pd.DatetimeIndex(common_idx).sort_values()
    common_idx = common_idx[(common_idx >= pd.Timestamp(start)) &
                             (common_idx <= pd.Timestamp(end))]
    if len(common_idx) == 0:
        return pd.Series(dtype=float), {}

    trading_days = common_idx
    month_ends = pd.Series(index=trading_days).resample('ME').last().index

    # 预计算 BTC 200dMA 和 60日波动率（全序列）
    btc_log_ret = np.log(btc_p / btc_p.shift(1))
    btc_200ma   = btc_p.rolling(ma_window, min_periods=int(ma_window*0.8)).mean()
    btc_vol60   = btc_log_ret.rolling(vol_window, min_periods=int(vol_window*0.5)).std() * np.sqrt(252)

    # BTC 峰值追踪（用于回撤保护）
    btc_running_peak = btc_p.expanding().max()
    btc_dd_series    = (btc_p - btc_running_peak) / btc_running_peak

    val = 1.0
    equity_vals, equity_dates = [], []

    # 月频信号状态
    monthly_weights = {}   # BTC/ETH 月频动量信号
    prev_monthly_w  = {}
    processed_months = set()

    # 日频过滤状态
    crypto_blocked = False  # True = 当前 BTC 趋势/回撤过滤生效
    btc_dd_blocked = False  # 专门追踪 DD 阻断状态（需要恢复确认才重入）

    # 波动率调整因子（月频更新）
    vol_scale = 1.0

    filter_log = []  # 记录过滤触发日期

    for day_idx, day in enumerate(trading_days):
        # 第一天无 prev_day，跳过
        if day_idx == 0:
            equity_vals.append(val); equity_dates.append(day); continue
        # 修复前瞻偏差：所有日频信号用昨天收盘价
        prev_day_crypto = trading_days[day_idx - 1]

        # ── 月频信号更新 ──────────────────────────────────────────────────
        past_me = month_ends[month_ends < day]
        if len(past_me) > 0:
            last_me = past_me[-1]
            next_days = trading_days[trading_days > last_me]
            exec_day = next_days[0] if len(next_days) > 0 else None

            if exec_day is not None and day == exec_day and last_me not in processed_months:
                # 使用月末前数据（严禁前瞻偏差）
                btc_hist = btc_p.loc[:last_me].dropna()
                eth_hist = eth_p.loc[:last_me].dropna()

                btc_mom = (btc_hist.iloc[-1] / btc_hist.iloc[-mom_lb] - 1
                           if len(btc_hist) >= mom_lb else -1)
                eth_mom = (eth_hist.iloc[-1] / eth_hist.iloc[-mom_lb] - 1
                           if len(eth_hist) >= mom_lb else -1)

                if max(btc_mom, eth_mom) > 0:
                    if btc_mom >= eth_mom:
                        new_mw = {'BTC': 0.70, 'ETH': 0.30}
                    else:
                        new_mw = {'ETH': 0.70, 'BTC': 0.30}
                else:
                    new_mw = {'GLD': 1.00}  # 熊市 → 黄金

                # Layer 4: 波动率目标化（调整 crypto 仓位比例）
                btc_vol_hist = btc_vol60.loc[:last_me].dropna()
                if len(btc_vol_hist) > 0:
                    cur_vol = float(btc_vol_hist.iloc[-1])
                    if cur_vol > 0.01:
                        raw_scale = vol_target / cur_vol
                        vol_scale = float(np.clip(raw_scale, BTC_VOL_MIN, BTC_VOL_MAX))
                    else:
                        vol_scale = 1.0
                else:
                    vol_scale = 1.0

                # 换仓成本
                all_t = set(new_mw) | set(prev_monthly_w)
                turnover = sum(abs(new_mw.get(t, 0) - prev_monthly_w.get(t, 0)) for t in all_t) / 2
                val *= (1 - turnover * cost * 2)

                monthly_weights = new_mw.copy()
                prev_monthly_w  = new_mw.copy()
                processed_months.add(last_me)

        # ── 日频过滤层（用 prev_day 价格，避免前瞻偏差）────────────────────
        if prev_day_crypto in btc_200ma.index and prev_day_crypto in btc_dd_series.index:
            btc_price_now = btc_p.loc[prev_day_crypto] if prev_day_crypto in btc_p.index else np.nan
            btc_ma_now    = btc_200ma.loc[prev_day_crypto]
            btc_dd_now    = btc_dd_series.loc[prev_day_crypto]

            # Layer 2: BTC 200dMA 趋势过滤
            trend_blocked = bool(pd.notna(btc_ma_now) and pd.notna(btc_price_now) and
                                 btc_price_now < btc_ma_now)

            # Layer 3: BTC 回撤保护（迟滞型：进入需 DD<-25%，退出需 DD>-15%）
            if not btc_dd_blocked:
                if pd.notna(btc_dd_now) and btc_dd_now < dd_exit:
                    btc_dd_blocked = True
                    filter_log.append({'date': str(day.date()), 'trigger': 'DD_exit',
                                       'btc_dd': round(float(btc_dd_now), 3)})
            else:
                if pd.notna(btc_dd_now) and btc_dd_now > dd_reentry:
                    btc_dd_blocked = False
                    filter_log.append({'date': str(day.date()), 'trigger': 'DD_reentry',
                                       'btc_dd': round(float(btc_dd_now), 3)})

            crypto_blocked = trend_blocked or btc_dd_blocked

        # ── 构建当日实际持仓 ─────────────────────────────────────────────
        if crypto_blocked:
            # 趋势/回撤过滤触发 → 切换到 GLD（不使用 vol_scale）
            current_weights = {'GLD': 1.00}
        else:
            # 正常模式：月频信号 + 波动率调整
            cw = {}
            crypto_tickers = [t for t in monthly_weights if t in ('BTC', 'ETH')]
            gld_frac = monthly_weights.get('GLD', 0.0)

            if crypto_tickers and vol_scale != 1.0:
                # 波动率调整：缩放 crypto 仓位，剩余补 GLD
                raw_crypto = sum(monthly_weights.get(t, 0) for t in crypto_tickers)
                adj_crypto = min(raw_crypto * vol_scale, 0.95)  # 最多 95% crypto
                gld_supplement = gld_frac + max(raw_crypto - adj_crypto, 0)

                scale_factor = adj_crypto / raw_crypto if raw_crypto > 0 else 1.0
                for t in crypto_tickers:
                    cw[t] = monthly_weights[t] * scale_factor
                if gld_supplement > 0.01:
                    cw['GLD'] = gld_supplement
            else:
                cw = monthly_weights.copy()

            current_weights = cw

        # ── 日频净值追踪 ──────────────────────────────────────────────────
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
            elif ticker == 'SHY': series = shy_p
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
        equity_vals.append(val)
        equity_dates.append(day)

    crypto_eq = pd.Series(equity_vals, index=equity_dates, name='CryptoV3')
    return crypto_eq, {'filter_log': filter_log}


# ══════════════════════════════════════════════════════════════════════════════
# 组合指标计算（固定日期 Walk-Forward）
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
    # 新 Composite 公式（CAGR 权重 0.50）
    composite = sharpe * 0.25 + calmar * 0.25 + min(cagr, 2.0) * 0.50
    return dict(cagr=round(cagr, 4), maxdd=round(maxdd, 4),
                sharpe=round(sharpe, 4), calmar=round(calmar, 4),
                composite=round(composite, 4), ann_vol=round(ann_vol, 4),
                years=round(years, 2))


def walk_forward_fixed(equity: pd.Series,
                       is_end='2020-12-31',
                       oos_start='2021-01-01',
                       rf=0.04):
    """固定日期边界 Walk-Forward（IS=2015-2020，OOS=2021-2025）"""
    eq = equity.dropna()
    is_end_ts   = pd.Timestamp(is_end)
    oos_start_ts = pd.Timestamp(oos_start)

    is_eq  = eq[eq.index <= is_end_ts]
    oos_eq = eq[eq.index >= oos_start_ts]

    if len(is_eq) < 30 or len(oos_eq) < 30:
        return dict(is_sharpe=0, oos_sharpe=0, wf_ratio=0,
                    is_cagr=0, oos_cagr=0)

    is_m  = calc_metrics(is_eq,  rf)
    oos_m = calc_metrics(oos_eq, rf)

    is_sh  = is_m.get('sharpe', 0)
    oos_sh = oos_m.get('sharpe', 0)
    wf = (oos_sh / is_sh if is_sh > 0 else 0)

    return dict(is_sharpe=round(is_sh, 3),
                oos_sharpe=round(oos_sh, 3),
                wf_ratio=round(wf, 3),
                is_cagr=round(is_m.get('cagr', 0), 4),
                oos_cagr=round(oos_m.get('cagr', 0), 4))


# ══════════════════════════════════════════════════════════════════════════════
# 混合组合合成（简洁版）
# ══════════════════════════════════════════════════════════════════════════════
def build_hybrid(stock_eq: pd.Series, crypto_eq: pd.Series, w_crypto: float):
    """按固定权重合成日频净值（日频再平衡等价）"""
    common = stock_eq.index.intersection(crypto_eq.index).sort_values()
    s_ret = stock_eq.loc[common].pct_change().fillna(0)
    c_ret = crypto_eq.loc[common].pct_change().fillna(0)
    w_stock = 1.0 - w_crypto
    port_ret = w_stock * s_ret + w_crypto * c_ret
    port_vals = (1 + port_ret).cumprod()
    port_vals.iloc[0] = 1.0
    return pd.Series(port_vals.values, index=common, name=f'Hybrid_c{int(w_crypto*100)}')


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 75)
    print("Hybrid v3 Daily — Stock v9j + Crypto v3 (趋势过滤+波动率目标化) 🐻")
    print("改进：BTC 200dMA 过滤 + 回撤保护 + vol targeting")
    print("=" * 75)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print("\n[1/5] 加载市场数据...")
    tickers_file = CACHE / "sp500_tickers.txt"
    with open(tickers_file) as f:
        tickers = [t.strip() for t in f if t.strip()]

    close_df = load_stocks(tickers + ['SPY'])
    if 'SPY' not in close_df.columns:
        raise ValueError("SPY 数据缺失！")
    print(f"  ✓ 股票数据：{len(close_df.columns)} 只，{len(close_df)} 个交易日")

    btc_p  = load_csv(CACHE / "BTC_USD.csv")['Close'].dropna()
    eth_p  = load_csv(CACHE / "ETH_USD.csv")['Close'].dropna()
    gld_p  = load_csv(CACHE / "GLD.csv")['Close'].dropna()
    gdx_p  = load_csv(CACHE / "GDX.csv")['Close'].dropna() if (CACHE/"GDX.csv").exists() else pd.Series(dtype=float)
    gdxj_p = load_csv(CACHE / "GDXJ.csv")['Close'].dropna() if (CACHE/"GDXJ.csv").exists() else pd.Series(dtype=float)
    shy_p  = load_csv(CACHE / "SHY.csv")['Close'].dropna() if (CACHE/"SHY.csv").exists() else pd.Series(dtype=float)
    tlt_p  = load_csv(CACHE / "TLT.csv")['Close'].dropna() if (CACHE/"TLT.csv").exists() else pd.Series(dtype=float)
    print(f"  ✓ 加密数据：BTC={len(btc_p)}d, ETH={len(eth_p)}d")
    print(f"  ✓ BTC 数据范围：{btc_p.index[0].date()} → {btc_p.index[-1].date()}")

    # ── 预计算股票信号 ────────────────────────────────────────────────────────
    print("[2/5] 预计算股票信号...")
    stock_cols = [t for t in close_df.columns if t not in ('GLD', 'GDX', 'GDXJ', 'SHY', 'TLT')]
    sig = precompute(close_df[stock_cols])
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
    stock_m  = calc_metrics(stock_eq)
    stock_wf = walk_forward_fixed(stock_eq)
    print(f"  ✓ Stock v9j 基准：")
    print(f"    CAGR={stock_m['cagr']:.1%}  MaxDD={stock_m['maxdd']:.1%}  "
          f"Sharpe={stock_m['sharpe']:.3f}  WF={stock_wf['wf_ratio']:.3f}")
    print(f"    IS  CAGR={stock_wf['is_cagr']:.1%}  OOS CAGR={stock_wf['oos_cagr']:.1%}")

    # ── 运行 Crypto v3 日频回测 ───────────────────────────────────────────────
    print("[4/5] 运行 Crypto v3 日频回测（趋势过滤+波动率目标化）...")
    crypto_eq, crypto_meta = run_crypto_v3_daily(
        btc_p, eth_p, gld_p, shy_p,
        start='2015-01-01', end='2025-12-31'
    )
    crypto_m  = calc_metrics(crypto_eq)
    crypto_wf = walk_forward_fixed(crypto_eq)
    print(f"  ✓ Crypto v3 结果：")
    print(f"    CAGR={crypto_m.get('cagr',0):.1%}  MaxDD={crypto_m.get('maxdd',0):.1%}  "
          f"Sharpe={crypto_m.get('sharpe',0):.3f}  WF={crypto_wf.get('wf_ratio',0):.3f}")
    print(f"    IS  CAGR={crypto_wf.get('is_cagr',0):.1%}  "
          f"OOS CAGR={crypto_wf.get('oos_cagr',0):.1%}")

    # 过滤触发记录
    filter_log = crypto_meta.get('filter_log', [])
    print(f"  ✓ 过滤触发记录（前10条）：")
    for entry in filter_log[:10]:
        print(f"    {entry['date']}: {entry['trigger']}  BTC_DD={entry['btc_dd']:.1%}")
    if len(filter_log) > 10:
        print(f"    ... 共 {len(filter_log)} 条过滤记录")

    # ── 对比：Crypto v2（原始，无过滤）────────────────────────────────────────
    print("\n[4b] 对比：Crypto v2（无趋势过滤，纯动量）...")
    from hybrid_v2_daily import run_crypto_daily as run_crypto_v2
    crypto_v2_eq = run_crypto_v2(btc_p, eth_p, gld_p, shy_p,
                                  start='2015-01-01', end='2025-12-31')
    cv2_m  = calc_metrics(crypto_v2_eq)
    cv2_wf = walk_forward_fixed(crypto_v2_eq)
    print(f"  Crypto v2（基准）：")
    print(f"    CAGR={cv2_m.get('cagr',0):.1%}  MaxDD={cv2_m.get('maxdd',0):.1%}  "
          f"Sharpe={cv2_m.get('sharpe',0):.3f}  WF={cv2_wf.get('wf_ratio',0):.3f}")
    print(f"  → Crypto v3 改进：")
    cagr_diff = crypto_m.get('cagr',0) - cv2_m.get('cagr',0)
    dd_diff   = crypto_m.get('maxdd',0) - cv2_m.get('maxdd',0)
    wf_diff   = crypto_wf.get('wf_ratio',0) - cv2_wf.get('wf_ratio',0)
    print(f"    CAGR: {cv2_m.get('cagr',0):.1%} → {crypto_m.get('cagr',0):.1%} "
          f"({'+'if cagr_diff>=0 else ''}{cagr_diff:.1%})")
    print(f"    MaxDD: {cv2_m.get('maxdd',0):.1%} → {crypto_m.get('maxdd',0):.1%} "
          f"({'+'if dd_diff>=0 else ''}{dd_diff:.1%})")
    print(f"    WF: {cv2_wf.get('wf_ratio',0):.3f} → {crypto_wf.get('wf_ratio',0):.3f} "
          f"({'+'if wf_diff>=0 else ''}{wf_diff:.3f})")

    # ── 混合组合扫描 ──────────────────────────────────────────────────────────
    print("\n[5/5] 混合组合扫描（w_crypto 从 0% → 40%）...")
    print()

    results = []
    w_list = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]

    for w_c in w_list:
        if w_c == 0.0:
            eq = stock_eq
        else:
            eq = build_hybrid(stock_eq, crypto_eq, w_c)
        m  = calc_metrics(eq)
        wf = walk_forward_fixed(eq)
        row = dict(w_crypto=w_c, **m, **wf)
        results.append(row)

    # 对比：Crypto v2 组合
    results_v2 = []
    for w_c in [0.10, 0.15, 0.20, 0.25, 0.30]:
        eq_v2 = build_hybrid(stock_eq, crypto_v2_eq, w_c)
        m_v2  = calc_metrics(eq_v2)
        wf_v2 = walk_forward_fixed(eq_v2)
        results_v2.append(dict(w_crypto=w_c, **m_v2, **wf_v2))

    # ── 结果输出 ──────────────────────────────────────────────────────────────
    print("=" * 110)
    print(f"{'策略':>15} {'w_crypto':>8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Calmar':>8} {'Composite':>10} {'WF':>6} {'IS_Sh':>7} {'OOS_Sh':>7} {'OOS_CAGR':>9}")
    print("-" * 110)

    best_composite = -99; best_row = None

    for r in results:
        wf_ok = "✅" if r['wf_ratio'] >= 0.6 else "❌"
        flag = ""
        if r['composite'] > best_composite and r['wf_ratio'] >= 0.6:
            best_composite = r['composite']
            best_row = r
        if r['w_crypto'] == 0.0:
            flag = " ← v9j 纯股票基准"
        wf_str = f"{wf_ok}{r['wf_ratio']:.2f}"
        print(f"{'Hybrid v3':>15}  {r['w_crypto']:>7.0%}  {r['cagr']:>7.1%}  {r['maxdd']:>7.1%}  "
              f"{r['sharpe']:>7.3f}  {r['calmar']:>7.3f}  {r['composite']:>9.3f}  "
              f"{wf_str:>7}  {r.get('is_sharpe',0):>6.3f}  "
              f"{r.get('oos_sharpe',0):>6.3f}  {r.get('oos_cagr',0):>8.1%}{flag}")

    print()
    print("── 对比：Hybrid v2（原始 Crypto，无过滤）────")
    for r in results_v2:
        wf_ok = "✅" if r['wf_ratio'] >= 0.6 else "❌"
        wf_str = f"{wf_ok}{r['wf_ratio']:.2f}"
        print(f"{'Hybrid v2':>15}  {r['w_crypto']:>7.0%}  {r['cagr']:>7.1%}  {r['maxdd']:>7.1%}  "
              f"{r['sharpe']:>7.3f}  {r['calmar']:>7.3f}  {r['composite']:>9.3f}  "
              f"{wf_str:>7}  {r.get('is_sharpe',0):>6.3f}  "
              f"{r.get('oos_sharpe',0):>6.3f}  {r.get('oos_cagr',0):>8.1%}")

    print("=" * 110)
    print()

    # 重大突破判断
    if best_row:
        baseline = results[0]
        print(f"🏆 Hybrid v3 最优（WF≥0.6）：w_crypto={best_row['w_crypto']:.0%}")
        print(f"   CAGR={best_row['cagr']:.1%} / MaxDD={best_row['maxdd']:.1%} / "
              f"WF={best_row['wf_ratio']:.3f} / Sharpe={best_row['sharpe']:.3f} / "
              f"Composite={best_row['composite']:.3f}")
        print(f"\n   vs 纯股票基准：CAGR={baseline['cagr']:.1%} / MaxDD={baseline['maxdd']:.1%} / "
              f"WF={baseline['wf_ratio']:.3f}")

        cagr_imp = best_row['cagr'] - baseline['cagr']
        comp_imp = best_row['composite'] - baseline['composite']

        if (best_row['cagr'] > 0.40 and best_row['wf_ratio'] >= 0.60
                and best_row['maxdd'] > -0.30):
            print("\n🚀🚀🚀 【重大突破】 CAGR>40% + WF>0.60 + MaxDD<30% 三目标达成！")
        elif cagr_imp > 0.05 and best_row['wf_ratio'] >= 0.60:
            print(f"\n✅ 显著改进：CAGR+{cagr_imp:.1%}，Composite+{comp_imp:.3f}，WF达标")
        elif best_row['wf_ratio'] >= 0.60:
            print(f"\n✅ WF 达标：WF={best_row['wf_ratio']:.3f}，CAGR 提升 {cagr_imp:+.1%}")
        else:
            print("\n⚠️  WF 未达标或无改进")
    else:
        print("⚠️  没有找到 WF≥0.60 的方案")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent
    results_path = out_dir / "hybrid_v3_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'stock_v9j':  {**stock_m,  **stock_wf},
            'crypto_v3':  {**crypto_m, **crypto_wf},
            'crypto_v2':  {**cv2_m,    **cv2_wf},
            'hybrid_v3_sweep': results,
            'hybrid_v2_compare': results_v2,
            'best_v3': best_row,
            'filter_log': filter_log,
            'meta': {
                'date':   '2026-02-22',
                'author': '代码熊 🐻',
                'desc':   'Hybrid v3: BTC 200dMA + 回撤保护 + vol targeting',
                'wf_is':  '2015-01-01 ~ 2020-12-31',
                'wf_oos': '2021-01-01 ~ 2025-12-31',
                'composite_formula': 'Sharpe*0.25 + Calmar*0.25 + min(CAGR,2.0)*0.50',
                'btc_ma_window': BTC_MA_WINDOW,
                'btc_dd_exit':   BTC_DD_EXIT,
                'btc_dd_reentry': BTC_DD_REENTRY,
                'btc_vol_window': BTC_VOL_WINDOW,
                'btc_vol_target': BTC_VOL_TARGET,
            }
        }, f, indent=2, default=str)
    print(f"\n📁 结果已保存：{results_path}")

    # 净值曲线
    eq_df = pd.DataFrame({
        'StockV9j':  stock_eq,
        'CryptoV3':  crypto_eq,
        'CryptoV2':  crypto_v2_eq,
    }).dropna()
    if best_row and best_row['w_crypto'] > 0:
        best_eq = build_hybrid(stock_eq, crypto_eq, best_row['w_crypto'])
        eq_df['HybridV3_Best'] = best_eq
    eq_path = out_dir / "hybrid_v3_equity.csv"
    eq_df.to_csv(eq_path)
    print(f"📁 净值曲线已保存：{eq_path}")

    return results, best_row


if __name__ == '__main__':
    results, best = main()
