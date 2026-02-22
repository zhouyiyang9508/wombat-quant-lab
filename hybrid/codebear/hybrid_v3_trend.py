#!/usr/bin/env python3
"""
Hybrid v3 — 股票(v11b_final) + 加密货币(200dMA趋势过滤)
代码熊 🐻 | 2026-02-22

核心创新 vs v2:
  v2 问题: 原始BTC/ETH无过滤 → 2022年暴跌摧毁OOS性能 → WF=0.17
  v3 方案: 200日均线趋势过滤
    - BTC > BTC_200dMA → 持有BTC；否则 → SHY
    - ETH > ETH_200dMA → 持有ETH；否则 → SHY
    - 月末信号，次月执行（严格无前瞻）

策略结构:
  股票组合 (1 - w_crypto): v11b_final 信号
    ✅ v10d 利率自适应债券
    ✅ v10b 防御行业桥接
    ✅ v9m SPY软对冲
  加密货币 (w_crypto):
    60% BTC (趋势过滤后) + 40% ETH (趋势过滤后)
    任一跌破200dMA → 该部分转SHY

扫描范围:
  w_crypto ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  BTC_ETH_SPLIT ∈ [0.6, 0.7, 0.8] (BTC占crypto内比例)

日频P&L追踪（精确MaxDD）

新Composite公式(CAGR权重提升):
  Sharpe×0.25 + Calmar×0.25 + CAGR×0.50
目标: CAGR>40%, WF>0.60, 日频MaxDD<30%
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# 路径
BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"
STOCK_MOD = BASE / "stocks" / "codebear" / "momentum_v11b_final.py"

# ─────────────────────────────────────────────
# 参数（v11b_final 基准）
# ─────────────────────────────────────────────
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
DEFENSIVE_FRAC = 0.12; DEFENSIVE_EACH = DEFENSIVE_FRAC / 3
SPY_SOFT_HI_THRESH = -0.07; SPY_SOFT_HI_FRAC = 0.08
HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'XLV', 'XLP', 'XLU'}

# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────
def load_csv(fp):
    df = pd.read_csv(fp)
    c = 'Date' if 'Date' in df.columns else df.columns[0]
    df[c] = pd.to_datetime(df[c])
    df = df.set_index(c).sort_index()
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df

def load_series(name, subdir=None):
    fp = CACHE / name if subdir is None else CACHE / subdir / name
    df = load_csv(fp)
    return df['Close'].dropna()

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

# ─────────────────────────────────────────────
# v11b_final 股票信号复现
# ─────────────────────────────────────────────
def precompute_stock(close_df):
    r1  = close_df / close_df.shift(22) - 1
    r3  = close_df / close_df.shift(63) - 1
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
    c = sig['close'].loc[:date].dropna(how='all')
    s = sig['sma50'].loc[:date].dropna(how='all')
    if len(c) < 50: return 1.0
    lc = c.iloc[-1]; ls = s.iloc[-1]
    mask = (lc > ls).dropna()
    return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0

def get_regime(sig, date):
    breadth = compute_breadth(sig, date)
    spy = sig['spy']; s200 = sig['s200']
    if spy is None or s200 is None: return 'bull_hi', breadth
    sv = float(spy.loc[:date].iloc[-1]); sm = float(s200.loc[:date].iloc[-1])
    bear = (sv < sm) and (breadth < BREADTH_NARROW)
    if bear: return 'bear', breadth
    return ('bull_hi', breadth) if breadth >= BREADTH_CONC else ('soft_bull', breadth)

def raw_mom(sig, t, date):
    try:
        r1  = float(sig['r1'][t].loc[:date].iloc[-1])
        r3  = float(sig['r3'][t].loc[:date].iloc[-1])
        r6  = float(sig['r6'][t].loc[:date].iloc[-1])
        r12 = float(sig['r12'][t].loc[:date].iloc[-1])
        if any(np.isnan(x) for x in [r1,r3,r6,r12]): return np.nan
        return MOM_W[0]*r1 + MOM_W[1]*r3 + MOM_W[2]*r6 + MOM_W[3]*r12
    except: return np.nan

def hi52(sig, t, date):
    try:
        hi = float(sig['r52w_hi'][t].loc[:date].iloc[-1])
        c  = float(sig['close'][t].loc[:date].iloc[-1])
        return c/hi if hi > 0 else 0.0
    except: return 0.0

def select_stocks(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices):
    regime, breadth = get_regime(sig, date)
    tickers = [t for t in sig['close'].columns if t not in HEDGE_KEYS and t != 'SPY']
    weights = {}; bond_type = 'none'

    if regime == 'bear':
        tlt_m = float(tlt_p.loc[:date].iloc[-1] / tlt_p.loc[:date].iloc[-BOND_MOM_LB] - 1) \
            if len(tlt_p.loc[:date]) >= BOND_MOM_LB else 0.0
        ief_m = float(ief_p.loc[:date].iloc[-1] / ief_p.loc[:date].iloc[-BOND_MOM_LB] - 1) \
            if len(ief_p.loc[:date]) >= BOND_MOM_LB else 0.0
        if tlt_m > ief_m and tlt_m > 0: weights['TLT'] = TLT_BEAR_FRAC; bond_type = 'TLT'
        elif ief_m > 0:                  weights['IEF'] = IEF_BEAR_FRAC; bond_type = 'IEF'
        gld_s = raw_mom(sig, 'GLD', date)
        gld_a = float(gld_p.loc[:date].iloc[-1] / gld_p.loc[:date].iloc[-252] - 1) \
            if len(gld_p.loc[:date]) >= 252 else 0.0
        if not np.isnan(gld_s) and gld_a > GLD_AVG_THRESH: weights['GLD'] = GLD_COMPETE_FRAC
        bond_alloc = sum(weights.values())
        stock_cap = max(0.0, 1.0 - bond_alloc)
        scored = [(t, raw_mom(sig, t, date)) for t in tickers
                  if not np.isnan(raw_mom(sig, t, date))]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in scored[:BEAR_SPS]]
        for t in top: weights[t] = stock_cap / len(top) if top else 0.0
    else:
        n_secs = N_BULL_SECS_HI if regime == 'bull_hi' else N_BULL_SECS
        sec_scores = {}
        for t in tickers:
            sec = sectors.get(t, 'Unknown')
            if sec not in sec_scores: sec_scores[sec] = []
            s = raw_mom(sig, t, date)
            if not np.isnan(s):
                if hi52(sig, t, date) >= HI52_FRAC:
                    if t in (prev_hold or {}): s += CONT_BONUS
                    sec_scores[sec].append((t, s))
        sec_avg = {s: np.mean([x[1] for x in lst]) for s, lst in sec_scores.items() if lst}
        gld_s = raw_mom(sig, 'GLD', date)
        gld_a = float(gld_p.loc[:date].iloc[-1] / gld_p.loc[:date].iloc[-252] - 1) \
            if len(gld_p.loc[:date]) >= 252 else 0.0
        if not np.isnan(gld_s) and gld_a > GLD_AVG_THRESH: sec_avg['__GLD__'] = gld_s
        gdx_s = raw_mom(sig, 'GDX', date)
        gdx_a = float(gdx_p.loc[:date].iloc[-1] / gdx_p.loc[:date].iloc[-252] - 1) \
            if len(gdx_p.loc[:date]) >= 252 else 0.0
        if not np.isnan(gdx_s) and gdx_a > GDX_AVG_THRESH:
            sec_avg['__GDX__'] = gdx_s * GDX_COMPETE_FRAC / GLD_COMPETE_FRAC
        top_secs = sorted(sec_avg.items(), key=lambda x: x[1], reverse=True)[:n_secs]
        stock_w = 1.0
        for sec, _ in top_secs:
            if sec == '__GLD__': weights['GLD'] = GLD_COMPETE_FRAC; stock_w -= GLD_COMPETE_FRAC
            elif sec == '__GDX__': weights['GDX'] = GDX_COMPETE_FRAC; stock_w -= GDX_COMPETE_FRAC
            else:
                lst = sorted(sec_scores.get(sec, []), key=lambda x: x[1], reverse=True)[:BULL_SPS]
                for t, _ in lst: weights[t] = 0.0
        real_stocks = [t for t in weights if t not in HEDGE_KEYS]
        w_per = stock_w / len(real_stocks) if real_stocks else 0.0
        for t in real_stocks: weights[t] = w_per
        if regime == 'soft_bull':
            def_avail = [e for e in DEFENSIVE_ETFS if e in def_prices and len(def_prices[e].loc[:date]) > 1]
            if def_avail:
                scale = 1.0 - DEFENSIVE_FRAC
                for k in list(weights.keys()): weights[k] *= scale
                for e in def_avail: weights[e] = DEFENSIVE_EACH
    return weights, regime, bond_type

def apply_overlays_stock(weights, sig, date, gdxj_p, shy_p, dd, port_vol_ann, spy_1m_ret):
    # GDXJ
    try:
        slog = np.log(sig['spy'] / sig['spy'].shift(1)).loc[:date].dropna()
        spy_vol = float(slog.iloc[-63:].std() * np.sqrt(252)) if len(slog) >= 10 else None
    except: spy_vol = None
    if spy_vol and spy_vol > GDXJ_VOL_HI_THRESH: frac = GDXJ_VOL_HI_FRAC
    elif spy_vol and spy_vol > GDXJ_VOL_LO_THRESH: frac = GDXJ_VOL_LO_FRAC
    else: frac = 0.0
    if frac > 0 and 'GDXJ' not in weights:
        for k in list(weights): weights[k] *= (1 - frac)
        weights['GDXJ'] = frac
    # DD保护
    if dd is not None and dd < -0.08:
        for thr in sorted(DD_PARAMS.keys()):
            if dd <= thr:
                existing = weights.pop('SHY', 0.0)
                new_shy = max(existing, DD_PARAMS[thr])
                boost = new_shy - existing
                for k in list(weights):
                    if k != 'SHY': weights[k] *= (1 - boost)
                weights['SHY'] = new_shy
                break
    # Vol目标
    if port_vol_ann and port_vol_ann > VOL_TARGET_ANN:
        scale = VOL_TARGET_ANN / port_vol_ann
        shy_add = 1.0 - scale
        for k in list(weights): weights[k] *= scale
        weights['SHY'] = weights.get('SHY', 0.0) + shy_add
    # SPY软对冲
    if spy_1m_ret and spy_1m_ret < SPY_SOFT_HI_THRESH and 'GLD' not in weights:
        scale = 1.0 - SPY_SOFT_HI_FRAC
        for k in list(weights): weights[k] *= scale
        weights['GLD'] = SPY_SOFT_HI_FRAC
    return weights

# ─────────────────────────────────────────────
# 加密货币信号 (200dMA趋势过滤)
# ─────────────────────────────────────────────
def get_crypto_weights(btc_p, eth_p, shy_p, date, btc_eth_split=0.6):
    """月末信号：BTC/ETH各自检查200dMA，不满足→转SHY"""
    btc_ma200 = btc_p.rolling(200).mean()
    eth_ma200 = eth_p.rolling(200).mean()

    btc_now = btc_p.loc[:date].iloc[-1] if len(btc_p.loc[:date]) > 0 else np.nan
    btc_ma  = btc_ma200.loc[:date].iloc[-1] if len(btc_ma200.loc[:date].dropna()) > 0 else np.nan
    eth_now = eth_p.loc[:date].iloc[-1] if len(eth_p.loc[:date]) > 0 else np.nan
    eth_ma  = eth_ma200.loc[:date].iloc[-1] if len(eth_ma200.loc[:date].dropna()) > 0 else np.nan

    btc_trend = (not np.isnan(btc_now)) and (not np.isnan(btc_ma)) and (btc_now > btc_ma)
    eth_trend = (not np.isnan(eth_now)) and (not np.isnan(eth_ma)) and (eth_now > eth_ma)

    weights = {}
    if btc_trend:
        weights['BTC'] = btc_eth_split
    else:
        weights['SHY_BTC'] = btc_eth_split  # BTC bucket → SHY

    if eth_trend:
        weights['ETH'] = 1.0 - btc_eth_split
    else:
        weights['SHY_ETH'] = 1.0 - btc_eth_split  # ETH bucket → SHY

    return weights, btc_trend, eth_trend

# ─────────────────────────────────────────────
# 日频P&L追踪（精确MaxDD）
# ─────────────────────────────────────────────
def run_hybrid_daily(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                     tlt_p, ief_p, def_prices, btc_p, eth_p,
                     w_crypto, btc_eth_split,
                     start='2015-01-01', end='2025-12-31', cost=0.0015):
    """
    月末调仓信号，日频净值追踪
    w_crypto: 加密货币总权重
    """
    all_dates = close_df.loc[start:end].index
    month_ends = close_df.loc[start:end].resample('ME').last().index

    portfolio = 1.0; peak = 1.0
    daily_rets = []; monthly_rets = []
    prev_weights_full = {}
    dd = 0.0; port_vol_ann = None
    spy_1m_ret = 0.0

    # 月末 → 全组合权重映射
    monthly_weights = {}

    for i, me in enumerate(month_ends):
        if i == 0:
            monthly_weights[me] = {}
            continue

        signal_date = month_ends[i-1]

        # ── 1. 股票信号 ──
        stock_w, regime, bond_type = select_stocks(sig, sectors, signal_date, prev_weights_full,
                                                    gld_p, gdx_p, tlt_p, ief_p, def_prices)
        stock_w = apply_overlays_stock(stock_w, sig, signal_date, gdxj_p, shy_p, dd, port_vol_ann, spy_1m_ret)

        # ── 2. 加密货币信号 ──
        crypto_w, btc_trend, eth_trend = get_crypto_weights(btc_p, eth_p, shy_p, signal_date, btc_eth_split)

        # ── 3. 组合权重 = (1-w_crypto)*股票 + w_crypto*加密 ──
        full_w = {}
        w_stock = 1.0 - w_crypto
        for k, v in stock_w.items():
            full_w[k] = full_w.get(k, 0.0) + v * w_stock
        for k, v in crypto_w.items():
            if k.startswith('SHY'):
                full_w['SHY'] = full_w.get('SHY', 0.0) + v * w_crypto
            elif k == 'BTC':
                full_w['BTC'] = full_w.get('BTC', 0.0) + v * w_crypto
            elif k == 'ETH':
                full_w['ETH'] = full_w.get('ETH', 0.0) + v * w_crypto

        monthly_weights[me] = full_w
        prev_weights_full = full_w

    # ── 日频P&L ──
    current_w = {}
    current_me_idx = 0
    me_list = list(monthly_weights.keys())

    for date in all_dates:
        # 找当前月份的权重
        applicable_me = None
        for me in me_list:
            if me <= date: applicable_me = me
            else: break
        if applicable_me is None or not monthly_weights.get(applicable_me):
            daily_rets.append(0.0); continue

        new_w = monthly_weights[applicable_me]

        # 月初换仓：计算交易成本
        if current_me_idx < len(me_list) - 1 and date == me_list[current_me_idx + 1]:
            tc = sum(abs(new_w.get(k, 0.0) - current_w.get(k, 0.0))
                     for k in set(new_w) | set(current_w)) * cost
            current_w = new_w.copy()
            current_me_idx += 1
            # 扣除交易成本（当天）
            daily_rets.append(-tc)
            portfolio *= (1 - tc)
            peak = max(peak, portfolio)
            dd = (portfolio - peak) / peak
            continue

        if not current_w:
            current_w = new_w.copy()

        # 计算日收益
        dr = 0.0
        for ticker, w in current_w.items():
            if w == 0: continue
            if ticker == 'SHY':
                # SHY ≈ 4%/年
                dr += w * (0.04 / 252)
                continue
            try:
                if ticker == 'BTC':
                    p_s = btc_p.loc[:date]
                    if len(p_s) < 2: continue
                    dr += w * float(p_s.iloc[-1] / p_s.iloc[-2] - 1)
                elif ticker == 'ETH':
                    p_s = eth_p.loc[:date]
                    if len(p_s) < 2: continue
                    dr += w * float(p_s.iloc[-1] / p_s.iloc[-2] - 1)
                else:
                    p_s = close_df[ticker].loc[:date]
                    if len(p_s) < 2: continue
                    dr += w * float(p_s.iloc[-1] / p_s.iloc[-2] - 1)
            except: pass

        daily_rets.append(dr)
        portfolio *= (1 + dr)
        peak = max(peak, portfolio)
        dd = (portfolio - peak) / peak

    # ── 计算指标 ──
    rets = np.array(daily_rets)
    if len(rets) < 50: return None

    years = len(rets) / 252
    cagr  = (1 + rets).prod() ** (1 / years) - 1
    max_dd = min(
        (np.maximum.accumulate(np.cumprod(1+rets)) - np.cumprod(1+rets)) /
         np.maximum.accumulate(np.cumprod(1+rets))
    ) * -1
    excess = rets - 0.04/252
    sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0

    # WF: IS 2015-2021 vs OOS 2022-2025
    def sub_metrics(s, e):
        mask = (all_dates >= s) & (all_dates <= e)
        r = rets[mask]
        if len(r) < 50: return 0.0
        ex = r - 0.04/252
        return float(np.mean(ex)/np.std(ex)*np.sqrt(252)) if np.std(ex) > 0 else 0.0
    is_s  = sub_metrics('2015-01-01', '2021-12-31')
    oos_s = sub_metrics('2022-01-01', '2025-12-31')
    wf = float(oos_s / is_s) if is_s > 0 else 0.0

    # 新Composite公式 (CAGR权重提升)
    composite_new = sharpe * 0.25 + calmar * 0.25 + min(cagr, 1.5) * 0.50

    return dict(
        cagr=cagr, max_dd=max_dd, sharpe=sharpe, calmar=calmar,
        wf=wf, is_sharpe=is_s, oos_sharpe=oos_s,
        composite_new=composite_new,
        composite_old=sharpe*0.4 + calmar*0.4 + min(cagr,1.0)*0.2,
        years=years,
    )


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    print("="*70)
    print("🐻 Hybrid v3 — v11b_final + Crypto 200dMA 趋势过滤")
    print("="*70)

    # 加载板块
    sectors_raw = json.load(open(CACHE / "sp500_sectors.json"))
    def norm(s): return s[:5] if isinstance(s, str) else 'Unknown'
    sectors = {t: norm(v) for t, v in sectors_raw.items()}

    # 加载股票
    tickers = list(sectors.keys())
    close_df = load_stocks(tickers)
    print(f"  Loaded {len(close_df.columns)} stocks")

    # 加载 ETF
    def le(name): return load_series(f"{name}.csv")
    gld_p  = le('GLD'); gdx_p = le('GDX'); gdxj_p = le('GDXJ')
    shy_p  = le('SHY'); tlt_p = le('TLT'); ief_p  = le('IEF')
    spy_p  = le('SPY')
    close_df['SPY'] = spy_p
    for e in ['GLD','GDX','GDXJ','TLT','IEF']: close_df[e] = le(e)
    def_prices = {e: le(e) for e in DEFENSIVE_ETFS}

    # 加载加密货币
    btc_p = le('BTC_USD')
    eth_p = le('ETH_USD')
    print(f"  BTC: {btc_p.index[0].date()} → {btc_p.index[-1].date()}")
    print(f"  ETH: {eth_p.index[0].date()} → {eth_p.index[-1].date()}")

    # 预计算
    sig = precompute_stock(close_df)

    # ── 基准: 纯股票 v11b_final ──
    print("\n📊 基准 v11b_final (股票, 不含Crypto):")
    base = run_hybrid_daily(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                             tlt_p, ief_p, def_prices, btc_p, eth_p,
                             w_crypto=0.0, btc_eth_split=0.6)
    if base:
        print(f"  CAGR={base['cagr']:.1%} MaxDD={base['max_dd']:.1%} "
              f"Sharpe={base['sharpe']:.2f} Calmar={base['calmar']:.2f} WF={base['wf']:.3f}")
        print(f"  Composite(new)={base['composite_new']:.3f}  IS={base['is_sharpe']:.2f} OOS={base['oos_sharpe']:.2f}")

    # ── 扫描加密货币配置 ──
    w_crypto_vals    = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    btc_eth_splits   = [0.60, 0.70, 0.80]

    print("\n🔍 扫描 w_crypto × BTC_ETH_SPLIT...")
    print(f"{'w_c':>5} {'btc':>5} | {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'Calmar':>7} {'WF':>6} {'IS':>6} {'OOS':>6} {'CompNEW':>8}")
    print("-"*80)

    results = []
    for wc in w_crypto_vals:
        for split in btc_eth_splits:
            m = run_hybrid_daily(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p,
                                  tlt_p, ief_p, def_prices, btc_p, eth_p,
                                  w_crypto=wc, btc_eth_split=split)
            if m is None: continue
            flag = "🏆" if m['wf'] >= 0.60 and m['cagr'] >= 0.40 else \
                   ("✅" if m['wf'] >= 0.60 else "  ")
            print(f"  {flag} {wc:.0%} {split:.0%} | "
                  f"{m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
                  f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f} "
                  f"{m['wf']:>6.3f} {m['is_sharpe']:>6.2f} {m['oos_sharpe']:>6.2f} "
                  f"{m['composite_new']:>8.3f}")
            results.append(dict(w_crypto=wc, btc_eth_split=split, **m))

    # ── 汇总 ──
    print("\n" + "="*70)
    print("📊 TOP 10 by Composite(new) [CAGR权重×0.50]:")
    results.sort(key=lambda x: x['composite_new'], reverse=True)
    for r in results[:10]:
        flag = "🏆" if r['wf'] >= 0.60 and r['cagr'] >= 0.40 else \
               ("✅" if r['wf'] >= 0.60 else "  ")
        print(f"  {flag} w={r['w_crypto']:.0%} btc={r['btc_eth_split']:.0%} → "
              f"CAGR={r['cagr']:.1%} WF={r['wf']:.3f} Sharpe={r['sharpe']:.2f} "
              f"MaxDD={r['max_dd']:.1%} CompNEW={r['composite_new']:.3f}")

    print("\n📊 TOP 5 by WF (WF>=0.60):")
    wf_results = [r for r in results if r['wf'] >= 0.60]
    wf_results.sort(key=lambda x: x['composite_new'], reverse=True)
    if wf_results:
        for r in wf_results[:5]:
            print(f"  ✅ w={r['w_crypto']:.0%} btc={r['btc_eth_split']:.0%} → "
                  f"CAGR={r['cagr']:.1%} WF={r['wf']:.3f} Sharpe={r['sharpe']:.2f} "
                  f"MaxDD={r['max_dd']:.1%} CompNEW={r['composite_new']:.3f}")
    else:
        print("  ❌ 无 WF>=0.60 的配置 — 需要更强的趋势过滤")

    best = results[0] if results else None
    if best:
        print(f"\n🏆 综合最优: w_crypto={best['w_crypto']:.0%} BTC_split={best['btc_eth_split']:.0%}")
        print(f"   CAGR={best['cagr']:.1%} MaxDD={best['max_dd']:.1%} "
              f"Sharpe={best['sharpe']:.2f} WF={best['wf']:.3f}")

    # 保存
    out = {
        'strategy': 'Hybrid v3: v11b_final + Crypto 200dMA Trend Filter',
        'baseline': base,
        'results': results,
        'best_composite': best,
        'best_wf': wf_results[0] if wf_results else None,
        'meta': {'date': '2026-02-22', 'author': '代码熊 🐻',
                 'composite_formula': 'Sharpe*0.25 + Calmar*0.25 + CAGR*0.50'}
    }
    out_path = Path(__file__).with_name('hybrid_v3_results.json')
    json.dump(out, open(out_path, 'w'), indent=2, default=float)
    print(f"\n💾 结果保存 → {out_path.name}")


if __name__ == '__main__':
    main()
