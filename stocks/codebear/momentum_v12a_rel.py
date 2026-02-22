#!/usr/bin/env python3
"""
动量轮动 v12a — 行业相对动量 (Industry-Relative Momentum)
代码熊 🐻 | 2026-02-22

核心创新:
  在 v11b_final 基础上，引入行业相对动量得分：
    score = raw_mom - REL_W * sector_etf_mom
  
  其中:
    raw_mom      = 当前绝对动量得分 (0.20*r1m + 0.50*r3m + 0.20*r6m + 0.10*r12m)
    sector_etf   = 对应行业 ETF（XLK/XLF/XLC/XLI/XLE/XLB/XLV/XLU/XLRE）
    REL_W ∈ [0, 1.0] = 相对化权重（0=纯绝对，1=纯相对）

理论依据:
  - 捕捉个股相对行业的超额动量（真正的 Alpha）
  - 减少对市场整体趋势的依赖（改善 OOS 稳健性）
  - 行业轮动时，相对强者更可能持续跑赢
  - 预期效果: WF 从 0.74 → 0.76+，Composite 保持 >2.15

v11b_final 基准: Composite 2.190, WF 0.74, MaxDD -9.4% (月频)

行业 ETF 映射:
  Techn → XLK   CommS → XLC   Indus → XLI
  Finan → XLF   Energ → XLE   Mater → XLB
  Healt → XLV   Utili → XLU   RealE → XLRE
  Consu → XLY (Consumer Disc) / XLP (Consumer Staples) — 混合用 XLY

严格无前瞻: 所有信号使用历史收盘价
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE  = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data_cache"
STOCK_CACHE = CACHE / "stocks"

# v11b_final 最优参数
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
TLT_BEAR_FRAC = 0.25; IEF_BEAR_FRAC = 0.20; BOND_MOM_LB = 126
DEFENSIVE_ETFS = ['XLV', 'XLP', 'XLU']
DEFENSIVE_FRAC = 0.12; DEFENSIVE_EACH = DEFENSIVE_FRAC / 3
SPY_SOFT_HI_THRESH = -0.07; SPY_SOFT_HI_FRAC = 0.08

# v12a 新参数
REL_W = 0.30  # 行业相对化权重 (sweep 中会覆盖)

# 行业 ETF 映射
SECTOR_ETF_MAP = {
    'Techn': 'XLK',
    'CommS': 'XLC',
    'Indus': 'XLI',
    'Finan': 'XLF',
    'Energ': 'XLE',
    'Mater': 'XLB',
    'Healt': 'XLV',   # 注意: XLV 也是 DEFENSIVE_ETF, 但只用于相对计算
    'Utili': 'XLU',
    'RealE': 'XLRE',
    'Consu': 'XLY',   # Consumer Disc (混合用)
}

HEDGE_KEYS = {'GLD', 'GDX', 'GDXJ', 'TLT', 'IEF', 'XLV', 'XLP', 'XLU'}


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


def precompute(close_df, sector_etf_close, ticker_sector):
    """预计算动量信号（含行业相对动量）"""
    r1  = close_df / close_df.shift(22)  - 1
    r3  = close_df / close_df.shift(63)  - 1
    r6  = close_df / close_df.shift(126) - 1
    r12 = close_df / close_df.shift(252) - 1

    # 行业 ETF 动量
    etf_r1  = sector_etf_close / sector_etf_close.shift(22)  - 1
    etf_r3  = sector_etf_close / sector_etf_close.shift(63)  - 1
    etf_r6  = sector_etf_close / sector_etf_close.shift(126) - 1
    etf_r12 = sector_etf_close / sector_etf_close.shift(252) - 1

    r52w_hi = close_df.rolling(252).max()
    log_r   = np.log(close_df / close_df.shift(1))
    vol5    = log_r.rolling(5).std() * np.sqrt(252)
    vol30   = log_r.rolling(30).std() * np.sqrt(252)
    spy     = close_df['SPY'] if 'SPY' in close_df.columns else None
    s200    = spy.rolling(200).mean() if spy is not None else None
    sma50   = close_df.rolling(50).mean()

    return dict(r1=r1, r3=r3, r6=r6, r12=r12, r52w_hi=r52w_hi,
                vol5=vol5, vol30=vol30, spy=spy, s200=s200,
                sma50=sma50, close=close_df,
                etf_r1=etf_r1, etf_r3=etf_r3, etf_r6=etf_r6, etf_r12=etf_r12,
                ticker_sector=ticker_sector, sector_etf_close=sector_etf_close)


def compute_breadth(sig, date):
    close = sig['close'].loc[:date].dropna(how='all')
    sma50 = sig['sma50'].loc[:date].dropna(how='all')
    if len(close) < 50: return 1.0
    lc = close.iloc[-1]; ls = sma50.iloc[-1]
    mask = (lc > ls).dropna()
    return float(mask.sum() / len(mask)) if len(mask) > 0 else 1.0


def get_three_regime(sig, date):
    """三态制度: bull_hi / soft_bull / bear"""
    breadth = compute_breadth(sig, date)
    spy  = sig['spy']
    s200 = sig['s200']
    if spy is None or s200 is None:
        return 'bull_hi', breadth
    spy_now = spy.loc[:date].iloc[-1]
    s200_now = s200.loc[:date].iloc[-1]
    bear = (spy_now < s200_now) and (breadth < BREADTH_NARROW)
    if bear:
        return 'bear', breadth
    elif breadth >= BREADTH_CONC:
        return 'bull_hi', breadth
    else:
        return 'soft_bull', breadth


def mom_score_raw(sig, ticker, date):
    """原始绝对动量得分"""
    def safe(df, col):
        try:
            v = df[col].loc[:date]
            return float(v.iloc[-1]) if len(v) > 0 else np.nan
        except: return np.nan
    r1  = safe(sig['r1'],  ticker)
    r3  = safe(sig['r3'],  ticker)
    r6  = safe(sig['r6'],  ticker)
    r12 = safe(sig['r12'], ticker)
    if any(np.isnan(x) for x in [r1, r3, r6, r12]): return np.nan
    return MOM_W[0]*r1 + MOM_W[1]*r3 + MOM_W[2]*r6 + MOM_W[3]*r12


def sector_etf_mom_score(sig, sector_name, date):
    """行业 ETF 动量得分"""
    etf = SECTOR_ETF_MAP.get(sector_name)
    if etf is None: return 0.0
    def safe(df, col):
        try:
            v = df[col].loc[:date]
            return float(v.iloc[-1]) if len(v) > 0 else np.nan
        except: return np.nan
    r1  = safe(sig['etf_r1'],  etf)
    r3  = safe(sig['etf_r3'],  etf)
    r6  = safe(sig['etf_r6'],  etf)
    r12 = safe(sig['etf_r12'], etf)
    if any(np.isnan(x) for x in [r1, r3, r6, r12]): return 0.0
    return MOM_W[0]*r1 + MOM_W[1]*r3 + MOM_W[2]*r6 + MOM_W[3]*r12


def mom_score(sig, ticker, date, rel_w=None):
    """动量得分 = 绝对动量 - REL_W * 行业ETF动量"""
    if rel_w is None: rel_w = REL_W
    raw = mom_score_raw(sig, ticker, date)
    if np.isnan(raw): return np.nan
    if rel_w == 0.0: return raw
    sector = sig['ticker_sector'].get(ticker)
    etf_score = sector_etf_mom_score(sig, sector, date) if sector else 0.0
    return raw - rel_w * etf_score


def hi52_ratio(sig, ticker, date):
    try:
        hi = sig['r52w_hi'][ticker].loc[:date].iloc[-1]
        c  = sig['close'][ticker].loc[:date].iloc[-1]
        return float(c / hi) if hi > 0 else 0.0
    except: return 0.0


def select(sig, sectors, date, prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices, rel_w=None):
    """选股逻辑（三制度框架 + 行业相对动量）"""
    regime, breadth = get_three_regime(sig, date)
    tickers = [t for t in sig['close'].columns if t not in HEDGE_KEYS and t != 'SPY']
    weights = {}
    bond_type = 'none'

    if regime == 'bear':
        # 债券动量适应性选择
        tlt_mom = float(tlt_p.loc[:date].iloc[-1] / tlt_p.loc[:date].iloc[-BOND_MOM_LB] - 1) \
            if len(tlt_p.loc[:date]) >= BOND_MOM_LB else 0.0
        ief_mom = float(ief_p.loc[:date].iloc[-1] / ief_p.loc[:date].iloc[-BOND_MOM_LB] - 1) \
            if len(ief_p.loc[:date]) >= BOND_MOM_LB else 0.0

        if tlt_mom > ief_mom and tlt_mom > 0:
            weights['TLT'] = TLT_BEAR_FRAC; bond_type = 'TLT'
        elif ief_mom > 0:
            weights['IEF'] = IEF_BEAR_FRAC; bond_type = 'IEF'

        # GLD 竞争
        gld_s = mom_score_raw(sig, 'GLD', date)
        gld_avg = float(gld_p.loc[:date].iloc[-1] / gld_p.loc[:date].iloc[-252] - 1) \
            if len(gld_p.loc[:date]) >= 252 else 0.0
        if not np.isnan(gld_s) and gld_avg > GLD_AVG_THRESH:
            weights['GLD'] = GLD_COMPETE_FRAC

        bond_alloc = sum(weights.values())
        stock_cap = max(0.0, 1.0 - bond_alloc)

        # Bear 期股票选择（不用相对动量，熊市时行业中性更重要）
        secs = sorted(set(sectors.get(t, 'Unknown') for t in tickers))
        scored = []
        for t in tickers:
            s = mom_score_raw(sig, t, date)  # bear 期用绝对动量
            if not np.isnan(s): scored.append((t, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in scored[:BEAR_SPS]]
        for t in top: weights[t] = stock_cap / len(top) if top else 0.0

    else:
        # Bull: 分行业选股（使用相对动量）
        n_secs = N_BULL_SECS_HI if regime == 'bull_hi' else N_BULL_SECS
        sec_scores = {}
        for t in tickers:
            sec = sectors.get(t, 'Unknown')
            if sec not in sec_scores: sec_scores[sec] = []
            s = mom_score(sig, t, date, rel_w)  # 使用行业相对动量
            if not np.isnan(s):
                hi = hi52_ratio(sig, t, date)
                if hi >= HI52_FRAC:
                    if t in (prev_hold or {}): s += CONT_BONUS
                    sec_scores[sec].append((t, s))

        # 行业平均分排名
        sec_avg = {}
        for sec, lst in sec_scores.items():
            if lst: sec_avg[sec] = np.mean([x[1] for x in lst])

        # GLD 行业竞争
        gld_s = mom_score_raw(sig, 'GLD', date)
        gld_avg = float(gld_p.loc[:date].iloc[-1] / gld_p.loc[:date].iloc[-252] - 1) \
            if len(gld_p.loc[:date]) >= 252 else 0.0
        if not np.isnan(gld_s) and gld_avg > GLD_AVG_THRESH:
            sec_avg['__GLD__'] = gld_s

        # GDX 竞争
        gdx_s = mom_score_raw(sig, 'GDX', date)
        gdx_avg = float(gdx_p.loc[:date].iloc[-1] / gdx_p.loc[:date].iloc[-252] - 1) \
            if len(gdx_p.loc[:date]) >= 252 else 0.0
        if not np.isnan(gdx_s) and gdx_avg > GDX_AVG_THRESH:
            sec_avg['__GDX__'] = gdx_s * GDX_COMPETE_FRAC / GLD_COMPETE_FRAC

        top_secs = sorted(sec_avg.items(), key=lambda x: x[1], reverse=True)[:n_secs]
        stock_w = 1.0
        for sec, _ in top_secs:
            if sec == '__GLD__':
                weights['GLD'] = GLD_COMPETE_FRAC; stock_w -= GLD_COMPETE_FRAC
            elif sec == '__GDX__':
                weights['GDX'] = GDX_COMPETE_FRAC; stock_w -= GDX_COMPETE_FRAC
            else:
                lst = sorted(sec_scores.get(sec, []), key=lambda x: x[1], reverse=True)[:BULL_SPS]
                for t, _ in lst: weights[t] = 0.0

        real_stocks = [t for t in weights if t not in HEDGE_KEYS]
        w_per = stock_w / len(real_stocks) if real_stocks else 0.0
        for t in real_stocks: weights[t] = w_per

        # v10b: 软牛期防御行业桥接
        if regime == 'soft_bull':
            def_avail = []
            for etf in DEFENSIVE_ETFS:
                if etf in def_prices and len(def_prices[etf].loc[:date]) > 1:
                    def_avail.append(etf)
            if def_avail:
                scale = 1.0 - DEFENSIVE_FRAC
                for k in list(weights.keys()): weights[k] *= scale
                for etf in def_avail: weights[etf] = DEFENSIVE_EACH

    return weights, regime, bond_type


def apply_overlays(weights, spy_vol, dd, port_vol_ann, spy_1m_ret, sig, date, gdxj_p, shy_p):
    """叠加层: GDXJ + DD保护 + Vol目标 + SPY软对冲"""
    shy_boost = 0.0

    # GDXJ 波动率对冲
    if spy_vol is not None and not np.isnan(spy_vol):
        if spy_vol > GDXJ_VOL_HI_THRESH:
            frac = GDXJ_VOL_HI_FRAC
        elif spy_vol > GDXJ_VOL_LO_THRESH:
            frac = GDXJ_VOL_LO_FRAC
        else:
            frac = 0.0
        if frac > 0.0 and 'GDXJ' not in weights:
            scale = 1.0 - frac
            for k in list(weights.keys()): weights[k] *= scale
            weights['GDXJ'] = frac

    # DD 保护
    if dd is not None and dd < -0.08:
        for thr in sorted(DD_PARAMS.keys()):
            if dd <= thr:
                shy_frac = DD_PARAMS[thr]
                existing_shy = weights.pop('SHY', 0.0)
                new_shy = max(existing_shy, shy_frac)
                boost = new_shy - existing_shy
                for k in list(weights.keys()):
                    if k != 'SHY': weights[k] *= (1.0 - boost)
                weights['SHY'] = new_shy
                shy_boost = boost
                break

    # 波动率目标化
    if port_vol_ann is not None and port_vol_ann > VOL_TARGET_ANN:
        scale = VOL_TARGET_ANN / port_vol_ann
        shy_add = 1.0 - scale
        for k in list(weights.keys()): weights[k] *= scale
        weights['SHY'] = weights.get('SHY', 0.0) + shy_add

    # v9m: SPY 软对冲 (月跌>7%)
    if spy_1m_ret is not None and spy_1m_ret < SPY_SOFT_HI_THRESH:
        if 'GLD' not in weights:
            scale = 1.0 - SPY_SOFT_HI_FRAC
            for k in list(weights.keys()): weights[k] *= scale
            weights['GLD'] = SPY_SOFT_HI_FRAC

    return weights, shy_boost


def run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                 start, end, cost, rel_w=None):
    if rel_w is None: rel_w = REL_W
    dates = close_df.loc[start:end].resample('ME').last().index
    portfolio = 1.0
    peak = 1.0
    returns = []
    prev_hold = {}
    dd = 0.0
    port_vol_ann = None
    monthly_rets = []
    weights_hist = []

    for i, date in enumerate(dates):
        if i == 0:
            weights_hist.append({}); returns.append(0.0); continue

        weights, regime, bond_type = select(sig, sectors, dates[i-1], prev_hold, gld_p, gdx_p, tlt_p, ief_p, def_prices, rel_w)

        # SPY 月收益（信号）
        try:
            spy_vals = sig['spy'].loc[:dates[i-1]]
            spy_prev = float(spy_vals.iloc[-1])
            spy_prev_prev = float(sig['spy'].loc[:dates[i-2]].iloc[-1]) if i > 1 else spy_prev
            spy_1m_ret = (spy_prev - spy_prev_prev) / spy_prev_prev if spy_prev_prev > 0 else 0.0
        except: spy_1m_ret = 0.0

        # SPY 短期波动率
        try:
            spy_log = np.log(sig['spy'] / sig['spy'].shift(1)).loc[:dates[i-1]].dropna()
            spy_vol = float(spy_log.iloc[-63:].std() * np.sqrt(252)) if len(spy_log) >= 10 else None
        except: spy_vol = None

        # 组合波动率
        if len(monthly_rets) >= VOL_LOOKBACK:
            port_vol_ann = float(np.std(monthly_rets[-VOL_LOOKBACK:]) * np.sqrt(12))

        weights, _ = apply_overlays(weights, spy_vol, dd, port_vol_ann, spy_1m_ret, sig, dates[i-1], gdxj_p, shy_p)

        # 计算月收益
        prev_weights = weights_hist[-1] if weights_hist else {}
        pret = 0.0
        for ticker, w in weights.items():
            try:
                p_now  = float(close_df[ticker].loc[:date].iloc[-1])
                p_prev = float(close_df[ticker].loc[:dates[i-1]].iloc[-1])
                if p_prev > 0: pret += w * (p_now - p_prev) / p_prev
            except: pass

        # 交易成本
        tc = sum(abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in
                 set(weights) | set(prev_weights)) * cost
        pret -= tc

        portfolio *= (1 + pret)
        peak = max(peak, portfolio)
        dd = (portfolio - peak) / peak
        monthly_rets.append(pret)
        returns.append(pret)
        weights_hist.append(weights.copy())
        prev_hold = weights.copy()

    returns = np.array(returns[1:])
    if len(returns) < 12:
        return {'cagr': 0, 'max_dd': 0, 'sharpe': 0, 'calmar': 0}

    n_years = len(returns) / 12
    cagr = (1 + returns).prod() ** (1 / n_years) - 1
    max_dd = min((np.maximum.accumulate(np.cumprod(1 + returns)) - np.cumprod(1 + returns))
                 / np.maximum.accumulate(np.cumprod(1 + returns))) * -1
    excess = returns - 0.04 / 12
    sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(12)) if np.std(excess) > 0 else 0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0
    return {'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe, 'calmar': calmar}


def get_metrics(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices, cost, rel_w=None):
    full = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                        '2015-01-01', '2025-12-31', cost, rel_w)
    is_  = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                        '2015-01-01', '2021-12-31', cost, rel_w)
    oos  = run_backtest(close_df, sig, sectors, gld_p, gdx_p, gdxj_p, shy_p, tlt_p, ief_p, def_prices,
                        '2022-01-01', '2025-12-31', cost, rel_w)
    wf = float(oos['sharpe'] / is_['sharpe']) if is_['sharpe'] > 0 else 0.0
    composite = full['sharpe'] * 0.4 + full['calmar'] * 0.4 + min(full['cagr'], 1.0) * 0.2
    return dict(full=full, is_=is_, oos=oos, wf=wf, composite=composite)


def main():
    import itertools

    print("="*70)
    print("🐻 v12a 行业相对动量 — REL_W 扫描")
    print("="*70)

    # 加载板块信息
    sectors_raw = json.load(open(CACHE / "sp500_sectors.json"))
    # sp500_sectors.json 格式: {ticker: sector_string}
    ticker_sector = {t: v if isinstance(v, str) else list(v.keys())[0]
                     for t, v in sectors_raw.items()}

    # 加载所有数据
    tickers = list(ticker_sector.keys())
    close_df = load_stocks(tickers)
    print(f"  Loaded {len(close_df.columns)} stock tickers")

    # 加载 ETF 和辅助数据
    def load_etf(name):
        fp = CACHE / f"{name}.csv"
        df = load_csv(fp)
        return df['Close'].dropna()

    gld_p  = load_etf('GLD')
    gdx_p  = load_etf('GDX')
    gdxj_p = load_etf('GDXJ')
    shy_p  = load_etf('SHY')
    tlt_p  = load_etf('TLT')
    ief_p  = load_etf('IEF')
    spy_p  = load_etf('SPY')

    # 防御 ETF 价格
    def_prices = {}
    for etf in DEFENSIVE_ETFS:
        try: def_prices[etf] = load_etf(etf)
        except: pass

    # 行业 ETF 价格（用于相对动量计算）
    sector_etf_close = pd.DataFrame()
    for sector, etf in SECTOR_ETF_MAP.items():
        try:
            series = load_etf(etf)
            sector_etf_close[etf] = series
        except: pass
    print(f"  Loaded {len(sector_etf_close.columns)} sector ETFs: {list(sector_etf_close.columns)}")

    # 合并 SPY 到 close_df
    close_df['SPY'] = spy_p
    for etf in ['GLD', 'GDX', 'GDXJ', 'TLT', 'IEF']:
        try: close_df[etf] = load_etf(etf)
        except: pass

    # 预计算
    sig = precompute(close_df, sector_etf_close, ticker_sector)
    sectors = ticker_sector

    # 修复 sectors 格式: 提取行业名
    def norm_sector(s):
        if isinstance(s, str): return s[:5]
        return 'Unknown'
    sectors_norm = {t: norm_sector(v) for t, v in sectors.items()}

    # REL_W 扫描
    rel_w_vals = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
    results = []

    print(f"\n🔍 扫描 REL_W ({len(rel_w_vals)} 个配置)...")
    for rw in rel_w_vals:
        m = get_metrics(close_df, sig, sectors_norm, gld_p, gdx_p, gdxj_p, shy_p,
                        tlt_p, ief_p, def_prices, 0.0015, rw)
        f = m['full']
        tag = "🏆" if m['composite'] > 2.19 else ("✅" if m['wf'] > 0.75 else "  ")
        print(f"  {tag} REL_W={rw:.2f}: Comp={m['composite']:.4f} Sharpe={f['sharpe']:.2f} "
              f"WF={m['wf']:.3f} CAGR={f['cagr']:.1%} MaxDD={f['max_dd']:.1%} "
              f"Calmar={f['calmar']:.2f}")
        print(f"         IS={m['is_']['sharpe']:.2f} OOS={m['oos']['sharpe']:.2f}")
        results.append({
            'rel_w': rw,
            'composite': m['composite'],
            'sharpe': f['sharpe'],
            'calmar': f['calmar'],
            'cagr': f['cagr'],
            'max_dd': f['max_dd'],
            'wf': m['wf'],
            'is_sharpe': m['is_']['sharpe'],
            'oos_sharpe': m['oos']['sharpe'],
        })

    results.sort(key=lambda x: x['composite'], reverse=True)
    print("\n" + "="*70)
    print("📊 最终排名 (by Composite):")
    for r in results:
        flag = "🏆" if r['composite'] > 2.19 else ("✅" if r['wf'] > 0.75 else "  ")
        print(f"  {flag} REL_W={r['rel_w']:.2f} → Comp={r['composite']:.4f} WF={r['wf']:.3f} "
              f"Sharpe={r['sharpe']:.2f} MaxDD={r['max_dd']:.1%}")

    best = results[0]
    print(f"\n🏆 最佳 REL_W={best['rel_w']:.2f}: Composite {best['composite']:.4f}, WF {best['wf']:.3f}")

    if best['composite'] > 2.19:
        print("🚀🚀 新冠军！超越 v11b_final (2.190)！")
    elif best['composite'] > 2.18 and best['wf'] > 0.75:
        print("🚀 WF 提升！Composite+WF 双提升！")

    # 保存结果
    out = {
        'strategy': 'v12a Industry-Relative Momentum Sweep',
        'baseline': {'composite': 2.190, 'wf': 0.74},
        'results': results,
        'best': best,
    }
    out_path = Path(__file__).with_name('momentum_v12a_results.json')
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f"\n💾 结果保存到 {out_path.name}")


if __name__ == '__main__':
    main()
